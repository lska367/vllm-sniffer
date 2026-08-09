"""Worker hook tests: forward timing and the greedy argmax-margin probe.

The sampler margin hook runs real torch logic (topk, masking, reductions),
so these tests verify both behavior invariance (sampling results unchanged)
and the flip-zone detection that underpins the temperature=0
nondeterminism diagnosis.
"""

from types import SimpleNamespace

import torch
from conftest import fake_vllm_module

from vllm_sniffer.config import get_config
from vllm_sniffer.hooks.worker import install_worker_hooks


class FakeSampler:
    @staticmethod
    def greedy_sample(logits):
        return logits.argmax(dim=-1).view(-1)


def _install_sampler(monkeypatch):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.sample.sampler")
    mod.Sampler = FakeSampler
    install_worker_hooks()
    from vllm.v1.sample.sampler import Sampler

    return Sampler


def _logits_with_one_flip():
    """2 rows; row 0 has a tiny 5e-4 gap (flip zone), row 1 a clear gap."""
    logits = torch.full((2, 8), -10.0)
    logits[0, 3] = 1e-3   # winner by a hair
    logits[0, 5] = 5e-4   # gap 5e-4 < eps=1e-3 -> flip zone
    logits[1, 1] = 1.0
    logits[1, 7] = -2.0   # gap 3.0 -> safe
    return logits


def test_greedy_sample_behavior_invariant(monkeypatch):
    torch.manual_seed(0)
    logits = torch.randn(16, 64)
    expected = FakeSampler.greedy_sample(logits)
    Sampler = _install_sampler(monkeypatch)
    got = Sampler.greedy_sample(logits)
    assert torch.equal(got, expected)


def test_flip_zone_detected(monkeypatch, read_events):
    Sampler = _install_sampler(monkeypatch)
    logits = _logits_with_one_flip()
    sampled = Sampler.greedy_sample(logits)
    assert sampled.tolist() == [3, 1]  # argmax unchanged by the probe

    events = read_events()
    flips = [e for e in events if e["type"] == "sample_flip"]
    assert len(flips) == 1
    assert flips[0]["data"]["top1"] == 3
    assert flips[0]["data"]["top2"] == 5
    assert abs(flips[0]["data"]["margin"] - 5e-4) < 1e-9

    stats = [e for e in events if e["type"] == "sample_stats"][0]
    assert stats["data"]["n_greedy"] == 2
    assert stats["data"]["n_flips"] == 1
    assert abs(stats["data"]["margin_min"] - 5e-4) < 1e-9
    assert stats["data"]["margin_max"] == 3.0


def test_no_flip_no_event(monkeypatch, read_events):
    Sampler = _install_sampler(monkeypatch)
    logits = torch.zeros(4, 8)
    logits[:, 0] = 2.0  # every row: clear winner
    logits[2, 3] = 1.9  # still gap 0.1 > eps
    Sampler.greedy_sample(logits)

    events = read_events()
    flips = [e for e in events if e["type"] == "sample_flip"]
    assert flips == []
    stats = [e for e in events if e["type"] == "sample_stats"][0]
    assert stats["data"]["n_flips"] == 0


def test_margin_disabled(monkeypatch, read_events):
    monkeypatch.setenv("VLLM_SNIFFER_MARGIN", "0")
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    try:
        Sampler = _install_sampler(monkeypatch)
        logits = _logits_with_one_flip()
        sampled = Sampler.greedy_sample(logits)
        assert sampled.tolist() == [3, 1]
        assert read_events() == []  # no probe, no events
    finally:
        get_config.cache_clear()


def test_nan_rows_do_not_crash(monkeypatch, read_events):
    Sampler = _install_sampler(monkeypatch)
    logits = torch.full((2, 8), float("nan"))
    logits[1, :] = 0.0
    logits[1, 2] = 1.0
    out = Sampler.greedy_sample(logits)
    assert out.tolist()[1] == 2  # valid row still sampled correctly
    assert [e["type"] for e in read_events()].count("sample_flip") == 0


def test_install_twice_is_idempotent(monkeypatch, read_events):
    """Double installation must not double-wrap (would double-emit)."""
    mod = fake_vllm_module(monkeypatch, "vllm.v1.sample.sampler")
    mod.Sampler = FakeSampler
    install_worker_hooks()
    install_worker_hooks()  # second install: no-op
    from vllm.v1.sample.sampler import Sampler

    Sampler.greedy_sample(_logits_with_one_flip())
    events = read_events()
    assert len([e for e in events if e["type"] == "sample_flip"]) == 1
    assert len([e for e in events if e["type"] == "sample_stats"]) == 1


def test_execute_model_forward_event(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.worker.gpu_model_runner")

    class FakeModelRunner:
        def __init__(self):
            self.input_batch = SimpleNamespace(num_tokens=128, num_seqs=4)

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "result"

    mod.GPUModelRunner = FakeModelRunner
    install_worker_hooks()

    sched_out = SimpleNamespace(total_num_scheduled_tokens=128)
    r = FakeModelRunner().execute_model(sched_out)
    assert r == "result"

    events = read_events()
    fwd = [e for e in events if e["type"] == "forward"][0]
    assert fwd["sampled"] is True
    assert fwd["data"]["num_tokens"] == 128
    assert fwd["data"]["num_seqs"] == 4
    assert fwd["data"]["total_num_scheduled_tokens"] == 128
    assert fwd["data"]["dur_ns"] >= 0


def test_forward_num_seqs_from_scheduler_output(monkeypatch, read_events):
    """Batch composition falls back to scheduler_output.num_scheduled_tokens
    (the real-vLLM path: input_batch arrays reset after execution)."""
    mod = fake_vllm_module(monkeypatch, "vllm.v1.worker.gpu_model_runner")

    class FakeModelRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "result"

    mod.GPUModelRunner = FakeModelRunner
    install_worker_hooks()

    sched_out = SimpleNamespace(
        total_num_scheduled_tokens=100,
        num_scheduled_tokens={"r1": 60, "r2": 40},
    )
    FakeModelRunner().execute_model(sched_out)

    fwd = [e for e in read_events() if e["type"] == "forward"][0]
    assert fwd["data"]["total_num_scheduled_tokens"] == 100
    assert fwd["data"]["num_seqs"] == 2


def test_worker_events_carry_current_step(monkeypatch, read_events):
    """forward / sample_flip / sample_stats carry the engine step index
    (shared process-local counter; TP=1 runs model in the engine process)."""
    from vllm_sniffer.core.step_counter import next_step

    mod = fake_vllm_module(monkeypatch, "vllm.v1.worker.gpu_model_runner")

    class FakeModelRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "result"

    mod.GPUModelRunner = FakeModelRunner
    install_worker_hooks()

    next_step()  # engine iteration 1 in progress
    sched_out = SimpleNamespace(
        total_num_scheduled_tokens=19,
        num_scheduled_tokens={"r1": 19},
    )
    FakeModelRunner().execute_model(sched_out)
    events = read_events()
    fwd = [e for e in events if e["type"] == "forward"][0]
    assert fwd["step"] == 1
    # sampler events (same iteration)
    Sampler = _install_sampler(monkeypatch)
    Sampler.greedy_sample(_logits_with_one_flip())
    events = read_events()
    assert all(e["step"] == 1 for e in events if e["type"] in
               ("sample_flip", "sample_stats"))


def test_execute_model_missing_batch_graceful(monkeypatch, read_events):
    mod = fake_vllm_module(monkeypatch, "vllm.v1.worker.gpu_model_runner")

    class FakeModelRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "result"

    mod.GPUModelRunner = FakeModelRunner
    install_worker_hooks()

    sched_out = SimpleNamespace(total_num_scheduled_tokens=None)
    assert FakeModelRunner().execute_model(sched_out) == "result"
    events = read_events()
    assert [e["type"] for e in events] == ["forward"]


# ----------------------------------------------------------------------
# logits bit-level fingerprint (deep-dive mode, P1-1)
# ----------------------------------------------------------------------
def _enable_logits_fp(monkeypatch, rows=8):
    monkeypatch.setenv("VLLM_SNIFFER_LOGITS_FP", "1")
    monkeypatch.setenv("VLLM_SNIFFER_LOGITS_FP_ROWS", str(rows))
    get_config.cache_clear()


def _bits_of_row(fp_event, row):
    for r in fp_event["data"]["rows"]:
        if r["row"] == row:
            return r
    raise AssertionError(f"row {row} not fingerprinted")


def _expected_bit_counts(x: torch.Tensor):
    """Reference per-bit set counts computed with pure Python."""
    xi = x.flatten().view(torch.int32).tolist()
    nbits = 32
    counts = [0] * nbits
    for v in xi:
        u = v & 0xFFFFFFFF  # unsigned semantics
        for b in range(nbits):
            counts[b] += (u >> b) & 1
    return counts


def test_logits_fp_off_by_default(monkeypatch, read_events):
    """Deep-dive mode is OFF by default: no logits_fp events at all."""
    Sampler = _install_sampler(monkeypatch)
    logits = torch.randn(4, 16)
    Sampler.greedy_sample(logits)
    types = [e["type"] for e in read_events()]
    assert "logits_fp" not in types
    assert "sample_stats" in types  # margin probe still on


def test_logits_fp_event_exact_bits(monkeypatch, read_events):
    """fp32 fingerprints are exact per-bit set counts."""
    _enable_logits_fp(monkeypatch, rows=4)
    Sampler = _install_sampler(monkeypatch)
    # 2 rows x 4 logits, bit patterns chosen to hit sign/exponent/mantissa.
    x = torch.tensor(
        [
            [1.0, -10.0, 0.5, 1e-30],   # row 0
            [3.14159, -2.0, 1e20, 0.0],  # row 1
        ],
        dtype=torch.float32,
    )
    out = Sampler.greedy_sample(x)
    assert out.tolist() == [0, 2]  # behavior invariant

    events = read_events()
    fp = [e for e in events if e["type"] == "logits_fp"]
    assert len(fp) == 1
    data = fp[0]["data"]
    assert data["n_rows"] == 2
    assert data["dtype"] == "torch.float32"
    assert data["sampled_rows"] == 2
    assert [r["row"] for r in data["rows"]] == [0, 1]
    assert [r["top1"] for r in data["rows"]] == [0, 2]
    for row in (0, 1):
        assert _bits_of_row(fp[0], row)["bits"] == _expected_bit_counts(x[row:row + 1])


def test_logits_fp_bit_counts_match_known_pattern(monkeypatch, read_events):
    """1.0 = 0x3F800000: exponent 127 -> bits 23..29 set, nothing else."""
    _enable_logits_fp(monkeypatch, rows=2)
    Sampler = _install_sampler(monkeypatch)
    x = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    Sampler.greedy_sample(x)
    fp = [e for e in read_events() if e["type"] == "logits_fp"][0]
    bits = _bits_of_row(fp, 0)["bits"]
    assert bits[30] == 0  # exponent 127 has bit 30 clear
    assert bits[23] == 3
    assert bits[29] == 3
    assert sum(bits) == 7 * 3  # bits 23..29, three logits


def test_logits_fp_strided_rows(monkeypatch, read_events):
    """More rows than logits_fp_rows -> strided sampling, evenly spread."""
    _enable_logits_fp(monkeypatch, rows=3)
    Sampler = _install_sampler(monkeypatch)
    x = torch.full((8, 4), -10.0)
    x[0, 0] = 1.0
    Sampler.greedy_sample(x)
    fp = [e for e in read_events() if e["type"] == "logits_fp"][0]
    rows = [r["row"] for r in fp["data"]["rows"]]
    assert len(rows) == 3
    assert rows[0] == 0 and rows[-1] == 7  # covers both ends of the batch


def test_logits_fp_with_margin_off(monkeypatch, read_events):
    """Fingerprint works standalone (margin probe disabled)."""
    _enable_logits_fp(monkeypatch, rows=4)
    monkeypatch.setenv("VLLM_SNIFFER_MARGIN", "0")
    get_config.cache_clear()
    Sampler = _install_sampler(monkeypatch)
    x = torch.randn(4, 16)
    Sampler.greedy_sample(x)
    events = read_events()
    types = [e["type"] for e in events]
    assert "logits_fp" in types
    assert "sample_flip" not in types
    assert "sample_stats" not in types


def test_logits_fp_fp16_16bits(monkeypatch, read_events):
    """fp16 input -> 16-bit fingerprint (defensive path)."""
    _enable_logits_fp(monkeypatch, rows=4)
    Sampler = _install_sampler(monkeypatch)
    x = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float16)
    Sampler.greedy_sample(x)
    fp = [e for e in read_events() if e["type"] == "logits_fp"][0]
    data = fp["data"]
    assert data["dtype"] == "torch.float16"
    bits = data["rows"][0]["bits"]
    assert len(bits) == 16
    # 1.0 in fp16 = 0x3C00: exponent 01111 -> bits 13..10 set, bit 14 clear
    assert bits[14] == 0 and bits[13] == 3 and bits[10] == 3
    assert bits[11] == 3 and bits[12] == 3
    assert sum(bits) == 12


def test_logits_fp_unsupported_dtype_no_event(monkeypatch, read_events):
    _enable_logits_fp(monkeypatch, rows=4)
    Sampler = _install_sampler(monkeypatch)
    x = torch.ones(2, 4, dtype=torch.int64)  # not a float dtype
    Sampler.greedy_sample(x)  # must not crash
    types = [e["type"] for e in read_events()]
    assert "logits_fp" not in types


def test_logits_fp_behavior_invariant(monkeypatch, read_events):
    """Fingerprinting never changes the sampled output."""
    _enable_logits_fp(monkeypatch, rows=8)
    Sampler = _install_sampler(monkeypatch)
    torch.manual_seed(7)
    x = torch.randn(5, 32)
    expected = FakeSampler.greedy_sample(x)
    assert torch.equal(Sampler.greedy_sample(x), expected)
    # and no crash on a nan-heavy batch either
    x2 = torch.full((3, 8), float("nan"))
    x2[1, :] = 0.0
    x2[1, 2] = 1.0
    assert Sampler.greedy_sample(x2)[1] == 2
