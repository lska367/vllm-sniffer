"""Worker hook tests: forward timing and the greedy argmax-margin probe.

The sampler margin hook runs real torch logic (topk, masking, reductions),
so these tests verify both behavior invariance (sampling results unchanged)
and the flip-zone detection that underpins the temperature=0
nondeterminism diagnosis.
"""

from types import SimpleNamespace

import torch
from conftest import fake_vllm_module

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
