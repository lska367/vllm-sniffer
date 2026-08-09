"""logits_fp_compare.py tests (P1-1): diff-bit distribution analysis.

Synthetic fingerprint runs:

- ``fp-solo``: 3 logits_fp events x 2 rows of fixed bit patterns (the
  "same batch" control).
- ``fp-solo-identical``: byte-identical copy (delta must be ~0).
- ``fp-lowbit``: same shape, mantissa bits perturbed (batch-composition
  style low-bit noise).
- ``fp-systematic``: same shape, exponent bits perturbed (scale-level
  difference).

The tool runs as a subprocess (real CLI bootstrap, like test_tools.py).
"""

import json
import subprocess
import sys

from test_tools import _write_event

TOOL = "tools/logits_fp_compare.py"

# 32-bit fp32 fingerprint for a "clean" row (e.g. a row of constant logits).
CLEAN_BITS = [0] * 23 + [3, 3, 3, 3, 3, 3, 3, 0, 0]  # bits 23..29 set
CLEAN_BITS2 = [0] * 22 + [1, 3, 3, 3, 3, 3, 3, 3, 0, 0]  # bit 22 (mantissa)


def _lowbit_bits(extra=0):
    """CLEAN_BITS with low-bit (mantissa) perturbations."""
    b = list(CLEAN_BITS)
    b[0] += 4 + extra
    b[1] += 2
    b[5] += 1
    return b


def _systematic_bits():
    """CLEAN_BITS with exponent-bit perturbations."""
    b = list(CLEAN_BITS)
    b[25] += 3
    b[24] += 1
    b[0] += 1
    return b


def _fp_event(ts_ns, n_rows=2):
    return {
        "schema_ver": 1, "ts_ns": ts_ns, "pid": 1002, "group": "worker",
        "type": "logits_fp", "req_id": None, "step": None, "sampled": True,
        "data": {
            "n_rows": n_rows, "dtype": "torch.float32",
            "sampled_rows": 2,
            "rows": [
                {"row": 0, "top1": 5, "bits": list(CLEAN_BITS)},
                {"row": 1, "top1": 9, "bits": list(CLEAN_BITS2)},
            ],
        },
    }


def make_fp_run(tmp_path, name, events):
    run = tmp_path / name
    run.mkdir()
    for ev in events:
        _write_event(run / "1002.jsonl", **ev)
    return run


def solo_run(tmp_path):
    t = 1_000_000_000
    return make_fp_run(
        tmp_path, "fp-solo",
        [_fp_event(t), _fp_event(t + 10_000_000), _fp_event(t + 20_000_000)],
    )


def _run_tool(*args):
    return subprocess.run(
        [sys.executable, TOOL, *map(str, args)],
        capture_output=True, text=True, timeout=60,
    )


def test_identical_runs_verdict_identical(tmp_path):
    a = solo_run(tmp_path)
    r = _run_tool(a, a)
    assert r.returncode == 0, r.stderr
    assert "IDENTICAL" in r.stdout
    assert "total bit delta : 0" in r.stdout
    assert "pairs=3" in r.stdout
    assert "rows matched=6" in r.stdout


def test_low_bit_noise_verdict(tmp_path):
    a = solo_run(tmp_path)
    t = 1_000_000_000
    b = make_fp_run(
        tmp_path, "fp-lowbit",
        [
            {**_fp_event(t), "data": {
                "n_rows": 2, "dtype": "torch.float32", "sampled_rows": 2,
                "rows": [
                    {"row": 0, "top1": 5, "bits": _lowbit_bits()},
                    {"row": 1, "top1": 9, "bits": list(CLEAN_BITS2)},
                ],
            }},
            _fp_event(t + 10_000_000),
            _fp_event(t + 20_000_000),
        ],
    )
    r = _run_tool(a, b)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "LOW-BIT NOISE" in out
    # mantissa share: 7/7 of delta is in bits 0..22 -> 100%
    assert "mantissa share  : 100.0%" in out
    # the diff-bit distribution chart lists the perturbed bits
    assert "bit  0 [man ]" in out
    assert "bit  1 [man ]" in out
    assert "bit  5 [man ]" in out
    assert "top1 mismatch=0" in out


def test_systematic_verdict(tmp_path):
    a = solo_run(tmp_path)
    t = 1_000_000_000
    b = make_fp_run(
        tmp_path, "fp-systematic",
        [
            {**_fp_event(t), "data": {
                "n_rows": 2, "dtype": "torch.float32", "sampled_rows": 2,
                "rows": [
                    {"row": 0, "top1": 5, "bits": _systematic_bits()},
                    {"row": 1, "top1": 9, "bits": list(CLEAN_BITS2)},
                ],
            }},
            _fp_event(t + 10_000_000),
            _fp_event(t + 20_000_000),
        ],
    )
    r = _run_tool(a, b)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "SYSTEMATIC" in out
    # exponent share: 4/5 of delta is in bits 23..30 -> 80%
    assert "exponent share  : 80.0%" in out


def test_ts_align_mode(tmp_path):
    """--align ts matches events by nearest ts_ns within the window."""
    a = solo_run(tmp_path)
    t = 1_000_000_000
    # Same stream, but every ts shifted by +1ms (still within 50ms window).
    b = make_fp_run(
        tmp_path, "fp-ts-shift",
        [_fp_event(t + 1_000_000), _fp_event(t + 11_000_000),
         _fp_event(t + 21_000_000)],
    )
    r = _run_tool(a, b, "--align", "ts")
    assert r.returncode == 0, r.stderr
    assert "align : ts" in r.stdout
    assert "pairs=3" in r.stdout
    assert "IDENTICAL" in r.stdout


def test_unmatched_rows_reported(tmp_path):
    """Rows only present in one run are reported (batch grew/shrunk)."""
    t = 1_000_000_000
    ev = _fp_event(t)
    ev["data"]["rows"] = [
        {"row": 0, "top1": 5, "bits": list(CLEAN_BITS)},
        {"row": 1, "top1": 9, "bits": list(CLEAN_BITS2)},
        {"row": 2, "top1": 3, "bits": list(CLEAN_BITS)},
        {"row": 3, "top1": 7, "bits": list(CLEAN_BITS2)},
    ]
    a = make_fp_run(tmp_path, "fp-more-rows", [ev])
    b = solo_run(tmp_path)
    r = _run_tool(a, b)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "rows matched=2" in out
    assert "rows only in A=2" in out
    assert "rows only in B=0" in out


def test_missing_fp_events_friendly_error(tmp_path):
    a = solo_run(tmp_path)
    run = tmp_path / "fp-empty"
    run.mkdir()
    _write_event(run / "1001.jsonl", group="core", type="step",
                 data={"dur_ns": 1000})
    r = _run_tool(a, run)
    assert r.returncode == 1
    assert "no logits_fp events" in r.stderr
    assert "VLLM_SNIFFER_LOGITS_FP" in r.stderr
