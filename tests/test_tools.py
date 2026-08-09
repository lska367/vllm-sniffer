"""Analysis tools tests (P0-2): common helpers + the three CLIs.

The tools are exercised as subprocesses (python tools/<tool>.py <input>)
so the tests cover the real CLI bootstrap (sys.path insert, argparse).

Synthetic run data:
- 2 pid files (multi-process run), events interleaved across files
- 2 requests with the same prompt length but *different* n_output_tokens
  (nondeterminism signal for repro_compare)
- 1 flip event attributed to req-001 only (outside req-002's window)
- step / sample_stats events for engine + sampler views
"""

import json
import subprocess
import sys

import pytest

TOOLS = "tools"


def _write_event(path, **kw):
    ev = {
        "schema_ver": 1,
        "ts_ns": 0,
        "pid": 1,
        "group": "api",
        "type": "request_start",
        "req_id": None,
        "step": None,
        "sampled": False,
        "data": {},
    }
    ev.update(kw)
    with open(path, "a") as f:
        f.write(json.dumps(ev) + "\n")


def make_synthetic_run(tmp_path):
    """Build a run dir with two pid files; return (run_dir, expected stats)."""
    run = tmp_path / "run-20260809-120000"
    run.mkdir()
    f1 = run / "1001.jsonl"
    f2 = run / "1002.jsonl"

    t = 1_000_000_000  # base ts_ns

    # ---- pid 1001: api server + engine core --------------------------
    _write_event(f1, ts_ns=t, pid=1001, group="api", type="env_snapshot",
                 data={"run_id": "run-20260809-120000", "python_version": "3.12"})
    # req-001: TTFT = 50ms, 64 output tokens over 640ms -> TPOT = 10.16ms
    _write_event(f1, ts_ns=t + 10_000_000, pid=1001, group="api", type="request_start",
                 req_id="req-001", data={"kind": "str", "n_chars": 42})
    _write_event(f1, ts_ns=t + 60_000_000, pid=1001, group="api", type="request_first_token",
                 req_id="req-001", data={"n_prompt_tokens": 42})
    # req-002: starts later, TTFT = 20ms, 65 output tokens (differs from
    # req-001's 64 -> "VARYING" group verdict)
    _write_event(f1, ts_ns=t + 80_000_000, pid=1001, group="api", type="request_start",
                 req_id="req-002", data={"kind": "str", "n_chars": 42})
    _write_event(f1, ts_ns=t + 100_000_000, pid=1001, group="api", type="request_first_token",
                 req_id="req-002", data={"n_prompt_tokens": 42})
    # engine steps
    _write_event(f1, ts_ns=t + 61_000_000, pid=1001, group="core", type="step", step=1,
                 data={"dur_ns": 8_000_000, "n_outputs": 1})
    _write_event(f1, ts_ns=t + 69_000_000, pid=1001, group="core", type="step", step=2,
                 data={"dur_ns": 12_000_000, "n_outputs": 1})
    # ---- pid 1002: worker ---------------------------------------------
    _write_event(f2, ts_ns=t + 62_000_000, pid=1002, group="worker", type="forward",
                 sampled=True, step=1, data={"dur_ns": 7_000_000, "num_tokens": 42, "num_seqs": 1})
    # flip inside req-001's window [t+10ms, t+710ms], outside req-002's
    # window start (t+80ms)?? -> must be >= req-002 start to be ambiguous;
    # keep it inside req-001 only: req-002 window is [t+80ms, t+740ms],
    # flip at t+70ms -> inside req-001 only.
    _write_event(f2, ts_ns=t + 70_000_000, pid=1002, group="worker", type="sample_flip",
                 step=1, data={"margin": 0.0004, "top1": 12, "top2": 7})
    _write_event(f2, ts_ns=t + 70_000_001, pid=1002, group="worker", type="sample_stats",
                 sampled=True, step=1, data={"n_greedy": 32, "n_flips": 1,
                                             "margin_min": 0.0004, "margin_max": 3.0, "margin_mean": 1.5})
    # finishes
    _write_event(f1, ts_ns=t + 710_000_000, pid=1001, group="api", type="request_finish",
                 req_id="req-001", data={"finish_reason": "length", "n_output_tokens": 64})
    _write_event(f1, ts_ns=t + 740_000_000, pid=1001, group="api", type="request_finish",
                 req_id="req-002", data={"finish_reason": "stop", "n_output_tokens": 65})
    return run


def _run_tool(tool: str, *args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, tool, *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=60,
    )


@pytest.fixture
def synthetic_run(tmp_path):
    return make_synthetic_run(tmp_path)


# ----------------------------------------------------------------------
# common helpers
# ----------------------------------------------------------------------
def test_load_events_merges_and_sorts(tmp_path):
    run = make_synthetic_run(tmp_path)
    sys.path.insert(0, TOOLS)
    from common import load_events

    events = load_events(str(run))
    assert len(events) == 12
    ts = [e["ts_ns"] for e in events]
    assert ts == sorted(ts)
    # both pid files merged
    assert {e["pid"] for e in events} == {1001, 1002}


def test_request_timelines_and_latency(tmp_path):
    run = make_synthetic_run(tmp_path)
    sys.path.insert(0, TOOLS)
    from common import load_events, request_timelines, ttft_ms, tpot_ms

    events = load_events(str(run))
    timelines = request_timelines(events)
    tl1 = timelines["req-001"]
    assert abs(ttft_ms(tl1) - 50.0) < 1e-6
    # (710 - 60) ms / 63 intervals
    assert abs(tpot_ms(tl1) - (650.0 / 63.0)) < 1e-6
    tl2 = timelines["req-002"]
    assert abs(ttft_ms(tl2) - 20.0) < 1e-6


def test_summarize_and_histogram():
    sys.path.insert(0, TOOLS)
    from common import summarize, ascii_histogram

    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0], "ms")
    assert "n=5" in s and "p50=" in s and "(ms)" in s
    lines = ascii_histogram([1.0, 2.0, 3.0], bins=3)
    assert len(lines) == 4  # header + 3 bins


# ----------------------------------------------------------------------
# CLI tools
# ----------------------------------------------------------------------
def test_export_parquet_and_roundtrip(tmp_path, synthetic_run):
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    out = tmp_path / "exported.parquet"
    r = _run_tool("tools/export_parquet.py", synthetic_run, "-o", out)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    table = pq.read_table(out)
    assert table.column("group").to_pylist() == ["api"] * 0 or len(table) == 12
    assert "data_json" in table.column_names
    assert table.column("req_id").to_pylist().count("req-001") == 3

    # parquet -> load_events gives the same sorted stream
    sys.path.insert(0, TOOLS)
    from common import load_events

    from_jsonl = load_events(str(synthetic_run))
    from_parquet = load_events(str(out))
    assert [e["type"] for e in from_jsonl] == [e["type"] for e in from_parquet]
    assert from_parquet[0]["ts_ns"] == from_jsonl[0]["ts_ns"]


def test_export_parquet_missing_pyarrow_message(tmp_path, synthetic_run, monkeypatch):
    """Without pyarrow the tool fails with a friendly message (exit 2)."""
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "pyarrow":
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sys, "argv", ["export_parquet.py", str(synthetic_run)])

    sys.path.insert(0, TOOLS)
    import export_parquet

    assert export_parquet.main() == 2


def test_latency_report_output(tmp_path, synthetic_run):
    r = _run_tool("tools/latency_report.py", synthetic_run)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "TTFT" in out and "TPOT" in out
    assert "p50=" in out
    # per-request table rows
    assert "req-001" in out and "req-002" in out
    # step summary present
    assert "step  : n=2" in out
    # TTFT values: 50.0 and 20.0 -> p50 = 35
    assert "TTFT : n=2 p50=35.00" in out


def test_repro_compare_output(tmp_path, synthetic_run):
    r = _run_tool("tools/repro_compare.py", synthetic_run)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # output-length variance detected for prompt_len=42
    assert "VARYING" in out and "values=[64, 65]" in out
    # flip density
    assert "flips (from stats): 1" in out
    # attribution: req-001 got the flip, req-002 none
    lines = out.splitlines()
    req1 = [ln for ln in lines if "req-001" in ln]
    req2 = [ln for ln in lines if "req-002" in ln]
    assert req1 and req1[0].split()[1] == "1"  # flips column
    assert req2 and req2[0].split()[1] == "0"
