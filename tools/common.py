"""Shared helpers for the vllm-sniffer analysis tools (tools/).

Everything here is pure Python with only the package's own dependency
(msgspec). pyarrow is required *only* by export_parquet.py and degrades
with a friendly error when absent.

Inputs accepted by every tool:
- a run directory (``<out_dir>/<run_id>/``, one JSONL file per process), or
- a parquet file produced by ``export_parquet.py``.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Iterable

import msgspec

# Columns written by export_parquet.py (flat schema; data payload is a JSON
# string column so the schema never needs to change when data fields grow).
PARQUET_COLUMNS = (
    "schema_ver",
    "ts_ns",
    "pid",
    "group",
    "type",
    "req_id",
    "step",
    "sampled",
    "data_json",
)


class ToolError(Exception):
    """User-facing error (bad input path, missing optional dep, ...)."""


# ----------------------------------------------------------------------
# Event loading
# ----------------------------------------------------------------------
def _iter_jsonl_dir(run_dir: str) -> Iterable[dict[str, Any]]:
    found = False
    for root, _dirs, files in os.walk(run_dir):
        for name in sorted(files):
            if not name.endswith(".jsonl"):
                continue
            found = True
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield msgspec.json.decode(line)
    if not found:
        raise ToolError(f"no .jsonl files found under {run_dir!r}")


def load_events(path: str) -> list[dict[str, Any]]:
    """Load events from a run directory or a parquet file, sorted by ts_ns.

    The returned dicts always contain the flat ``data_json`` column plus
    the common fields (same shape as the parquet output).
    """
    if os.path.isdir(path):
        events = list(_iter_jsonl_dir(path))
        if not events:
            raise ToolError(f"run directory {path!r} contains no events")
        # Normalize: keep the original data dict under a JSON string column
        # so downstream code is identical for jsonl and parquet inputs.
        for ev in events:
            ev["data_json"] = json.dumps(ev.get("data") or {}, ensure_ascii=False)
        return sorted(events, key=lambda e: e["ts_ns"])

    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover - depends on env
            raise ToolError(
                "reading parquet requires pyarrow: "
                "uv pip install --python .venv/bin/python 'pyarrow>=14'"
            ) from e
        table = pq.read_table(path)
        events: list[dict[str, Any]] = []
        for row in zip(*[table.column(c).to_pylist() for c in PARQUET_COLUMNS]):
            events.append(dict(zip(PARQUET_COLUMNS, row)))
        return sorted(events, key=lambda e: e["ts_ns"])

    raise ToolError(f"unsupported input {path!r}: expected a run directory or .parquet")


def events_by_type(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        out.setdefault(ev["type"], []).append(ev)
    return out


def data_of(ev: dict[str, Any]) -> dict[str, Any]:
    """The event's data payload as a dict (jsonl or parquet input)."""
    d = ev.get("data")
    if isinstance(d, dict):
        return d
    dj = ev.get("data_json")
    if isinstance(dj, str) and dj:
        try:
            return json.loads(dj)
        except ValueError:
            return {}
    return {}


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy-compatible default)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(values: list[float], unit: str = "ms", scale: float = 1.0) -> str:
    """'n=42 p50=1.2 p90=2.1 p99=3.4 max=5.0 (ms)' summary line."""
    if not values:
        return f"n=0 ({unit})"
    scaled = [v * scale for v in values]
    parts = [
        f"n={len(scaled)}",
        f"p50={percentile(scaled, 0.50):.2f}",
        f"p90={percentile(scaled, 0.90):.2f}",
        f"p99={percentile(scaled, 0.99):.2f}",
        f"max={max(scaled):.2f}",
        f"mean={statistics.fmean(scaled):.2f}",
    ]
    return " ".join(parts) + f" ({unit})"


def ascii_histogram(
    values: list[float],
    *,
    bins: int = 16,
    width: int = 50,
    fmt: str = "{:.1f}",
) -> list[str]:
    """Simple ASCII histogram lines (no external plotting deps)."""
    if not values:
        return ["(no data)"]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [f"all values = {fmt.format(lo)}"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1
    lines = [f"range [{fmt.format(lo)}, {fmt.format(hi)}]  {bins} bins"]
    for i, c in enumerate(counts):
        bar = "#" * max(1, round(c / peak * width)) if c else " " * width
        lines.append(f"{fmt.format(lo + i * step):>10} |{bar}| {c}")
    return lines


# ----------------------------------------------------------------------
# Request timeline assembly
# ----------------------------------------------------------------------
def request_timelines(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group api events by req_id into per-request timelines.

    Returns {req_id: {start, first_token, finish, abort, data...}} with the
    raw event dicts. Offline batch markers (data.mode == "offline") are
    included too but flagged (no req_id correlation in the worker/core).
    """
    by_type = events_by_type(events)
    timelines: dict[str, dict[str, Any]] = {}
    for req_id in sorted({e.get("req_id") or "" for e in by_type.get("request_start", [])}):
        timelines[req_id] = {"start": None, "first_token": None, "finish": None, "abort": None}
    for t, key in (
        ("request_start", "start"),
        ("request_first_token", "first_token"),
        ("request_finish", "finish"),
        ("request_abort", "abort"),
    ):
        for ev in by_type.get(t, []):
            rid = ev.get("req_id")
            if rid is None:
                continue
            tl = timelines.setdefault(rid, {"start": None, "first_token": None, "finish": None, "abort": None})
            # first_token / finish: keep the earliest / latest occurrence.
            if key in ("first_token", "finish"):
                cur = tl[key]
                if cur is None or ev["ts_ns"] > cur["ts_ns"]:
                    tl[key] = ev
            else:
                tl[key] = ev
    return timelines


def ttft_ms(tl: dict[str, Any]) -> float | None:
    if tl["start"] and tl["first_token"]:
        return (tl["first_token"]["ts_ns"] - tl["start"]["ts_ns"]) / 1e6
    return None


def tpot_ms(tl: dict[str, Any]) -> float | None:
    """Per-token output time (finish - first_token) / (n_output_tokens - 1)."""
    if not (tl["first_token"] and tl["finish"]):
        return None
    n_out = data_of(tl["finish"]).get("n_output_tokens") or 0
    n_intervals = max(n_out - 1, 1)
    return (tl["finish"]["ts_ns"] - tl["first_token"]["ts_ns"]) / 1e6 / n_intervals
