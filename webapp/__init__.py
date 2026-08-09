"""vllm-sniffer visualization frontend (roadmap P2-1).

Offline analysis UI: a FastAPI aggregation layer over the event stream
(JSONL run directories or exported parquet files) plus a self-contained
static frontend (vanilla JS + canvas, no CDN / no charting dependency).

Views (one API call each):
- request timeline  : per-request TTFT / TPOT bars      -> /api/runs/{id}/timeline
- step duration     : engine step dur_ns series         -> /api/runs/{id}/steps
- flip heatmap      : sample_flip margins by step       -> /api/runs/{id}/flips
- batch vs latency  : forward num_tokens vs dur_ns      -> /api/runs/{id}/scatter

Run:
    python -m webapp.server --dir /tmp/vllm-sniffer --port 8080
    # open http://127.0.0.1:8080

API documentation: doc/WEBAPP.md
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tools import common

# run_id must be a plain identifier: it becomes part of a filesystem path.
_RUN_ID_RE = re.compile(r"^[0-9A-Za-z._-]+$")

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Tiny event cache: {abs_path: (mtime_ns, events)}. Offline analysis data is
# small (MBs); the cache keeps the four views of one run from re-reading the
# same files four times. Invalidated on mtime change.
_cache: dict[str, tuple[int, list[dict[str, Any]]]] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 8


def _load_cached(path: str) -> list[dict[str, Any]]:
    mtime = os.path.getmtime(path)
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    events = common.load_events(path)
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[path] = (mtime, events)
    return events


def list_runs(out_dir: str) -> list[dict[str, Any]]:
    """Every run under out_dir: JSONL run directories + exported .parquet."""
    if not os.path.isdir(out_dir):
        return []
    runs: list[dict[str, Any]] = []
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            has_jsonl = any(
                f.endswith(".jsonl")
                for root, _d, files in os.walk(path)
                for f in files
            )
            if has_jsonl:
                runs.append(_run_meta(name, "dir", path))
        elif name.endswith(".parquet"):
            runs.append(_run_meta(name[: -len(".parquet")], "parquet", path))
    runs.sort(key=lambda r: r["run_id"])
    return runs


def _run_meta(run_id: str, kind: str, path: str) -> dict[str, Any]:
    try:
        events = _load_cached(path)
    except common.ToolError:
        return {"run_id": run_id, "kind": kind, "n_events": 0,
                "span_s": 0.0, "types": {}, "error": True}
    types: dict[str, int] = {}
    for ev in events:
        types[ev["type"]] = types.get(ev["type"], 0) + 1
    span_s = (
        (events[-1]["ts_ns"] - events[0]["ts_ns"]) / 1e9 if len(events) > 1 else 0.0
    )
    return {
        "run_id": run_id,
        "kind": kind,
        "n_events": len(events),
        "span_s": round(span_s, 3),
        "types": types,
    }


def _resolve(out_dir: str, run_id: str) -> str:
    """Resolve run_id to an input path, rejecting path traversal."""
    if not _RUN_ID_RE.fullmatch(run_id or ""):
        raise HTTPException(status_code=404, detail=f"run not found: {run_id!r}")
    dir_path = os.path.realpath(os.path.join(out_dir, run_id))
    if not dir_path.startswith(os.path.realpath(out_dir) + os.sep):
        raise HTTPException(status_code=404, detail=f"run not found: {run_id!r}")
    if os.path.isdir(dir_path):
        return dir_path
    parquet = os.path.realpath(os.path.join(out_dir, run_id + ".parquet"))
    if parquet.startswith(os.path.realpath(out_dir) + os.sep) and os.path.isfile(
        parquet
    ):
        return parquet
    raise HTTPException(status_code=404, detail=f"run not found: {run_id!r}")


def _env_snapshot(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for ev in events:
        if ev["type"] == "env_snapshot":
            d = common.data_of(ev)
            return {
                "vllm_version": (d.get("vllm") or {}).get("vllm_version"),
                "vllm_commit": (d.get("vllm") or {}).get("vllm_commit"),
                "torch_version": (d.get("torch") or {}).get("torch_version"),
                "python_version": d.get("python_version"),
                "sniffer": d.get("sniffer"),
                "env": d.get("env"),
            }
    return None


def create_app(out_dir: str | None = None) -> FastAPI:
    """Build the FastAPI app. ``out_dir`` defaults to VLLM_SNIFFER_DIR."""
    out_dir = os.path.realpath(
        out_dir or os.environ.get("VLLM_SNIFFER_DIR") or "/tmp/vllm-sniffer"
    )

    app = FastAPI(
        title="vllm-sniffer visualization",
        description="Offline aggregation API over vllm-sniffer event runs",
        version="0.1.0",
    )
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "out_dir": out_dir}

    @app.get("/api/runs")
    def runs() -> dict:
        return {"runs": list_runs(out_dir)}

    def _events(run_id: str) -> list[dict[str, Any]]:
        path = _resolve(out_dir, run_id)
        try:
            return _load_cached(path)
        except common.ToolError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict:
        events = _events(run_id)
        meta = next((r for r in list_runs(out_dir) if r["run_id"] == run_id), None)
        return {
            "run_id": run_id,
            "n_events": len(events),
            "span_s": meta["span_s"] if meta else None,
            "types": {t: sum(1 for e in events if e["type"] == t)
                      for t in sorted({e["type"] for e in events})},
            "env": _env_snapshot(events),
        }

    @app.get("/api/runs/{run_id}/timeline")
    def timeline(run_id: str) -> dict:
        events = _events(run_id)
        timelines = common.request_timelines(events)
        requests = []
        for req_id in sorted(timelines, key=lambda r: (
                timelines[r]["start"]["ts_ns"] if timelines[r]["start"] else 0,
                r)):
            tl = timelines[req_id]
            finish = tl["finish"]
            d_finish = common.data_of(finish) if finish else {}
            start = tl["start"]
            requests.append({
                "req_id": req_id,
                "start_ts": start["ts_ns"] if start else None,
                "finish_ts": finish["ts_ns"] if finish else None,
                "ttft_ms": round(common.ttft_ms(tl), 3) if common.ttft_ms(tl) is not None else None,
                "tpot_ms": round(common.tpot_ms(tl), 3) if common.tpot_ms(tl) is not None else None,
                "n_output_tokens": d_finish.get("n_output_tokens"),
                "finish_reason": d_finish.get("finish_reason"),
                "aborted": tl["abort"] is not None,
            })
        return {"requests": requests}

    @app.get("/api/runs/{run_id}/steps")
    def steps(run_id: str) -> dict:
        events = _events(run_id)
        out = []
        for ev in sorted(
            (e for e in events if e["type"] == "step"), key=lambda e: e["ts_ns"]
        ):
            d = common.data_of(ev)
            out.append({
                "step": ev.get("step"),
                "ts_ns": ev["ts_ns"],
                "dur_ms": round((d.get("dur_ns") or 0) / 1e6, 3),
                "n_outputs": d.get("n_outputs"),
            })
        return {"steps": out}

    @app.get("/api/runs/{run_id}/flips")
    def flips(run_id: str) -> dict:
        events = _events(run_id)
        out = []
        for ev in (e for e in events if e["type"] == "sample_flip"):
            d = common.data_of(ev)
            out.append({
                "ts_ns": ev["ts_ns"],
                "step": ev.get("step"),
                "margin": d.get("margin"),
                "top1": d.get("top1"),
                "top2": d.get("top2"),
            })
        return {"total": len(out), "flips": out}

    @app.get("/api/runs/{run_id}/scatter")
    def scatter(run_id: str) -> dict:
        events = _events(run_id)
        out = []
        for ev in (e for e in events if e["type"] == "forward"):
            d = common.data_of(ev)
            out.append({
                "ts_ns": ev["ts_ns"],
                "dur_ms": round((d.get("dur_ns") or 0) / 1e6, 3),
                "num_tokens": d.get("num_tokens"),
                "num_seqs": d.get("num_seqs"),
            })
        return {"points": out}

    return app
