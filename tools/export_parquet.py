#!/usr/bin/env python3
"""export_parquet.py -- merge a run directory's JSONL event files into one
parquet table (P0-2 analysis pipeline, step 1).

Usage:
    python tools/export_parquet.py <run_dir> [-o out.parquet]

- Merges every <pid>.jsonl under <run_dir> (one file per vLLM process),
  sorts globally by ts_ns (wall clock).
- Flat schema (columns): schema_ver, ts_ns, pid, group, type, req_id,
  step, sampled, data_json (the event's data payload as a JSON string, so
  the table schema stays stable as data fields grow).
- Requires pyarrow (optional dependency):
      uv pip install --python .venv/bin/python 'pyarrow>=14'
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PARQUET_COLUMNS, ToolError, load_events  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="run directory (<out_dir>/<run_id>)")
    ap.add_argument("-o", "--out", default=None, help="output .parquet path")
    args = ap.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        print(
            "export_parquet requires pyarrow (optional dependency).\n"
            "  uv pip install --python .venv/bin/python 'pyarrow>=14'",
            file=sys.stderr,
        )
        return 2

    try:
        events = load_events(args.run_dir)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not events:
        print("error: no events", file=sys.stderr)
        return 1

    run_id = os.path.basename(os.path.normpath(args.run_dir))
    out = args.out or f"{run_id}.parquet"

    rows = {
        col: [ev.get(col) for ev in events]
        for col in PARQUET_COLUMNS[:-1]
    }
    rows["data_json"] = [json.dumps(ev.get("data") or {}, ensure_ascii=False) for ev in events]
    table = pa.table(rows)

    pq.write_table(table, out)
    n_files = sum(
        1
        for root, _d, files in os.walk(args.run_dir)
        for f in files
        if f.endswith(".jsonl")
    )
    by_type: dict[str, int] = {}
    for ev in events:
        by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1
    span_s = (events[-1]["ts_ns"] - events[0]["ts_ns"]) / 1e9
    print(f"run_id            : {run_id}")
    print(f"pid files merged  : {n_files}")
    print(f"events            : {len(events)}")
    print(f"time span         : {span_s:.2f}s")
    print(f"types             : {', '.join(f'{k}={v}' for k, v in sorted(by_type.items()))}")
    print(f"output            : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
