#!/usr/bin/env python3
"""latency_report.py -- TTFT / TPOT / step-duration report (P0-2, step 2).

Usage:
    python tools/latency_report.py <run_dir_or_parquet>

Per request (correlated by req_id across api events):
    TTFT = request_first_token.ts - request_start.ts
    TPOT = (request_finish.ts - request_first_token.ts) / (n_output_tokens - 1)

Also reports engine heartbeat (step) and forward-pass durations, so the
report covers both the request view and the engine view of one run.

Offline batch markers (data.mode == "offline") are reported separately:
they have no first_token anchor and carry only an overall duration.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ToolError,
    ascii_histogram,
    data_of,
    events_by_type,
    load_events,
    request_timelines,
    summarize,
    ttft_ms,
    tpot_ms,
)


def _fmt_ms(v: float | None) -> str:
    return "--" if v is None else f"{v:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="run directory or .parquet file")
    ap.add_argument("--quiet", action="store_true", help="skip per-request table")
    args = ap.parse_args()

    try:
        events = load_events(args.input)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    by_type = events_by_type(events)
    timelines = request_timelines(events)
    offline = [
        ev for ev in by_type.get("request_finish", [])
        if data_of(ev).get("mode") == "offline"
    ]
    per_req: list[tuple[str, float | None, float | None, int | None]] = []
    for rid, tl in sorted(timelines.items(), key=lambda kv: (kv[1]["start"] or {}).get("ts_ns", 0)):
        if not tl["start"]:
            continue
        per_req.append(
            (rid, ttft_ms(tl), tpot_ms(tl), data_of(tl["finish"]).get("n_output_tokens") if tl["finish"] else None)
        )

    ttfts = [v for _, v, _, _ in per_req if v is not None]
    tpots = [v for _, _, v, _ in per_req if v is not None]
    steps = [data_of(e)["dur_ns"] / 1e6 for e in by_type.get("step", []) if "dur_ns" in data_of(e)]
    forwards = [data_of(e)["dur_ns"] / 1e6 for e in by_type.get("forward", []) if "dur_ns" in data_of(e)]

    print(f"== {args.input}")
    print(f"events: {len(events)}  ({', '.join(f'{k}={len(v)}' for k, v in sorted(by_type.items()))})")

    if per_req:
        print(f"\n== per-request latency (n={len(per_req)})")
        if not args.quiet:
            print(f"{'req_id':<38} {'TTFT(ms)':>9} {'TPOT(ms)':>9} {'out_tok':>8}")
            for rid, ttft, tpot, n_out in per_req:
                print(f"{rid:<38} {_fmt_ms(ttft):>9} {_fmt_ms(tpot):>9} {str(n_out or '--'):>8}")
        print(f"TTFT : {summarize(ttfts, 'ms')}")
        print(f"TPOT : {summarize(tpots, 'ms')}")
        print("\nTTFT histogram:")
        print("\n".join(ascii_histogram(ttfts)))
        print("\nTPOT histogram:")
        print("\n".join(ascii_histogram(tpots)))
    else:
        print("\nno request_start/first_token pairs found (offline batch run?)")

    if offline:
        durs = [data_of(e)["dur_ns"] / 1e6 for e in offline if "dur_ns" in data_of(e)]
        print(f"\n== offline batches (n={len(offline)})")
        print(f"batch duration : {summarize(durs, 'ms')}")

    if steps:
        print(f"\n== engine heartbeat")
        print(f"step  : {summarize(steps, 'ms')}")
        print("\nstep-duration histogram:")
        print("\n".join(ascii_histogram(steps)))
    if forwards:
        print(f"\n== worker forward")
        print(f"forward: {summarize(forwards, 'ms')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
