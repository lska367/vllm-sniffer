#!/usr/bin/env python3
"""repro_compare.py -- temperature=0 nondeterminism comparison (P0-2, step 3).

Usage:
    python tools/repro_compare.py <run_dir_or_parquet> [--req-ids r1 r2 ...]

What it answers, from one run (or several runs exported into one parquet):

1. Output-length consistency: requests are grouped by prompt length
   (request_first_token.n_prompt_tokens); if two requests with the same
   prompt length produced different n_output_tokens, that is direct
   evidence of nondeterminism (vLLM samples the stop decision).
2. Flip density: sample_flip events (greedy argmax with margin < eps) vs
   total greedy rows (sample_stats.n_greedy) -- the "how many positions
   are in the flip zone" rate.
3. Per-request flip attribution: every flip's wall-clock ts is matched
   against each request's active window [start.ts, finish.ts]. A flip
   inside exactly one window is attributed to that request; a flip inside
   several windows is reported as ambiguous (v0 limitation: the sampler
   does not yet carry req_id, only ts correlation).

The estimated position within the output is a linear interpolation
(first_token..finish), which is exact when the request was the only one in
the batch and approximate under batching.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ToolError,
    data_of,
    events_by_type,
    load_events,
    request_timelines,
    summarize,
)


def _groups_by_prompt_len(timelines) -> dict[str, list[tuple[str, dict]]]:
    """Group requests by prompt length (n_prompt_tokens)."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    for rid, tl in timelines.items():
        if not tl["first_token"]:
            continue
        n = data_of(tl["first_token"]).get("n_prompt_tokens")
        if n is None:
            continue
        groups.setdefault(str(n), []).append((rid, tl))
    return groups


def _active_windows(timelines) -> list[tuple[str, int, int | None]]:
    """[(req_id, start_ts, end_ts_or_None)] for requests with a start."""
    out = []
    for rid, tl in timelines.items():
        if not tl["start"]:
            continue
        end = tl["finish"]["ts_ns"] if tl["finish"] else tl["abort"]["ts_ns"] if tl["abort"] else None
        out.append((rid, tl["start"]["ts_ns"], end))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="run directory or .parquet file")
    ap.add_argument(
        "--req-ids",
        nargs="*",
        default=None,
        help="restrict the per-request table to these req_ids",
    )
    args = ap.parse_args()

    try:
        events = load_events(args.input)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    by_type = events_by_type(events)
    timelines = request_timelines(events)
    flips = by_type.get("sample_flip", [])
    stats = by_type.get("sample_stats", [])
    n_greedy = sum(data_of(e).get("n_greedy") or 0 for e in stats)
    n_flips_reported = sum(data_of(e).get("n_flips") or 0 for e in stats)

    print(f"== {args.input}")
    print(f"requests with start: {len(timelines)}   sample_flip events: {len(flips)}")

    # --- 1. output-length consistency per prompt-length group -------------
    print("\n== output-length consistency (grouped by prompt length)")
    groups = _groups_by_prompt_len(timelines)
    if not groups:
        print("no request_first_token events with n_prompt_tokens found")
    for n_prompt in sorted(groups, key=lambda k: int(k)):
        reqs = groups[n_prompt]
        n_out = [data_of(tl["finish"]).get("n_output_tokens") for _, tl in reqs if tl["finish"]]
        n_out = [v for v in n_out if v is not None]
        unique = sorted(set(n_out))
        verdict = "VARYING" if len(unique) > 1 else "consistent"
        extra = f"values={unique}" if unique else "no finishes"
        print(f"  prompt_len={n_prompt:<6} n_reqs={len(reqs):<4} n_output_tokens={extra} -> {verdict}")

    # --- 2. flip density ---------------------------------------------------
    print("\n== flip-zone density (greedy argmax, margin < eps)")
    if stats:
        rate = n_flips_reported / n_greedy if n_greedy else 0.0
        print(f"  greedy rows      : {n_greedy}")
        print(f"  flips (from stats): {n_flips_reported}  ({rate:.2%})")
        print(f"  raw flip events  : {len(flips)}")
        margins = [data_of(e).get("margin", 0.0) for e in flips]
        if margins:
            print(f"  flip margin      : {summarize(margins, 'logit')}")
    else:
        print("  no sample_stats events (margin probe disabled? VLLM_SNIFFER_MARGIN=0)")

    # --- 3. per-request flip attribution -----------------------------------
    print("\n== per-request flip attribution (by active-window overlap)")
    windows = _active_windows(timelines)
    attr: dict[str, list[dict]] = {rid: [] for rid, _, _ in windows}
    ambiguous = 0
    for ev in flips:
        ts = ev["ts_ns"]
        hits = [rid for rid, start, end in windows if start <= ts and (end is None or ts <= end)]
        if len(hits) == 1:
            attr[hits[0]].append(ev)
        elif len(hits) > 1:
            ambiguous += 1

    rows = []
    for rid, _start, _end in windows:
        if args.req_ids and rid not in args.req_ids:
            continue
        tl = timelines[rid]
        fl = attr[rid]
        # approximate output position: linear interpolation over the stream
        n_out = data_of(tl["finish"]).get("n_output_tokens") if tl["finish"] else None
        spans = []
        if fl and tl["first_token"] and tl["finish"] and n_out:
            span = tl["finish"]["ts_ns"] - tl["first_token"]["ts_ns"]
            for e in fl:
                pos = (e["ts_ns"] - tl["first_token"]["ts_ns"]) / span * (n_out - 1)
                spans.append(pos)
        rows.append((rid, len(fl), spans, n_out))

    if rows:
        print(f"  {'req_id':<38} {'flips':>5} {'~positions (output idx)':<40} {'out_tok':>7}")
        for rid, nf, spans, n_out in rows:
            pos_str = ",".join(f"{p:.0f}" for p in sorted(spans)) if spans else "--"
            print(f"  {rid:<38} {nf:>5} {pos_str:<40} {n_out if n_out is not None else '--':>7}")
        total_attr = sum(len(fl) for fl in attr.values())
        print(f"  attributed: {total_attr}  ambiguous (multi-request window): {ambiguous}")
    else:
        print("  no requests with start events (offline batch run?)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
