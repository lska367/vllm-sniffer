#!/usr/bin/env python3
"""logits_fp_compare.py -- compare logits bit-level fingerprints of two runs
(P1-1 deep-dive analysis).

Usage:
    python tools/logits_fp_compare.py <run_a> <run_b> [--align index|ts]
                                      [--ts-window-ms 50]

Input: two run directories (JSONL) or two parquet files exported by
export_parquet.py. Both runs must contain ``logits_fp`` events, which are
only produced when the deep-dive mode VLLM_SNIFFER_LOGITS_FP=1 was on.

What it does:
1. Collects the ``logits_fp`` events of each run (sorted by ts_ns) and
   aligns them event-to-event (by sequence index -- the i-th fingerprint
   event of run A vs the i-th of run B -- or by nearest ts_ns within a
   window).
2. For every matched event, matches rows by their absolute ``row`` index
   and computes, per fp32 bit position b, the absolute difference of the
   per-bit set counts between the two runs: |counts_A[b] - counts_B[b]|.
3. Aggregates into a per-bit difference distribution (ASCII chart) and
   classifies it:
   - total delta ~= 0            -> IDENTICAL (same batch, same numerics)
   - mantissa bits (0..22) dom.  -> low-bit noise (numerical-path jitter,
                                    e.g. batch-composition effects)
   - exponent/sign bits dom.     -> systematic scale-level difference
                                    (different logits entirely)

This is the "diff-bit distribution chart" of the roadmap acceptance: a
same-batch control run (rerun the identical workload) should come out
IDENTICAL; a different-batch-composition run should show a low-bit-noise
pattern.

Limitations (v0):
- Row correspondence is by absolute row index within the aligned event.
  With different batch compositions, row i may be a different request in
  each run; the per-bit *distribution* is still meaningful, but per-row
  verdicts are not. top1 token ids are reported so you can judge
  correspondence confidence.
- fp32 vs fp16/bf16 runs are compared over min(nbits) positions with a
  warning.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ToolError, data_of, load_events  # noqa: E402

# fp32 bit layout (IEEE 754): bit 31 sign, 23..30 exponent, 0..22 mantissa.
MANTISSA_BITS = range(23)
EXPONENT_BITS = range(23, 31)
SIGN_BIT = 31


def collect_fingerprints(events: list[dict]) -> list[dict]:
    """logits_fp events of a run, sorted by ts_ns, rows keyed by row idx."""
    out = []
    for ev in sorted(
        (e for e in events if e["type"] == "logits_fp"), key=lambda e: e["ts_ns"]
    ):
        d = data_of(ev)
        rows = {
            r["row"]: r
            for r in d.get("rows", [])
            if isinstance(r, dict) and "bits" in r
        }
        out.append(
            {
                "ts_ns": ev["ts_ns"],
                "n_rows": d.get("n_rows"),
                "dtype": d.get("dtype"),
                "rows": rows,
            }
        )
    return out


def align_events(a: list[dict], b: list[dict], mode: str, window_ns: int):
    """Yield (event_a, event_b) pairs.

    mode="index": i-th fingerprint event of A vs i-th of B (the default:
    for identical workloads the fingerprint streams have the same shape).
    mode="ts": nearest ts_ns within window, each event used at most once.
    """
    if mode == "index":
        for ea, eb in zip(a, b):
            yield ea, eb
        return
    used = set()
    for ea in a:
        best, best_d = None, None
        for i, eb in enumerate(b):
            if i in used:
                continue
            d = abs(eb["ts_ns"] - ea["ts_ns"])
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None and best_d <= window_ns:
            used.add(best)
            yield ea, b[best]


def compare_runs(a_path: str, b_path: str, align: str = "index",
                 window_ms: float = 50.0) -> dict:
    fa = collect_fingerprints(load_events(a_path))
    fb = collect_fingerprints(load_events(b_path))
    if not fa:
        raise ToolError(
            f"{a_path!r} has no logits_fp events (was VLLM_SNIFFER_LOGITS_FP=1 "
            "enabled for that run?)"
        )
    if not fb:
        raise ToolError(
            f"{b_path!r} has no logits_fp events (was VLLM_SNIFFER_LOGITS_FP=1 "
            "enabled for that run?)"
        )

    n_bits = 32
    delta = [0] * n_bits          # total |delta| per bit position
    rows_diff = [0] * n_bits      # rows where bit b differs at all
    n_pairs = n_rows_matched = 0
    n_rows_unmatched_a = n_rows_unmatched_b = n_top1_mismatch = 0
    dtype_mismatch = False

    for ea, eb in align_events(fa, fb, align, int(window_ms * 1e6)):
        n_pairs += 1
        if ea["dtype"] != eb["dtype"]:
            dtype_mismatch = True
        for row_idx, ra in ea["rows"].items():
            rb = eb["rows"].get(row_idx)
            if rb is None:
                n_rows_unmatched_a += 1
                continue
            n_rows_matched += 1
            if "top1" in ra and "top1" in rb and ra["top1"] != rb["top1"]:
                n_top1_mismatch += 1
            bits_a, bits_b = ra["bits"], rb["bits"]
            nbits = min(len(bits_a), len(bits_b), n_bits)
            for b in range(nbits):
                d = abs(bits_a[b] - bits_b[b])
                if d:
                    delta[b] += d
                    rows_diff[b] += 1
        # Rows present in B but not in A (e.g. batch composition grew).
        n_rows_unmatched_b += len(set(eb["rows"]) - set(ea["rows"]))

    total_delta = sum(delta)
    frac_mantissa = sum(delta[b] for b in MANTISSA_BITS) / total_delta \
        if total_delta else 0.0
    frac_exponent = sum(delta[b] for b in EXPONENT_BITS) / total_delta \
        if total_delta else 0.0

    if total_delta == 0:
        verdict = "IDENTICAL: no bit differences in matched rows " \
                  "(same batch, same numerics)"
    elif frac_mantissa >= 0.8:
        verdict = (
            f"LOW-BIT NOISE: {frac_mantissa * 100:.1f}% of bit delta is in "
            "mantissa bits 0..22 -- consistent with numerical-path jitter "
            "(e.g. batch-composition-dependent kernel order)"
        )
    elif frac_exponent >= 0.5:
        verdict = (
            f"SYSTEMATIC: {frac_exponent * 100:.1f}% of bit delta is in "
            "exponent bits 23..30 -- logits differ at scale level "
            "(different values, not just rounding)"
        )
    else:
        verdict = (
            f"MIXED: mantissa {frac_mantissa * 100:.1f}%, "
            f"exponent {frac_exponent * 100:.1f}% -- both noise and "
            "systematic differences present"
        )

    return {
        "run_a": a_path, "run_b": b_path, "align": align,
        "n_events_a": len(fa), "n_events_b": len(fb),
        "n_pairs": n_pairs, "n_rows_matched": n_rows_matched,
        "n_rows_unmatched_a": n_rows_unmatched_a,
        "n_rows_unmatched_b": n_rows_unmatched_b,
        "n_top1_mismatch": n_top1_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "total_delta": total_delta, "delta": delta, "rows_diff": rows_diff,
        "frac_mantissa": frac_mantissa, "frac_exponent": frac_exponent,
        "verdict": verdict,
    }


def render_report(r: dict) -> str:
    lines = []
    lines.append(f"run A : {r['run_a']}  ({r['n_events_a']} logits_fp events)")
    lines.append(f"run B : {r['run_b']}  ({r['n_events_b']} logits_fp events)")
    lines.append(f"align : {r['align']}  pairs={r['n_pairs']}  "
                 f"rows matched={r['n_rows_matched']}  "
                 f"rows only in A={r['n_rows_unmatched_a']}  "
                 f"rows only in B={r['n_rows_unmatched_b']}  "
                 f"top1 mismatch={r['n_top1_mismatch']}")
    if r["dtype_mismatch"]:
        lines.append("WARNING: runs have different logits dtypes; compared "
                     "over min(nbits) positions")
    lines.append("")
    lines.append("per-bit |delta| distribution (fp32 bit positions):")
    peak = max(r["delta"]) or 1
    for b, d in enumerate(r["delta"]):
        if d == 0:
            continue
        bar = "#" * max(1, round(d / peak * 40))
        tag = "sign" if b == SIGN_BIT else \
            ("exp " if b in EXPONENT_BITS else "man ")
        lines.append(f"bit {b:2d} [{tag}] {bar} |d|={d} rows={r['rows_diff'][b]}")
    lines.append("")
    lines.append(f"total bit delta : {r['total_delta']}")
    lines.append(f"mantissa share  : {r['frac_mantissa'] * 100:.1f}%  "
                 f"(bits 0..22)")
    lines.append(f"exponent share  : {r['frac_exponent'] * 100:.1f}%  "
                 f"(bits 23..30)")
    lines.append(f"VERDICT         : {r['verdict']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a", help="run directory or .parquet")
    ap.add_argument("run_b", help="run directory or .parquet")
    ap.add_argument("--align", choices=("index", "ts"), default="index",
                    help="event alignment: by sequence index (default) or "
                         "nearest ts_ns")
    ap.add_argument("--ts-window-ms", type=float, default=50.0,
                    help="ts alignment window (default 50)")
    args = ap.parse_args()

    try:
        result = compare_runs(args.run_a, args.run_b, args.align,
                              args.ts_window_ms)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
