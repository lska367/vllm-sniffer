"""exp_determinism.py -- temperature=0 reproducibility experiment (real machine).

The core research question: same prompt, temperature=0, N requests -- do the
outputs differ? If yes, WHERE do they differ, and does the sniffer's
sample_flip map (margin < eps positions) explain the flip positions?

Usage (GPU machine with vLLM installed, e.g. .venv-gpu):
    .venv-gpu/bin/python scripts/exp_determinism.py --n 8 --max-tokens 64

What it does:
1. Runs N identical-prompt requests (batched in one LLM.generate call, so
   batch composition is identical across the N outputs -- isolates
   per-position variance under a fixed batch).
2. Computes per-position agreement: positions where all N outputs agree vs
   differ; also unique-sequence count.
3. Writes results.json (outputs, positions diff, per-position counts) for
   later analysis.
4. Sets VLLM_SNIFFER_DIR to <out>/exp-determinism/<ts> before importing
   vLLM, so the run's events (incl. sample_flip) are captured in one place
   and can be cross-referenced with tools/repro_compare.py.

Caveats:
- With identical batch composition, identical input and greedy sampling,
  outputs usually match on a single GPU; variance shows up across runs /
  batch compositions / prefix-cache states. Use --runs 3 to repeat the
  whole batch and compare *across* runs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys


def _default_prompt() -> str:
    return (
        "Explain the concept of gravity in one sentence, "
        "then list its effects on a falling apple."
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default=None, help="default: fixed physics prompt")
    ap.add_argument("--n", type=int, default=8, help="requests per run")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=1, help="repeat the whole batch")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--out-dir", default="/tmp/vllm-sniffer")
    ap.add_argument("--sniffer-dir", default=None, help="VLLM_SNIFFER_DIR override")
    return ap


def run_experiment(args) -> dict:
    # Configure the sniffer output dir *before* importing vLLM (the plugin
    # fixes the run_id at import time).
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sniffer_dir = args.sniffer_dir or os.path.join(args.out_dir, f"exp-determinism-{ts}")
    os.environ.setdefault("VLLM_SNIFFER_DIR", sniffer_dir)

    from vllm import LLM, SamplingParams

    prompt = args.prompt or _default_prompt()
    llm = LLM(
        model=args.model,
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    runs: list[dict] = []
    for run_i in range(args.runs):
        outs = llm.generate([prompt] * args.n, sampling_params=params)
        token_ids = [list(o.outputs[0].token_ids) for o in outs]
        texts = [o.outputs[0].text for o in outs]
        runs.append({"run": run_i, "token_ids": token_ids, "texts": texts})
        print(f"run {run_i}: {len(set(map(tuple, token_ids)))} unique output(s) of {args.n}")

    # Per-position agreement across the *last* run's outputs.
    token_ids = runs[-1]["token_ids"]
    n_pos = max((len(t) for t in token_ids), default=0)
    diff_positions = []
    for pos in range(n_pos):
        vals = {t[pos] for t in token_ids if pos < len(t)}
        if len(vals) > 1:
            diff_positions.append(pos)
    unique_sequences = len(set(map(tuple, token_ids)))

    result = {
        "model": args.model,
        "prompt": prompt,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "sniffer_dir": sniffer_dir,
        "diff_positions": diff_positions,
        "n_diff_positions": len(diff_positions),
        "unique_sequences_last_run": unique_sequences,
        "output_lengths": [len(t) for t in token_ids],
        "per_run": runs,
    }
    return result


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_experiment(args)
    except ImportError as e:
        print(f"error: this experiment needs a GPU environment with vLLM installed "
              f"(e.g. .venv-gpu): {e}", file=sys.stderr)
        return 2

    out_path = os.path.join(
        args.out_dir, f"exp-determinism-{result['sniffer_dir'].rsplit('-', 1)[-1]}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n== summary ==")
    print(f"n={result['n']} max_tokens={result['max_tokens']} runs={result['runs']}")
    print(f"output lengths : {result['output_lengths']}")
    print(f"unique seqs (last run): {result['unique_sequences_last_run']}")
    print(f"diff positions : {result['n_diff_positions']} -> {result['diff_positions'][:20]}")
    print(f"events dir     : {result['sniffer_dir']}")
    print(f"results        : {out_path}")
    print("hint: correlate with the flip map via:")
    print(f"  python tools/repro_compare.py {result['sniffer_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
