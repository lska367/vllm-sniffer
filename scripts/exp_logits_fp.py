"""exp_logits_fp.py -- logits bit-level fingerprint experiment (P1-1, GPU).

Research question: where do the logits differences come from when the same
prompt is served in different batch compositions? The tracer's deep-dive
mode (VLLM_SNIFFER_LOGITS_FP=1) records per-step bit-level fingerprints of
the raw logits; this script runs a controlled A/B:

- Run A ("solo"): the probe prompt alone in the batch (batch size 1).
- Run B ("mixed"): the probe prompt alongside N-1 filler prompts, so the
  kernel paths / GEMM shapes differ at the probe's row.

The two runs are then compared with tools/logits_fp_compare.py: a solo-vs-
solo control should be IDENTICAL (~0 bit delta), solo-vs-mixed typically
shows low-bit (mantissa) noise -- the "diff-bit distribution chart".

Usage (GPU machine with vLLM installed, e.g. .venv-gpu):
    .venv-gpu/bin/python scripts/exp_logits_fp.py --n 8 --max-tokens 32

Events land in <out_dir>/exp-logits-fp-*/ (one run dir per arm) and the
compare report is printed at the end.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys


def _default_probe() -> str:
    return (
        "Explain the concept of gravity in one sentence, "
        "then list its effects on a falling apple."
    )


def _default_filler(i: int) -> str:
    topics = ["rain", "winds", "oceans", "mountains", "forests", "deserts",
              "volcanoes", "rivers", "planets", "stars", "moons", "tides",
              "clouds", "seasons", "glaciers", "canyons"]
    return f"Describe {topics[i % len(topics)]} in a few words."


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--probe-prompt", default=None,
                    help="default: fixed physics prompt")
    ap.add_argument("--n", type=int, default=8,
                    help="total requests in the mixed arm")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--out-dir", default="/tmp/vllm-sniffer")
    return ap


def run_arm(args, tag: str, prompts: list[str]) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sniffer_dir = os.path.join(args.out_dir, f"exp-logits-fp-{tag}-{ts}")
    os.environ["VLLM_SNIFFER_DIR"] = sniffer_dir
    os.environ["VLLM_SNIFFER_LOGITS_FP"] = "1"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    llm.generate(prompts, sampling_params=params)
    print(f"arm {tag}: {len(prompts)} greedy request(s) -> {sniffer_dir}")
    return sniffer_dir


def main() -> int:
    args = build_parser().parse_args()
    probe = args.probe_prompt or _default_probe()
    # Restore our env changes on every exit path: the script is also
    # importable in-process (tests), and leaked VLLM_SNIFFER_* vars would
    # poison later runs in the same process.
    _saved = {k: os.environ.get(k)
              for k in ("VLLM_SNIFFER_DIR", "VLLM_SNIFFER_LOGITS_FP")}
    try:
        # Arm A: probe alone (batch composition = 1 row).
        solo_dir = run_arm(args, "solo", [probe])
        # Arm B: probe + N-1 fillers (batch composition changes the kernel
        # paths the probe's row goes through).
        mixed = [probe] + [_default_filler(i) for i in range(args.n - 1)]
        mixed_dir = run_arm(args, "mixed", mixed)
    except ImportError as e:
        print(f"error: this experiment needs a GPU environment with vLLM "
              f"installed (e.g. .venv-gpu): {e}", file=sys.stderr)
        return 2
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print("\n== control: solo vs solo (expect ~IDENTICAL) ==")
    os.system(
        f"{sys.executable} tools/logits_fp_compare.py {solo_dir} {solo_dir} "
        f"--align index"
    )
    print("\n== hypothesis: solo vs mixed (expect low-bit noise) ==")
    os.system(
        f"{sys.executable} tools/logits_fp_compare.py {solo_dir} {mixed_dir} "
        f"--align index"
    )
    print(f"\nrun dirs: {solo_dir}\n          {mixed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
