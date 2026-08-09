"""exp_overhead.py -- quantify the tracer's runtime overhead (real machine).

Answers the "does the tracer cost anything" question with numbers: the same
offline workload is run under three configurations, each in its own
subprocess (env isolation):

    baseline      VLLM_SNIFFER=0            tracer fully off
    margin-on     VLLM_SNIFFER=1            default (incl. flip probe)
    margin-off    VLLM_SNIFFER=1, VLLM_SNIFFER_MARGIN=0   no flip probe

The flip probe (extra topk(2) kernel per greedy step) is the only per-step
GPU cost, so comparing margin-on vs margin-off isolates it; comparing
baseline vs margin-off isolates the queue+JSONL+io overhead.

Usage (GPU machine with vLLM installed):
    .venv-gpu/bin/python scripts/exp_overhead.py --n 16 --max-tokens 64 --repeat 3
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

# The workload runs in a subprocess with a given env; it must print
# "TOKENS <n> SECS <t>" on stdout so the parent can parse throughput.
WORKLOAD = r"""
import json, os, sys, time

model, n, max_tokens, gpu_mem = sys.argv[1:5]
n, max_tokens, gpu_mem = int(n), int(max_tokens), float(gpu_mem)

from vllm import LLM, SamplingParams

llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=gpu_mem)
prompts = [f"Write a {i}-sentence story about a robot." for i in range(n)]
t0 = time.monotonic()
outs = llm.generate(prompts, sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens))
dur = time.monotonic() - t0
tokens = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"TOKENS {tokens} SECS {dur:.3f}")
sys.stdout.flush()
"""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=16, help="prompts per run")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2, help="repetitions per config")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--out-dir", default="/tmp/vllm-sniffer")
    return ap


def run_config(args, env: dict[str, str], label: str) -> dict:
    """Run the workload once under ``env``; return throughput stats."""
    cmd = [
        sys.executable,
        "-c",
        WORKLOAD,
        args.model,
        str(args.n),
        str(args.max_tokens),
        str(args.gpu_memory_utilization),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        return {"label": label, "error": proc.stderr[-500:]}
    tokens = secs = None
    for line in proc.stdout.splitlines():
        # workload prints: "TOKENS <n> SECS <t>" (4 fields)
        if line.startswith("TOKENS ") and len(line.split()) == 4:
            _, t, _, s = line.split()
            tokens, secs = int(t), float(s)
    if tokens is None:
        return {"label": label, "error": f"unparsable stdout: {proc.stdout[-300:]}"}
    return {"label": label, "tokens": tokens, "secs": secs, "tps": tokens / secs}


def main() -> int:
    args = build_parser().parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_env = os.environ.copy()
    base_env.pop("VLLM_SNIFFER", None)
    base_env["VLLM_SNIFFER_DIR"] = os.path.join(args.out_dir, f"exp-overhead-{ts}")
    # Keep children quiet: the tracer's unknown-env warning is expected.
    configs = [
        ("baseline  (VLLM_SNIFFER=0)", {"VLLM_SNIFFER": "0"}),
        ("margin-off(VLLM_SNIFFER=1, MARGIN=0)", {"VLLM_SNIFFER": "1", "VLLM_SNIFFER_MARGIN": "0"}),
        ("margin-on (VLLM_SNIFFER=1, default)", {"VLLM_SNIFFER": "1"}),
    ]

    results = {label: [] for label, _ in configs}
    for rep in range(args.repeat):
        print(f"-- repetition {rep + 1}/{args.repeat} --", flush=True)
        for label, extra in configs:
            env = {**base_env, **extra}
            r = run_config(args, env, label)
            results[label].append(r)
            if "error" in r:
                print(f"  {label:<38} ERROR: {r['error'][-200:]}")
            else:
                print(f"  {label:<38} {r['tokens']} tokens in {r['secs']:.2f}s -> {r['tps']:.0f} tok/s")

    print(f"\n== overhead summary (out_dir: {base_env['VLLM_SNIFFER_DIR']}) ==")
    baseline_tps = None
    print(f"{'config':<38} {'tps':>9} {'vs baseline':>12}")
    for label, _ in configs:
        ok = [r for r in results[label] if "error" not in r]
        if not ok:
            print(f"{label:<38} failed")
            continue
        tps = sum(r["tps"] for r in ok) / len(ok)
        if label.startswith("baseline"):
            baseline_tps = tps
            print(f"{label:<38} {tps:>9.0f} {'--':>12}")
        else:
            delta = (tps / baseline_tps - 1) * 100 if baseline_tps else float("nan")
            print(f"{label:<38} {tps:>9.0f} {delta:>+11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
