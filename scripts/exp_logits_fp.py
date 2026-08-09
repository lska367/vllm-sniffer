"""exp_logits_fp.py -- logits bit-level fingerprint experiment (P1-1, GPU).

Research question: where do the logits differences come from when the same
prompt is served in different batch compositions? The tracer's deep-dive
mode (VLLM_SNIFFER_LOGITS_FP=1) records per-step bit-level fingerprints of
the raw logits; this script runs a controlled A/B:

- Arm "solo": the probe prompt alone in the batch (batch size 1).
- Arm "mixed": the probe prompt alongside N-1 filler prompts, so the
  kernel paths / GEMM shapes differ at the probe's row.

Each arm runs in its own subprocess (env isolation -- required: a forked
engine process inherits the parent's cached VLLM_SNIFFER_DIR, so two arms
in one process would write both into the first arm's run dir; 2026-08-09
real-machine finding). The two runs are then compared with
tools/logits_fp_compare.py:

- control: solo vs solo  -> expect IDENTICAL (bit delta = 0)
- hypothesis: solo vs mixed -> typically LOW-BIT NOISE (mantissa bits)

Usage (GPU machine with vLLM installed, e.g. .venv-gpu):
    .venv-gpu/bin/python scripts/exp_logits_fp.py --n 8 --max-tokens 32
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

# Runs in a fresh subprocess per arm. vLLM is imported only here, so the
# plugin's load() picks up the arm's VLLM_SNIFFER_DIR from the env passed
# by the parent. Prints "DONE <n>" on success.
ARM_WORKLOAD = r"""
import json, sys

model, prompts_json, max_tokens, gpu_mem = sys.argv[1:5]
prompts = json.loads(prompts_json)
max_tokens, gpu_mem = int(max_tokens), float(gpu_mem)

from vllm import LLM, SamplingParams

llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=gpu_mem)
outs = llm.generate(prompts, sampling_params=SamplingParams(
    temperature=0, max_tokens=max_tokens))
print(f"DONE {len(outs)}")
sys.stdout.flush()
"""


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
    """Run one arm in a subprocess; return its sniffer run directory."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sniffer_dir = os.path.join(args.out_dir, f"exp-logits-fp-{tag}-{ts}")
    env = os.environ.copy()
    # Scrub tracer state that would otherwise leak into the child (a forked
    # engine process would inherit the parent's cached config/run_id).
    for k in list(env):
        if k.startswith("VLLM_SNIFFER"):
            env.pop(k)
    env["VLLM_SNIFFER_DIR"] = sniffer_dir
    env["VLLM_SNIFFER_LOGITS_FP"] = "1"
    cmd = [
        sys.executable, "-c", ARM_WORKLOAD,
        args.model, json.dumps(prompts, ensure_ascii=False),
        str(args.max_tokens), str(args.gpu_memory_utilization),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=3600)
    if proc.returncode != 0 or "DONE" not in proc.stdout:
        tail = (proc.stderr or proc.stdout)[-500:]
        raise RuntimeError(
            f"arm {tag} failed (exit {proc.returncode}): {tail}"
        )
    print(f"arm {tag}: {len(prompts)} greedy request(s) -> {sniffer_dir}")
    return sniffer_dir


def compare(label: str, a: str, b: str) -> None:
    print(f"\n== {label} ==")
    subprocess.run(
        [sys.executable, "tools/logits_fp_compare.py", a, b, "--align", "index"],
        timeout=300,
    )


def main() -> int:
    args = build_parser().parse_args()
    probe = args.probe_prompt or _default_probe()
    try:
        # Arm A: probe alone (batch composition = 1 row).
        solo_dir = run_arm(args, "solo", [probe])
        # Arm B: probe + N-1 fillers (batch composition changes the kernel
        # paths the probe's row goes through).
        mixed = [probe] + [_default_filler(i) for i in range(args.n - 1)]
        mixed_dir = run_arm(args, "mixed", mixed)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        print("this experiment needs a GPU environment with vLLM installed "
              "(e.g. .venv-gpu)", file=sys.stderr)
        return 2

    compare("control: solo vs solo (expect ~IDENTICAL)", solo_dir, solo_dir)
    compare("hypothesis: solo vs mixed (expect low-bit noise)",
            solo_dir, mixed_dir)
    print(f"\nrun dirs: {solo_dir}\n          {mixed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
