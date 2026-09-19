# vllm-sniffer

[![CI](https://img.shields.io/github/actions/workflow/status/lska367/vllm-sniffer/ci.yml?branch=master&label=CI&logo=github)](https://github.com/lska367/vllm-sniffer/actions)
[![Python 3.10 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](https://github.com/lska367/vllm-sniffer/blob/master/pyproject.toml)
[![License](https://img.shields.io/github/license/lska367/vllm-sniffer)](LICENSE)

A non-invasive runtime tracer for [vLLM](https://github.com/vllm-project/vllm) inference:
observes the hot paths (scheduling, forward, sampling, KV-cache pressure) and helps
localize the root cause of the most annoying production issue — **the same prompt
with `temperature=0` produces different outputs** (float nondeterminism).

- **Zero code intrusion**: auto-loaded through vLLM's official plugin mechanism
  (the `vllm.general_plugins` entry point). No vLLM source changes, no behavior change
- **Full process coverage**: hooks are active in all three processes — API server,
  engine core, and GPU workers
- **JSONL event stream**: `/tmp/vllm-sniffer/<run_id>/<pid>.jsonl`, stable schema,
  designed for a visualization frontend
- **Transparent, configurable overhead**: the hot path collects only lightweight
  fields; the argmax-margin probe (one extra topk(2) kernel) can be turned off with
  `VLLM_SNIFFER_MARGIN=0`; deep-dive fingerprinting is off by default
  (`VLLM_SNIFFER_LOGITS_FP=0`); `VLLM_SNIFFER=0` disables everything

## Install

```bash
pip install -e .
# Active immediately; disable with: export VLLM_SNIFFER=0
```

No changes to launch commands. vLLM loads the plugin automatically in every
process at import time.

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `VLLM_SNIFFER` | `1` | Master switch |
| `VLLM_SNIFFER_DIR` | `/tmp/vllm-sniffer` | Output root directory |
| `VLLM_SNIFFER_SAMPLE_RATE` | `1.0` | Sampling rate for hot events (step/schedule/forward/sample_stats) |
| `VLLM_SNIFFER_MARGIN` | `1` | Greedy argmax-margin observation (one extra topk(2) kernel) |
| `VLLM_SNIFFER_FLIP_EPS` | `1e-3` | Flip-zone threshold (top1-top2 logit gap) |
| `VLLM_SNIFFER_LOGITS_FP` | `0` | **Deep-dive mode**: bit-level logits fingerprint (per-row 32-bit bit counts; off by default — zero-overhead promise unchanged) |
| `VLLM_SNIFFER_LOGITS_FP_ROWS` | `8` | Fingerprint rows sampled per step (strided, cap 64) |

## Event types

| Event | Group | Frequency | Contents |
|---|---|---|---|
| `env_snapshot` | core | once per run | **Reference frame**: vLLM/torch/CUDA/python versions, determinism-related env vars, tracer config (emitted by the main process; children inherit run_id without duplicating) |
| `request_start` / `request_first_token` / `request_finish` / `request_abort` | api | per request | Request lifecycle (the skeleton for TTFT/TPOT debugging) |
| `step` | core | per step | EngineCore.step duration, output count (engine heartbeat) |
| `schedule` | core | sampled | Scheduled token count, queue size, KV-block allocations |
| `preempt` | core | per event | Preemption (recompute mode in v1) |
| `forward` | worker | sampled | Forward duration, batch composition |
| `sample_flip` | worker | per event | **argmax in flip zone**: margin < eps (root-cause localization for temp=0 instability) |
| `sample_stats` | worker | sampled | Greedy-batch margin aggregates (min/max/mean, flip count) |
| `logits_fp` | worker | sampled | **Bit-level logits fingerprint** (deep mode `VLLM_SNIFFER_LOGITS_FP=1`): per-sampled-row bit counts to separate low-bit mantissa noise from systematic exponent drift |

Common event fields: `schema_ver` / `ts_ns` (wall clock) / `pid` / `group` / `type` /
`req_id` / `step` / `sampled` / `data`. **Prompts are never recorded** (only type/length).

## Architecture

```
vllm-sniffer (pip package, entry point: vllm.general_plugins)
├─ API server process: request lifecycle hooks
├─ EngineCore process: EngineCore.step / Scheduler.schedule hooks
├─ Worker process:     GPUModelRunner.execute_model / Sampler.greedy_sample hooks
└─ per process: lock-free bounded queue → daemon thread → JSONL (drop on full, never blocks inference)
```

Cross-process correlation keys: `req_id` + `step` + `ts_ns`. Multi-process files
are aggregated per `run_id` directory.

## Design principles

1. **Behavior-preserving**: every hook returns the original value; hook exceptions
   degrade silently (logged, then that hook point is disabled) and are never raised
   into the inference path; installation is idempotent per class (repeated loading
   never double-wraps)
2. **Transparent overhead**: hot paths collect only lightweight fields; flip detection
   completes on GPU and only rare flips are copied back to host; a full queue drops
   events instead of blocking. Expensive modes (margin probe, bit fingerprints) have
   their own switches and are off unless explicitly enabled
3. **Disableable**: with `VLLM_SNIFFER=0` the plugin entry returns immediately,
   equivalent to not being installed

## Float-nondeterminism observation (temp=0 output diversity)

With `temperature=0`, vLLM takes `logits.argmax` — sampling itself is deterministic;
different outputs mean slightly different logits. Known root-cause directions: batch
composition change (vLLM's official `VLLM_BATCH_INVARIANT` exists for exactly this),
prefix-cache hit/miss, preemption recomputation, cudagraph graph switching, and
multi-GPU AllReduce ordering.

The tracer observes the **argmax margin (top1−top2)** — when margin < `FLIP_EPS`,
floating-point noise can flip the output. `sample_flip` events mark these high-risk
positions; pairwise comparison (same prompt across requests) happens in the analysis
layer — the tracer itself stays stateless.

## Analysis tools (`tools/`)

The tracer produces raw event streams; the analysis layer tells the story.
Four standalone CLIs, each taking a run directory or a parquet file:

```bash
# 1) Merge multi-pid JSONL → single parquet (optional dep: pip install -e '.[analyze]')
python tools/export_parquet.py /tmp/vllm-sniffer/<run_id> -o run.parquet

# 2) TTFT / TPOT / step distributions (p50/p90/p99 + ASCII histograms)
python tools/latency_report.py run.parquet            # or pass a run directory

# 3) temp=0 pairwise comparison: output-length consistency + flip density + per-request attribution
python tools/repro_compare.py run.parquet

# 4) logits bit-fingerprint comparison (deep-mode output): per-bit difference
#    distribution + verdict (IDENTICAL ≈0 / LOW-BIT NOISE / SYSTEMATIC exponent bits)
python tools/logits_fp_compare.py run-a run-b         # two run dirs or parquet files
```

Sample output:

```
TTFT : n=42 p50=154.00 p90=210.00 p99=330.00 max=412.00 mean=170.00 (ms)
flips (from stats): 19  (29.69%)     # 19 of 64 output positions in flip zone
```

## Visualization frontend (`webapp/`)

Offline-analysis web app: FastAPI aggregation endpoints + self-built frontend
(vanilla JS + canvas, no CDN). Reads run directories or parquet, four views:
request timeline (TTFT/TPOT), step-duration series, flip-distribution heatmap,
batch-composition vs latency scatter.

```bash
uv pip install --python .venv/bin/python -e '.[web]'    # optional dependency
.venv/bin/python -m webapp.server --dir /tmp/vllm-sniffer --port 8080
# open http://127.0.0.1:8080 → pick a run_id → four charts
```

API docs: [doc/WEBAPP.md](doc/WEBAPP.md) (`/api/runs`, `/api/runs/{id}/timeline`, and 5 more).

## Experiment scripts (`scripts/`)

```bash
# temp=0 determinism experiment: same prompt × N, per-position diff + unique-sequence count
.venv-gpu/bin/python scripts/exp_determinism.py --n 8 --max-tokens 64

# Overhead quantification: baseline / margin-off / margin-on arms (subprocess env isolation)
.venv-gpu/bin/python scripts/exp_overhead.py --n 16 --max-tokens 64 --repeat 3
```

Smoke scripts: `scripts/offline_smoke.py` (eager), `scripts/offline_smoke_cudagraph.py`,
`scripts/online_smoke.py` (api_server + streaming + client disconnect).

## Roadmap

- [x] Injection layer (plugin auto-load, idempotent, silent degradation)
- [x] Request lifecycle (online + offline)
- [x] step / schedule / preempt / forward
- [x] Greedy argmax-margin float observation (flip detection + aggregate stats)
- [x] env_snapshot (vLLM commit / determinism env vars / cudagraph state)
- [x] Analysis tools: JSONL→parquet export, TTFT/TPOT distributions, repro comparison (flip attribution)
- [x] Logits bit-level fingerprint (deep mode, off by default) — [EXTENSION_ROADMAP](doc/EXTENSION_ROADMAP.md) P1-1
- [x] Visualization frontend (self-built, reads JSONL/aggregation APIs) — P2-1, [WEBAPP.md](doc/WEBAPP.md)
- [ ] Multi-GPU TP/PP (rank field reserved in the event schema) and Ray clusters (node_id + local disk)
- [ ] OTLP sink (pluggable, for existing observability stacks)

## Documentation

- `doc/README.md` — doc index with a suggested reading path
- `doc/ARCHITECTURE.md` — architecture analysis (process model, data flow, hook points, design principles)
- `doc/CODE_WALKTHROUGH.md` — code walkthrough guide (with a vLLM source map and self-test questions)
- `doc/EVENT_SCHEMA.md` — event-stream schema reference
- `doc/EXTENSION_ROADMAP.md` — extension roadmap (prioritized)
- `doc/WEBAPP.md` — frontend usage and aggregation API docs
- `doc/RESEARCH_PLATFORM.md` — research-platform planning (collect/analyze/visualize)
- `doc/RESUME_PROJECT.md` — resume materials (bilingual), incl. interview Q&A with honest boundaries
- `doc/VALIDATION_LOG.md` — real-machine validation log (internal evidence archive)
- `doc/CONTRIBUTING.md` — contributing guide
- `CHANGELOG.md` — changelog

## Development

```bash
uv venv .venv && uv pip install --python .venv/bin/python pytest msgspec torch --index-url https://download.pytorch.org/whl/cpu
# analysis tools (parquet export) need: uv pip install --python .venv/bin/python 'pyarrow>=14'
# visualization frontend needs: uv pip install --python .venv/bin/python -e '.[web]'
.venv/bin/python -m pytest
```

Tests do not require a real vLLM: fake modules are injected into `sys.modules`
while exercising the real installed code paths; torch validates the sampling probes.
CI (GitHub Actions) runs the full suite on Python 3.10/3.12 × CPU, including the
webapp API tests.