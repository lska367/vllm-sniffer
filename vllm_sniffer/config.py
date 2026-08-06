"""vllm-sniffer configuration, read from environment variables.

All knobs are env-based so that no code change is required in the target
deployment. The tracer is *on* by default once the package is installed;
set ``VLLM_SNIFFER=0`` to fully disable (the plugin entry point then returns
immediately and vLLM behavior is untouched).

Env vars
--------
VLLM_SNIFFER : 0|1
    Master switch. Default 1 (installed == enabled).
VLLM_SNIFFER_DIR : str
    Root output directory. Default /tmp/vllm-sniffer.
VLLM_SNIFFER_SAMPLE_RATE : float in (0, 1]
    Sampling rate for high-frequency event types (step/schedule/forward).
    Request-level events (start/first_token/finish) and preemption events
    are never sampled. Default 1.0.
VLLM_SNIFFER_MARGIN : 0|1
    Enable greedy argmax-margin observation (top1-top2 logit gap) in the
    sampler. Costs one extra GPU topk(2) kernel per greedy sampling step.
    Default 1.
VLLM_SNIFFER_FLIP_EPS : float
    Logit gap below which a token is considered a "flip zone" (the argmax
    can be toggled by float noise). Default 1e-3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    out_dir: str = "/tmp/vllm-sniffer"
    sample_rate: float = 1.0
    margin: bool = True
    flip_eps: float = 1e-3
    # Filled in lazily once the writer is created (first event).
    run_id: str | None = field(default=None, init=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_config() -> Config:
    sample_rate = _env_float("VLLM_SNIFFER_SAMPLE_RATE", 1.0)
    sample_rate = max(0.0, min(1.0, sample_rate))
    return Config(
        enabled=_env_bool("VLLM_SNIFFER", True),
        out_dir=os.getenv("VLLM_SNIFFER_DIR", "/tmp/vllm-sniffer"),
        sample_rate=sample_rate,
        margin=_env_bool("VLLM_SNIFFER_MARGIN", True),
        flip_eps=_env_float("VLLM_SNIFFER_FLIP_EPS", 1e-3),
    )
