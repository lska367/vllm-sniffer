"""Environment snapshot: the reference frame of a run.

Float-nondeterminism attribution needs to know *what environment* a run
happened in: vLLM version/commit, torch/CUDA versions, determinism-related
env vars (``VLLM_BATCH_INVARIANT``, ``VLLM_FLOAT32_MATMUL_PRECISION``,
``VLLM_USE_CUDA_GRAPH``, ...), and the tracer's own configuration. Without
it, any repro comparison is missing its baseline.

Design:

- Emitted exactly **once per run**, from the process that *creates* the run
  directory (the API-server / offline main process). Forked or spawned
  children inherit ``VLLM_SNIFFER_RUN_ID`` and skip emission, so the
  snapshot is never duplicated.
- Emitted as a normal ``core`` event (``type=env_snapshot``) so it lives in
  the same JSONL timeline as everything else (``schema_ver`` applies; the
  analysis layer can also use it as the run's identity record).
- All field extraction is defensive: missing packages / attributes degrade
  to ``None`` and never raise (tracer must never break inference).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from ..config import get_config
from ..core.event import GROUP_CORE, make_event
from ..core.writer import RUN_ID_ENV, emit

logger = logging.getLogger("vllm_sniffer")

# Env vars that matter for float determinism / execution path selection.
# Only *present* ones are recorded (values are short strings).
DETERMINISM_ENV_VARS = (
    # vLLM official determinism switches
    "VLLM_BATCH_INVARIANT",
    "VLLM_FLOAT32_MATMUL_PRECISION",
    "VLLM_USE_CUDA_GRAPH",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_TORCH_COMPILE_LEVEL",
    # CUDA / torch numerics
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "CUDA_LAUNCH_BLOCKING",
    # sniffer itself
    "VLLM_SNIFFER_MARGIN",
    "VLLM_SNIFFER_FLIP_EPS",
    "VLLM_SNIFFER_SAMPLE_RATE",
    "VLLM_SNIFFER_LOGITS_FP",
    "VLLM_SNIFFER_LOGITS_FP_ROWS",
)

_emitted: bool = False


def _pkg_version(pkg: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(pkg)
    except Exception:
        return None


def _vllm_info() -> dict[str, Any]:
    """vLLM version + commit, if importable."""
    out: dict[str, Any] = {}
    try:
        import vllm

        v = getattr(vllm, "__version__", None)
        out["vllm_version"] = str(v) if v else None
        # Some builds expose the git commit (vllm.version.__commit__ or
        # __git_version__); probe defensively.
        try:
            from vllm import version as _vmod

            for attr in ("__commit__", "commit", "__git_version__"):
                v = getattr(_vmod, attr, None)
                if isinstance(v, str) and v:
                    out["vllm_commit"] = v
                    break
        except Exception:
            pass
    except Exception:
        pass
    out.setdefault("vllm_version", _pkg_version("vllm"))
    out.setdefault("vllm_commit", None)
    return out


def _torch_info() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch

        # torch.__version__ is a TorchVersion object, not a plain str;
        # normalize so msgspec can encode it.
        out["torch_version"] = str(torch.__version__)
        out["torch_cuda"] = str(torch.version.cuda) if torch.version.cuda else None
        out["torch_git"] = (
            str(torch.version.git_version) if torch.version.git_version else None
        )
    except Exception:
        out["torch_version"] = _pkg_version("torch")
        out["torch_cuda"] = None
        out["torch_git"] = None
    return out


def collect_env_snapshot() -> dict[str, Any]:
    """Collect the run's reference-frame fields (pure; no side effects)."""
    cfg = get_config()
    env: dict[str, str] = {}
    for name in DETERMINISM_ENV_VARS:
        v = os.environ.get(name)
        if v is not None:
            env[name] = v
    return {
        "run_id": os.environ.get(RUN_ID_ENV),
        "python_version": sys.version.split()[0],
        "vllm": _vllm_info(),
        "torch": _torch_info(),
        "env": env,
        "sniffer": {
            "enabled": cfg.enabled,
            "sample_rate": cfg.sample_rate,
            "margin": cfg.margin,
            "flip_eps": cfg.flip_eps,
            "logits_fp": cfg.logits_fp,
            "logits_fp_rows": cfg.logits_fp_rows,
        },
    }


def emit_env_snapshot(*, is_root: bool = False) -> bool:
    """Emit one ``env_snapshot`` event, but only from the run-root process.

    ``is_root`` must be decided *before* ``prepare_run_dir()`` exports
    ``VLLM_SNIFFER_RUN_ID`` (the plugin entry point does exactly that: the
    process that finds the env var unset is the run root). Children --
    spawned processes that re-import the plugin -- inherit the env var,
    pass ``is_root=False`` and emit nothing, so the snapshot is emitted
    exactly once per run. The module-level ``_emitted`` guard additionally
    protects against double emission within one process (e.g. plugin
    loaded twice in tests).
    """
    global _emitted
    if _emitted:
        return False
    if not is_root:
        return False
    _emitted = True
    try:
        emit(make_event(GROUP_CORE, "env_snapshot", data=collect_env_snapshot()))
        return True
    except Exception:
        # Observation must never raise into plugin load.
        logger.exception("vllm-sniffer: failed to emit env_snapshot")
        return False
