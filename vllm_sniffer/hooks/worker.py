"""GPU-worker hooks: forward pass timing and float-nondeterminism probes.

These classes run in per-GPU worker processes. The plugin entry point is
loaded there as well (vllm/v1/worker/worker_base.py calls
``load_general_plugins()``), so the patches apply automatically.

Patch points (vLLM v0.19, V1 engine):
- ``GPUModelRunner.execute_model``  (vllm/v1/worker/gpu_model_runner.py)
- ``Sampler.greedy_sample``         (vllm/v1/sample/sampler.py)

Events emitted (group=worker):
- ``forward``      sampled: model forward duration + batch composition
- ``sample_flip``  never sampled: a greedy argmax sits in the "flip zone"
                   (top1-top2 logit gap < VLLM_SNIFFER_FLIP_EPS) -- the
                   token where temperature=0 output instability originates
- ``sample_stats`` sampled: aggregate margin statistics over greedy rows

Nondeterminism design notes
---------------------------
temperature=0 sampling is ``logits.argmax(dim=-1)`` (deterministic given the
logits). Different outputs for the same prompt therefore imply *different
logits*, caused by e.g. batch-composition-dependent kernel paths, prefix
cache hit/miss, or preemption recompute. The tracer records:
- argmax margin (top1 - top2): if margin < eps, float noise of that size
  can flip the output -- the "flip zone";
- full logits fingerprinting (bit-level hash) is deliberately NOT in the
  hot path; it is a future deep-dive mode (analysis-side repro compare).
All GPU work here is one extra topk(2) kernel + scalar reductions per greedy
sampling step. Disable with VLLM_SNIFFER_MARGIN=0.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import get_config
from ..core.event import GROUP_WORKER, make_event
from ..core.writer import emit
from . import is_our_wrapper, mark_wrapper

logger = logging.getLogger("vllm_sniffer")

_installed: set[str] = set()


def _warn_hook_failed(name: str, exc: Exception) -> None:
    logger.warning("vllm-sniffer: hook '%s' disabled (%s)", name, exc)


def _should_sample(cfg) -> bool:
    import random

    return cfg.sample_rate >= 1.0 or random.random() < cfg.sample_rate


# ----------------------------------------------------------------------
# GPUModelRunner.execute_model
# ----------------------------------------------------------------------
def install_execute_model_hook() -> bool:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as e:  # pragma: no cover
        return False

    orig = GPUModelRunner.execute_model
    if is_our_wrapper(orig):
        _installed.add("execute_model")
        return True

    def execute_model_wrapper(self, scheduler_output, *args, **kwargs):
        t0 = time.monotonic_ns()
        try:
            result = orig(self, scheduler_output, *args, **kwargs)
        except Exception:
            raise
        dur_ns = time.monotonic_ns() - t0
        try:
            cfg = get_config()
            if _should_sample(cfg):
                data: dict[str, Any] = {"dur_ns": dur_ns}
                try:
                    data["total_num_scheduled_tokens"] = (
                        scheduler_output.total_num_scheduled_tokens
                    )
                except Exception:
                    pass
                try:
                    batch = self.input_batch
                    data["num_tokens"] = batch.num_tokens
                except Exception:
                    pass
                try:
                    num_seqs = getattr(self.input_batch, "num_seqs", None)
                    if num_seqs is not None:
                        data["num_seqs"] = num_seqs
                except Exception:
                    pass
                emit(make_event(GROUP_WORKER, "forward", sampled=True, data=data))
        except Exception:
            pass
        return result

    try:
        GPUModelRunner.execute_model = mark_wrapper(execute_model_wrapper)
        _installed.add("execute_model")
        return True
    except Exception as e:
        _warn_hook_failed("execute_model", e)
        return False


# ----------------------------------------------------------------------
# Sampler.greedy_sample: argmax-margin observation
# ----------------------------------------------------------------------
def install_sampler_margin_hook() -> bool:
    cfg = get_config()
    if not cfg.margin:
        return False
    try:
        import torch  # noqa: F401
        from vllm.v1.sample.sampler import Sampler
    except ImportError as e:  # pragma: no cover
        return False

    orig = Sampler.greedy_sample
    if is_our_wrapper(orig):
        _installed.add("sampler_margin")
        return True

    def greedy_sample_wrapper(logits):
        sampled = orig(logits)
        cfg = get_config()
        # Runtime check in addition to the install-time check: the wrapper
        # may persist on a class (e.g. re-install in tests), and re-reading
        # the cached config is free.
        if not cfg.margin:
            return sampled
        try:
            top2v, top2i = torch.topk(logits, 2, dim=-1)
            margin = top2v[:, 0] - top2v[:, 1]  # [B] float32, on GPU
            eps = cfg.flip_eps
            # Flip-zone detection stays on GPU; only the rare flips are
            # copied to host (avoids per-step device sync for every row).
            mask = margin < eps
            n_flips = int(mask.sum().item())  # single scalar sync
            if n_flips > 0:
                flip_rows = mask.nonzero(as_tuple=False).flatten()
                flip_top1 = top2i[flip_rows, 0].tolist()
                flip_top2 = top2i[flip_rows, 1].tolist()
                flip_margin = margin[flip_rows].tolist()
                for tok1, tok2, m in zip(flip_top1, flip_top2, flip_margin):
                    emit(
                        make_event(
                            GROUP_WORKER,
                            "sample_flip",
                            data={
                                "margin": m,
                                "top1": tok1,
                                "top2": tok2,
                            },
                        )
                    )
            if _should_sample(cfg):
                emit(
                    make_event(
                        GROUP_WORKER,
                        "sample_stats",
                        sampled=True,
                        data={
                            "n_greedy": int(margin.numel()),
                            "margin_min": float(margin.min().item()),
                            "margin_max": float(margin.max().item()),
                            "margin_mean": float(margin.mean().item()),
                            "n_flips": n_flips,
                        },
                    )
                )
        except Exception:
            # Observation must never alter or break sampling.
            pass
        return sampled

    try:
        Sampler.greedy_sample = staticmethod(mark_wrapper(greedy_sample_wrapper))
        _installed.add("sampler_margin")
        return True
    except Exception as e:
        _warn_hook_failed("sampler_margin", e)
        return False


def install_worker_hooks() -> list[str]:
    for fn in (install_execute_model_hook, install_sampler_margin_hook):
        try:
            fn()
        except Exception as e:
            _warn_hook_failed("worker", e)
    return sorted(_installed)
