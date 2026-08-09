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
- ``logits_fp``    sampled, deep-dive mode only (VLLM_SNIFFER_LOGITS_FP=1):
                   bit-level fingerprint of the raw logits -- per sampled
                   row, the set-bit count of each fp32 bit position. Lets
                   analysis localize logit differences to mantissa bits
                   (low-bit noise) vs exponent/sign bits (systematic scale
                   differences) -- see tools/logits_fp_compare.py.

Nondeterminism design notes
---------------------------
temperature=0 sampling is ``logits.argmax(dim=-1)`` (deterministic given the
logits). Different outputs for the same prompt therefore imply *different
logits*, caused by e.g. batch-composition-dependent kernel paths, prefix
cache hit/miss, or preemption recompute. The tracer records:
- argmax margin (top1 - top2): if margin < eps, float noise of that size
  can flip the output -- the "flip zone";
- logits bit-level fingerprints (deep-dive mode, default OFF): strided-row
  fp32 bit-pattern counts. Deliberately NOT in the default hot path -- the
  zero-overhead promise holds unless VLLM_SNIFFER_LOGITS_FP=1.
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
def _logits_fp_data(cfg, logits, top1_all):
    """Bit-level fingerprint of the raw logits (deep-dive mode).

    Returns the event ``data`` dict or ``None`` (unsupported dtype / empty
    batch / degenerate shape). For every strided-sampled row the fp32 bit
    pattern is decomposed into per-bit set counts: ``bits[b]`` = number of
    logits in that row whose bit ``b`` is set. Two logits vectors that
    differ anywhere produce different per-bit counts; the *distribution*
    of the differences across bit positions then tells mantissa noise
    (bits 0..22) apart from exponent/sign-level systematic differences
    (bits 23..31) -- that is what tools/logits_fp_compare.py consumes.

    Cost (deep-dive mode only, default off): one view + nbits small
    reductions over <= cfg.logits_fp_rows strided rows + one device sync
    of k x nbits ints per sampled greedy step.
    """
    try:
        import torch
    except ImportError:
        return None
    nbits = {
        torch.float32: 32,
        torch.float16: 16,
        torch.bfloat16: 16,
    }.get(logits.dtype)
    if nbits is None or logits.ndim != 2 or logits.shape[0] == 0:
        return None
    if not logits.is_contiguous():
        logits = logits.contiguous()
    b = logits.shape[0]
    n_rows = max(1, min(cfg.logits_fp_rows, b))
    # Strided row sampling: evenly spaced rows across the batch, so the
    # fingerprint covers the batch composition without copying it.
    rows_idx = (
        torch.linspace(0, b - 1, n_rows, device=logits.device)
        .round()
        .long()
    )
    xs = logits.view(torch.int32 if nbits == 32 else torch.int16)[rows_idx]
    bits = torch.zeros((n_rows, nbits), dtype=torch.int64, device=logits.device)
    for bi in range(nbits):
        bits[:, bi] = ((xs >> bi) & 1).sum(dim=1)
    rows = []
    rows_idx_l = rows_idx.tolist()
    for i, r in enumerate(rows_idx_l):
        row: dict[str, Any] = {"row": int(r), "bits": bits[i].tolist()}
        if top1_all is not None:
            row["top1"] = int(top1_all[r])
        rows.append(row)
    return {
        "n_rows": int(b),
        "dtype": str(logits.dtype),
        "sampled_rows": n_rows,
        "rows": rows,
    }


def install_sampler_margin_hook() -> bool:
    """Install the greedy-sampler observation wrapper.

    Installed when *either* probe is enabled (``VLLM_SNIFFER_MARGIN=1`` or
    ``VLLM_SNIFFER_LOGITS_FP=1``); the runtime wrapper re-checks both so
    leftover wrappers never do work they should not. ``logits_fp`` is the
    deep-dive mode (default off): strided-row fp32 bit-pattern counts,
    emitted as ``logits_fp`` events under the same sampling gate as
    ``sample_stats``.

    Note on logits dtype: vLLM V1 ``Sampler.forward`` converts logits to
    float32 before sampling, so production fingerprints are 32-bit; fp16 /
    bf16 are handled defensively with 16-bit fingerprints (same contract,
    shorter ``bits`` list).
    """
    cfg = get_config()
    if not (cfg.margin or cfg.logits_fp):
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
        if not (cfg.margin or cfg.logits_fp):
            return sampled
        try:
            if cfg.margin:
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
            if cfg.logits_fp and _should_sample(cfg):
                top1_all = None
                if cfg.margin:
                    # Reuse the margin probe's topk output (top1 per row is
                    # alignment context for the compare tool).
                    top1_all = top2i[:, 0]
                else:
                    top1_all = torch.argmax(logits, dim=-1)
                fp_data = _logits_fp_data(cfg, logits, top1_all)
                if fp_data is not None:
                    emit(
                        make_event(
                            GROUP_WORKER,
                            "logits_fp",
                            sampled=True,
                            data=fp_data,
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
