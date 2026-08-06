"""Engine-core hooks: per-step timing, scheduler details, preemptions.

These classes run in the EngineCoreProc process (spawned by vLLM for both
online and offline modes). The plugin entry point is loaded in that process
too, so the patches apply automatically.

Patch points (vLLM v0.19, V1 engine):
- ``EngineCore.step``        (vllm/v1/engine/core.py) -- hot loop, per iteration
- ``Scheduler.schedule``     (vllm/v1/core/sched/scheduler.py) -- scheduling

Events emitted (group=core):
- ``step``      every iteration: wall duration; lightweight counts only
                (never sampled -- it is the heartbeat of the engine)
- ``schedule``  sampled: token counts, preemption count, KV block allocation
- ``preempt``   every preemption (never sampled; low frequency)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import get_config
from ..core.event import GROUP_CORE, make_event
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
# EngineCore.step / step_with_batch_queue
# ----------------------------------------------------------------------
def install_engine_core_step_hook() -> bool:
    try:
        from vllm.v1.engine.core import EngineCore
    except ImportError as e:  # pragma: no cover
        return False

    # v0.19 runs *asynchronous* scheduling by default: the hot loop calls
    # step_with_batch_queue (selected in __init__ as ``step_fn``), while
    # plain ``step`` is only used when batch_queue is None. Patch both.
    attrs = ("step", "step_with_batch_queue")
    if all(is_our_wrapper(getattr(EngineCore, a, None)) for a in attrs):
        _installed.add("engine_core_step")
        return True

    def _make_step_wrapper(orig):
        def step_wrapper(self, *args, **kwargs):
            t0 = time.monotonic_ns()
            try:
                result = orig(self, *args, **kwargs)
            except Exception:
                raise
            dur_ns = time.monotonic_ns() - t0
            try:
                data: dict[str, Any] = {"dur_ns": dur_ns}
                outputs, _ = result
                n_outputs = 0
                for engine_outputs in outputs.values():
                    try:
                        n_outputs += len(engine_outputs.outputs)
                    except Exception:
                        pass
                data["n_outputs"] = n_outputs
                emit(make_event(GROUP_CORE, "step", data=data))
            except Exception:
                pass
            return result

        return mark_wrapper(step_wrapper)

    try:
        for attr in attrs:
            orig = getattr(EngineCore, attr, None)
            if orig is None or is_our_wrapper(orig):
                continue
            setattr(EngineCore, attr, _make_step_wrapper(orig))
        _installed.add("engine_core_step")
        return True
    except Exception as e:
        _warn_hook_failed("engine_core_step", e)
        return False


# ----------------------------------------------------------------------
# Scheduler.schedule
# ----------------------------------------------------------------------
def install_scheduler_hook() -> bool:
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except ImportError as e:  # pragma: no cover
        return False

    orig = Scheduler.schedule
    if is_our_wrapper(orig):
        _installed.add("scheduler")
        return True

    def schedule_wrapper(self, *args, **kwargs):
        t0 = time.monotonic_ns()
        try:
            out = orig(self, *args, **kwargs)
        except Exception:
            raise
        dur_ns = time.monotonic_ns() - t0
        try:
            cfg = get_config()
            # Preemption events: never sampled, low frequency.
            preempted = getattr(out, "preempted_req_ids", None) or set()
            for req_id in preempted:
                emit(
                    make_event(
                        GROUP_CORE,
                        "preempt",
                        req_id=req_id,
                        data={"mode": "recompute"},
                    )
                )
            # Sampled schedule detail.
            if _should_sample(cfg):
                data: dict[str, Any] = {"dur_ns": dur_ns}
                for attr in (
                    "total_num_scheduled_tokens",
                    "num_scheduled_tokens",
                    "finished_req_ids",
                    "new_block_ids_to_zero",
                ):
                    try:
                        v = getattr(out, attr)
                        if attr in ("finished_req_ids", "new_block_ids_to_zero"):
                            # Sets/lists -> counts (keep the event compact).
                            v = None if v is None else len(v)
                        data[attr] = v
                    except Exception:
                        pass
                if "num_scheduled_tokens" in data and isinstance(
                    data["num_scheduled_tokens"], dict
                ):
                    data["n_scheduled_reqs"] = len(data["num_scheduled_tokens"])
                data["n_preempted"] = len(preempted)
                emit(
                    make_event(
                        GROUP_CORE, "schedule", sampled=True, data=data
                    )
                )
        except Exception:
            pass
        return out

    try:
        Scheduler.schedule = mark_wrapper(schedule_wrapper)
        _installed.add("scheduler")
        return True
    except Exception as e:
        _warn_hook_failed("scheduler", e)
        return False


def install_engine_hooks() -> list[str]:
    for fn in (
        install_engine_core_step_hook,
        install_scheduler_hook,
    ):
        try:
            fn()
        except Exception as e:
            _warn_hook_failed("engine", e)
    return sorted(_installed)
