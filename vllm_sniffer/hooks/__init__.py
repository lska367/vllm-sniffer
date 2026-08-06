"""Hook installation. Idempotent; safe to call in every vLLM process.

vLLM loads the ``vllm.general_plugins`` entry point in every process
(API server via engine/arg_utils.py, engine core via v1/engine/core.py,
GPU workers via v1/worker/worker_base.py, model registry). Each hook is
installed defensively: if the target class is missing or patching fails,
we log once and continue -- the tracer degrades gracefully and never
changes vLLM behavior.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vllm_sniffer")

_installed: bool = False


def is_our_wrapper(fn: object) -> bool:
    """True if ``fn`` is a wrapper previously installed by vllm-sniffer.

    Used to make hook installation idempotent per class: installing twice
    must not double-wrap (which would double-emit events).
    """
    return bool(getattr(fn, "_sniffer_wrapper", False))


def mark_wrapper(fn):
    fn._sniffer_wrapper = True
    return fn


def install_all() -> list[str]:
    global _installed
    if _installed:
        return []
    _installed = True

    installed: list[str] = []
    from .api import install_api_hooks
    from .engine import install_engine_hooks
    from .worker import install_worker_hooks

    for fn in (install_api_hooks, install_engine_hooks, install_worker_hooks):
        try:
            installed.extend(fn())
        except Exception as e:
            logger.warning("vllm-sniffer: hook group %s failed (%s)", fn.__name__, e)
    return installed
