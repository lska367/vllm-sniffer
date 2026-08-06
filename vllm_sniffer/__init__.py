"""vllm-sniffer: non-invasive runtime tracer for vLLM.

Entry point for the ``vllm.general_plugins`` plugin group. vLLM calls
``load()`` in every process at import time; the function must be idempotent
and must never raise (a failing tracer must never break inference).
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("vllm_sniffer")

_loaded: bool = False


def load() -> None:
    """Plugin entry point invoked by vLLM (vllm.general_plugins)."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    from .config import get_config

    if not get_config().enabled:
        logger.debug("vllm-sniffer disabled (VLLM_SNIFFER=0)")
        return

    import os

    if os.environ.get("VLLM_SNIFFER_DEBUG"):
        print(
            f"[vllm-sniffer] load() in pid={os.getpid()} "
            f"run_id_env={os.environ.get('VLLM_SNIFFER_RUN_ID')!r} "
            f"dir={os.environ.get('VLLM_SNIFFER_DIR')!r}",
            flush=True,
        )

    try:
        from .hooks import install_all

        installed = install_all()
        # Fix the run_id and export it (VLLM_SNIFFER_RUN_ID) *before* vLLM
        # forks/spawns the engine core / worker processes, so all processes
        # of one launch land in the same run directory.
        from .core.writer import prepare_run_dir

        prepare_run_dir()
        if os.environ.get("VLLM_SNIFFER_DEBUG"):
            print(
                f"[vllm-sniffer] pid={os.getpid()} hooks_installed={installed}",
                flush=True,
            )
        if installed:
            logger.info("vllm-sniffer: hooks installed: %s", ", ".join(installed))
    except Exception:  # pragma: no cover - last-resort safety
        import traceback

        # vLLM's logging config may not surface the "vllm_sniffer" logger;
        # always dump to stderr so a failing tracer is diagnosable.
        traceback.print_exc()
        logger.exception("vllm-sniffer: failed to install hooks; tracing disabled")


__all__ = ["load"]
