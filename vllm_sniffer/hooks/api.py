"""API-server hooks: request lifecycle events.

Patch points (vLLM v0.19, V1 engine):
- ``AsyncLLMEngine.generate``  (vllm/v1/engine/async_llm.py)  -- online serving
- ``LLM.generate``             (vllm/entrypoints/llm.py)      -- offline

Events emitted (group=api):
- ``request_start``        request accepted
- ``request_first_token``  first output token observed on the stream
- ``request_finish``       stream finished (with finish_reason)

Note: the v0.19 ``AsyncLLMEngine.generate`` signature is
``(self, prompt, sampling_params, request_id, *, ...)`` -- request_id is the
3rd positional argument. We never read the prompt text (privacy; only its
type/length may be recorded). All hooks are defensive: any failure silently
disables the hook (recorded once) and never propagates into inference.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.event import GROUP_API, make_event
from ..core.writer import emit
from . import is_our_wrapper, mark_wrapper

logger = logging.getLogger("vllm_sniffer")

_installed: set[str] = set()


def _warn_hook_failed(name: str, exc: Exception) -> None:
    logger.warning("vllm-sniffer: hook '%s' disabled (%s)", name, exc)


def _summary_of_prompt(prompt: Any) -> dict[str, Any]:
    """Record prompt shape only -- never the prompt text (privacy)."""
    if isinstance(prompt, str):
        return {"kind": "str", "n_chars": len(prompt)}
    if isinstance(prompt, (list, tuple)):
        return {"kind": "list", "n": len(prompt)}
    if isinstance(prompt, dict):
        return {"kind": "dict", "n": len(prompt)}
    return {"kind": type(prompt).__name__}


# ----------------------------------------------------------------------
# Online: AsyncLLM.generate / AsyncLLMEngine.generate (async generators)
# ----------------------------------------------------------------------
def install_async_generate_hook() -> bool:
    """Patch the streaming generate() entry points.

    v0.19 API server uses ``AsyncLLM`` (vllm/v1/engine/async_llm.py) as the
    engine client; ``AsyncLLMEngine`` is the legacy class kept for
    in-process/debug paths. Patch both, each with its own wrapper (the
    closure must capture that class's original method).
    """
    targets: list[type] = []
    try:
        from vllm.v1.engine.async_llm import AsyncLLM

        targets.append(AsyncLLM)
    except ImportError as e:  # pragma: no cover - vLLM absent in tests
        pass
    try:
        from vllm.v1.engine.async_llm import AsyncLLMEngine

        targets.append(AsyncLLMEngine)
    except ImportError:
        pass
    if not targets:
        return False

    if all(
        is_our_wrapper(getattr(cls, "generate", None)) for cls in targets
    ):
        _installed.add("async_generate")
        return True

    def _make_generate_wrapper(orig):
        async def generate_wrapper(self, prompt, sampling_params, request_id, *args, **kwargs):
            import os as _os

            if _os.environ.get("VLLM_SNIFFER_DEBUG"):
                print(
                    f"[vllm-sniffer] {_os.getpid()} AsyncLLM.generate called "
                    f"req={request_id} cls={type(self).__name__}",
                    flush=True,
                )
            emit(
                make_event(
                    GROUP_API,
                    "request_start",
                    req_id=request_id,
                    data=_summary_of_prompt(prompt),
                )
            )
            first_token_seen = False
            finished = False
            total_out_tokens = 0
            try:
                async for output in orig(self, prompt, sampling_params, request_id, *args, **kwargs):
                    if _os.environ.get("VLLM_SNIFFER_DEBUG"):
                        try:
                            print(
                                f"[vllm-sniffer] {_os.getpid()} output: finished={output.finished} "
                                f"n_out={len(output.outputs)}",
                                flush=True,
                            )
                        except Exception:
                            pass
                    if not first_token_seen:
                        try:
                            if output.outputs and len(output.outputs[0].token_ids) > 0:
                                first_token_seen = True
                                emit(
                                    make_event(
                                        GROUP_API,
                                        "request_first_token",
                                        req_id=request_id,
                                        data={
                                            "n_prompt_tokens": (
                                                len(output.prompt_token_ids)
                                                if output.prompt_token_ids is not None
                                                else None
                                            ),
                                        },
                                    )
                                )
                        except Exception:
                            pass
                    # Streaming (DELTA) outputs carry only the increment;
                    # accumulate for a total at finish.
                    try:
                        if output.outputs:
                            total_out_tokens += len(output.outputs[0].token_ids)
                    except Exception:
                        pass
                    if not finished and output.finished:
                        finished = True
                        try:
                            fr = (
                                output.outputs[0].finish_reason
                                if output.outputs
                                else None
                            )
                            emit(
                                make_event(
                                    GROUP_API,
                                    "request_finish",
                                    req_id=request_id,
                                    data={
                                        "finish_reason": (
                                            fr.value if hasattr(fr, "value") else fr
                                        ),
                                        "n_output_tokens": total_out_tokens,
                                    },
                                )
                            )
                        except Exception:
                            pass
                    yield output
            except BaseException:
                # Client abort (GeneratorExit, i.e. the client disconnected) or
                # engine error: record an abort if we never reached finish, then
                # re-raise unchanged so vLLM behavior is preserved.
                if not finished:
                    try:
                        emit(make_event(GROUP_API, "request_abort", req_id=request_id))
                    except Exception:
                        pass
                raise

        return mark_wrapper(generate_wrapper)

    try:
        for cls in targets:
            orig = getattr(cls, "generate", None)
            if orig is None or is_our_wrapper(orig):
                continue
            setattr(cls, "generate", _make_generate_wrapper(orig))
        _installed.add("async_generate")
        return True
    except Exception as e:
        _warn_hook_failed("async_generate", e)
        return False


# ----------------------------------------------------------------------
# Online: AsyncLLM.abort (explicit client-abort path)
# ----------------------------------------------------------------------
def install_async_abort_hook() -> bool:
    """Patch AsyncLLM.abort(): the API server calls this explicitly when a
    client disconnects, so it is the most reliable abort signal."""
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except ImportError as e:  # pragma: no cover
        return False

    orig = AsyncLLM.abort
    if is_our_wrapper(orig):
        _installed.add("async_abort")
        return True

    def abort_wrapper(self, request_id, *args, **kwargs):
        try:
            result = orig(self, request_id, *args, **kwargs)
        except Exception:
            raise
        try:
            request_ids = (
                (request_id,) if isinstance(request_id, str) else tuple(request_id)
            )
            for rid in request_ids:
                emit(
                    make_event(
                        GROUP_API,
                        "request_abort",
                        req_id=rid,
                        data={"source": "explicit_abort"},
                    )
                )
        except Exception:
            pass
        return result

    try:
        AsyncLLM.abort = mark_wrapper(abort_wrapper)
        _installed.add("async_abort")
        return True
    except Exception as e:
        _warn_hook_failed("async_abort", e)
        return False


# ----------------------------------------------------------------------
# Offline: LLM.generate (synchronous, batch-oriented)
# ----------------------------------------------------------------------
def install_offline_generate_hook() -> bool:
    try:
        from vllm.entrypoints.llm import LLM
    except ImportError as e:  # pragma: no cover
        return False

    orig = LLM.generate
    if is_our_wrapper(orig):
        _installed.add("offline_generate")
        return True

    def generate_wrapper(self, *args, **kwargs):
        # Offline generation is batch-oriented; emit one start/finish pair.
        n_requests = _guess_batch_size(args, kwargs)
        emit(
            make_event(
                GROUP_API,
                "request_start",
                req_id=_offline_run_id(),
                data={"mode": "offline", "n_requests": n_requests},
            )
        )
        t0 = time.monotonic_ns()
        try:
            result = orig(self, *args, **kwargs)
        except Exception:
            emit(make_event(GROUP_API, "request_abort", data={"mode": "offline"}))
            raise
        dur_ns = time.monotonic_ns() - t0
        n_finished = 0
        try:
            if result is not None:
                if isinstance(result, (list, tuple)):
                    n_finished = len(result)
                else:
                    n_finished = 1
        except Exception:
            pass
        emit(
            make_event(
                GROUP_API,
                "request_finish",
                data={"mode": "offline", "n_finished": n_finished, "dur_ns": dur_ns},
            )
        )
        return result

    try:
        LLM.generate = mark_wrapper(generate_wrapper)
        _installed.add("offline_generate")
        return True
    except Exception as e:
        _warn_hook_failed("offline_generate", e)
        return False


def _guess_batch_size(args: tuple, kwargs: dict) -> int:
    """Heuristic: first positional arg is prompts (str or list of str)."""
    if args:
        p = args[0]
    elif "prompts" in kwargs:
        p = kwargs["prompts"]
    else:
        return 1
    if isinstance(p, (list, tuple)):
        return len(p)
    return 1


def _offline_run_id() -> str:
    import threading

    return f"offline-{threading.get_ident()}"


def install_api_hooks() -> list[str]:
    for fn in (
        install_async_generate_hook,
        install_async_abort_hook,
        install_offline_generate_hook,
    ):
        try:
            fn()
        except Exception as e:
            _warn_hook_failed("api", e)
    return sorted(_installed)
