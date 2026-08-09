"""Process-local engine step counter, shared by core and worker hooks.

Why: worker-side events (forward / sample_flip / sample_stats / logits_fp)
need the engine iteration index for cross-process correlation, but the
Sampler and GPUModelRunner do not know it. With TP=1 (single GPU) the model
execution runs *in the EngineCore process*, so a process-global counter
incremented by the engine step wrapper is visible to all hooks:

- the step wrapper calls ``next_step()`` when an iteration starts;
- schedule / forward / sampler hooks read ``current_step()`` (they fire
  inside the same iteration, so they see the right index).

Warmup model executions happen outside the step loop -> they read 0, which
is exactly the "virtual batch" marker the analysis layer already splits on.

TP>1: worker processes are separate, the counter does not propagate;
per-rank step correlation stays future work (see EXTENSION_ROADMAP P2-2).
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_counter = 0


def next_step() -> int:
    """Increment and return the current engine iteration index."""
    global _counter
    with _lock:
        _counter += 1
        return _counter


def current_step() -> int:
    """The iteration index of the step currently executing (0 = outside)."""
    return _counter


def reset_step_counter() -> None:
    """Reset to 0. Tests only (conftest isolates hook state per test)."""
    global _counter
    with _lock:
        _counter = 0
