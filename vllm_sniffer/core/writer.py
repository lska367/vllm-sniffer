"""Per-process JSONL writer.

Design constraints (budget: <1% overhead, never block the inference path):

- A bounded queue decouples hooks (producer) from disk I/O (consumer).
  If the queue is full, events are dropped and a counter is bumped; the
  inference path is never blocked by slow disk.
- One file per process: ``<out_dir>/<run_id>/<pid>.jsonl``. All vLLM
  processes in one launch share the same ``run_id`` directory (they are
  spawned within the same second in practice). The analysis layer merges
  files by ``run_id`` and correlates by ``req_id``/``step``/``ts_ns``.
- A daemon thread owns the file handle; hooks only enqueue.
"""

from __future__ import annotations

import datetime
import os
import queue
import threading
import time

import msgspec

from ..config import get_config
from .event import Event, encode_line

_SENTINEL = object()

# Max queued events per process. Full queue -> drop, never block.
_MAX_QUEUE = 8192
# Flush the file every N lines or every _FLUSH_SECS seconds (whichever
# comes first). Frequent flushing limits tail loss: vLLM engine-core/worker
# processes can exit without running atexit (os._exit paths), so buffered
# events at shutdown may be lost -- keep the buffer small.
_FLUSH_EVERY = 32
_FLUSH_SECS = 1.0

# Env var used by parent processes to propagate the run_id to spawned
# vLLM processes (engine core / workers are spawned seconds later and must
# join the same run directory).
RUN_ID_ENV = "VLLM_SNIFFER_RUN_ID"


class JsonlWriter:
    """One writer per process. Lazily opens the file on first event."""

    def __init__(self, out_dir: str, run_id: str):
        self._q: "queue.Queue[Event | object]" = queue.Queue(maxsize=_MAX_QUEUE)
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._closed = False
        self._run_id = run_id
        self._dir = os.path.join(out_dir, run_id)
        self._path = os.path.join(self._dir, f"{os.getpid()}.jsonl")
        self._thread = threading.Thread(
            target=self._run, name="vllm-sniffer-writer", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Producer side (called from hooks; must be non-blocking).
    # ------------------------------------------------------------------
    def emit(self, event: Event) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait(event)
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1

    @property
    def dropped(self) -> int:
        with self._drop_lock:
            return self._dropped

    @property
    def path(self) -> str:
        return self._path

    def close(self) -> None:
        """Flush remaining events and stop the writer thread.

        Called atexit; safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        # Join so buffered events are flushed before we return (the analysis
        # layer may read the file right after shutdown).
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Consumer side (daemon thread).
    # ------------------------------------------------------------------
    def _run(self) -> None:
        os.makedirs(self._dir, exist_ok=True)
        lines = 0
        last_flush = time.monotonic()
        try:
            with open(self._path, "ab", buffering=0) as f:
                buf = bytearray()
                while True:
                    try:
                        item = self._q.get(timeout=_FLUSH_SECS)
                    except queue.Empty:
                        # Time-based flush: also fires when no events arrive
                        # (a quiet tail of the stream must not sit in the
                        # buffer forever).
                        if buf:
                            f.write(buf)
                            f.flush()
                            buf.clear()
                            lines = 0
                        continue
                    if item is _SENTINEL:
                        f.write(buf)
                        f.flush()
                        return
                    buf += encode_line(item) + b"\n"
                    lines += 1
                    now = time.monotonic()
                    if lines >= _FLUSH_EVERY or now - last_flush >= _FLUSH_SECS:
                        f.write(buf)
                        f.flush()
                        buf.clear()
                        lines = 0
                        last_flush = now
        except Exception:
            # Writer thread must never take the process down.
            import os as _os

            if _os.environ.get("VLLM_SNIFFER_DEBUG"):
                import traceback as _tb

                _tb.print_exc()
            pass


# ----------------------------------------------------------------------
# Module-level singleton: every vLLM process gets exactly one writer.
# ----------------------------------------------------------------------
_writer: JsonlWriter | None = None
_writer_lock = threading.Lock()


def ensure_writer() -> JsonlWriter:
    """Get-or-create the process-local writer.

    The run_id directory is shared by all processes of one launch: the
    first process to call this creates ``<out_dir>/<ts>``; later processes
    (engine core / workers, forked or spawned within the same second)
    reuse it via the exported ``VLLM_SNIFFER_RUN_ID`` env var.
    """
    global _writer
    if _writer is not None:
        return _writer
    with _writer_lock:
        if _writer is not None:
            return _writer
        cfg = get_config()
        run_id = _resolve_run_id(cfg.out_dir)
        _writer = JsonlWriter(cfg.out_dir, run_id)
        return _writer


def prepare_run_dir() -> None:
    """Create the run directory and export the run_id to child processes.

    Called from the plugin entry point: the run_id must be fixed and
    visible via ``VLLM_SNIFFER_RUN_ID`` *before* vLLM forks/spawns the
    engine core and workers, or each process would pick its own directory.

    Deliberately does NOT create the writer: the writer stays lazy so a
    forked child never inherits a writer whose consumer thread does not
    exist in the child.
    """
    cfg = get_config()
    run_id = _resolve_run_id(cfg.out_dir)
    os.makedirs(os.path.join(cfg.out_dir, run_id), exist_ok=True)


def _resolve_run_id(out_dir: str) -> str:
    # Children (engine core / workers) reuse the run_id chosen by the
    # parent process; the parent sets it before spawning. Only reuse it if
    # the directory actually exists under the current out_dir (it may have
    # been set for a different output location).
    inherited = os.environ.get(RUN_ID_ENV)
    if inherited and os.path.isdir(os.path.join(out_dir, inherited)):
        return inherited
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # If a directory for this second already exists (sibling processes),
    # reuse it so one launch lands in one directory.
    if os.path.isdir(os.path.join(out_dir, ts)):
        return ts
    try:
        os.makedirs(os.path.join(out_dir, ts), exist_ok=True)
    except FileExistsError:
        pass
    # Make the run_id visible to spawned vLLM processes.
    os.environ[RUN_ID_ENV] = ts
    return ts


def emit(event: Event) -> None:
    """Non-blocking emit used by hooks."""
    import os as _os

    try:
        w = ensure_writer()
        if _os.environ.get("VLLM_SNIFFER_DEBUG"):
            print(
                f"[vllm-sniffer] {_os.getpid()} emit {event.type} "
                f"writer={id(w)} thread_alive={w._thread.is_alive()}",
                flush=True,
            )
        w.emit(event)
        if _os.environ.get("VLLM_SNIFFER_DEBUG") and w._closed is False:
            print(
                f"[vllm-sniffer] {_os.getpid()} after emit {event.type} "
                f"qsize={w._q.qsize()} dropped={w.dropped} "
                f"thread_ident={w._thread.ident} cur_ident={threading.get_ident()}",
                flush=True,
            )
    except Exception as e:
        if _os.environ.get("VLLM_SNIFFER_DEBUG"):
            import traceback as _tb

            _tb.print_exc()
        # Hook layer must never raise into the inference path.
        pass


def close_writer() -> None:
    global _writer
    w = _writer
    _writer = None
    if w is not None:
        w.close()


# Flush remaining events at interpreter exit.
import atexit

atexit.register(close_writer)


# ----------------------------------------------------------------------
# fork safety
# ----------------------------------------------------------------------
# vLLM starts engine core / workers with multiprocessing *fork* by default
# (VLLM_WORKER_MULTIPROC_METHOD=fork). A forked child inherits the parent's
# memory: if the parent already created a writer, the child inherits a
# writer whose consumer thread does not exist in the child -- events pushed
# into it would be lost forever. Reset the singleton after every fork so
# the child lazily creates its own writer on first emit (reusing the run_id
# via the exported VLLM_SNIFFER_RUN_ID env var).
def _reset_for_fork() -> None:
    global _writer
    w = _writer
    _writer = None
    if w is not None:
        # The consumer thread is gone in the child; stop enqueueing.
        # Do NOT call w.close(): joining a thread that does not exist
        # would block the child for the join timeout.
        w._closed = True


import os

os.register_at_fork(after_in_child=_reset_for_fork)
