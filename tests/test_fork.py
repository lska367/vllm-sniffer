"""Fork-safety tests: vLLM starts engine core / workers with fork by
default; a forked child inheriting a live writer singleton (whose consumer
thread does not exist in the child) would silently lose events."""

import multiprocessing
import os

from vllm_sniffer.core import writer as w
from vllm_sniffer.core.event import make_event


def _child_proc(out_dir, run_id, q):
    # Simulate a forked vLLM child: it inherits the parent's writer
    # singleton state, then the after_fork hook should have reset it.
    os.environ["VLLM_SNIFFER_DIR"] = out_dir
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    w.ensure_writer().emit(make_event("core", "step", data={"from": "child"}))
    w.close_writer()
    q.put(w.ensure_writer().path)


def test_fork_resets_writer_singleton(tmp_path):
    # Parent creates a writer first (as if events were already emitted).
    os.environ["VLLM_SNIFFER_DIR"] = str(tmp_path)
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    parent_writer = w.ensure_writer()
    parent_writer.emit(make_event("core", "step", data={"from": "parent"}))
    parent_pid = os.getpid()

    q = multiprocessing.get_context("fork").Queue()
    p = multiprocessing.get_context("fork").Process(
        target=_child_proc, args=(str(tmp_path), w._resolve_run_id(str(tmp_path)), q)
    )
    p.start()
    child_path = q.get(timeout=10)
    p.join(timeout=10)
    assert p.exitcode == 0

    # The child must have written its own file (not the parent's).
    assert child_path != parent_writer.path
    assert os.path.exists(child_path)
    import json

    ev = json.loads(open(child_path).read())
    assert ev["data"]["from"] == "child"

    get_config.cache_clear()


def test_after_fork_marks_inherited_writer_closed():
    os.environ["VLLM_SNIFFER_DIR"] = "/tmp/vllm-sniffer-fork-test"
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    try:
        w.ensure_writer()
        writer_obj = w._writer
        # Simulate what the after-fork hook does in the child.
        w._reset_for_fork()
        assert w._writer is None
        assert writer_obj._closed is True
        # Emitting into the closed (inherited) writer is a silent no-op.
        writer_obj.emit(make_event("core", "step"))
    finally:
        get_config.cache_clear()
