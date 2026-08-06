"""JsonlWriter behavior tests."""

import json
import threading

from vllm_sniffer.core import writer as w
from vllm_sniffer.core.event import make_event


def test_writer_persists_events(tmp_path):
    wr = w.JsonlWriter(str(tmp_path), "run1")
    for i in range(10):
        wr.emit(make_event("core", "step", data={"i": i}))
    wr.close()
    assert wr.path.endswith(".jsonl")
    lines = (tmp_path / "run1" / f"{wr.path.split('/')[-1]}").read_text().splitlines()
    assert len(lines) == 10
    ev = json.loads(lines[0])
    assert ev["group"] == "core"
    assert ev["type"] == "step"
    assert ev["data"]["i"] == 0


def test_emit_after_close_is_noop(tmp_path):
    wr = w.JsonlWriter(str(tmp_path), "run2")
    wr.close()
    wr.emit(make_event("core", "step"))  # must not raise
    (tmp_path / "run2").mkdir(exist_ok=True)


def test_drop_when_queue_full(tmp_path, monkeypatch):
    release = threading.Event()

    def slow_run(self):
        release.wait()

    monkeypatch.setattr(w.JsonlWriter, "_run", slow_run)
    wr = w.JsonlWriter(str(tmp_path), "run3")
    n = w._MAX_QUEUE + 20
    for _ in range(n):
        wr.emit(make_event("core", "step"))
    assert wr.dropped >= 20
    release.set()
    wr.close()


def test_ensure_writer_singleton_and_run_id_shared(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER_DIR", str(tmp_path))
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    a = w.ensure_writer()
    b = w.ensure_writer()
    assert a is b
    assert (tmp_path / a.path.split("/")[-2]).is_dir()
    get_config.cache_clear()
