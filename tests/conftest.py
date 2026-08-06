"""Shared test fixtures.

The test environment has no vLLM installed, so vLLM classes are simulated
with fake modules injected into sys.modules. This exercises the *real*
install paths of the hooks (same import statements, same patching logic).
"""

from __future__ import annotations

import sys
import types

import pytest

from vllm_sniffer.config import get_config
from vllm_sniffer.core import writer as writer_mod


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    """Reset the process-local singletons and point output at tmp_path."""
    monkeypatch.setenv("VLLM_SNIFFER_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_SNIFFER_RUN_ID", raising=False)
    get_config.cache_clear()
    writer_mod.close_writer()
    # Reset hook bookkeeping so per-test installs never leak across tests.
    from vllm_sniffer.hooks import api, engine, worker

    api._installed.clear()
    engine._installed.clear()
    worker._installed.clear()
    yield
    writer_mod.close_writer()
    get_config.cache_clear()


def fake_vllm_module(monkeypatch, path: str) -> types.ModuleType:
    """Create ``path`` (e.g. 'vllm.v1.sample.sampler') as a fake module.

    Parent packages are created on demand. ``monkeypatch`` restores
    sys.modules and parent attributes after the test.

    Any stale ``vllm*`` entries left in sys.modules by failed hook imports
    are wiped first so tests never see half-imported packages.
    """
    for name in [n for n in sys.modules if n == "vllm" or n.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    parts = path.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            parent = sys.modules[".".join(parts[: i - 1])] if i > 1 else None
            mod = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, mod)
            if parent is not None:
                monkeypatch.setattr(parent, parts[i - 1], mod, raising=False)
    leaf = types.ModuleType(path)
    monkeypatch.setitem(sys.modules, path, leaf)
    parent = sys.modules[".".join(parts[:-1])]
    monkeypatch.setattr(parent, parts[-1], leaf, raising=False)
    return leaf


@pytest.fixture
def read_events(tmp_path):
    """Read all events written under the current run dir as msgspec dicts."""

    def _read():
        writer_mod.close_writer()  # flush
        import json

        events = []
        for f in sorted(tmp_path.glob("*/*.jsonl")):
            for line in f.read_text().splitlines():
                if line:
                    events.append(json.loads(line))
        return events

    return _read
