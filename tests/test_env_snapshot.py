"""env_snapshot tests: the per-run reference-frame event (P0-1).

Verifies:
- field collection is defensive (no vLLM/torch installed in CI -> None)
- root process emits exactly one env_snapshot event with the reference
  fields; children (inheriting VLLM_SNIFFER_RUN_ID) skip emission
- the event is emitted through the real plugin entry point (load())
"""

import json

import pytest
from conftest import fake_vllm_module

from vllm_sniffer import load
from vllm_sniffer.core import env_snapshot
from vllm_sniffer.core.writer import RUN_ID_ENV, prepare_run_dir, close_writer


def test_collect_env_snapshot_defensive():
    """With no vllm/torch installed, collection degrades to None, never raises."""
    data = env_snapshot.collect_env_snapshot()
    assert "python_version" in data
    assert "vllm" in data and "torch" in data and "env" in data and "sniffer" in data
    # missing packages -> None, not an exception
    assert data["vllm"]["vllm_version"] is None or isinstance(
        data["vllm"]["vllm_version"], str
    )
    assert data["run_id"] is None  # no run dir fixed yet


def test_collect_env_snapshot_with_fake_vllm(monkeypatch):
    mod = fake_vllm_module(monkeypatch, "vllm")
    mod.__version__ = "0.19.1"
    mod.version = None  # no commit module -> still fine

    data = env_snapshot.collect_env_snapshot()
    assert data["vllm"]["vllm_version"] == "0.19.1"


def test_collect_env_records_determinism_vars(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_CUDA_GRAPH", "1")
    data = env_snapshot.collect_env_snapshot()
    assert data["env"]["VLLM_BATCH_INVARIANT"] == "1"
    assert data["env"]["VLLM_USE_CUDA_GRAPH"] == "1"
    # unrelated vars are not recorded
    monkeypatch.setenv("VLLM_SNIFFER_DEBUG", "1")
    data = env_snapshot.collect_env_snapshot()
    assert "VLLM_SNIFFER_DEBUG" not in data["env"]


def test_root_process_emits_once(tmp_path, monkeypatch, read_events):
    monkeypatch.delenv(RUN_ID_ENV, raising=False)
    # Simulate the plugin flow: prepare_run_dir then emit (root path).
    prepare_run_dir()
    assert env_snapshot.emit_env_snapshot(is_root=True) is True
    assert env_snapshot.emit_env_snapshot(is_root=True) is False  # once per proc

    events = read_events()
    snapshots = [e for e in events if e["type"] == "env_snapshot"]
    assert len(snapshots) == 1
    ev = snapshots[0]
    assert ev["group"] == "core"
    assert ev["req_id"] is None
    assert ev["data"]["run_id"] == ev["data"]["run_id"]  # present
    assert ev["data"]["run_id"]
    assert ev["data"]["sniffer"]["margin"] is True


def test_child_process_skips_emission(tmp_path, monkeypatch, read_events):
    # Child: the run_id env var is already fixed by the parent.
    monkeypatch.setenv(RUN_ID_ENV, "child-run")
    monkeypatch.setenv("VLLM_SNIFFER_DIR", str(tmp_path))
    assert env_snapshot.emit_env_snapshot(is_root=False) is False
    assert read_events() == []


def test_load_emits_env_snapshot_for_root(tmp_path, monkeypatch, read_events):
    """End-to-end: the plugin entry point emits env_snapshot for the root."""
    monkeypatch.delenv(RUN_ID_ENV, raising=False)
    monkeypatch.setenv("VLLM_SNIFFER_DIR", str(tmp_path))
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    try:
        load()
        events = read_events()
        snaps = [e for e in events if e["type"] == "env_snapshot"]
        assert len(snaps) == 1
        assert snaps[0]["data"]["python_version"].startswith("3.")
    finally:
        get_config.cache_clear()


def test_load_skips_env_snapshot_for_child(tmp_path, monkeypatch, read_events):
    """A spawned child re-imports and re-runs load(); must not duplicate."""
    monkeypatch.setenv(RUN_ID_ENV, "existing-run")
    monkeypatch.setenv("VLLM_SNIFFER_DIR", str(tmp_path))
    # Make the run dir exist so the inherited run_id is accepted.
    (tmp_path / "existing-run").mkdir()
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    try:
        load()
        events = read_events()
        assert [e["type"] for e in events if e["type"] == "env_snapshot"] == []
    finally:
        get_config.cache_clear()
