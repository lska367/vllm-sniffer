"""Plugin entry point tests: load() idempotency and graceful degradation."""

from vllm_sniffer import load
from vllm_sniffer.hooks import install_all


def test_load_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER", "0")
    from vllm_sniffer.config import get_config

    get_config.cache_clear()
    try:
        load()  # must return silently, no hooks installed
        assert install_all() == []
    finally:
        get_config.cache_clear()


def test_load_idempotent(monkeypatch):
    # Without vLLM installed, hooks degrade gracefully to [].
    load()
    assert install_all() == []
    assert install_all() == []  # second call: no-op


def test_install_all_without_vllm_is_safe():
    # In a bare environment (no vLLM package) nothing raises.
    from vllm_sniffer.hooks import install_all

    result = install_all()
    assert isinstance(result, list)
