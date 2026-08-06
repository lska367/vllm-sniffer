"""Config parsing tests."""

from vllm_sniffer.config import get_config


def test_defaults():
    cfg = get_config()
    assert cfg.enabled is True
    assert cfg.sample_rate == 1.0
    assert cfg.margin is True
    assert cfg.flip_eps == 1e-3


def test_master_switch_off(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER", "0")
    get_config.cache_clear()
    assert get_config().enabled is False
    get_config.cache_clear()


def test_margin_off(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER_MARGIN", "0")
    get_config.cache_clear()
    assert get_config().margin is False
    get_config.cache_clear()


def test_sample_rate_clamped(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER_SAMPLE_RATE", "1.5")
    get_config.cache_clear()
    assert get_config().sample_rate == 1.0
    monkeypatch.setenv("VLLM_SNIFFER_SAMPLE_RATE", "-1")
    get_config.cache_clear()
    assert get_config().sample_rate == 0.0
    get_config.cache_clear()


def test_flip_eps_parse(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER_FLIP_EPS", "1e-4")
    get_config.cache_clear()
    assert get_config().flip_eps == 1e-4
    get_config.cache_clear()


def test_dir_env(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER_DIR", "/tmp/custom")
    get_config.cache_clear()
    assert get_config().out_dir == "/tmp/custom"
    get_config.cache_clear()
