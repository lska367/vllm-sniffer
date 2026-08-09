"""Config parsing tests."""

from vllm_sniffer.config import get_config


def test_defaults():
    cfg = get_config()
    assert cfg.enabled is True
    assert cfg.sample_rate == 1.0
    assert cfg.margin is True
    assert cfg.flip_eps == 1e-3
    # deep-dive mode is OFF by default (zero-overhead promise)
    assert cfg.logits_fp is False
    assert cfg.logits_fp_rows == 8


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


def test_logits_fp_env(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER_LOGITS_FP", "1")
    get_config.cache_clear()
    assert get_config().logits_fp is True
    monkeypatch.setenv("VLLM_SNIFFER_LOGITS_FP", "0")
    get_config.cache_clear()
    assert get_config().logits_fp is False
    get_config.cache_clear()


def test_logits_fp_rows_clamped(monkeypatch):
    monkeypatch.setenv("VLLM_SNIFFER_LOGITS_FP_ROWS", "128")
    get_config.cache_clear()
    assert get_config().logits_fp_rows == 64  # clamped
    monkeypatch.setenv("VLLM_SNIFFER_LOGITS_FP_ROWS", "0")
    get_config.cache_clear()
    assert get_config().logits_fp_rows == 1  # clamped
    monkeypatch.setenv("VLLM_SNIFFER_LOGITS_FP_ROWS", "abc")
    get_config.cache_clear()
    assert get_config().logits_fp_rows == 8  # fallback
    get_config.cache_clear()
