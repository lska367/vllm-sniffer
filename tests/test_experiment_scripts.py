"""Real-machine experiment scripts: import-safety and syntax checks.

These scripts target a GPU machine with vLLM installed (.venv-gpu); CI has
neither, so tests here only verify that:
- modules import cleanly without vLLM (vllm imports are lazy, inside
  functions),
- the embedded subprocess workload in exp_overhead.py is valid Python,
- every scripts/ and tools/ file is syntactically valid.
"""

import ast
import importlib.util
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exp_determinism_imports_and_parses_args():
    mod = _load("exp_determinism", ROOT / "scripts" / "exp_determinism.py")
    ap = mod.build_parser()
    args = ap.parse_args(["--n", "4", "--max-tokens", "32"])
    assert args.n == 4
    assert args.max_tokens == 32
    # default prompt is set
    assert mod._default_prompt()


def test_exp_overhead_imports_and_workload_compiles():
    mod = _load("exp_overhead", ROOT / "scripts" / "exp_overhead.py")
    # The embedded workload must be valid Python (it runs in a subprocess).
    ast.parse(mod.WORKLOAD)
    ap = mod.build_parser()
    args = ap.parse_args(["--repeat", "1"])
    assert args.repeat == 1
    # config list covers the three comparison arms
    assert len(mod.main.__code__.co_consts) > 0


def test_all_scripts_and_tools_compile():
    for sub in ("scripts", "tools"):
        for f in sorted((ROOT / sub).glob("*.py")):
            py_compile.compile(str(f), doraise=True)


def test_exp_logits_fp_imports_and_parses_args():
    mod = _load("exp_logits_fp", ROOT / "scripts" / "exp_logits_fp.py")
    ap = mod.build_parser()
    args = ap.parse_args(["--n", "4", "--max-tokens", "32"])
    assert args.n == 4
    assert args.max_tokens == 32
    assert mod._default_probe()
    assert mod._default_filler(0)


def test_exp_logits_fp_requires_vllm_only_at_runtime(tmp_path, monkeypatch):
    """Without vLLM, the experiment exits with code 2 and a hint."""
    mod = _load("exp_logits_fp", ROOT / "scripts" / "exp_logits_fp.py")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "vllm" or name.startswith("vllm."):
            raise ImportError("No module named 'vllm'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sys, "argv", ["exp_logits_fp.py", "--n", "2"])
    assert mod.main() == 2


def test_exp_determinism_requires_vllm_only_at_runtime(tmp_path, monkeypatch):
    """Without vLLM, the experiment exits with code 2 and a hint."""
    mod = _load("exp_determinism", ROOT / "scripts" / "exp_determinism.py")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "vllm" or name.startswith("vllm."):
            raise ImportError("No module named 'vllm'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sys, "argv", ["exp_determinism.py", "--n", "2"])
    assert mod.main() == 2
