# Changelog

All notable changes to vllm-sniffer are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `env_snapshot` 事件（P0-1）：每次 run 由主进程发出一次参照系事件 —
  vLLM/torch/CUDA/python 版本、determinism 相关 env（`VLLM_BATCH_INVARIANT`、
  `VLLM_FLOAT32_MATMUL_PRECISION`、`VLLM_USE_CUDA_GRAPH` 等）、tracer 自身配置。
  子进程（继承 `VLLM_SNIFFER_RUN_ID`）不重复发出。
- 分析工具 v0（P0-2，`tools/` 目录）：
  - `export_parquet.py`：run 目录多 pid JSONL 合并 → 单 parquet（按 ts_ns 排序；
    需要可选依赖 pyarrow，`pip install vllm-sniffer[analyze]`）
  - `latency_report.py`：TTFT / TPOT / step / forward 分布（p50/p90/p99 + ASCII 直方图）
  - `repro_compare.py`：temp=0 对拍 — 输出长度一致性（按 prompt 长度分组）、
    flip 密度、按活动窗口的 per-request flip 归因
- 真机实验脚本（`scripts/`）：
  - `exp_determinism.py`：同 prompt × N、temperature=0 的输出一致性实验
    （逐位置 diff + 唯一序列计数 + 结果 JSON）
  - `exp_overhead.py`：开销量化（baseline / margin-off / margin-on 三臂对比，
    子进程 env 隔离）
- CI（GitHub Actions）：Python 3.10/3.12 × CPU torch + pyarrow 全量测试
- `LICENSE`（Apache-2.0）；`pyproject.toml` 增加 `[analyze]` extras
- 测试：+14 个（env_snapshot 7、tools 7、实验脚本 4，含 pyarrow parquet 往返）

### Changed

- 插件入口 `load()`：修复 run-root 检测并发出 `env_snapshot`（每个 run 一次）
- `tests/conftest.py`：`fake_vllm_module` 支持顶层模块路径（如 `vllm`）

### Fixed

- 移除调试残留 `breakpoint()`（vllm_sniffer/__init__.py）
- `scripts/offline_smoke_cudagraph.py`：清理注释掉的调试代码

## [0.1.0] - 2026-08-06

### Added

- 非侵入式插件入口（`vllm.general_plugins` entry point），幂等、静默降级
- 请求生命周期 hooks：`request_start` / `request_first_token` / `request_finish` /
  `request_abort`（online `AsyncLLM` + offline `LLM.generate`）
- engine-core hooks：`step`（心跳，不采样）/ `schedule`（采样）/ `preempt`
- worker hooks：`forward`（采样）/ `sample_flip`（margin<eps 高危位置）/
  `sample_stats`（margin 聚合）
- 每进程无锁有界队列 + daemon 线程 JSONL writer（满丢弃不阻塞；定时 flush 防尾丢失；
  fork 后重置单例）
- 真机验证（2026-08-06，V100 + Qwen2.5-0.5B + vLLM 0.19.1）：offline（eager +
  cudagraph）、online（api_server + 流式 + abort）全通过
- 文档集：ARCHITECTURE / CODE_WALKTHROUGH / EVENT_SCHEMA / EXTENSION_ROADMAP /
  RESEARCH_PLATFORM / RESUME_PROJECT / VALIDATION_LOG / CONTRIBUTING
