# Changelog

All notable changes to vllm-sniffer are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- README 改为英文纯净版（`README.md`）：删除全部验证/实测记录，只保留功能描述
  与使用方法；`README_EN.md` 移除（主 README 即英文）

### Added

- **logits 位级指纹（P1-1，深挖模式）**：`VLLM_SNIFFER_LOGITS_FP=1` 时
  `logits_fp` 事件逐采样行输出 fp32 bit 置位数指纹（`VLLM_SNIFFER_LOGITS_FP_ROWS`
  控制行数，默认 8）；默认关，零开销承诺不变。vLLM V1 采样前统一转 fp32，
  生产恒为 32 位指纹（fp16/bf16 防御性 16 位）。
- `tools/logits_fp_compare.py`：两 run 指纹对拍——逐 bit 差异分布图（ASCII）+
  结论分类（IDENTICAL / LOW-BIT NOISE / SYSTEMATIC / MIXED），报告 batch
  规模变化与 top1 不一致数
- **可视化前端（P2-1，`webapp/`）**：FastAPI 聚合接口（runs/summary/
  timeline/steps/flips/scatter）+ 单文件 vanilla-JS canvas 前端四视图
  （请求时间线 / step 序列 / flip 热图 / batch×耗时散点）；
  `python -m webapp.server --dir … --port …` 启动，`pip install -e '.[web]'`
- `scripts/exp_logits_fp.py`：GPU 对拍实验（solo vs mixed batch 组成），
  自动调用 logits_fp_compare 输出差异位分布
- `doc/WEBAPP.md`：前端使用 + 聚合接口文档
- 测试：+29 个（config 2、worker hook 9、指纹对拍工具 6、webapp 10、
  实验脚本 2）；CI 增加 web 依赖与工具 --help 检查

### Changed

- `install_sampler_margin_hook`：安装条件改为 margin 或 logits_fp 任一开启；
  `env_snapshot` 记录新增 `logits_fp`/`logits_fp_rows` 配置与相关 env
- 文档：EXTENSION_ROADMAP（P1-1/P2-1 完成）、EVENT_SCHEMA（logits_fp 登记）、
  README（配置/事件/工具/前端/Roadmap）

### Fixed

- `scripts/exp_logits_fp.py` 恢复进程内 env（`finally`），避免污染同进程后续 run
- `scripts/exp_logits_fp.py` 改为**每臂 subprocess + env 清洗**（真机发现：
  fork 的 engine 进程继承主进程 lru_cached Config，两臂同进程会串台）
- `scripts/exp_overhead.py` 修复 TOKENS 行解析（4 字段非 3——真机首跑必崩）

### 真机验证（2026-08-09，V100）

- 确定性对拍：同 batch 8 请求 → 2 个唯一输出，分叉点稳定在位置 13
  （token 476 "of" vs 13 "\n"，margin=2⁻¹⁰）；flip 事件精确命中 (476,13)×21
- logits 位级指纹：solo-vs-solo IDENTICAL（Δ=0）；solo-vs-mixed LOW-BIT NOISE
  （92.4% 尾数位 13..22）——P1-1 真机验收达成
- 开销量化：baseline 1700 tok/s / margin-off +2.0% / margin-on −21.7%
- online（spawn）：TTFT p50=28.9ms、TPOT p50=8.9ms；env_snapshot 在 fork 与
  spawn 两条路径均为 JSONL 首事件且只发一次；webapp 四视图真机数据全通

### Added

- `vllm_sniffer/core/step_counter.py`：进程内 step 计数器（真机发现 step 字段
  恒为 null；engine 入口自增，core/worker 事件全部携带迭代号；warmup=0）
- 测试：+9 个（step 单调/仅调度迭代心跳/schedule 标记/worker 事件标记/
  scheduler_output 兜底/scatter 兜底/TOKENS 解析/exp_logits_fp env 清洗与失败退出）

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
