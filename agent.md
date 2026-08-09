# vllm-sniffer Agent Guide

> 供后续 agent / 开发者使用的工作档案。包含项目定位、架构、代码详解、
> 真机验证中已验证的事实与踩坑记录、以及下一步规划。
> 最后更新：2026-08-09（v0.1 + env_snapshot + 分析工具 v0 + logits 位级指纹 + 可视化前端）

---

## 1. 项目定位

非侵入式 vLLM 运行时 tracer，用于**生产环境推理排障**与**KV Cache 研究数据采集**。

- **核心约束：行为不变**——不改 vLLM 源码、不改推理行为、hook 异常绝不进入推理路径
- **形态**：L2 层 monkey-patch 运行时注入，通过 vLLM 官方插件机制（`vllm.general_plugins` entry point）自动加载，零代码侵入
- **输出**：JSONL 事件流（`/tmp/vllm-sniffer/<run_id>/<pid>.jsonl`），后续自研可视化前端
- **开销预算**：<1%（热路径只采计数/耗时/轻量摘要；队列满丢弃不阻塞；flip 检测在 GPU 上完成只拷回罕见值）

## 2. 架构总览

### 2.1 进程模型（vLLM v0.19.1，真机验证确认）

```
API server 进程 (AsyncLLM / AsyncMPClient)
   │  ZMQ
   ▼
EngineCoreProc 进程 (EngineCore 热循环 + Scheduler + GPUModelRunner[单卡])
```

- **offline（LLM.generate）子进程是 fork**（`VLLM_WORKER_MULTIPROC_METHOD` 默认 fork）：
  hooks 靠 fork 内存继承传播（子进程不重新执行插件加载）
- **online（api_server）子进程是 spawn**（uvicorn 多线程触发 vLLM `_maybe_force_spawn`）：
  子进程重新 import vllm → 重新执行插件加载
- 单卡 tp=1 时模型执行在 engine core 进程内（无独立 worker 进程）
- 两种模型下插件自动加载均真机验证通过

### 2.2 数据流

```
vLLM 热路径 ──> hook wrapper（只读观测，try/except 全包裹）
      │ emit()（非阻塞 put_nowait，满则丢弃计数）
      ▼
每进程: 有界队列(8192) ──> daemon 线程 ──> JSONL 文件
      （q.get(timeout=1s) 定时 flush，尾部数据不滞留）
```

跨进程关联键：`req_id` + `step` + `ts_ns`(wall clock)。同一次启动的所有进程共享一个
`run_id` 目录（主进程在 load() 时通过 `VLLM_SNIFFER_RUN_ID` env 导出，子进程继承）。

### 2.3 Hook 点（v0.19.1 源码位置已核实）

| 进程 | Hook | 模块位置 | 事件 |
|---|---|---|---|
| API server | `AsyncLLM.generate` | vllm/v1/engine/async_llm.py | request_start/first_token/finish/abort |
| API server | `AsyncLLM.abort` | vllm/v1/engine/async_llm.py | request_abort (explicit) |
| API server | `LLM.generate` | vllm/entrypoints/llm.py | offline 批处理 start/finish |
| EngineCore | `EngineCore.step` + `step_with_batch_queue` | vllm/v1/engine/core.py | step 心跳 |
| EngineCore | `Scheduler.schedule` | vllm/v1/core/sched/scheduler.py | schedule 详情 / preempt |
| Worker | `GPUModelRunner.execute_model` | vllm/v1/worker/gpu_model_runner.py | forward 计时 |
| Worker | `Sampler.greedy_sample` | vllm/v1/sample/sampler.py | sample_flip / sample_stats |

## 3. 代码详解

### 3.1 `vllm_sniffer/__init__.py` — 插件入口

- `load()`：`vllm.general_plugins` entry point 函数，vLLM 在每个进程 import 时调用
  （API server 经 arg_utils、engine core 经 core.py:99、worker 经 worker_base.py:232）
- 必须幂等（模块级 `_loaded` 标志）；`VLLM_SNIFFER=0` 时直接返回
- 流程：install_all() → prepare_run_dir()（建目录+导出 RUN_ID_ENV，**不创建 writer**）
- 所有异常兜底：traceback.print_exc() 到 stderr（vLLM 日志系统不显示 vllm_sniffer logger）
- `VLLM_SNIFFER_DEBUG=1` 时打印：load 时的 env 快照、hooks 安装列表

### 3.2 `vllm_sniffer/config.py` — 配置

env 解析，`get_config()` lru_cache。全部 `VLLM_SNIFFER_*` 前缀。
新增（2026-08-09）：`logits_fp`（深挖模式，默认关）、`logits_fp_rows`
（每步指纹行数，clamp 1..64）。

### 3.3 `vllm_sniffer/core/event.py` — 事件 schema

- `Event`：msgspec Struct（`kw_only=True` 必需——msgspec 要求必填字段不能跟在默认字段后）
- 统一字段：`schema_ver/ts_ns/pid/group/type/req_id/step/sampled/data`
- data 只允许 JSON 类型（msgspec 编码）；**不记录 prompt 原文**（隐私）
- `encode_line()`：msgspec.json.encode，快且分配少

### 3.4.5 `vllm_sniffer/core/step_counter.py` — 进程内 step 计数器（2026-08-09）

- 真机发现：engine/worker 事件 step 字段恒为 null（schema 承诺但从未写入）。
  现由 engine step 包装器入口 `next_step()` 自增，schedule/forward/sample_*/
  logits_fp 读 `current_step()`（TP=1 模型执行在 engine core 进程内，天然共享）
- warmup 在 step 循环外 → step=0 = 虚拟批标记；异步调度返回 (None, False)
  的"仅调度"迭代也发心跳（n_outputs=0），不再静默丢事件
- 限制：TP>1 独立 worker 进程不共享，需 rank 关联（待 P2-2）

### 3.4 `vllm_sniffer/core/writer.py` — 每进程 JSONL writer（核心组件）

- `JsonlWriter`：有界队列(8192) + daemon 线程；`_FLUSH_EVERY=32` 行或 `_FLUSH_SECS=1s`
  **定时 flush 用 `q.get(timeout=1s)` + Empty 分支实现**（坑：只在收到事件时检查时间
  会漏掉事件流尾部——主进程 api 事件只有几条、1s 内结束，曾导致文件永远 0 字节）
- `close()`：哨兵 + join(5s)；`emit()`：put_nowait，Full → dropped 计数，绝不阻塞
- `ensure_writer()`：惰性单例（**不能预创建**——见 fork 坑）
- `prepare_run_dir()`：load() 调用，建目录 + setenv `VLLM_SNIFFER_RUN_ID`
- `_reset_for_fork()`：`os.register_at_fork(after_in_child=...)`——fork 继承的 writer
  单例其消费者线程在子进程不存在，事件进死队列永远不落盘；子进程重置后惰性重建。
  **不能 close()**（join 不存在的线程会阻塞子进程 5s），只标记 `_closed=True`

### 3.5 `vllm_sniffer/core/env_snapshot.py` — run 参照系事件（2026-08-09 新增）

- `collect_env_snapshot()`：纯收集——vllm（version+commit，防御性探测）、torch
  （version/cuda/git，**注意 torch.__version__ 是 TorchVersion 对象需 str()**——msgspec
  编码会炸）、python_version、`env`（只记**存在**的 determinism 相关变量白名单）、
  sniffer 自身配置
- `emit_env_snapshot(is_root=...)`：run 根进程（load() 里 `VLLM_SNIFFER_RUN_ID` 未设置者）
  发出一次 `env_snapshot`（group=core）。**root 判定必须在 prepare_run_dir() 之前**——
  它会把 env 写进去，事后无法区分根/子

### 3.6 `vllm_sniffer/hooks/__init__.py` — 安装基础设施

- `install_all()`：幂等（`_installed` 标志），三组 hooks 全 try/except
- `is_our_wrapper()` / `mark_wrapper()`：**按类幂等**——wrapper 打 `_sniffer_wrapper` 标记，
  重复安装检测到标记直接返回（坑：重复安装会双重包装 → 双重发事件，曾导致 flip 事件 ×2）

### 3.6 `vllm_sniffer/hooks/api.py` — 请求生命周期

- `install_async_generate_hook`：**patch `AsyncLLM.generate` 和 `AsyncLLMEngine.generate` 两个类**
  （坑：v0.19 API server 用 AsyncLLM，AsyncLLMEngine 已不是入口；每个类独立 wrapper 闭包）
- generate wrapper：async generator 包装——请求流式事件 + 原样 yield
  - `request_start`：req_id + prompt 形状（str/list/dict 类型与长度，不记内容）
  - `request_first_token`：首个含 token 的 chunk；`n_prompt_tokens`
  - `request_finish`：`output.finished`（RequestOutput 属性）判断；
    **finish_reason 在 `output.outputs[0]` 上**（RequestOutput 本身没有该字段——坑）
  - `n_output_tokens`：**流式 DELTA 模式的 token_ids 是增量，需累计**（坑）
  - `request_abort`：`except BaseException`（**GeneratorExit 继承 BaseException，Exception 抓不到**）
    + 未 finish 才发
- `install_async_abort_hook`：patch `AsyncLLM.abort`——server 检测到客户端断开后显式调用，
  是最可靠的 abort 信号（req_id 可能是内部 id，带后缀）
- `install_offline_generate_hook`：LLM.generate 同步包装，start/finish/abort（无 first_token，
  非流式批处理语义弱，TTFT 分析走 online）

### 3.7 `vllm_sniffer/hooks/engine.py` — engine core 进程

- `install_engine_core_step_hook`：**同时 patch `step` 和 `step_with_batch_queue`**
  （坑：v0.19 默认异步调度，热循环走 step_with_batch_queue，step 根本不执行；
  用工厂 `_make_step_wrapper(orig)` 每方法独立闭包——共享闭包会让两个方法的 orig 都指向
  最后一次 getattr 的方法）；step 事件不采样（引擎心跳）
- `install_scheduler_hook`：preempt 事件永不采样（低频，从 `SchedulerOutput.preempted_req_ids`）；
  schedule 详情按采样率（total_num_scheduled_tokens / num_scheduled_tokens 字典长度 /
  finished_req_ids 计数 / new_block_ids_to_zero 计数）

### 3.8 `vllm_sniffer/hooks/worker.py` — 浮点不确定性观测（研究核心）

- `forward` 事件数据：2026-08-09 起 batch 组成取自
  `scheduler_output.num_scheduled_tokens`（input_batch 数组执行后即重置——
  真机发现）；num_tokens 与 total_num_scheduled_tokens 语义等价
- `install_sampler_margin_hook`：**argmax-margin 探针**（temp=0 不稳定性的量化观测）
  - 事实：temp=0 走 `logits.argmax(dim=-1)`（sampler.py:144），采样本身确定，
    输出不同 = logits 微差 → argmax 翻转。根因方向：batch 组成变化（官方
    `VLLM_BATCH_INVARIANT` 存在即证据）、前缀缓存命中差异、preemption 重算、
    cudagraph 图切换、多卡 AllReduce 顺序
  - 实现：wrapper 先调 orig（行为不变），再 `torch.topk(logits, 2)` 一次 GPU kernel
    算 margin；flip 检测（margin < `VLLM_SNIFFER_FLIP_EPS` 默认 1e-3）在 GPU 上 mask +
    nonzero，只拷回罕见 flip 行（避免全量 D2H 同步）；sample_stats 每步聚合
    （min/max/mean margin、flip 数，3 个标量 .item() 同步一次）
  - **运行时也检查 cfg.margin**（不只安装时——wrapper 可能因共享类而残留）
  - 开销：每次 greedy 采样一次 topk(2) + 一次标量同步，对 decode 步（ms 级）<1%
  - `VLLM_SNIFFER_MARGIN=0` 完全关闭
- **logits 位级指纹（P1-1，深挖模式，2026-08-09）**：`VLLM_SNIFFER_LOGITS_FP=1`
  - 事实：vLLM V1 `Sampler.forward` 采样前 `logits.to(torch.float32)`（sampler.py:90）
    → 生产指纹恒为 fp32 32 位；fp16/bf16 防御性 16 位
  - `_logits_fp_data()`：strided 行采样（linspace 覆盖 batch 两端，≤rows 行），
    每行 32 个 bit 位的置位数 `bits[b]`（int32 view + `(x>>b)&1` 归约，
    算术右移对负数取位正确）；每行附 top1（margin 开启时复用 topk，否则
    一次 argmax）供对拍对齐；事件 `logits_fp`（sampled=True，与 sample_stats
    同采样门）
  - 成本：每采样步 ≤8 行 × 32 个小归约 + 一次 k×32 D2H 同步——深挖模式
    专用，默认关（零开销承诺不变）
  - 安装条件：`cfg.margin or cfg.logits_fp`（任一开启即装 wrapper，运行时
    分别检查）
- 注意：TP>1 时每 rank 都跑 sampler（margin 重复计算），未来按 rank 过滤

### 3.9 `scripts/` — 真机验证脚本

- `offline_smoke.py` / `offline_smoke_cudagraph.py`：offline 冒烟（eager / cudagraph）
- `online_smoke.py`：httpx 流式 + abort 场景（`stream_then_abort` 读几个 chunk 后断开）
  - 注意 SSE 结束符 `data: [DONE]` 需跳过；finish_reason 可能是 "length"（截断）不是 "stop"

### 3.10 `tools/` — 分析工具（2026-08-09 新增，P0-2）

- `common.py`：共享层——`load_events()`（run 目录 jsonl 或 parquet 统一成 dict 流，
  **按 ts_ns 全局排序**；parquet 输入时把 data_json 解出来，两种输入下游一致）、
  percentile / ascii_histogram（无 matplotlib 依赖）、`request_timelines()`（api 事件
  按 req_id 组装 start/first_token/finish/abort）
- `export_parquet.py`：多 pid jsonl 合并 → 单 parquet（扁平列 + `data_json` JSON 字符串列，
  表 schema 永不随 data 增长变化）；pyarrow 是可选依赖（`pip install -e '.[analyze]'`）
- `latency_report.py`：TTFT = first_token.ts - start.ts；TPOT = (finish - first_token) /
  (n_output_tokens - 1)；offline 批（mode=offline）无 first_token 锚点单独报时长
- `repro_compare.py`：按 prompt 长度分组查输出长度一致性（VARYING = 确定性证据）；
  flip 密度 = 统计 flips / n_greedy；**per-request flip 归因是 ts 窗口匹配**
  （[start.ts, finish.ts]，多个请求窗口重叠 → 标 ambiguous）——v0 限制：
  sample_flip 事件没有 req_id（sampler 不知道），未来让 worker 侧带 req 关联
- `logits_fp_compare.py`（2026-08-09，P1-1）：两 run `logits_fp` 事件对拍
  - 对齐：默认按序列序号（index，i-th 事件 ↔ i-th）；`--align ts` 按 ts 窗口
  - 匹配行按绝对 row 号；逐 bit 置位数差 |Δ| 聚合 → 差异位分布图（ASCII）
  - 结论分类：IDENTICAL（Δ=0）/ LOW-BIT NOISE（尾数位 0..22 ≥80%）/
    SYSTEMATIC（阶码位 23..30 ≥50%）/ MIXED；报告 rows only in A/B
    （batch 规模变化）与 top1 不一致数
  - 限制：不同 batch 组成下 row i 未必是同一请求——分布级结论可靠，行级不可

### 3.12 `webapp/` — 可视化前端（2026-08-09，P2-1）

- `webapp/__init__.py`：FastAPI `create_app(out_dir)` 工厂；端点：
  `/api/runs`（run 列表，目录+parquet）、`/api/runs/{id}/summary|timeline|
  steps|flips|scatter`、`/healthz`；run_id 白名单 `[0-9A-Za-z._-]+` 防路径
  穿越（`os.path.realpath` 二次校验）；`_load_cached` 小缓存（mtime 失效，
  max 8 项）
- `webapp/server.py`：`python -m webapp.server --dir … --port …`（uvicorn）
- `webapp/static/index.html`：单文件前端（vanilla JS + canvas，无 CDN）；
  四视图：请求时间线（TTFT 橙 + TPOT 蓝段）、step 序列（warmup 红点）、
  flip 热图（x=step/序号，y=log10 margin，对数色深）、batch×耗时散点
- 依赖：`pip install -e '.[web]'`（fastapi/uvicorn/httpx——httpx 供
  TestClient）；测试 `tests/test_webapp.py`（10 个）
- 文档：`doc/WEBAPP.md`（接口结构 + 前端说明 + 扩展指南）

### 3.11 `scripts/exp_*.py` — 真机实验脚本（2026-08-09 全部真机跑通）

- `exp_determinism.py`：同 prompt × N、temperature=0 的批量对拍；逐位置 diff +
  唯一序列计数；在 import vLLM **之前** setdefault VLLM_SNIFFER_DIR，保证事件与
  结果 JSON 落在同一 run
- `exp_overhead.py`：三臂开销量化（VLLM_SNIFFER=0 / =1+MARGIN=0 / =1 默认），
  每臂子进程跑同一 workload（env 隔离），解析 TOKENS/SECS 行算 tok/s
  （**坑：TOKENS 行是 4 字段，旧 3 字段解析必崩——2026-08-09 真机首跑修复**）
- `exp_logits_fp.py`：solo vs mixed 两臂 logits 指纹对拍（P1-1）。**必须每臂
  subprocess + env 清洗**：fork 的 engine 进程继承主进程 lru_cached Config
  （out_dir 还是第一臂的）→ 同进程两臂会串台（真机发现）

## 4. 事件类型速查

| type | group | 采样 | data 关键字段 |
|---|---|---|---|
| env_snapshot | core | 否 | vllm/torch/python/env/sniffer/run_id（每 run 一次） |
| request_start | api | 否 | kind/n（prompt 形状） |
| request_first_token | api | 否 | n_prompt_tokens |
| request_finish | api | 否 | finish_reason, n_output_tokens（累计） |
| request_abort | api | 否 | source（explicit_abort 或空） |
| step | core | 否 | dur_ns, n_outputs |
| schedule | core | 是 | dur_ns, total_num_scheduled_tokens, n_scheduled_reqs, n_preempted, new_block_ids_to_zero |
| preempt | core | 否 | mode=recompute |
| forward | worker | 是 | dur_ns, num_tokens, num_seqs |
| sample_flip | worker | 否 | margin, top1, top2 |
| sample_stats | worker | 是 | n_greedy, margin_min/max/mean, n_flips |
| logits_fp | worker | 是 | n_rows, dtype, sampled_rows, rows[{row, top1, bits}]（深挖模式） |

## 5. 真机验证记录（2026-08-06 + 2026-08-09，V100 32GB + Qwen2.5-0.5B-Instruct + vLLM 0.19.1）

- 环境：`/home/lskam/work/vllm-sniffer/.venv-gpu`（uv 创建，vllm==0.19.1 + torch 2.10.0+cu128）；
  CPU 测试 venv：`.venv`（torch cpu + pytest + msgspec）
- 覆盖场景：offline（eager + cudagraph）、online（api_server + httpx 流式 + abort）；
  7 种事件全产出；TTFT 可算（实测 154ms/26ms）
- 观测数据：0.5B 模型 step 中位 8.7ms（cudagraph）；**64 token 输出中 19 个 sample_flip**
  （margin<1e-3 占比可观——temp=0 不稳定性的量化证据，研究可用素材）
- warmup 虚拟批（sample_stats n_greedy=256）混在事件流中，分析层需按"首个 step 事件"
  时间戳分割（engine_ready hook 已废弃：EngineCore.__init__ 第一行就调 load_general_plugins，
  patch __init__ 对当前实例无效）
- V100 事实：无 FA2（fallback TRITON_ATTN）、bf16 自动降 fp16、`VLLM_BATCH_INVARIANT`
  需 cc>=9.0 不可用（V100 上 batch 组成影响无法用官方开关消除 → tracer 观测更有价值）

### 5.1 2026-08-09 真机实验全集（全部跑通，run 目录保留在 /tmp/vllm-sniffer/）

| 实验 | 结果摘要 |
|---|---|
| `exp_determinism --n 8 --max-tokens 64 --runs 3` | 3 run 均 8 请求 → **2 个唯一输出**；分叉点稳定在位置 13（token 476 "of" vs 13 "\n"，margin=2⁻¹⁰）；分叉后永不汇合（51/64 位）；flip 对 (476,13) 命中 21 次 |
| `exp_logits_fp --n 8 --max-tokens 32` | solo-vs-solo **IDENTICAL Δ=0**（63 事件/287 行，同 batch 位级确定）；solo-vs-mixed **LOW-BIT NOISE 92.4% 尾数位 13..22** |
| `exp_overhead --n 16 --max-tokens 64 --repeat 3` | baseline 1700 tok/s；margin-off **+2.0%**；margin-on **−21.7%** |
| online api_server（spawn） | 8 请求 + 2 abort；TTFT p50=28.9ms / TPOT p50=8.9ms；flip 2.37% |
| 工具全链路 | export_parquet（818 事件/2 pid）、latency_report、repro_compare、logits_fp_compare、webapp 四视图（9 runs）全部真机数据可用 |

- env_snapshot：fork/spawn 两条路径均验证 **JSONL 首事件且只发一次**
- step 计数器：真机验证 step 1..196 覆盖 schedule/forward/sample_*/logits_fp

## 6. 踩坑清单（#lesson，改代码前必读）

1. **msgspec Struct**：必填字段不能跟在默认字段后 → `kw_only=True`
2. **vLLM 子进程 fork**：继承的 writer 单例线程不存在 → `os.register_at_fork` 重置；
   **不能预创建 writer**（保持惰性）
3. **v0.19 异步调度**：热循环是 `step_with_batch_queue` 不是 `step`，两个都 patch
4. **v0.19 API server 入口是 AsyncLLM** 不是 AsyncLLMEngine；server 显式调 `AsyncLLM.abort`
5. **GeneratorExit 是 BaseException**：`except Exception` 抓不到客户端断开
6. **流式 DELTA token_ids 是增量**：累计；**RequestOutput 无 finish_reason**（在 outputs[0]）
7. **writer 定时 flush**：用 `q.get(timeout)` 实现，否则事件流尾部永不落盘
8. **hook 内一切代码（含 DEBUG 打印）必须 try/except**：曾因 DEBUG 访问不存在属性
   导致 AttributeError 传播到 serving 层、请求失败
9. **重复安装双重包装**：按类幂等（mark_wrapper 标记）
10. **pkill -f 自杀**：pkill 模式会匹配 bash -c 命令行自身；用 pgrep 查 pid 再 kill
11. **vLLM 对未知 `VLLM_*` env 警告**（无害，一次性）
12. **子进程退出可能不走 atexit**（os._exit 路径）：尾部数据靠定时 flush 兜底

## 7. 开发与测试

```bash
# 单元测试（无 vLLM 依赖，假模块注入 sys.modules 走真实安装路径）
.venv/bin/python -m pytest -q          # 82 tests（含 tools parquet 往返 + webapp 接口）

# 分析工具（parquet 导出需要 pyarrow，可选）
uv pip install --python .venv/bin/python 'pyarrow>=14'

# 可视化前端（可选）
uv pip install --python .venv/bin/python -e '.[web]'

# 真机验证（GPU venv）
uv pip install --python .venv-gpu/bin/python -e .
VLLM_SNIFFER_DIR=/tmp/sniffer-test .venv-gpu/bin/python scripts/offline_smoke.py
VLLM_SNIFFER_DIR=/tmp/sniffer-online .venv-gpu/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct --port 8099 --no-enable-log-requests &
.venv-gpu/bin/python scripts/online_smoke.py
```

测试要点：
- `tests/conftest.py`：`fake_vllm_module()` 把假 vLLM 模块注入 sys.modules（**必须先清掉
  残留的 vllm* 模块**——hook 安装失败会留下半导入包）；`monkeypatch.setattr(..., raising=False)`
- 共享模块级 FakeSampler 类会让 wrapper 跨测试残留 → wrapper 运行时检查 cfg
- fork 测试：真 fork 子进程验证 writer 重置

## 8. Roadmap / 下一步

- [x] 注入层（插件自动加载、幂等、静默降级）
- [x] 请求生命周期（online：AsyncLLM generate/abort；offline：LLM.generate）
- [x] step/schedule/preempt/forward
- [x] greedy argmax-margin 浮点观测（flip 检测 + 聚合统计）
- [x] 真机验证（offline + online、fork + spawn、eager + cudagraph）
- [x] **env_snapshot**：启动时记录 vLLM commit/版本、determinism 相关 env、torch/cuda 版本
  （2026-08-09 完成，`vllm_sniffer/core/env_snapshot.py`）
- [x] **分析工具 v0**：export_parquet / latency_report / repro_compare（2026-08-09 完成，
  `tools/`；合成数据测试全绿，真机数据待复跑）
- [x] **logits 位级指纹**（采样模式，深挖数值差异来源；默认关闭保持零开销）
  （2026-08-09 完成：`VLLM_SNIFFER_LOGITS_FP` + `logits_fp` 事件 +
  `tools/logits_fp_compare.py` + `scripts/exp_logits_fp.py`）
- [x] **可视化前端**（自研，读 JSONL/聚合接口；schema 已稳定 schema_ver=1）
  （2026-08-09 完成：`webapp/` FastAPI + 单文件 canvas 前端，doc/WEBAPP.md）
- [ ] **多卡 TP/PP**（rank 字段预留；sampler margin 按 rank 去重）与 **Ray 集群**
  （node_id + 本地落盘）
- [ ] OTLP sink（可插拔）
- [ ] 真机补充：A100/H100 上重测开销量化（0.5B 上 margin 探针 −22% 是上限，
  大模型相对成本应显著更低）；`VLLM_BATCH_INVARIANT` 对比实验（需 cc≥9.0）；
  行为不变逐 token 回归固化为 CI

## 9. 相关文件

- `doc/papers/`、`doc/PLAN.md`（研究线，vllm-sniffer 是 KV Cache 研究的数据采集工具基础）
- 真机环境：`/home/lskam/work/vllm-sniffer/.venv-gpu`；模型 Qwen2.5-0.5B-Instruct 在 HF 缓存
- vLLM 源码参考：`/home/lskam/work/kvcache_research/vllm`（v0.19.1）
