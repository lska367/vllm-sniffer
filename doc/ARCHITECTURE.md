# vllm-sniffer 架构设计解析

> 本文档回答三个问题：**它是什么、它怎么工作、为什么这样设计**。
> 配合 [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) 一起读：本文讲设计意图，
> 走读文档讲代码位置。

## 1. 定位与核心约束

非侵入式 vLLM 运行时 tracer：在不改 vLLM 源码、不改推理行为的前提下，
观测推理热路径（调度、前向、采样、KV cache 压力），输出 JSONL 事件流。

三条不可违背的约束（所有设计决策的出发点）：

1. **行为不变（no behavior change）**：hook 返回原值、不干预推理；
   hook 内部任何异常静默降级，绝不抛进推理路径
2. **开销预算 <1%**：热路径只采计数/耗时/轻量摘要；队列满丢弃不阻塞
3. **可完全关闭**：`VLLM_SNIFFER=0` 时插件入口直接返回，等价于未安装

## 2. 进程模型（vLLM v0.19.1，真机验证）

```
┌───────────────────────────┐
│ API server 进程            │   ← plugin 自动加载（spawn 后重新加载）
│  AsyncLLM.generate        │
│  AsyncLLM.abort           │
└─────────────┬─────────────┘
              │ ZMQ
┌─────────────▼─────────────┐
│ EngineCore 进程            │   ← 热循环 + Scheduler + GPUModelRunner（单卡）
│  EngineCore.step          │      子进程默认 **fork**（hooks 靠内存继承传播）
│  step_with_batch_queue    │
│  Scheduler.schedule       │
└─────────────┬─────────────┘
              │（TP>1 时经 Ray/多进程）
┌─────────────▼─────────────┐
│ Worker 进程（每 GPU）       │
│  GPUModelRunner.execute_model
│  Sampler.greedy_sample    │
└───────────────────────────┘
```

关键事实（均真机验证）：

- **offline（LLM.generate）**：子进程 fork，hook 靠 fork 内存继承传播，
  子进程不重新执行插件加载
- **online（api_server）**：uvicorn 多线程触发 vLLM `_maybe_force_spawn`，
  子进程重新 import vllm → 重新执行插件加载
- 单卡 tp=1 时模型执行在 engine core 进程内，无独立 worker 进程
- vLLM 在每个进程 import 时调用 `load_general_plugins()`（API server 经
  arg_utils、engine core 经 core.py:99、worker 经 worker_base.py:232），
  这是"零代码侵入"得以成立的基础

## 3. 数据流

```
vLLM 热路径
   │  emit()：put_nowait 非阻塞，队列满则丢弃并计数
   ▼
每进程: 有界队列(8192) ──daemon 线程──▶ JSONL 文件
   （q.get(timeout=1s) 定时 flush：保证事件流尾部不滞留）
```

- **每进程一个 writer 单例**，一个文件 `<out_dir>/<run_id>/<pid>.jsonl`
- **run_id 传播**：主进程在 load() 时把 run_id 写入
  `VLLM_SNIFFER_RUN_ID` env，子进程（fork 继承 / spawn 继承 env）复用，
  同一次启动的所有进程落在同一目录
- **run 参照系**：run 根进程（创建 run 目录者）发出一次 `env_snapshot` 事件
  ——vLLM/torch 版本、determinism 相关 env、tracer 配置，供对拍归因
- **跨进程关联键**：`req_id` + `step` + `ts_ns`(wall clock)
- **消费者线程 daemon 化**：不阻塞进程退出；子进程 os._exit 路径丢失的
  尾部数据由定时 flush 兜底

## 4. Hook 点一览

| 进程 | Hook | vLLM 源码位置 | 产出事件 |
|---|---|---|---|
| API server | `AsyncLLM.generate` | vllm/v1/engine/async_llm.py | request_start / first_token / finish / abort |
| API server | `AsyncLLM.abort` | 同上 | request_abort (source=explicit_abort) |
| API server | `LLM.generate` | vllm/entrypoints/llm.py | offline 批处理 start/finish/abort |
| EngineCore | `EngineCore.step` + `step_with_batch_queue` | vllm/v1/engine/core.py | step 心跳（不采样） |
| EngineCore | `Scheduler.schedule` | vllm/v1/core/sched/scheduler.py | schedule 详情 / preempt |
| Worker | `GPUModelRunner.execute_model` | vllm/v1/worker/gpu_model_runner.py | forward 计时 + batch 组成 |
| Worker | `Sampler.greedy_sample` | vllm/v1/sample/sampler.py | sample_flip / sample_stats |

**为什么 patch 这些点**：它们恰好覆盖一条请求的完整生命线——
入口（generate）→ 调度（schedule/step）→ 执行（execute_model）→
采样（greedy_sample）→ 出口（first_token/finish/abort）。
任何性能瓶颈或行为异常都能在这条线上定位到环节。

## 5. 关键设计机制

### 5.1 插件加载（零代码侵入的载体）

`pyproject.toml` 声明 `vllm.general_plugins` entry point 指向
`vllm_sniffer.load()`。pip 安装后 vLLM 自动发现，无需改启动命令。

- **幂等**：模块级 `_loaded` 标志 + 按类 `_sniffer_wrapper` 标记，
  防重复加载双重包装（vLLM 官方文档明确警告多进程重复加载）
- **静默降级**：load() 整体 try/except，失败打 stderr 后禁用，绝不炸掉 vLLM

### 5.2 惰性 writer + fork 安全

- **不预创建 writer**：fork 子进程继承父进程内存，若父进程已创建 writer，
  子进程继承的 writer 其消费者线程在子进程不存在 → 事件进死队列永远不落盘
- `os.register_at_fork(after_in_child=...)`：fork 后重置单例（只标记 closed，
  **不能 close()**——join 不存在的线程会阻塞子进程 5s），首事件时惰性重建

### 5.3 事件 schema（msgspec）

- msgspec Struct：编码快、分配少；`kw_only=True`（必填字段不能跟在默认字段后）
- `data` 只允许 JSON 类型；**不记录 prompt 原文**（隐私设计）
- `schema_ver=1` 稳定字段，为分析层/前端预留演进空间

### 5.4 采样与降频

- 高频事件（schedule/forward/sample_stats）按 `VLLM_SNIFFER_SAMPLE_RATE` 采样
- 低频事件（preempt/sample_flip/step）永不采样
- flip 检测在 GPU 上完成（topk(2) + mask + nonzero），只有罕见 flip 行拷回
  host，避免全量 D2H 同步

### 5.5 浮点不确定性观测（研究核心）

事实链：temp=0 走 `logits.argmax`，采样本身确定 → 输出不同 = logits 微差 →
观测 **argmax margin（top1-top2）**，margin < eps 即"flip 区"，浮点噪声
可翻转输出。已知根因方向：batch 组成变化（官方 `VLLM_BATCH_INVARIANT`
存在即证据）、前缀缓存命中差异、preemption 重算、cudagraph 图切换、
多卡 AllReduce 顺序。

tracer 只记录 margin/flip 位置，**对拍对比在分析层做**（tracer 无状态）。

**深挖模式（logits 位级指纹，默认关）**：`VLLM_SNIFFER_LOGITS_FP=1` 时
sampler wrapper 额外对原始 logits 做 strided 行的 fp32 bit 置位数指纹
（`logits_fp` 事件），分析层（`tools/logits_fp_compare.py`）据此区分
"同 batch ≈0 差异"与"换 batch 组成后的尾数位噪声"。深挖模式明确不在
默认热路径内——零开销承诺以默认配置为准。

## 6. 离线分析层与可视化前端

采集层产出 JSONL 事件流后，分析/展示与推理进程完全分离：

- `tools/`（CLI，纯 stdlib + msgspec，pyarrow 可选）：export_parquet /
  latency_report / repro_compare / logits_fp_compare，共享 `tools/common.py`
  （事件加载统一 jsonl/parquet 输入、百分位/直方图、请求时间线组装）
- `webapp/`（FastAPI + 单文件 canvas 前端，离线分析型，不要求实时）：
  `/api/runs/{id}/summary|timeline|steps|flips|scatter` 五个聚合端点 +
  四视图页面；run_id 白名单防路径穿越；事件小缓存（mtime 失效）。
  详见 [WEBAPP.md](WEBAPP.md)

## 7. 设计原则总结

1. 观测与推理路径**完全解耦**（队列 + daemon 线程，满则丢弃）
2. 一切防御式编程：hook 内任何一行都可能抛异常，全部 try/except
3. 幂等是安全网：重复加载、重复安装、跨版本残留都不该造成行为差异
4. 隐私默认：不记 prompt 原文，只记形状
5. 开销换价值的权衡显式化：每次新增观测都要问"成本多少、采样还是全量"

## 8. 局限性（诚实清单）

- 只覆盖 V1 engine（v0.19）；V0 engine 需要额外 hook 点
- TP>1 时每个 rank 都跑 sampler，margin 重复计算（rank 字段已预留未用）
- cudagraph 捕获路径的 batch 形状固定，flip 观测可能漏掉图内采样
  （待验证）
- 离线 LLM.generate 是批语义，无 first_token 事件，TTFT 分析走 online
