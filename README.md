# vllm-sniffer

非侵入式 vLLM 运行时 tracer：观测推理热路径（调度、前向、采样、KV cache 压力），
以及生产排障最棘手的"同样 prompt + temperature=0 却输出不同结果"的浮点不确定性。

- **零代码侵入**：通过 vLLM 官方插件机制（`vllm.general_plugins` entry point）自动加载，
  不改 vLLM 源码、不改推理行为
- **全进程覆盖**：API server / engine core / GPU worker 三个进程的 hook 自动生效
- **JSONL 事件流**：`/tmp/vllm-sniffer/<run_id>/<pid>.jsonl`，schema 稳定，为可视化前端预留
- **默认开销预算 <1%**：热路径只采集计数/耗时/轻量摘要，队列满时丢弃而非阻塞

## 安装

```bash
pip install -e .
# 装上即生效；关闭：export VLLM_SNIFFER=0
```

不需要改启动命令。vLLM import 时会在每个进程自动加载插件。

## 真机验证（2026-08-06，V100 32GB + Qwen2.5-0.5B-Instruct + vLLM 0.19.1）

- offline `LLM.generate`（eager + cudagraph 两种模式）与 online `api_server` 均验证通过
- **v0.19 API server 入口是 `AsyncLLM`（vllm/v1/engine/async_llm.py）而非 `AsyncLLMEngine`**：
  请求生命周期 hook 必须 patch `AsyncLLM.generate` 与 `AsyncLLM.abort`（server 显式调 abort）
- **vLLM 子进程默认 fork**（`VLLM_WORKER_MULTIPROC_METHOD`）：hooks 靠 fork 内存继承传播，
  但继承的 writer 单例消费者线程不存在 → `os.register_at_fork` 在子进程重置 writer
- **异步调度默认开启**：热循环走 `EngineCore.step_with_batch_queue` 而非 `step`，两个方法都 patch
- 单卡 tp=1 时模型执行在 engine core 进程内；V100 无 FA2（fallback TRITON_ATTN）、bf16 降级 fp16
- `VLLM_SNIFFER_*` env 触发 vLLM 的一次性 "Unknown vLLM environment variable" 警告（无害）
- #lesson writer 的定时 flush 用 `q.get(timeout)` 实现：仅在收到事件时检查时间会漏掉
  "事件流尾部"（<32 行且 1s 内结束的请求序列永远不 flush）
- #lesson 流式（DELTA）输出 `token_ids` 是增量：n_output_tokens 需累计；
  `RequestOutput` 无 `finish_reason` 字段（在 `outputs[0]` 上），`finished` 在 RequestOutput 上

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_SNIFFER` | `1` | 总开关 |
| `VLLM_SNIFFER_DIR` | `/tmp/vllm-sniffer` | 输出根目录 |
| `VLLM_SNIFFER_SAMPLE_RATE` | `1.0` | 高频事件（step/schedule/forward/sample_stats）采样率 |
| `VLLM_SNIFFER_MARGIN` | `1` | greedy argmax-margin 观测（一次额外 topk(2) kernel） |
| `VLLM_SNIFFER_FLIP_EPS` | `1e-3` | flip 区判定阈值（top1-top2 logit 差） |

## 事件类型

| 事件 | 组 | 频率 | 内容 |
|---|---|---|---|
| `request_start` / `request_first_token` / `request_finish` / `request_abort` | api | 每请求 | 请求生命周期（TTFT/TPOT 排障的骨架） |
| `step` | core | 每步 | EngineCore.step 耗时、输出数（引擎心跳） |
| `schedule` | core | 采样 | 调度 token 数、队列规模、KV 块分配数 |
| `preempt` | core | 每事件 | preemption（v1 为 recompute 模式） |
| `forward` | worker | 采样 | 前向耗时、batch 组成 |
| `sample_flip` | worker | 每事件 | **argmax 处于 flip 区**：margin < eps 的位置（temp=0 不稳定的根因定位） |
| `sample_stats` | worker | 采样 | greedy 批的 margin 聚合（min/max/mean、flip 数） |

事件统一字段：`schema_ver` / `ts_ns`(wall clock) / `pid` / `group` / `type` /
`req_id` / `step` / `sampled` / `data`。**不记录 prompt 原文**（只记类型/长度）。

## 架构

```
vllm-sniffer (pip 包, entry point: vllm.general_plugins)
├─ API server 进程: 请求生命周期 hooks
├─ EngineCore 进程: EngineCore.step / Scheduler.schedule hooks
├─ Worker 进程:     GPUModelRunner.execute_model / Sampler.greedy_sample hooks
└─ 每进程: 无锁有界队列 → daemon 线程 → JSONL（队列满丢弃，不阻塞推理）
```

跨进程关联键：`req_id` + `step` + `ts_ns`。多进程文件按 `run_id` 目录聚合。

## 设计原则

1. **行为不变**：所有 hook 返回原值；hook 异常静默降级（记录后禁用该点），
   绝不把异常抛进推理路径；安装按类幂等（重复加载不会双重包装）
2. **开销预算 <1%**：热路径只采集轻量字段；flip 检测在 GPU 上完成，
   只有罕见 flip 才拷回 host；队列满丢事件不阻塞
3. **可关闭**：`VLLM_SNIFFER=0` 时插件入口直接返回，等价于未安装

## 浮点不确定性观测（temp=0 输出不同）

`temperature=0` 时 vLLM 走 `logits.argmax`，采样本身确定；输出不同 = logits 有微差。
已知根因方向：batch 组成变化（vLLM 官方 `VLLM_BATCH_INVARIANT` 即为此而生）、
前缀缓存命中/未命中、preemption 重算、cudagraph 图切换、多卡 AllReduce 顺序。

tracer 观测：**argmax margin（top1-top2）**——margin < `FLIP_EPS` 时浮点噪声即可翻转输出。
`sample_flip` 事件标记这些高危位置；对拍分析（同 prompt 多次请求对比）在分析层完成，
tracer 本身保持无状态。

## Roadmap

- [x] 注入层（插件自动加载、幂等、静默降级）
- [x] 请求生命周期（online + offline）
- [x] step / schedule / preempt / forward
- [x] greedy argmax-margin 浮点观测（flip 检测 + 聚合统计）
- [ ] env_snapshot（vLLM commit / determinism 相关 env / cudagraph 状态）
- [ ] 分析工具：jsonl → parquet 导出、TTFT/TPOT 分布、repro 对拍（同 tag 对比）
- [ ] logits 位级指纹（采样模式，深挖数值差异来源）
- [ ] 可视化前端（自研，读 JSONL/聚合接口）
- [ ] 多卡 TP/PP（rank 事件字段已预留）与 Ray 集群（node_id + 本地落盘）
- [ ] OTLP sink（可插拔，对接现有可观测栈）

## 开发

```bash
uv venv .venv && uv pip install --python .venv/bin/python pytest msgspec torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest
```

测试不依赖真实 vLLM：用假模块注入 `sys.modules` 走真实安装路径，torch 用于验证采样探针。
