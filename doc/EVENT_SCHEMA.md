# 事件流 Schema 参考手册

> 本文档是 `vllm_sniffer/core/event.py` 的完整语义参考，写给三类读者：
> 写分析工具的、写可视化前端的、新增 hook 的。schema 版本 `schema_ver=1`。

## 1. 统一字段

每条事件一行 JSONL，格式：

```json
{"schema_ver":1,"ts_ns":1754460965112345678,"pid":12345,"group":"api",
 "type":"request_start","req_id":"req-001","step":null,"sampled":false,
 "data":{...}}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_ver` | int | schema 版本（当前 1）；字段变更时递增，分析层据此兼容 |
| `ts_ns` | int | wall-clock 纳秒（time.time_ns()），跨进程可比，用于排序 |
| `pid` | int | 观测进程 pid |
| `group` | str | `api` / `core` / `worker`（进程树中的位置） |
| `type` | str | 事件类型（见下表） |
| `req_id` | str\|null | vLLM 请求 id，跨进程关联的主键；offline 为 `offline-<thread_id>` |
| `step` | int\|null | engine 迭代序号（engine-core 侧设置） |
| `sampled` | bool | 该事件是否经采样产生（高频事件的降频标记） |
| `data` | dict | 类型专属字段，只含 JSON 类型，**无 prompt 原文** |

## 2. 事件类型总表

| type | group | 采样 | data 关键字段 | 语义 |
|---|---|---|---|---|
| `env_snapshot` | core | 否 | `vllm`, `torch`, `python_version`, `env`, `sniffer`, `run_id` | **每 run 一次**的参照系：版本、determinism env、tracer 配置（主进程发出） |
| `request_start` | api | 否 | `kind`, `n`/`n_chars` | 请求被接受（prompt 形状，不记内容） |
| `request_first_token` | api | 否 | `n_prompt_tokens` | 流上出现首个输出 token（TTFT 锚点） |
| `request_finish` | api | 否 | `finish_reason`, `n_output_tokens` | 流正常结束 |
| `request_abort` | api | 否 | `source`（`explicit_abort` 或空） | 客户端断开/显式 abort |
| `step` | core | 否 | `dur_ns`, `n_outputs` | 引擎步进心跳（不采样） |
| `schedule` | core | 是 | `dur_ns`, `total_num_scheduled_tokens`, `n_scheduled_reqs`, `n_preempted`, `new_block_ids_to_zero` | 调度详情 |
| `preempt` | core | 否 | `mode`（v1 为 `recompute`） | 抢占（低频不采样） |
| `forward` | worker | 是 | `dur_ns`, `num_tokens`, `num_seqs`, `total_num_scheduled_tokens` | 模型前向计时 + batch 组成 |
| `sample_flip` | worker | 否 | `margin`, `top1`, `top2` | greedy argmax 处于 flip 区（margin<eps） |
| `sample_stats` | worker | 是 | `n_greedy`, `margin_min/max/mean`, `n_flips` | greedy 批 margin 聚合 |
| `logits_fp` | worker | 是 | `n_rows`, `dtype`, `sampled_rows`, `rows[{row, top1, bits}]` | **logits 位级指纹**（深挖模式，默认关） |

## 3. 逐类型字段语义

### api 组

**request_start**
- `data.kind`：`str`/`list`/`dict`/类型名 —— prompt 形状
- `data.n_chars`（str 时）/ `data.n`（list/dict 时）—— 长度
- offline 模式附加 `data.mode="offline"`、`data.n_requests`（批大小估算）

**request_first_token**
- `data.n_prompt_tokens`：`output.prompt_token_ids` 长度；流式下仅首个 chunk 触发

**request_finish**
- `data.finish_reason`：`stop` / `length` 等（取值自 `outputs[0].finish_reason`，
  注意 **RequestOutput 本身没有 finish_reason 字段**）
- `data.n_output_tokens`：**流式 DELTA 的 token_ids 是增量，必须累计**

**request_abort**
- `data.source`：`explicit_abort`（server 显式调 AsyncLLM.abort）或缺失
  （GeneratorExit 路径，即客户端断开）

### core 组

**env_snapshot**（run 参照系，每 run 一次）
- `data.run_id`：本次 run 的目录名
- `data.python_version`：Python 版本（`sys.version` 首段）
- `data.vllm.vllm_version` / `data.vllm.vllm_commit`：vLLM 版本与 commit（不可得时为 null）
- `data.torch.torch_version` / `torch_cuda` / `torch_git`：torch/CUDA 版本
- `data.env`：**存在**的 determinism 相关 env 的值：`VLLM_BATCH_INVARIANT`、
  `VLLM_FLOAT32_MATMUL_PRECISION`、`VLLM_USE_CUDA_GRAPH`、`VLLM_ATTENTION_BACKEND`、
  `VLLM_WORKER_MULTIPROC_METHOD`、`VLLM_TORCH_COMPILE_LEVEL`、`NVIDIA_TF32_OVERRIDE`、
  `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`、`CUDA_LAUNCH_BLOCKING`、`VLLM_SNIFFER_*`
- `data.sniffer`：tracer 自身配置（enabled/sample_rate/margin/flip_eps/
  logits_fp/logits_fp_rows）
- 发出方：**创建 run 目录的主进程**（load() 中判定 `VLLM_SNIFFER_RUN_ID` 未设置者）；
  子进程继承该 env，不重复发出。分析层把该事件当 run 的标识记录。

**step**
- `data.dur_ns`：step 方法总耗时（含调度+执行，单卡时含前向）
- `data.n_outputs`：本轮产出的输出数
- 引擎心跳，**不采样**；warmup 虚拟批也会产生，分析层需按首个真实 step
  时间戳分割（见 VALIDATION_LOG.md）

**schedule**
- `data.total_num_scheduled_tokens`：本轮调度的 token 总数
- `data.n_scheduled_reqs`：参与调度的请求数（`num_scheduled_tokens` 字典长度）
- `data.n_preempted`：被抢占请求数（对应 `SchedulerOutput.preempted_req_ids`）
- `data.new_block_ids_to_zero`：新分配待清零的 KV 块数

**preempt**
- `data.mode`：v1 engine 为 `recompute`

### worker 组

**forward**
- `data.dur_ns`：`execute_model` 全程耗时（含 GPU kernel 等待）
- `data.num_tokens` / `data.num_seqs`：`self.input_batch` 的 batch 组成
- `data.total_num_scheduled_tokens`：本次的调度 token 数（与 schedule 事件对拍）

**sample_flip**（研究核心）
- `data.margin`：top1 - top2 logit 差（float32，< `VLLM_SNIFFER_FLIP_EPS`）
- `data.top1` / `data.top2`：两个候选 token id
- **同一位置的 margin 只记录一次**（即使 top1/top2 对调也按行记）
- 语义：margin 小于 eps 时，浮点噪声的量级即可翻转 argmax ——
  这就是 temp=0 输出不稳定的"高危位置"标记

**sample_stats**
- `data.n_greedy`：本步 greedy 行数
- `data.margin_min/max/mean`：聚合（每步 3 个标量 .item() 同步一次）
- `data.n_flips`：本步 flip 行数

**logits_fp**（深挖模式，`VLLM_SNIFFER_LOGITS_FP=1` 才产出）
- 动机：flip 事件只标"哪里可能翻"，不解释"logits 为什么不同"。位级指纹把
  每个采样行的 fp32 bit-pattern 分解成 32 个 bit 位的置位数：
  `bits[b]` = 该行 logits 中第 b 位为 1 的个数。两行 logits 只要有一位不同，
  指纹就不同；差异落在**尾数位（0..22，低位噪声）**还是**阶码/符号位
  （23..31，系统性差异）**可区分"数值路径抖动"与"logits 整体不同"。
- `data.n_rows`：本步总行数（batch 规模；对拍时两 run 不一致 = batch 组成变了）
- `data.dtype`：`torch.float32`（vLLM V1 采样前统一转 fp32，生产恒为 32 位；
  fp16/bf16 防御性支持，16 位指纹）
- `data.sampled_rows`：实际采样的行数（≤ `VLLM_SNIFFER_LOGITS_FP_ROWS`）
- `data.rows[]`：`row`（绝对行号）、`top1`（该行 argmax token，对拍对齐用）、
  `bits`（长度 32 或 16 的置位数列表）
- 行采样是 strided（linspace 均匀铺满 batch），不是随机子集——保证覆盖
  batch 两端。成本：每采样步 k≤8 行 × 32 个 bit 归约 + 一次 D2H 同步，
  深挖模式专用，默认关（零开销承诺不变）
- 分析工具：`tools/logits_fp_compare.py`（两 run 逐位差异分布 + 结论分类）

## 4. 示例（真实数据形态）

```jsonl
{"schema_ver":1,"ts_ns":1754460965112345678,"pid":12345,"group":"api","type":"request_start","req_id":"b62e1f9e-...","step":null,"sampled":false,"data":{"kind":"str","n_chars":42}}
{"schema_ver":1,"ts_ns":1754460965234567890,"pid":12346,"group":"core","type":"step","req_id":null,"step":3,"sampled":false,"data":{"dur_ns":8700000,"n_outputs":1}}
{"schema_ver":1,"ts_ns":1754460965278901234,"pid":12346,"group":"worker","type":"sample_flip","req_id":null,"step":3,"sampled":false,"data":{"margin":0.000482,"top1":1234,"top2":5678}}
```

## 5. 跨进程关联方法

同一 run_id 目录下多个 pid 文件，按以下键关联：

- **请求生命周期**：`req_id` 在 api/core/worker 三组事件中保持一致
- **时序对齐**：`ts_ns` wall clock 全局可比，跨进程排序即得全局时间线
- **步骤对齐**：`step` 号在 engine core 与 worker 间一致（worker 的 step 由
  engine core 传入）；api 事件无 step（异步流），用 ts_ns 就近关联

分析层建议流程：按 `req_id` 分组 → 组内按 `ts_ns` 排序 →
TTFT = first_token.ts_ns - start.ts_ns；TPOT 用 finish/输出 token 数计算。

## 6. schema 演进策略

- 新增字段：向后兼容，`schema_ver` 不变
- 语义变更/字段改名：`schema_ver+1`，分析层按版本分叉
- `data` 内的字段名以 snake_case 为准，新增时在本文档登记
- **加新 hook 前必读**：[EXTENSION_ROADMAP.md](EXTENSION_ROADMAP.md) 的
  事件设计准则（成本/采样/隐私三问）
