# 代码走读指南

> 目标：按本文档走读一遍后，你能回答"每个设计决策为什么存在"。
> 全程约 2-3 小时。建议边读边对照 [agent.md](../agent.md) 第 6 节踩坑清单——
> 代码里每一处"看似多余"的防御，都对应一个真实踩过的坑。

## 0. 阅读顺序总览

```
vllm_sniffer/__init__.py      ← 入口：插件加载（先读，建立全局图）
vllm_sniffer/config.py        ← 配置（简单，热身）
vllm_sniffer/core/event.py    ← 事件 schema
vllm_sniffer/core/writer.py   ← 核心组件：队列 + daemon 线程 + fork 安全
vllm_sniffer/hooks/__init__.py← 安装基础设施（幂等标记）
vllm_sniffer/hooks/api.py     ← 请求生命周期
vllm_sniffer/hooks/engine.py  ← 引擎热循环
vllm_sniffer/hooks/worker.py  ← 浮点观测（研究核心）
tests/conftest.py + 测试      ← 理解"无 vLLM 也能测"的测试策略
scripts/*.py                  ← 真机验证脚本
```

## 1. 入口：`vllm_sniffer/__init__.py`

只有 `load()` 一个函数，回答三个问题：

- **什么时候被调用**：vLLM 每个进程 import 时调 `load_general_plugins()`
  （entry point 机制，见 pyproject.toml 的 `[project.entry-points."vllm.general_plugins"]`）
- **为什么必须幂等**：vLLM 文档警告插件可能被多进程重复加载；
  模块级 `_loaded` 标志防重复执行
- **失败怎么办**：整体 try/except → stderr 打 traceback → 禁用
  （vLLM 的日志系统不显示 vllm_sniffer logger，必须显式打 stderr）

**走读重点**：`load()` 里"先装 hooks 再 prepare_run_dir()"的顺序——
run_id 必须在 vLLM fork/spawn 子进程**之前**导出 env，否则各进程目录不一致。
注意 load() **不创建 writer**（惰性，fork 安全的需要）。

## 2. 配置：`vllm_sniffer/config.py`

`get_config()` 用 lru_cache 缓存。全部 env 前缀 `VLLM_SNIFFER_`。

**走读重点**：`margin` 配置被**运行时**读取（不只是安装时）——
见 worker.py 里 wrapper 内 `get_config()` 的注释：wrapper 可能因共享类
（测试场景）残留，运行时要重新检查开关。

## 3. 事件 schema：`vllm_sniffer/core/event.py`

- `Event` 是 msgspec Struct：`kw_only=True`（msgspec 要求必填字段不能
  跟在默认字段后——这是踩坑后的修正）
- `gc=False` + `frozen=True`：减少分配、防误改
- `data` 只允许 JSON 类型——msgspec 编码时的硬约束，从类型上保证隐私

**走读重点**：`make_event()` 统一填 ts_ns/pid——每个 hook 少写 3 行，
也保证字段口径一致。

## 4. 核心组件：`vllm_sniffer/core/writer.py`（必读，全项目最精巧）

四个设计点，每个都是坑的产物：

1. **有界队列 + put_nowait**：生产端（hook）绝不被慢磁盘阻塞；
   Full → dropped 计数。这是"热路径零阻塞、事件管线近零开销"承诺的实现基础
（真机实测：margin-off +2.0%，见 VALIDATION_LOG）
2. **daemon 线程 + q.get(timeout=1s)**：定时 flush 必须由 get 的
   timeout 触发，而不是"收到事件时检查时间"——否则事件流尾部
   （<32 行且 1s 内结束的请求序列）永远不 flush，文件 0 字节
   （真实踩过）
3. **惰性单例 + register_at_fork**：fork 子进程继承的 writer 其消费者
   线程不存在，事件进死队列。子进程重置单例（**只标 closed，不 close()**——
   join 不存在的线程会阻塞 5s），首事件时重建
4. **run_id 解析**：`_resolve_run_id` 先看 env 继承（子进程），再看
   同秒目录（兄弟进程），最后自建——三种进程来源都落同一目录

**走读重点**：`emit()` 是 hook 的唯一出口，它自身也整体 try/except——
"hook 层任何异常不进推理路径"在最后一公里也要守住。
`_run()` 消费者线程异常只打印不崩溃（daemon 线程崩了进程照跑，但事件全丢，
所以打日志可诊断）。

## 5. 安装基础设施：`vllm_sniffer/hooks/__init__.py`

- `mark_wrapper()` / `is_our_wrapper()`：wrapper 函数打 `_sniffer_wrapper`
  属性标记，**按类幂等**——重复安装检测到标记直接跳过
  （否则双重包装 → 事件 ×2，真实踩过）
- `install_all()`：三组 hook 各自 try/except，一组失败不影响其它组

## 6. API hooks：`vllm_sniffer/hooks/api.py`

三个 patch 点，三种语义：

| patch | 语义 | 关键实现 |
|---|---|---|
| `AsyncLLM.generate` + `AsyncLLMEngine.generate` | 流式请求生命周期 | async generator 包装，yield 原样透传 |
| `AsyncLLM.abort` | 显式 abort 信号 | 最简单可靠的断开检测（server 主动调） |
| `LLM.generate` | offline 批处理 | 同步包装，start/finish/abort 三元组 |

**走读重点**（每个都是坑）：

- **两个类都要 patch**：v0.19 API server 用 AsyncLLM，AsyncLLMEngine 是
  遗留类；每个类用工厂 `_make_generate_wrapper(orig)` 生成独立闭包——
  共享闭包会让两个 orig 都指向最后一次 getattr 的方法
- **request_id 是第 3 个位置参数**（v0.19 签名），不是 keyword
- **finish 判定**：`output.finished` 在 RequestOutput 上；
  `finish_reason` 在 `outputs[0]` 上
- **流式 DELTA token_ids 是增量**：total_out_tokens 必须累计
- **except BaseException**：GeneratorExit 继承 BaseException，
  `except Exception` 抓不到客户端断开
- **`_summary_of_prompt` 只记形状不记内容**：隐私设计的落地

## 7. Engine hooks：`vllm_sniffer/hooks/engine.py`

**走读重点**：

- **同时 patch `step` 和 `step_with_batch_queue`**：v0.19 异步调度默认开启，
  热循环走 step_with_batch_queue，step 根本不执行；同样用工厂函数
  `_make_step_wrapper(orig)` 每方法独立闭包
- step 事件**不采样**（引擎心跳，数量少）；preempt 不采样（低频）
- schedule 按采样率，data 从 `SchedulerOutput` 提取（total_num_scheduled_tokens、
  num_scheduled_tokens 字典长度、preempted_req_ids、new_block_ids_to_zero）

## 8. Worker hooks：`vllm_sniffer/hooks/worker.py`（研究核心）

### 8.1 execute_model 计时

wrapper 先调 orig 再计时——**只测原调用耗时，不计 hook 自身**。
batch 组成（num_tokens/num_seqs）优先从 `scheduler_output.num_scheduled_tokens`
取（真机发现 `self.input_batch` 数组执行后即重置）；`input_batch` 保留作兜底，
`total_num_scheduled_tokens` 与 schedule 事件对拍。

### 8.2 sampler margin 探针（最值得精读的部分）

背景链（代码注释里写得很清楚）：

```
temp=0 → logits.argmax(dim=-1)（采样确定）
  → 输出不同 ⇒ logits 不同
  → 观测 argmax margin（top1-top2）
  → margin < eps ⇒ 浮点噪声可翻转输出（"flip 区"）
```

实现要点：

1. **先调 orig 再观测**：行为不变的直接体现
2. **`torch.topk(logits, 2)` 一次 GPU kernel** 算 margin
3. **flip 检测全在 GPU 上**：mask + sum + nonzero，只把罕见 flip 行拷回
   host（`flip_rows` 索引 → tolist）；每步只有 1 个标量同步
   （`mask.sum().item()`）用于 n_flips——避免全量 D2H 同步
4. **sample_stats 每步聚合**：min/max/mean 三个标量 .item() 一次同步
5. **wrapper 内运行时再查 cfg.margin**：共享类残留时也能关闭
6. **整体 try/except**：观测失败绝不破坏采样

**开销账**：设计上 decode 步 ms 级、一次 topk(2) + 一次标量同步可忽略；
2026-08-09 真机实测（0.5B/V100）：事件管线 ≈0（margin-off +2.0%），margin 探针
**−21.7%**（topk 只随 vocab 规模，大模型/大 batch 相对成本显著下降，待 A100/H100）。
`VLLM_SNIFFER_MARGIN=0` 完全关闭（安装时就不 patch）。——这是"真实数字"的
唯一来源（VALIDATION_LOG §3/§10），讲开销时必须按配置区分，勿再说"<1%"。

## 9. 测试策略：`tests/`

核心 trick：**假 vLLM 模块注入 sys.modules**，测试走真实安装路径
（hook 里的 `from vllm.xxx import ...`），但类都是假的、行为可编程。

- `tests/conftest.py`：`fake_vllm_module()` 构造假 vllm 包树；
  必须先清掉残留的 vllm* 模块（hook 安装失败会留半导入包）
- `tests/test_fork.py`：真 fork 子进程验证 writer 重置（子进程写管道回报）
- `tests/test_hooks_worker.py`：用 CPU torch 验证 topk 探针
- `monkeypatch.setattr(..., raising=False)`：模块新属性不存在时兼容

## 10. 真机脚本：`scripts/`

- `offline_smoke.py` / `offline_smoke_cudagraph.py`：offline 冒烟
- `online_smoke.py`：httpx 流式 + `stream_then_abort` 场景
  （读几个 chunk 断开 → 验证 abort 双路径）
- `exp_determinism.py`：同 prompt × N、temp=0 输出一致性（逐位置 diff + 唯一序列）
- `exp_logits_fp.py`：P1-1 两臂对拍（solo vs mixed，子进程 env 隔离防串台）
- `exp_overhead.py`：三臂开销量化（baseline / margin-off / margin-on）
- 真机验证记录与命令见 [VALIDATION_LOG.md](VALIDATION_LOG.md) §10

## 11. 推荐对照阅读的 vLLM 源码（v0.19.1）

读 hook 时对照原函数，理解"我在观测什么"：

| 主题 | 文件 | 看什么 |
|---|---|---|
| 插件加载 | vllm/plugins/__init__.py | load_general_plugins 的调用链 |
| 请求入口 | vllm/v1/engine/async_llm.py | AsyncLLM vs AsyncLLMEngine 的关系 |
| 引擎热循环 | vllm/v1/engine/core.py | step / step_with_batch_queue 分工 |
| 调度器 | vllm/v1/core/sched/scheduler.py | SchedulerOutput 的字段语义 |
| 模型执行 | vllm/v1/worker/gpu_model_runner.py | input_batch 结构、cudagraph 路径 |
| 采样器 | vllm/v1/sample/sampler.py | greedy_sample 的 argmax 分支 |
| 批不变性 | vllm/config.py 搜 BATCH_INVARIANT | 官方对 batch 组成问题的解法 |

## 12. 自测题（读完应能回答）

1. 为什么 writer 必须惰性创建？fork 子进程里为什么不能 close() 旧 writer？
2. 为什么 step 和 step_with_batch_queue 都要 patch？为什么不能用共享闭包？
3. finish_reason 在哪里取？流式 n_output_tokens 为什么必须累计？
4. 客户端断开时 generate wrapper 会走哪个异常分支？为什么 except Exception 不够？
5. flip 检测为什么在 GPU 上完成？每步最少几次设备同步？
6. 为什么 hook 内连 DEBUG print 都要 try/except？
7. 如果 vLLM 版本升级后 `AsyncLLM.generate` 签名变了，会发生什么？
   （提示：wrapper 的 *args/**kwargs 兜底 + 异常静默降级）
8. 事件流里如何区分 warmup 虚拟批和真实请求？（提示：首个 step 时间戳）

答案都在 agent.md 第 6 节与本文档正文中；答不上来的题对应你还没读透的模块。
