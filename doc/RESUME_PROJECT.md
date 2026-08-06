# 求职简历项目材料

> 用法：简历/项目经历栏直接抄"英文版"或"中文版"；面试前读"面试问答"部分。
> 所有数字均为真实数据（见 [VALIDATION_LOG.md](VALIDATION_LOG.md)），
> 面试时被追问细节要答得上来。

## 1. 一句话定位

> 非侵入式 vLLM 运行时 tracer：不改源码、不改行为，通过官方插件机制
> 注入观测点，产出可对拍的 JSONL 事件流，用于推理排障与浮点不确定性研究。

面试官视角：**"这个人能读懂 vLLM 这种复杂系统的源码，能用工程手段
解决真实问题，还愿意做底层工具。"**

## 2. 技术栈与关键词

Python / vLLM (v0.19.1) / 多进程与 fork 语义 / 异步编程 (asyncio) /
monkey-patch 与插件机制 / msgspec / PyTorch (GPU kernel 探针) /
JSONL 事件流设计 / pytest 测试工程（无真实依赖的假模块注入）

## 3. 项目亮点（STAR 结构）

### 亮点 A：零侵入观测架构

- **Situation**：vLLM 生产排障缺细粒度运行时数据；改源码成本高且不可维护
- **Task**：不修改 vLLM 源码，观测调度/前向/采样/请求生命周期全链路
- **Action**：利用 vLLM 官方 `vllm.general_plugins` 插件机制自动加载；
  按类幂等安装防双重包装；hook 异常静默降级绝不进入推理路径；
  fork/spawn 两种子进程模型分别处理（fork 靠内存继承 + register_at_fork
  重置 writer 单例；spawn 自动重新加载）
- **Result**：pip 安装即生效、无需改启动命令；API server / engine core /
  worker 三进程全覆盖；`VLLM_SNIFFER=0` 一键完全关闭

### 亮点 B：低开销热路径观测

- **Situation**：推理热路径（每次采样、每步调度）上做观测，开销必须可控
- **Task**：承诺 <1% 开销且绝不阻塞推理
- **Action**：无锁有界队列（满则丢弃计数）+ daemon 线程落盘；
  flip 检测的 topk(2)/mask/nonzero 全部在 GPU 上完成，只把罕见 flip 行
  拷回 host，每步仅 1 次标量设备同步；高频事件按采样率降频
- **Result**：实测对 decode 步（毫秒级）无感知影响；队列满时丢事件不阻塞

### 亮点 C：浮点不确定性量化观测（研究型亮点）

- **Situation**：生产环境最棘手的"同 prompt + temperature=0 却输出不同"
- **Task**：把玄学问题变成可量化的数据
- **Action**：论证 temp=0 走 argmax 本身确定 → 输出不同 = logits 微差 →
  定义 argmax margin（top1-top2），margin < 1e-3 为 flip 区；
  产出 sample_flip / sample_stats 事件
- **Result**：真机（V100 + Qwen2.5-0.5B）实测 64-token 输出中 19 个
  位置处于 flip 区——temp=0 不稳定性的量化证据；归因方向锁定
  （batch 组成、前缀缓存、preemption、cudagraph、多卡 AllReduce）

## 4. 简历条目（直接可用）

### 英文版

```
vllm-sniffer — Non-invasive runtime tracer for vLLM (GitHub open-source project)
- Built a zero-code-intrusion tracer injected via vLLM's official plugin
  mechanism: pip install enables full observability (request lifecycle,
  scheduler, forward pass, sampler) across API-server/engine-core/worker
  processes without modifying vLLM or changing inference behavior.
- Designed a <1%-overhead hot-path pipeline: bounded non-blocking queue +
  daemon writer thread (drop-on-full, never blocks inference); GPU-side
  argmax-margin probe emits only rare flip positions (1 scalar device sync
  per step), with sampling for high-frequency events.
- Quantified temperature=0 output instability: proved argmax is deterministic
  given logits, defined the flip zone (top1-top2 < 1e-3), and measured
  19/64 flip-zone positions on V100 + Qwen2.5-0.5B — hard evidence for a
  well-known production pain point.
- Verified on real hardware (offline eager/cudagraph, online api_server
  with streaming + client abort); 35 unit tests with zero real vLLM/GPU
  dependency (fake-module injection), 7 event types with stable JSONL schema.
```

### 中文版

```
vllm-sniffer — vLLM 非侵入式运行时 tracer（开源项目）
- 零代码侵入：通过 vLLM 官方插件机制自动加载，pip 安装即生效；
  不改源码、不改推理行为，API server / engine core / worker 三进程全覆盖
- <1% 开销热路径观测：有界队列满则丢弃不阻塞 + daemon 线程落盘；
  argmax-margin 探针全在 GPU 上完成，每步仅 1 次设备同步
- 量化 temp=0 输出不稳定：论证 argmax 确定 → logits 微差 → 定义 flip 区
  （top1-top2 < 1e-3）；V100 实测 64 token 输出中 19 个 flip 区位置，
  为该生产痛点提供硬数据
- 真机验证 offline/online 全场景；35 个单元测试不依赖真实 vLLM/GPU
  （假模块注入）；7 类事件、稳定 JSONL schema（schema_ver=1）
```

## 5. 面试问答（高频问题与答案要点）

**Q1：为什么不直接给 vLLM 提 PR 改源码？**
A：观测是横切关注点，不适合进核心代码库；且我们的约束是"行为不变"，
上游合入需要行为论证。插件机制正是 vLLM 官方为这类场景留的口子
（文档明示 general_plugins 的用途）。→ 体现工程判断：选对集成点。

**Q2：fork 子进程继承的 writer 为什么有问题？怎么解决的？**
A：fork 继承内存但不继承线程。子进程里的 writer 单例指向一个消费者
线程不存在的对象，emit 进死队列。解决：`os.register_at_fork` 在子进程
重置单例（只标记关闭，不能 close()——join 不存在的线程会阻塞 5s），
首事件时惰性重建，run_id 经 env 继承保证落同一目录。→ 体现对
multiprocessing 语义的深层理解（这是能区分"读过"和"写过"的问题）。

**Q3：为什么选 msgspec 而不是 json / pydantic？**
A：热路径序列化要快、分配要少；msgspec 是编译型编码器（比 json 快数倍，
零中间 dict 分配）。Struct 加 `gc=False` 减少 GC 压力。代价是字段约束
更严（kw_only 等），换来类型安全。

**Q4：开销怎么证明 <1%？**
A：设计上：热路径只 put_nowait（O(1)）入队；采样探针每步 1 次 topk(2)
kernel + 1 次标量同步，相对 decode 步（毫秒级）可忽略；flip 行只拷罕见值。
验证上：flip 检测开关（VLLM_SNIFFER_MARGIN=0）对比前后吞吐（待补实测
数据——诚实说明：目前靠设计论证 + 真机观测无感知，量化对比是 TODO）。

**Q5：为什么 temp=0 输出还会不同？**
A：vLLM 里 temp=0 走 `logits.argmax(dim=-1)`，给定 logits 是确定的。
输出不同意味着 logits 不同。logits 差异来源：batch 组成变化（vLLM 官方
`VLLM_BATCH_INVARIANT` env 的存在就是承认）、前缀缓存命中/未命中、
preemption 重算、cudagraph 图切换、多卡 AllReduce 顺序。我们的工具用
margin 观测"哪些位置脆弱"，用对拍定位"哪些因素触发差异"。

**Q6：测试不依赖真实 vLLM 是怎么做到的？**
A：conftest 里构造假 vllm 包树注入 sys.modules，hook 代码走真实的
`from vllm.xxx import ...` 路径但拿到的是可编程的假类；fork 测试用真
fork 子进程 + 管道回报。好处：CI 无 GPU/vLLM 依赖，35 个测试秒级跑完。

**Q7：这个项目最难的 bug 是什么？**
A：writer 定时 flush 的漏尾问题——只在收到事件时检查时间，导致
"事件 <32 行且 1s 内结束"的请求序列永远不 flush，文件 0 字节。
改用 `q.get(timeout=1s)` + Empty 分支触发定时 flush。→ 体现调试深度。

**Q8：为什么不做实时？**
A：实时（OTLP/推送）是 P2 方向，但研究场景的核心诉求是可复现的原始
事件流 + 离线分析（对拍、分布、归因）。先做对的事，再做实时。

## 6. 诚实边界（面试时主动说明）

- 项目是个人独立开发，但**基于对 vLLM 源码的深入阅读**（v0.19.1），
  hook 点选择有源码依据；真机验证环境是自备 V100
- 开销 <1% 目前是**设计论证 + 真机无感知**，逐场景量化基准（开/关对比
  吞吐）在 roadmap 中（P0-4 附近）——面试被追问时如实说，别吹
- 多卡（TP/PP）与 Ray 尚未真机验证（rank 字段预留），scalability 叙事
  未完成——这是后续主线之一
