# 延伸拓展路线图

> 如何把 vllm-sniffer 从"能用"变成"简历上的亮点 + 科研平台"。
> 分三个优先级，每项给出**动机、实施要点、验收标准**。
> P0 是求职基本盘（建议先做完），P1 是科研价值，P2 是平台化。

## 决策框架：先想清楚"为什么做"

每个候选功能先过三问，再动手：

1. **求职价值**：面试官能一眼看懂吗？有量化指标吗？体现什么能力
   （系统设计 / 性能工程 / 数值分析 / 工具链）？
2. **科研价值**：能产出研究数据吗？与 [[KV Cache 研究]] 的问题对齐吗？
3. **成本**：工作量、GPU 时间、维护负担？

下表是当前评估：

| 功能 | 求职价值 | 科研价值 | 成本 | 优先级 |
|---|---|---|---|---|
| env_snapshot | 中 | **高**（浮点归因参照系） | 低 | P0 |
| 分析工具 v0（parquet + 分布） | **高**（可演示、可量化） | **高** | 中 | P0 |
| README 英文版 + 项目 polishing | **高**（开源门面） | — | 低 | P0 |
| CI + coverage | 中（工程规范感） | — | 低 | P0 |
| logits 位级指纹（深挖模式） | 中 | **高**（论文素材） | 中 | P1 |
| repro 对拍工具 | 中 | **高**（temp=0 复现实验） | 中 | P1 |
| warmup 分割 + 分析层净化 | 低 | 中（数据质量） | 低 | P1 |
| 可视化前端 | **高**（演示效果拉满） | 中 | **高** | P2 |
| TP/PP + Ray 支持 | **高**（规模叙事） | 中 | 高 | P2 |
| OTLP sink | 中 | 低 | 中 | P2 |

---

## P0：求职基本盘（1-2 周）

### P0-1 env_snapshot：启动时记录参照系 ✅（2026-08-09 已实现）

**动机**：浮点问题归因需要"这次运行在什么环境"——vLLM commit、determinism
相关 env（`VLLM_BATCH_INVARIANT`、`VLLM_FLOAT32_MATMUL_PRECISION`）、
cudagraph 开关、torch/CUDA 版本。没有它，任何对拍结论都缺参照系。

**实施要点**（已完成）：
- 新事件类型 `env_snapshot`（group=core，每 run 一次，主进程发出）
- 实现：`vllm_sniffer/core/env_snapshot.py`；`load()` 中判定 run-root
  （`VLLM_SNIFFER_RUN_ID` 未设置者）后发出，子进程继承 env 不重复
- data：`vllm`（version+commit）、`torch`（version/cuda/git）、`python_version`、
  `env`（determinism 相关）、`sniffer`（自身配置）、`run_id`
- 测试：`tests/test_env_snapshot.py`（7 个：收集防御性、fake vllm、env 过滤、
  根进程一次/子进程跳过、load() 端到端）

**验收**：✅ 一次 run 的 JSONL 里能直接读出全部参照系字段（见 EVENT_SCHEMA.md）；
字段名已登记。

### P0-2 分析工具 v0：`tools/` 目录起步 ✅（2026-08-09 已实现）

**动机**：tracer 只产原始数据，**"能讲出故事"全靠分析层**。这是简历上
"数据分析 + 性能工程"叙事的关键；也是科研平台的第一个支柱。

**实施要点**（已完成，三个脚本各自独立可跑，输入为 run 目录或 parquet）：
1. `tools/export_parquet.py`：jsonl → parquet（pyarrow，可选依赖
   `pip install -e '.[analyze]'`）。多 pid 文件合并，按 ts_ns 全局排序；
   扁平 schema + `data_json` 列，schema 稳定。
2. `tools/latency_report.py`：TTFT / TPOT / inter-token 分布。
   TTFT = first_token.ts_ns - start.ts_ns（按 req_id 关联）；
   TPOT = (finish - first_token) / (n_output_tokens - 1)；
   p50/p90/p99 + ASCII 直方图（无 matplotlib 依赖）。
3. `tools/repro_compare.py`：同 prompt 多请求对拍——按 prompt 长度分组做
   输出长度一致性检查，统计 flip 密度（sample_flip / n_greedy），按
   活动窗口 [start.ts, finish.ts] 做 per-request flip 归因并估算输出内位置
   （v0 限制：flip 事件尚无 req_id，ts 归因可能歧义，报告中标注 ambiguous）。

共享层 `tools/common.py`：事件加载（jsonl 目录/parquet 统一 dict 流）、
percentile/直方图、请求时间线组装。测试：`tests/test_tools.py`（7 个，含
parquet 往返与缺 pyarrow 的友好报错）。

**验收**：✅ `export_parquet.py <run_id> && latency_report.py <parquet>` 输出
真实数字；repro_compare 能从真实数据复现 flip 发现。
（合成数据测试全绿；真机数据待 GPU 环境复跑——见 VALIDATION_LOG 待补项。）

### P0-3 开源门面（低成本高回报）

**动机**：简历上的链接第一个被打开的就是 GitHub 主页。

**实施要点**：
- README 英文版（或中英双语）：重点重写"Motivation / How it works /
  Quickstart / Architecture diagram / Roadmap"
- 补 LICENSE（建议 Apache-2.0）、CHANGELOG.md、badges（CI 通过率）
- 整理 scripts 到 `examples/` 并写注释
- doc/ 已有 CONTRIBUTING.md，PR/issue 模板可选（有模板比没有强）
- 首屏 demo：放一张真实 JSONL 片段 + 一段 latency report 输出截图

**验收**：一个不认识项目的人（英文）5 分钟内能跑通安装 + 冒烟 + 看到事件。

### P0-4 CI + 覆盖率

GitHub Actions：`uv venv + pytest`（CPU 即可，测试不依赖 GPU/vLLM），
附 pytest-cov 报告。35 个测试 + 全绿 badge 是工程规范感的直接证据。

---

## P1：科研价值（对齐研究线）

### P1-1 logits 位级指纹（深挖模式）

**动机**：flip 事件只能告诉你"哪里可能翻"，不能告诉你"为什么 logits 不同"。
位级指纹（对 logits 张量做逐位 hash，记录每个位置 32 个 bit 的分布）能定位
差异来源：bit 低位噪声 vs 高位系统性差异 → batch 组成 vs 数值路径。

**实施要点**：
- 新 env `VLLM_SNIFFER_LOGITS_FP=0` 默认关（零开销承诺不变）
- 开启时：采样模式下对 logits 做轻量指纹（如 strided 采样行的 fp32
  bit-pattern hash），随 sample_stats 输出
- 设计成"模式"而非"默认行为"：文档明确这是深挖模式
- 对拍：同 prompt 在不同 batch 组成下跑，比较指纹差异位

**验收**：能区分"同 batch 内差异（≈0）"vs"换 batch 组成后差异（低位噪声）"；
产出一张差异位分布图（论文素材）。

### P1-2 repro 对拍：temp=0 复现实验

**动机**：研究叙事闭环——"观测到 flip 区 → 构造复现实验 → 验证根因"。

**实施要点**：
- `scripts/repro_same_prompt.py`：同 prompt 连发 N 次（固定 seed），
  输出 flip 密度分布；对照组：`VLLM_BATCH_INVARIANT=1`（若硬件支持，
  V100 不行，需 A100/H100）下重跑，看 flip 是否消失
- 分析层对比：同一 prompt 多次请求的输出序列 diff + flip 位置对应关系
- 结论模板化：输出"X% 位置处于 flip 区，其中 Y% 实际翻转"

**验收**：复现 2026-08-06 的观测（0.5B 模型、64 token、19 flips），
并给出翻转位置与 margin 分布的对应图。

### P1-3 分析层数据净化

- warmup 分割：首个真实 step 事件时间戳为界，之前的事件（虚拟批）剔除
- dropped 计数上报：writer 的 dropped 计数需要可见（事件丢了多少，
  结论可信度上限）
- 时间对齐工具：多 pid 文件时钟对齐（同一机器 time.time_ns 一致，无需做，
  但工具应校验 ts_ns 单调性）

---

## P2：平台化（求职展示 + 科研平台支柱）

### P2-1 可视化前端

**动机**：简历演示的"wow 因子"；科研平台的人机界面。
**方向**：自研轻量 Web（FastAPI 聚合接口 + 前端图表），读 parquet 聚合，
不要求实时（离线分析型）。核心视图：请求时间线（每请求 TTFT/TPOT）、
step 耗时序列、flip 分布热图、batch 组成 vs 耗时的散点。
**验收**：打开页面 → 选 run_id → 看到上面四张图；聚合接口有文档。

### P2-2 多卡 TP/PP + Ray

**动机**：规模叙事（"单机单卡" → "多卡多节点"）；多卡 AllReduce 顺序是
浮点差异的已知根因方向之一，科研上必须覆盖。
**要点**：rank 字段（事件 schema 已预留 `rank`）真正启用；sampler margin
按 rank 去重（每 rank 都跑 sampler，事件会重复）；Ray 场景 node_id +
本地落盘 + 聚合工具按 node 合并。
**注意**：多卡调试成本高，排到最后，先保证设计文档（schema 扩展）就位。

### P2-3 可插拔 sink（OTLP）

**动机**：对接现有可观测栈（Grafana 等），从"研究工具"走向"工程工具"。
**要点**：writer 抽象出 `Sink` 接口（emit(batch) / flush / close），
JSONL 是默认实现；OTLP sink 按 batch 推送。接口设计先行，实现后置。

---

## 执行顺序建议

1. ✅ **本周**：P0-1（env_snapshot）+ P0-2（分析工具 v0）——2026-08-09 完成
2. **下周**：P0-3（README 英文版）+ P0-4（CI 已就位，剩 coverage badge）
3. **之后**：P1-3 净化 → P1-2 复现实验（等 GPU 环境，脚本已备：
   `scripts/exp_determinism.py`）→ P1-1
4. P2 三项按求职时间线取舍：前端 > OTLP > 多卡

每完成一项，更新根 README 的 Roadmap 勾选状态与 CHANGELOG。
