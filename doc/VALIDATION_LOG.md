# 真机验证记录（Validation Log）

> 本文件是项目可信度的证据档案：什么环境、跑了什么、看到了什么、
> 踩了什么坑。所有"能讲的数字"都从这里来，引用时注明日期与场景。

## 1. 环境

| 项 | 值 |
|---|---|
| 日期 | 2026-08-06 |
| GPU | NVIDIA V100 32GB（sm_70） |
| 模型 | Qwen/Qwen2.5-0.5B-Instruct |
| vLLM | 0.19.1（源码参考 /home/lskam/work/kvcache_research/vllm） |
| torch | 2.10.0+cu128 |
| venv | `.venv-gpu`（uv 创建）；CPU 测试用 `.venv`（torch cpu + pytest + msgspec） |

## 2. 场景矩阵

| 场景 | 命令 | 结果 |
|---|---|---|
| offline 冒烟（eager） | `.venv-gpu/bin/python scripts/offline_smoke.py` | ✅ 7 种事件全产出 |
| offline 冒烟（cudagraph） | `scripts/offline_smoke_cudagraph.py` | ✅ |
| online api_server + 流式 | api_server + `scripts/online_smoke.py`（httpx） | ✅ TTFT 可算 |
| online + 客户端断开 | `online_smoke.py` 的 `stream_then_abort` | ✅ abort 双路径验证 |
| **确定性对拍**（2026-08-09） | `scripts/exp_determinism.py --n 8 --max-tokens 64 --runs 3` | ✅ 3 run 均 2/8 唯一输出，分叉点稳定位置 13 |
| **logits 指纹对拍**（2026-08-09） | `scripts/exp_logits_fp.py --n 8 --max-tokens 32` | ✅ solo-vs-solo IDENTICAL；solo-vs-mixed LOW-BIT NOISE 92.4% |
| **开销量化**（2026-08-09） | `scripts/exp_overhead.py --n 16 --max-tokens 64 --repeat 3` | ✅ margin-off +2.0% / margin-on −21.7% |
| **webapp 真机演示**（2026-08-09） | `python -m webapp.server --dir /tmp/vllm-sniffer` | ✅ 9 个 run 四视图数据全通 |

## 3. 关键观测数据

- **TTFT**：正常请求 154ms；abort 请求 26ms（2026-08-06）；
  2026-08-09 online 复测 TTFT p50=28.9ms / TPOT p50=8.9ms（固定短 prompt）
- **step 中位耗时**：8.7ms（2026-08-06 cudagraph）；8.9ms（2026-08-09）
- **flip 观测**：64-token 输出中 **19 个位置处于 flip 区**（2026-08-06）；
  2026-08-09 复现：flip 密度 2.0%（offline）/ 2.4%（online）
- **2026-08-09 确定性实验**：同 batch 内 8 个相同 prompt → **2 个唯一输出**；
  分叉点稳定在位置 13（token 476 "of" vs 13 "\n"，margin=2⁻¹⁰，fp16 精度下限），
  分叉后永不汇合（51/64 位置分歧）；flip 对 (476,13) 被 tracer 命中 21 次
- **2026-08-09 logits 指纹**：solo-vs-solo 位级 IDENTICAL（Δ=0，63 事件/287 行）；
  solo-vs-mixed 差异 92.4% 集中在尾数位 13..22 → batch 组成改变 → 低位数值
  路径抖动（论文素材级证据）
- **2026-08-09 开销**：baseline 1700 tok/s；margin-off +2.0%（事件管线≈免费）；
  margin-on −21.7%（topk(2)+每步同步，0.5B/V100；大模型上相对成本预计显著
  下降，待 A100/H100 实测）
- **warmup**：启动时虚拟批（sample_stats n_greedy=256）混入事件流，
  分析层需按首个真实 step 事件时间戳分割（2026-08-09 起 step=0 即虚拟批标记）
- **事件完整性**：request_start / first_token / finish / abort / step /
  schedule / preempt / forward / sample_flip / sample_stats / env_snapshot /
  logits_fp 全部真实产出

## 4. 平台特性验证

- **fork 路径（offline）**：hooks 靠 fork 内存继承传播到子进程，
  writer 单例经 register_at_fork 重置 —— ✅
- **spawn 路径（online）**：uvicorn 多线程触发 `_maybe_force_spawn`，
  子进程重新 import → 插件重新加载 —— ✅
- **插件自动加载**：三个进程（API server / engine core / worker）均验证
- **行为不变**：开/关 tracer 的推理输出一致（冒烟观测，未做逐 token
  回归——已列入 roadmap 质量底线）
- **VLLM_SNIFFER=0 完全关闭**：已验证（等价未安装）

## 5. V100 平台事实（影响结论适用范围）

- 无 FlashAttention-2 → fallback TRITON_ATTN
- bf16 自动降级 fp16
- **`VLLM_BATCH_INVARIANT` 不可用**（需要 cc>=9.0）→ batch 组成的影响
  无法用官方开关消除 → tracer 的观测在 V100 上更有价值
- 单卡 tp=1 时模型执行在 engine core 进程内（无独立 worker 进程）

## 6. 踩坑记录（全部为真机验证中发现，已修复）

1. writer 定时 flush 漏尾：只在收到事件时检查时间 → 文件 0 字节
   （修复：`q.get(timeout)` 触发定时 flush）
2. 流式 DELTA token_ids 是增量：需累计；`RequestOutput` 无
   `finish_reason` 字段（在 `outputs[0]` 上）
3. `GeneratorExit` 是 `BaseException`：`except Exception` 抓不到客户端断开
4. hook 内 DEBUG 打印访问不存在属性 → AttributeError 传播到 serving 层、
   请求失败（教训：hook 内一切代码必须 try/except）
5. `pkill -f` 匹配到 bash 命令行自身、杀掉 shell（用 pgrep 再 kill）
6. vLLM 对未知 `VLLM_SNIFFER_*` env 打一次性警告（无害）
7. 子进程 os._exit 路径不跑 atexit：尾部数据靠定时 flush 兜底
8. `EngineCore.__init__` 第一行就调用 load_general_plugins →
   patch `__init__` 做 engine_ready 事件无效（已废弃该思路）
9. msgspec Struct 必填字段不能跟在默认字段后 → `kw_only=True`
10. 重复安装双重包装 → 事件 ×2（按类 mark_wrapper 幂等修复）

## 7. 复现步骤（别人验证用）

```bash
# 1. GPU 环境
uv venv .venv-gpu
uv pip install --python .venv-gpu/bin/python -e .
# 2. offline
VLLM_SNIFFER_DIR=/tmp/sniffer-test .venv-gpu/bin/python scripts/offline_smoke.py
# 3. online（两个终端）
VLLM_SNIFFER_DIR=/tmp/sniffer-online .venv-gpu/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct --port 8099 --no-enable-log-requests &
.venv-gpu/bin/python scripts/online_smoke.py
# 4. 查看事件
ls /tmp/sniffer-test/*/
```

## 8. 待补验证项

- [ ] 开/关 tracer 的逐 token 输出一致性回归（固化为 CI 步骤）
- [ ] 真机确认 2026-08-06 的 TTFT 154ms 场景（本次 p50=28.9ms 为固定 prompt
      短请求；154ms 对应 2026-08-06 的首次请求/不同并发场景）
- [ ] cudagraph 捕获路径下 flip 观测的完整性（图内采样可能漏）
- [ ] TP>1 / PP 多卡验证（rank 字段启用；step 计数在独立 worker 进程
      不传播，需按 rank 关联）
- [ ] Ray 集群验证（node_id + 本地落盘）

## 9. 新增工具与脚本（2026-08-09 上午：合成数据验证）

> 2026-08-09 下午已全部在 GPU 真机复跑通过——见第 2 节（场景矩阵）新增行
> 与本文件底部「真机复跑结果归档」清单。本表保留作为合成数据阶段的历史记录。

| 项 | 状态 | 待办 |
|---|---|---|
| `env_snapshot` 事件（主进程发出） | 单测 ✅（fake 模块） | ✅ 真机确认：fork/spawn 两条路径 JSONL 首事件均为 env_snapshot 且只发一次 |
| `tools/export_parquet.py` | 合成数据 ✅（pyarrow 往返） | ✅ 真机 run 目录导出：818 事件 / 2 pid 文件；req_id/step 空值为设计语义（worker/core 无 req_id） |
| `tools/latency_report.py` | 合成数据 ✅（TTFT/TPOT 数值断言） | ✅ 真机 online run：TTFT p50=28.9ms / TPOT p50=8.9ms |
| `tools/repro_compare.py` | 合成数据 ✅（flip 归因断言） | ✅ 真机：flip 密度 2.0%（offline）/ 2.4%（online），逐请求归因可算 |
| `logits_fp` 指纹事件（P1-1） | 单测 ✅（精确 bit 计数断言、fp16 16 位） | ✅ 真机：采样前 fp32 确认（sampler.py:90）；solo 臂 63 事件/287 行 |
| `tools/logits_fp_compare.py` | 合成数据 ✅（三分断言） | ✅ 真机：solo-vs-solo IDENTICAL Δ=0；solo-vs-mixed LOW-BIT NOISE 92.4% 尾数位 |
| `scripts/exp_logits_fp.py` | 导入/语法 ✅ | ✅ 真机两臂对拍全通（含 subprocess 隔离修复后） |
| `webapp/` 可视化前端（P2-1） | 合成数据 ✅（10 个接口用例） | ✅ 真机：9 个 run 可列、四视图数据源全通 |
| `scripts/exp_determinism.py` | 导入/语法 ✅ | ✅ 真机：3 run × 8 req 全部 2 个唯一输出，分叉点稳定在位置 13 |
| `scripts/exp_overhead.py` | 导入/语法 ✅（TOKENS 解析修复后） | ✅ 真机三臂开销量化完成（见 §3） |

## 10. 真机复跑结果归档（2026-08-09 下午，V100）

全部实验在本节完成，run 目录保留在 `/tmp/vllm-sniffer/`：

| 实验 | run 目录 | 结果摘要 |
|---|---|---|
| 确定性对拍 ×3 | `exp-determinism-20260809-164331` / `-164839` | 每 run 8 请求 2 个唯一输出；**分叉点稳定在位置 13**（token 476 "of" vs 13 "\n"，margin=2⁻¹⁰）；51/64 位置分歧；flip 对 (476,13) 命中 21 次 |
| logits 指纹 solo | `exp-logits-fp-solo-20260809-165502` | 63 事件 / 287 行；solo-vs-solo **IDENTICAL Δ=0**（同 batch 位级确定） |
| logits 指纹 mixed | `exp-logits-fp-mixed-20260809-165538` | solo-vs-mixed **LOW-BIT NOISE：92.4% 差异在尾数位 13..22**（batch 组成 → 数值路径抖动） |
| online api_server | `exp-online-20260809-170508` | spawn 路径；8 请求 + 2 abort；TTFT p50=28.9ms / TPOT p50=8.9ms；flip 密度 2.37% |
| 开销量化 ×3 轮 | `exp-overhead-20260809-165927` | baseline 1700 tok/s；margin-off **+2.0%**；margin-on **−21.7%** |

真机复跑命令：

```bash
.venv-gpu/bin/python scripts/exp_determinism.py --n 8 --max-tokens 64 --runs 3
.venv-gpu/bin/python scripts/exp_logits_fp.py --n 8 --max-tokens 32   # P1-1 两臂对拍
python tools/export_parquet.py /tmp/vllm-sniffer/<run_id> -o run.parquet
python tools/latency_report.py run.parquet
python tools/repro_compare.py run.parquet
python tools/logits_fp_compare.py <solo_run> <mixed_run>               # 差异位分布
.venv-gpu/bin/python scripts/exp_overhead.py --n 16 --max-tokens 64 --repeat 3
python -m webapp.server --dir /tmp/vllm-sniffer --port 8080            # 四视图演示
```

## 11. 2026-08-09 真机新发现（代码已修复）

1. **step 字段恒为 null**（schema 承诺的迭代号从未写入）：engine/worker 事件
   现通过进程内共享计数器携带 step（TP=1 模型执行在 engine core 进程内）；
   warmup 在 step 循环外 → step=0 即虚拟批标记。新增
   `vllm_sniffer/core/step_counter.py`
2. **forward 丢 num_tokens/num_seqs**：`input_batch` 数组执行后即重置。改为
   从 `scheduler_output.num_scheduled_tokens` 取 batch 组成（与
   `total_num_scheduled_tokens` 等价），input_batch 保留作兜底
3. **异步调度返回 (None, False) 的迭代丢 step 事件**：仅调度无输出的迭代
   现在也发心跳（n_outputs=0）
4. **exp_logits_fp 两臂同进程会串台**：fork 的 engine 进程继承主进程
   lru_cached Config（out_dir 还是第一臂的）→ mixed 臂事件写进 solo 目录。
   改为每臂 subprocess + env 清洗（`VLLM_SNIFFER_*` 全剔除再设新值）
5. **exp_overhead TOKENS 行是 4 字段**（`TOKENS 1024 SECS 0.589`），旧解析
   期待 3 字段 → 真机首次运行必崩；已修复 + 解析单测
6. **开销数字修正设计预算**：事件管线本身 ~2%（1700→1734 tok/s），但 margin
   探针（topk(2)+每步同步）在 0.5B/V100 上 **−21.7%**（1331 tok/s）。README
   "<1% 预算" 表述已改为按配置区分；大模型/大 batch 上相对成本会显著下降
   （topk 只随 vocab 规模），待 A100/H100 实测
7. **pkill -f 自杀**再次复现（清理 webapp 时杀掉 bash 自身）——统一用
   `pgrep -f | grep -v $$` + kill
