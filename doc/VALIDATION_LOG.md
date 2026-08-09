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

## 3. 关键观测数据

- **TTFT**：正常请求 154ms；abort 请求 26ms（0.5B 模型，V100）
- **step 中位耗时**：8.7ms（cudagraph 模式）
- **flip 观测**：64-token 输出中 **19 个位置处于 flip 区**（margin < 1e-3）
  ——temp=0 不稳定性的量化证据
- **warmup**：启动时虚拟批（sample_stats n_greedy=256）混入事件流，
  分析层需按首个真实 step 事件时间戳分割
- **事件完整性**：request_start / first_token / finish / abort / step /
  schedule / preempt / forward / sample_flip / sample_stats 全部真实产出

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
- [ ] 开销量化对比（`VLLM_SNIFFER_MARGIN=0` vs 1 的吞吐差）——脚本已备：
      `scripts/exp_overhead.py`
- [ ] cudagraph 捕获路径下 flip 观测的完整性（图内采样可能漏）
- [ ] TP>1 / PP 多卡验证（rank 字段启用）
- [ ] Ray 集群验证（node_id + 本地落盘）

## 9. 新增工具与脚本（2026-08-09，合成数据验证，待真机复跑）

本批新增未经 GPU 真机验证的部分如下，**勿把合成数据结论当真机证据**：

| 项 | 状态 | 待办 |
|---|---|---|
| `env_snapshot` 事件（主进程发出） | 单测 ✅（fake 模块） | 真机确认 JSONL 首事件为 env_snapshot 且只发一次 |
| `tools/export_parquet.py` | 合成数据 ✅（pyarrow 往返） | 真机 run 目录导出 + 查空值列 |
| `tools/latency_report.py` | 合成数据 ✅（TTFT/TPOT 数值断言） | 用 2026-08-06 真实数据复算 TTFT p50=154ms |
| `tools/repro_compare.py` | 合成数据 ✅（flip 归因断言） | 复现"64 token 中 19 flip"；校验 ts 归因的歧义率 |
| `scripts/exp_determinism.py` | 导入/语法 ✅ | GPU 跑同 prompt × N 对拍 |
| `scripts/exp_overhead.py` | 导入/语法 ✅ | GPU 跑三臂开销量化 |

真机复跑命令：

```bash
.venv-gpu/bin/python scripts/exp_determinism.py --n 8 --max-tokens 64 --runs 3
python tools/export_parquet.py /tmp/vllm-sniffer/<run_id> -o run.parquet
python tools/latency_report.py run.parquet
python tools/repro_compare.py run.parquet
.venv-gpu/bin/python scripts/exp_overhead.py --n 16 --max-tokens 64 --repeat 3
```
