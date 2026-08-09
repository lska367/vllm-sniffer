# 可视化前端（webapp）— 使用与聚合接口文档

> 对应 [EXTENSION_ROADMAP.md](EXTENSION_ROADMAP.md) P2-1。离线分析型 Web：
> FastAPI 聚合层 + 自研前端（vanilla JS + canvas，无 CDN / 无图表库依赖）。
> 读取 run 目录（JSONL）或 `export_parquet.py` 导出的 parquet，不要求实时。

## 1. 启动

```bash
# 安装可选依赖
uv pip install --python .venv/bin/python -e '.[web]'

# 启动（事件根目录默认 $VLLM_SNIFFER_DIR 或 /tmp/vllm-sniffer）
.venv/bin/python -m webapp.server --dir /tmp/vllm-sniffer --port 8080
# 打开 http://127.0.0.1:8080
```

页面流程：选 run_id → 四张图（请求时间线 / step 耗时序列 / flip 分布热图 /
batch 组成 vs 耗时散点）。无数据视图显示空态提示。

## 2. 聚合接口总览

Base: `http://127.0.0.1:<port>`。所有接口只读、无鉴权（内网分析用）。

| 接口 | 返回 | 对应视图 |
|---|---|---|
| `GET /api/runs` | run 列表（目录/parquet，事件数与类型计数） | 下拉框 |
| `GET /api/runs/{run_id}/summary` | 事件统计 + `env_snapshot` 参照系 | 头部元信息 |
| `GET /api/runs/{run_id}/timeline` | 每请求 TTFT/TPOT 时间线 | 请求时间线 |
| `GET /api/runs/{run_id}/steps` | engine step 耗时序列 | step 序列 |
| `GET /api/runs/{run_id}/flips` | sample_flip 明细（margin/top1/top2） | flip 热图 |
| `GET /api/runs/{run_id}/scatter` | forward 事件（batch 组成 × 耗时） | 散点图 |
| `GET /healthz` | 服务健康 | — |

`run_id` 必须是 `[0-9A-Za-z._-]+`（防路径穿越；非法 id 一律 404）。
输入可以是 run 目录（`<out_dir>/<run_id>/` 含 .jsonl）或
`<out_dir>/<run_id>.parquet`（`kind` 字段区分）。

## 3. 响应结构

### `GET /api/runs`

```json
{"runs": [{"run_id": "run-20260809-120000", "kind": "dir",
           "n_events": 12, "span_s": 0.74,
           "types": {"step": 2, "request_start": 2, ...}}]}
```

### `GET /api/runs/{run_id}/summary`

```json
{"run_id": "...", "n_events": 12, "span_s": 0.74,
 "types": {"sample_flip": 1, ...},
 "env": {"vllm_version": "0.19.1", "vllm_commit": null,
         "torch_version": "2.10.0+cu128", "python_version": "3.12",
         "sniffer": {"margin": true, "logits_fp": false, ...},
         "env": {"VLLM_BATCH_INVARIANT": "1"}}}  // 无 env_snapshot 时为 null
```

### `GET /api/runs/{run_id}/timeline`

```json
{"requests": [{"req_id": "req-001", "start_ts": 1754460965,
               "finish_ts": 1754460976, "ttft_ms": 154.2,
               "tpot_ms": 10.3, "n_output_tokens": 64,
               "finish_reason": "length", "aborted": false}]}
```

- `ttft_ms` = first_token.ts − start.ts；`tpot_ms` = (finish − first_token) /
  (n_output_tokens − 1)；缺锚点为 `null`。按 start 时间排序。
- `aborted`：存在 `request_abort` 事件。

### `GET /api/runs/{run_id}/steps`

```json
{"steps": [{"step": 1, "ts_ns": 1754460965, "dur_ms": 8.7,
            "n_outputs": 1}]}
```

按 ts_ns 排序（warmup 虚拟批在前，前端红色标注）。

### `GET /api/runs/{run_id}/flips`

```json
{"total": 19,
 "flips": [{"ts_ns": 1754460965, "step": 3, "margin": 0.0004,
            "top1": 1234, "top2": 5678}]}
```

`step` 可能为 `null`（worker 侧尚未带 step 关联），前端按事件序号分桶。

### `GET /api/runs/{run_id}/scatter`

```json
{"points": [{"ts_ns": 1754460965, "dur_ms": 8.2, "num_tokens": 256,
             "num_seqs": 4}]}
```

## 4. 前端实现说明

- `webapp/static/index.html` 单文件：fetch 五个接口 → canvas 绘制。
- 图表全部自绘（轴、刻度、图例），无第三方前端依赖，可离线打开。
- flip 热图：x = step（或事件序号分桶），y = log10(margin)（−7..0），
  颜色深度 = 事件数（对数刻度）。
- 深色主题；空数据视图显示中文空态。

## 5. 扩展指南

- 新增视图：后端在 `webapp/__init__.py` 加一个只读端点（复用
  `tools/common.load_events` 的缓存 `_load_cached`），前端加 canvas 绘制函数。
- 数据量大时：`webapp/__init__.py` 的事件缓存（按 mtime 失效，最多 8 项）
  是 v0 级缓存；后续可换 parquet 列式读取。
- 聚合口径与 `tools/` 完全一致（共用 `tools/common.py`），
  前端看到的数字与 CLI 工具输出可对账。

## 6. 验收对照（EXTENSION_ROADMAP P2-1）

- ✅ 打开页面 → 选 run_id → 四张图（时间线/step 序列/flip 热图/散点）
- ✅ 聚合接口本文档登记（§2/§3）
- ✅ 合成数据端到端测试（tests/test_webapp.py，10 个用例）
- ⏳ 真机数据演示待 GPU 环境复跑（见 VALIDATION_LOG）
