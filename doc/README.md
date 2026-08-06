# doc/ 目录说明

vllm-sniffer 的深度文档。根目录 `README.md` 面向使用者（快速上手），
`agent.md` 面向 agent/开发者的工作档案，本目录面向**项目经营者**：
帮助你深入理解代码、规划延伸拓展、把项目经营成求职简历亮点与科研工具平台。

## 文档地图

| 文档 | 用途 | 什么时候读 |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构设计解析：进程模型、数据流、hook 点、设计原则 | 想整体把握项目时 |
| [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) | 代码走读指南：逐文件阅读路径 + vLLM 源码对照表 + 自测题 | 准备动手改代码前 |
| [EVENT_SCHEMA.md](EVENT_SCHEMA.md) | 10 种事件类型的完整字段语义 + JSONL 示例 | 写分析工具 / 前端 / 新 hook 时 |
| [EXTENSION_ROADMAP.md](EXTENSION_ROADMAP.md) | 延伸拓展路线图：按求职价值/科研价值/平台化分优先级 | 决定"下一步做什么"时 |
| [RESEARCH_PLATFORM.md](RESEARCH_PLATFORM.md) | 科研工具平台规划：KV Cache 研究需求 → 采集/分析/可视化三层能力映射 | 把项目接入研究线时 |
| [RESUME_PROJECT.md](RESUME_PROJECT.md) | 求职材料：项目亮点（中英双语）、STAR 话术、量化数据、面试问答 | 写简历 / 面试前 |
| [VALIDATION_LOG.md](VALIDATION_LOG.md) | 真机验证记录：环境、场景矩阵、观测数据、踩坑清单 | 向别人证明项目可信时 / 复现实验时 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 开源贡献指南：开发环境、测试、PR 流程 | 准备开源发布 / 接受外部贡献时 |

## 建议阅读路径

1. **第一遍（理解，约 1 小时）**：README.md → ARCHITECTURE.md → EVENT_SCHEMA.md
2. **第二遍（读代码，约 2-3 小时）**：按 CODE_WALKTHROUGH.md 的顺序走读，
   对照 `agent.md` 第 6 节的踩坑清单（每个坑都对应代码里一处防御性设计）
3. **第三遍（动手）**：挑 EXTENSION_ROADMAP.md 里的一个 P0 任务实现，
   这是把"读过"变成"掌握"的关键一步

## 与其它文档的关系

- `README.md`（根目录）：用户视角，安装/配置/事件一览/真机验证摘要
- `agent.md`（根目录）：agent 工作档案，含代码逐文件详解与踩坑记录；
  doc/ 下的文档是对它的**提炼、重组与深化**，两者内容互补不重复
- `doc/papers/`、`doc/PLAN.md`（研究线目录）：KV Cache 论文与研究计划，
  RESEARCH_PLATFORM.md 说明本工具与它们的衔接方式
