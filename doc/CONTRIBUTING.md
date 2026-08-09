# Contributing Guide

> vllm-sniffer 的开源协作规范。目前是个人项目，但**从第一天就按
> 开源标准经营**——这是求职简历上"开源项目"叙事成立的前提。

## 1. 开发环境

```bash
# CPU 测试环境（无 GPU/vLLM 依赖，全部测试可在此跑）
uv venv .venv
uv pip install --python .venv/bin/python pytest msgspec torch \
    --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest -q        # 90 tests

# 真机验证环境（GPU）
uv venv .venv-gpu
uv pip install --python .venv-gpu/bin/python -e .
```

## 2. 测试规范

- **单元测试不得依赖真实 vLLM 或 GPU**：假模块注入 sys.modules
  （见 tests/conftest.py），CI 在任何机器秒级可跑
- 新增 hook 必须带测试：至少覆盖"安装成功 + 事件产出 + 异常静默降级"
  三个面
- 涉及 fork 的行为必须写 fork 测试（真 fork + 管道回报，见 test_fork.py）
- 修改事件 schema 必须同步更新 `doc/EVENT_SCHEMA.md` 与 `schema_ver`
  演进规则

## 3. 代码风格与约束（铁律）

1. **hook 内任何代码不得抛异常进推理路径**：观测代码整体 try/except，
   失败只记录；DEBUG 打印也不例外
2. **行为不变**：hook 必须返回原值、不修改参数、不影响控制流
3. **开销意识**：热路径只做 O(1) 操作；GPU 观测点先算账（每步几次
   设备同步）再写代码；默认关闭的深挖模式要显式标注
4. **隐私默认**：不记录 prompt 原文，只记形状
5. **幂等**：所有安装函数按类幂等（mark_wrapper），重复加载无害
6. **新事件登记**：新事件类型必须先过三问（成本/采样/隐私，见
   EXTENSION_ROADMAP.md 决策框架），然后在 EVENT_SCHEMA.md 登记字段

## 4. PR 流程

```
1. 从 main 开分支：feature/<描述>
2. 小步提交，信息清晰（一句话说明"为什么"而非"改了什么"）
3. 跑全量测试：.venv/bin/python -m pytest -q
4. 真机相关改动：附验证记录（场景 + 命令 + 结果，格式参考
   doc/VALIDATION_LOG.md）
5. PR 描述：动机 → 改动 → 测试/验证 → 对 schema/文档的影响
6. 自己 review 一遍 diff 再请求 review
```

## 5. Issue 模板（建议启用）

**Bug 报告**：场景（offline/online）→ 复现命令 → 期望 vs 实际 →
事件文件路径（run_id）→ vLLM/torch 版本。

**Feature 请求**：动机（研究/排障场景）→ 期望的事件/数据 → 开销估算 →
验收标准。

## 6. 发布流程（里程碑式）

1. 更新 CHANGELOG.md（Unreleased → 版本号）
2. 更新 README Roadmap 勾选状态
3. 全量测试 + 真机冒烟（scripts/）
4. 打 tag：`git tag v0.x.y && git push --tags`
5. 发布说明：列出新事件/新配置/破坏性变更（如有）

## 7. 行为准则

简短版：对事不对人；技术讨论基于证据（数据/源码引用）；
本项目面向学习与研究，欢迎任何水平的贡献者提问。
