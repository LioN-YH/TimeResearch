# Superpowers 阶段规划目录

本目录用于保存阶段级细化规划，避免把临时设计散落在项目主文档中。

## 目录约定

- `specs/`：阶段设计文档。用于明确范围、设计决策、接口、输入输出、风险和完成标准。
- `plans/`：实现计划。用于在设计确认后拆分可执行步骤、测试顺序和验证命令。

## 命名约定

```text
docs/superpowers/specs/YYYY-MM-DD-<stage>-<topic>-design.md
docs/superpowers/plans/YYYY-MM-DD-<stage>-<topic>.md
```

## 使用规则

- 如果需要细化某个 Stage 的方案，优先写入 `docs/superpowers/specs/`。
- 如果设计已经确认并准备实现，再写入 `docs/superpowers/plans/`。
- 项目 `Doc/` 目录保留主计划、交接文档和面向项目阅读的索引；必要时可从 `Doc/` 链接到本目录中的细化设计。
- 实验执行结果仍写入 `experiment_logs/`，并同步更新 `experiment_logs/实验日志总览.md`。
