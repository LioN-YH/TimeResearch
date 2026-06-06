# Stage 0.6：QuitoBench 官方 TSF cluster 语义画像

## 1. 实验目的

解释 QuitoBench 官方 `official_cluster_code` 在 Stage 0.1 item 级全长 STL 精确质量指标和 Stage 0 light proxy 指标下的经验 TSF 含义。

本实验不重新构造主标签，不实现 router，不执行通道级全长 STL。官方 cluster 仍是路线 2 的主 TSF cell 标签；本实验只为论文/报告提供可解释的经验命名和置信度。

## 2. 实验计划

1. 读取 Stage 0.5 输出 `quitobench_tsf_cells_final.csv`。
2. 按 `official_cluster_code` 汇总 STL/proxy 的 forecastability、seasonality、trend 分布。
3. 统计每个官方 cluster 的 STL cell 众数、proxy cell 众数和众数占比。
4. 设计保守经验命名规则：
   - STL/proxy 众数一致且占比均不低于 0.60：high confidence。
   - STL/proxy 众数一致但占比不足，或两者有 2/3 个 TSF 维度一致：medium confidence。
   - STL/proxy 冲突明显：low confidence。
   - 冲突时以 Stage 0.1 全长 STL 众数为建议经验名，但降低置信度。
5. 输出 cluster 语义汇总、item 诊断表和 Markdown 报告。
6. 运行单元测试和输出完整性检查。

## 3. 执行命令

```bash
conda run -n quito python -m pytest tests/test_quitobench_official_cluster_semantics.py -q
conda run -n quito python tools/quitobench_official_cluster_semantics.py
wc -l outputs/data_audit/quitobench_official_cluster_semantics.csv outputs/data_audit/quitobench_official_cluster_item_diagnostics.csv outputs/data_audit/quitobench_official_cluster_semantics_report.md
python3 - <<'PY'
import pandas as pd
s = pd.read_csv('outputs/data_audit/quitobench_official_cluster_semantics.csv')
d = pd.read_csv('outputs/data_audit/quitobench_official_cluster_item_diagnostics.csv')
print(s.shape)
print(d.shape)
print(s[['official_cluster_code','item_count','suggested_semantic_name','confidence']])
print(d['stl_proxy_agree'].mean())
print(d['cluster_semantic_match'].mean())
PY
```

## 4. 输入数据与配置

输入：

- `outputs/data_audit/quitobench_tsf_cells_final.csv`
- `outputs/data_audit/quitobench_item_quality_stl.csv`
- `outputs/data_audit/quitobench_item_quality.csv`

脚本：

- `tools/quitobench_official_cluster_semantics.py`

测试：

- `tests/test_quitobench_official_cluster_semantics.py`

语义命名原则：

- `suggested_semantic_name` 是经验解释，不是官方 codebook。
- 不把 `official_cluster_code` 强行解释为 high/low bit mask。
- Stage 0.1 STL 精确指标优先于 Stage 0 light proxy。

## 5. 实验结果

生成输出：

- `outputs/data_audit/quitobench_official_cluster_semantics.csv`
- `outputs/data_audit/quitobench_official_cluster_semantics_report.md`
- `outputs/data_audit/quitobench_official_cluster_item_diagnostics.csv`

输出行数：

- `quitobench_official_cluster_semantics.csv`：9 行，含表头；8 个官方 cluster。
- `quitobench_official_cluster_item_diagnostics.csv`：1,291 行，含表头；1,290 个 item。
- `quitobench_official_cluster_semantics_report.md`：65 行。

单元测试：

- `tests/test_quitobench_official_cluster_semantics.py`：`4 passed`。

Cluster 语义画像：

| official_cluster_code | item_count | STL 众数 cell | STL 众数占比 | proxy 众数 cell | proxy 众数占比 | 建议经验名 | 置信度 |
| ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| 0 | 166 | highT_highS_highF | 73.49% | highT_highS_highF | 28.92% | highT_highS_highF | medium |
| 2 | 136 | highT_lowS_lowF | 54.41% | highT_lowS_lowF | 34.56% | highT_lowS_lowF | medium |
| 6 | 170 | highT_highS_highF | 61.18% | lowT_highS_highF | 68.82% | highT_highS_highF | medium |
| 8 | 157 | highT_lowS_lowF | 32.48% | highT_highS_highF | 23.57% | highT_lowS_lowF | low |
| 18 | 159 | lowT_lowS_lowF | 42.77% | lowT_lowS_lowF | 46.54% | lowT_lowS_lowF | medium |
| 20 | 166 | lowT_lowS_lowF | 60.24% | lowT_lowS_lowF | 45.18% | lowT_lowS_lowF | medium |
| 24 | 169 | highT_highS_highF | 63.31% | highT_highS_highF | 71.01% | highT_highS_highF | high |
| 26 | 167 | lowT_lowS_lowF | 71.86% | lowT_lowS_lowF | 58.68% | lowT_lowS_lowF | medium |

诊断摘要：

- item 级 STL/proxy cell 完全一致率：45.04%。
- item 的 STL cell 与所属 cluster 建议经验名一致率：57.83%。
- cluster 置信度分布：`medium=6`，`low=1`，`high=1`。

## 6. 问题与观察

- 官方 cluster 与 STL/proxy 自建 cell 不是一一对应关系；尤其 cluster 8 的 STL/proxy 众数占比都较低，因此只能给 low confidence 经验命名。
- cluster 24 的 STL/proxy 解释最一致，可作为 high confidence 示例。
- proxy cell 众数占比普遍不高，进一步证明 Stage 0 light proxy 只适合规模审计和辅助解释，不适合作为最终标签。
- 本阶段未做通道级全长 STL，因此不能解释同一个 item 内 5 个 channel 是否有不同 TSF 模式。
- 用户复核后指出一个关键问题：QuitoBench 论文/README 声明覆盖 `2 x 2 x 2 = 8` 类 TSF regime，但本阶段用 STL/proxy 众数得到的 `suggested_semantic_name` 并没有覆盖 8 种模式，而是把多个官方 cluster 压到了相同经验 cell：
  - `0/6/24` 都被解释为 `highT_highS_highF`。
  - `2/8` 都被解释为 `highT_lowS_lowF`。
  - `18/20/26` 都被解释为 `lowT_lowS_lowF`。
- 因此，Stage 0.6 当前结果只能作为“粗略经验画像”，不能作为最终官方 codebook，也不能用于声称官方 8 个 cluster 的具体 high/low TSF 语义。
- 进一步观察到官方 cluster code 并不像随机编号：

```text
0  -> base-3 000
2  -> base-3 002
6  -> base-3 020
8  -> base-3 022
18 -> base-3 200
20 -> base-3 202
24 -> base-3 220
26 -> base-3 222
```

- 这 8 个 code 正好是三位三进制编码中的 8 个角点，强烈暗示官方 cluster 可能来自 trend / seasonality / forecastability 三个维度的离散编码，并且只保留 low/high 两端，排除了中间 bin。
- 当前 Stage 0.6 没有利用这个编码结构，这是造成解释坍缩的主要方法缺陷之一。

## 7. 结论

Stage 0.6 已完成官方 TSF cluster 的经验语义画像。后续路线 2 可以继续使用官方 cluster 作为主标签，并在报告中使用本阶段的 `suggested_semantic_name` 和 `confidence` 做解释性描述。

但本阶段结论需要降级为 preliminary：

> `suggested_semantic_name` 只能作为粗略经验描述，不能作为官方 cluster code 到 8 个 TSF cell 的最终映射。

推荐表述：

```text
official_cluster_24 在 item 级 STL/proxy 审计下经验上接近 highT_highS_highF，置信度 high。
```

不推荐表述：

```text
官方定义 cluster 24 为 highT_highS_highF。
```

## 8. 下一步计划

1. 下个会话优先做 Stage 0.6b：官方 cluster codebook 反推与验证。
2. Stage 0.6b 建议目标：
   - 重读 QuitoBench 论文/README 中 TSF regime 构造方法，确认 trend、seasonality、forecastability 的 low/high 划分规则。
   - 利用官方 code 的 base-3 结构，推断三位 digit 分别对应哪个 TSF 维度，以及 `0/2` 分别代表 low 还是 high。
   - 枚举所有可能映射，并用已有 STL/proxy 指标评估哪种映射最一致。
   - 如果论文或代码能给出明确 codebook，则以官方 codebook 为准；如果没有，则输出候选映射和置信度，不强行定论。
3. Stage 0.6b 建议输出：
   - `outputs/data_audit/quitobench_official_cluster_codebook_candidates.csv`
   - `outputs/data_audit/quitobench_official_cluster_codebook_report.md`
4. Quito CLI evaluate 可作为辅助线索：
   - 其 cluster 分组评估可能帮助观察不同官方 cluster 上专家表现差异。
   - 但模型性能只能间接解释 TSF 语义，不能替代 codebook 或质量指标。
5. 通道级全长 STL 可作为 Stage 0.7 或 Stage 0.6b 的长实验分支：
   - 可以复用 `quito.utils.dataset_quality.evaluate_series`。
   - 每个 `(item_id, ind_k)` 是一条 1D single series。
   - 预计耗时 1.5-2 小时，必须保留分批进度和中间 CSV。
   - 优先用途是验证官方 codebook 或解释 channel 内异质性，不应一开始就替代官方 cluster。
6. 当前仍不要实现 router。
