# Stage 0.6b：QuitoBench 官方 cluster codebook 反推与验证

## 1. 实验目的

反推并验证 QuitoBench 官方 `cluster` code 到 `trend x seasonality x forecastability` 八个 TSF regime cell 的映射。

本实验只做 codebook 解释，不重新构造主标签，不实现 router，不执行通道级全长 STL。

## 2. 实验计划

1. 重读旧 Hugging Face README、QuitoBench 论文和本地 Quito 代码中关于 TSF regime / cluster / quality metrics 的说明。
2. 利用官方 cluster code 的三位三进制结构枚举所有候选映射：
   - 3 位 digit 到 trend / seasonality / forecastability 的 6 种排列。
   - 每个维度中 digit `0` 或 `2` 表示 high 的 8 种组合。
   - 合计 48 种候选。
3. 用论文 Table 23 的 regime 顺序和 evaluation instances 计数验证候选映射。
4. 用 Stage 0.1 STL 和 Stage 0 light proxy 指标计算候选映射的一致率和方向评分。
5. 输出候选 CSV、官方 codebook CSV 和中文报告。
6. 运行测试和完整性检查。

## 3. 执行命令

```bash
conda run -n quito python -m pytest tests/test_quitobench_official_cluster_codebook.py -q
conda run -n quito python tools/quitobench_official_cluster_codebook.py
python3 - <<'PY'
import pandas as pd
for p in [
    'outputs/data_audit/quitobench_official_cluster_codebook_candidates.csv',
    'outputs/data_audit/quitobench_official_cluster_codebook.csv',
]:
    df = pd.read_csv(p)
    print(p, df.shape)
    print(df.head(8).to_string(index=False))
PY
wc -l outputs/data_audit/quitobench_official_cluster_codebook_candidates.csv \
      outputs/data_audit/quitobench_official_cluster_codebook.csv \
      outputs/data_audit/quitobench_official_cluster_codebook_report.md
```

TDD 记录：

- 初次运行新测试失败，原因是 `tools.quitobench_official_cluster_codebook` 尚不存在。
- 实现脚本后，测试暴露最小 DataFrame 缺少 STL/proxy 指标列的边界问题，已改为缺指标时方向评分记 0 并保留备注。
- 测试样本随后改为按论文推断 item count 构造，以覆盖 Table 23 计数匹配逻辑。

## 4. 输入数据与配置

输入：

- `outputs/data_audit/quitobench_tsf_cells_final.csv`
- `data/hf/hq-bench/quitobench/revisions/17362dcb/README.md`
- QuitoBench 论文：<https://arxiv.org/abs/2603.26017>

新增脚本：

- `tools/quitobench_official_cluster_codebook.py`

新增测试：

- `tests/test_quitobench_official_cluster_codebook.py`

关键论文证据：

- TSF diagnostic 使用默认阈值 `tau=0.4`；三项指标 `> tau` 为 high，`<= tau` 为 low。
- Appendix H / Table 23 按 `high_high_high, high_high_low, high_low_high, high_low_low, low_high_high, low_high_low, low_low_high, low_low_low` 顺序列出 regime statistics。
- Table 23 的 evaluation instances 除以 `10 models x 18 task configurations = 180` 后得到 item 数：
  `166, 136, 170, 157, 159, 166, 169, 167`。

完整计算过程：

```text
1,290 test series x 10 models x 18 task configurations = 232,200 evaluation instances
232,200 / 1,290 = 180 evaluation instances per item
```

| 顺序 | paper regime | Table 23 eval count | /180 item count | cluster code 升序 | 本地官方 cluster item count | 是否匹配 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | highT_highS_highF | 29,880 | 166 | 0 | 166 | 是 |
| 2 | highT_highS_lowF | 24,480 | 136 | 2 | 136 | 是 |
| 3 | highT_lowS_highF | 30,600 | 170 | 6 | 170 | 是 |
| 4 | highT_lowS_lowF | 28,260 | 157 | 8 | 157 | 是 |
| 5 | lowT_highS_highF | 28,620 | 159 | 18 | 159 | 是 |
| 6 | lowT_highS_lowF | 29,880 | 166 | 20 | 166 | 是 |
| 7 | lowT_lowS_highF | 30,420 | 169 | 24 | 169 | 是 |
| 8 | lowT_lowS_lowF | 30,060 | 167 | 26 | 167 | 是 |

## 5. 实验结果

生成输出：

- `outputs/data_audit/quitobench_official_cluster_codebook_candidates.csv`
- `outputs/data_audit/quitobench_official_cluster_codebook.csv`
- `outputs/data_audit/quitobench_official_cluster_codebook_report.md`

输出行数：

- candidates CSV：49 行，含表头；48 个候选。
- codebook CSV：9 行，含表头；8 个官方 cluster code。
- report：65 行。

测试结果：

- `tests/test_quitobench_official_cluster_codebook.py`：`3 passed`。

最可信 codebook：

| official_cluster_code | base3 | official TSF cell |
| ---: | --- | --- |
| 0 | 000 | highT_highS_highF |
| 2 | 002 | highT_highS_lowF |
| 6 | 020 | highT_lowS_highF |
| 8 | 022 | highT_lowS_lowF |
| 18 | 200 | lowT_highS_highF |
| 20 | 202 | lowT_highS_lowF |
| 24 | 220 | lowT_lowS_highF |
| 26 | 222 | lowT_lowS_lowF |

解释：

- 三位三进制 digit 依次对应 `trend, seasonality, forecastability`。
- digit `0` 表示 high。
- digit `2` 表示 low。

候选枚举结果：

- 共 48 种候选。
- 只有 1 种候选同时满足论文 regime 顺序和 Table 23 item count 序列。
- 该候选的 Stage 0.1 STL item exact match 为 27.21%。
- 该候选的 Stage 0 proxy item exact match 为 17.83%。

## 6. 问题与观察

- Stage 0.1 STL / Stage 0 proxy 的 exact match 不高，说明当前本地 item 级中位数二分口径不能复现官方 TSF regime 构造。
- 这解释了 Stage 0.6 的经验命名坍缩：STL/proxy 众数只能作为辅助画像，不能作为官方 codebook。
- 官方证据链更强：
  1. 旧 README 明确 `cluster` 是 8-class TSF regime integer code。
  2. cluster code 是三位三进制角点。
  3. 论文 Table 23 的 regime 顺序和 item count 序列与官方 cluster code 升序完全匹配。
- 本阶段未找到官方代码中直接写死 `cluster -> high/low TSF cell` 的 codebook；当前结论是基于论文表格和官方 cluster 列的反推。

## 7. 结论

Stage 0.6b 已完成官方 cluster codebook 反推与验证。

可以在后续路线 2 中使用 high confidence codebook：

```text
0  -> highT_highS_highF
2  -> highT_highS_lowF
6  -> highT_lowS_highF
8  -> highT_lowS_lowF
18 -> lowT_highS_highF
20 -> lowT_highS_lowF
24 -> lowT_lowS_highF
26 -> lowT_lowS_lowF
```

Stage 0.6 的 `suggested_semantic_name` 保持 preliminary，仅用于经验画像；不要再把它作为官方 codebook。

## 8. 下一步计划

1. 新会话优先做 Stage 0.7：通道级全长 STL 官方 codebook 验证。
2. Stage 0.7 建议严格复现论文 multivariate TSF 口径：
   - 对每个 `(subset, item_id, ind_k)` 跑 full-length `quito.utils.dataset_quality.evaluate_series`。
   - 保留通道级中间结果，便于后续通道独立伪图像设计复用。
   - 对每个 `(subset, item_id)` 的 5 个 channel 的 `trend_strength`、`seasonality_strength`、`forecastability` 分别求均值。
   - 使用论文固定阈值 `0.4` 二值化，得到 `paper_like_tsf_cell`。
   - 与 Stage 0.6b 官方 codebook 映射出的 cell 比较 item exact match、逐维 match、cluster-level confusion matrix。
3. Stage 0.7 建议输出：

```text
outputs/data_audit/quitobench_channel_quality_stl_full.csv
outputs/data_audit/quitobench_item_quality_stl_channel_mean.csv
outputs/data_audit/quitobench_official_codebook_channel_stl_validation.csv
outputs/data_audit/quitobench_official_codebook_channel_stl_validation_report.md
```

4. Stage 0.7 必须作为长实验单独记录日志，带分批进度和中间 CSV；预计 1.5-2 小时。
5. Stage 0.7 不替代官方 codebook，只用于验证论文口径能否复现官方 cluster 语义，并解释 Stage 0.6 的 item 代表序列口径偏差。
6. 后续路线 2A 使用 `quitobench_official_cluster_codebook.csv` 给官方 cluster 加上可解释 cell 名。
7. 路线 1 仍不使用 TSF 标签训练 router，只按 cell 报告；当前仍不要实现 router。
