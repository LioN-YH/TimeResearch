# Stage 0.5：QuitoBench 官方 TSF regime/cell 标签来源审计

## 1. 实验目的

定位 QuitoBench benchmark 是否存在官方 TSF regime/cell 标签；如果找不到，则基于 Stage 0.1 STL 精确指标和 Stage 0 light proxy 指标固化最终 cell 构造规则。

本实验只做标签来源审计和最终 cell 文件整理，不做伪图像、不做 router、不使用 `hq-bench/quito-corpus` 预训练 corpus。

## 2. 实验计划

1. 检查当前本地 QuitoBench parquet schema。
2. 搜索本地 HF README、Quito 官方代码和示例中与 `cluster/regime/cell/TSF` 相关的信息。
3. 使用 Hugging Face Hub API 检查当前文件列表和提交历史。
4. 如果提交历史显示曾移除 `cluster` 列，则下载移除前 revision，确认 schema 和标签含义。
5. 抽取官方 `item_id -> cluster` 映射，验证每个 item 的 cluster 是否唯一、是否与当前 item_id 对齐。
6. 生成最终 cell CSV，并附加 Stage 0.1 STL 与 Stage 0 proxy 辅助解释字段。
7. 写出来源报告、实验日志并更新总览。

## 3. 执行命令

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
for subset in ['hour','min']:
    p = Path('data/hf/hq-bench/quitobench/v20260315') / f'test_{subset}-00001-of-00001.parquet'
    df = pd.read_parquet(p)
    print(subset, df.shape, list(df.columns))
PY

conda run -n quito python -c "from huggingface_hub import HfApi; api=HfApi(); repo='hq-bench/quitobench'; print(api.list_repo_files(repo, repo_type='dataset')); [print(c.commit_id[:8], c.created_at, c.title) for c in api.list_repo_commits(repo, repo_type='dataset')[:20]]"

conda run -n quito python -c "from huggingface_hub import hf_hub_download; from pathlib import Path; repo='hq-bench/quitobench'; rev='17362dcb'; out=Path('data/hf/hq-bench/quitobench/revisions/17362dcb'); out.mkdir(parents=True, exist_ok=True); [print(hf_hub_download(repo_id=repo, repo_type='dataset', revision=rev, filename=fn, local_dir=out, local_dir_use_symlinks=False)) for fn in ['v20260315/test_hour-00001-of-00001.parquet','v20260315/test_min-00001-of-00001.parquet','README.md']]"
```

另执行 Python 脚本片段抽取 `cluster` 标签并生成：

```text
outputs/data_audit/quitobench_tsf_cells_final.csv
outputs/data_audit/quitobench_official_tsf_cluster_summary.csv
outputs/data_audit/quitobench_tsf_label_source_report.md
```

## 4. 输入数据与配置

- 数据集：`hq-bench/quitobench` benchmark。
- 当前 parquet：
  - `data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
  - `data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`
- 官方标签来源：
  - HF dataset revision：`17362dcb`
  - 旧 parquet：
    - `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_hour-00001-of-00001.parquet`
    - `data/hf/hq-bench/quitobench/revisions/17362dcb/v20260315/test_min-00001-of-00001.parquet`
- 旧 README schema 证据：
  - `cluster | int64 | TSF regime label (8-class integer code)`
- 辅助指标：
  - Stage 0.1 STL：`outputs/data_audit/quitobench_item_quality_stl.csv`
  - Stage 0 light proxy：`outputs/data_audit/quitobench_item_quality.csv`

## 5. 实验结果

当前公开文件列表只有：

```text
.gitattributes
README.md
v20260315/test_hour-00001-of-00001.parquet
v20260315/test_min-00001-of-00001.parquet
```

当前 parquet schema：

- `hour`: `ind_1, ind_2, ind_3, ind_4, ind_5, date_time, item_id`
- `min`: `date_time, ind_1, ind_2, ind_3, ind_4, ind_5, item_id`

Hub 提交历史关键记录：

| commit | 时间 UTC | 标题 |
| --- | --- | --- |
| `ed4bf8ee` | 2026-03-30 06:20:47 | Remove cluster column from parquet files |
| `bbfa8c4d` | 2026-03-30 02:55:40 | Fix project page URL, remove cluster column from schema |
| `17362dcb` | 2026-03-30 02:45:19 | Update README: add arXiv link, cross-link to quito-corpus, update citation |

旧 revision `17362dcb` 的 README schema 明确包含：

```text
cluster | int64 | TSF regime label (8-class integer code)
```

官方 cluster 抽取结果：

| official_cluster_code | item 数 | hour item | min item |
| ---: | ---: | ---: | ---: |
| 0 | 166 | 166 | 0 |
| 2 | 136 | 129 | 7 |
| 6 | 170 | 0 | 170 |
| 8 | 157 | 4 | 153 |
| 18 | 159 | 85 | 74 |
| 20 | 166 | 133 | 33 |
| 24 | 169 | 0 | 169 |
| 26 | 167 | 0 | 167 |

生成输出：

- `outputs/data_audit/quitobench_tsf_label_source_report.md`
- `outputs/data_audit/quitobench_tsf_cells_final.csv`
- `outputs/data_audit/quitobench_official_tsf_cluster_summary.csv`

最终 cell 文件：

- shape：`(1290, 20)`。
- `official_cluster_code` 共 8 类。
- 每个 `(subset, item_id)` 仅一条标签。
- 当前 item_id 与旧 revision 标签 item_id 完全对齐。
- `final_cell_id` 使用按官方 code 升序映射的连续 0-7。
- `final_cell_name` 使用 `official_cluster_<code>`。

## 6. 问题与观察

- 当前最新版 parquet 已移除 `cluster` 列，因此直接加载当前 HF dataset 不会得到官方标签。
- 旧 revision 中的 `cluster` 是官方 TSF regime label，但当前公开 README 没有给出 code 到 `high/low trend x seasonality x forecastability` 的语义映射。
- 官方 cluster code 不是连续 0-7，而是 `0, 2, 6, 8, 18, 20, 24, 26`。
- 后续模型训练应使用 `official_cluster_index` 作为连续分类标签，报告中保留 `official_cluster_code` 以追溯官方原始标签。
- Stage 0.1 STL 和 Stage 0 proxy 不再作为主标签，只作为解释官方 cluster 的辅助指标。

## 7. 结论

QuitoBench 官方 TSF regime/cell 标签存在，来源为 Hugging Face dataset revision `17362dcb` 中的 `cluster` 列。当前最新版 parquet 已移除该列，但 item_id 与旧 revision 完全一致，因此可以安全抽取旧 revision 的 item 级官方 cluster 映射，并作为路线 2 的主 TSF cell 标签。

最终规则：

- 主标签：`official_cluster_code` / `official_cluster_index`。
- 路线 2 训练和分层：使用 `final_cell_id` / `final_cell_name`。
- 语义解释：使用 `quitobench_official_tsf_cluster_summary.csv` 中的 STL/proxy 指标汇总。
- 不强行把官方 cluster code 解释成 high/low bit mask。

## 8. 下一步计划

1. 先补做 Stage 0.6：官方 TSF cluster 语义画像与命名建议。
   - 目标不是重造标签，而是解释官方 `cluster=0/2/6/8/18/20/24/26` 在 STL/proxy 指标下分别更接近哪类 TSF 结构。
   - 主标签仍使用官方 `official_cluster_code` / `official_cluster_index`。
   - 语义命名只作为经验解释，例如“经验上接近 highT_highS_highF”，不能写成“官方定义为 highT_highS_highF”。
2. Stage 0.6 建议输入：
   - `outputs/data_audit/quitobench_tsf_cells_final.csv`
   - `outputs/data_audit/quitobench_item_quality_stl.csv`
   - `outputs/data_audit/quitobench_item_quality.csv`
3. Stage 0.6 建议输出：
   - `outputs/data_audit/quitobench_official_cluster_semantics.csv`
   - `outputs/data_audit/quitobench_official_cluster_semantics_report.md`
   - `outputs/data_audit/quitobench_official_cluster_item_diagnostics.csv`
4. Stage 0.6 建议分析：
   - 按 `official_cluster_code` 汇总 STL/proxy 的 forecastability、seasonality、trend 分布。
   - 统计每个官方 cluster 的 STL cell 众数、proxy cell 众数和众数占比。
   - 给每个官方 cluster 生成 `suggested_semantic_name`、`confidence` 和解释备注。
   - 若 STL/proxy 对某个 cluster 的解释冲突明显，标记为 low confidence，而不是强行命名。
5. 暂不在 Stage 0.6 跑通道级全长 STL；如果官方 cluster 解释出现明显 channel 内异质性，再单独安排 Stage 0.7 通道级长实验。
6. 后续路线 2A 专家画像使用 `outputs/data_audit/quitobench_tsf_cells_final.csv` 作为 TSF cell 标签来源。
7. 路线 1 仍不使用 TSF 标签训练 router，只按 cell 报告。
8. 当前仍不要实现 router。
