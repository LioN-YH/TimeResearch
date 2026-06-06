# QuitoBench HF 数据源与 schema 确认

## 1. 实验目的

确认 QuitoBench benchmark 的官方 Hugging Face 数据集 ID、config、split、文件版本、schema 和本地落盘路径，避免误用 Quito 预训练 corpus。

## 2. 实验计划

1. 查询 `hq-bench` 组织下的 collection 和 dataset。
2. 确认 `hq-bench/quitobench` 的 config/split。
3. 下载 `hour` / `min` benchmark parquet 到工作区。
4. 读取 parquet 元数据、schema 和少量样本。
5. 阅读下载后的数据卡，确认 cut-off、行数、序列数和字段含义。

## 3. 执行命令

```bash
conda run -n quito python -c "from huggingface_hub import HfApi; api=HfApi(); print('collections'); [print(c.slug, c.title) for c in api.list_collections(owner='hq-bench')]; print('datasets'); [print(ds.id) for ds in api.list_datasets(author='hq-bench')]"
conda run -n quito python -c $'from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder\nfor cid in ["hq-bench/quitobench"]:\n    print("DATASET", cid)\n    cfgs=get_dataset_config_names(cid)\n    print("configs", cfgs)\n    for cfg in cfgs:\n        print("config", cfg, "splits", get_dataset_split_names(cid, cfg))'
mkdir -p data/hf/hq-bench/quitobench
conda run -n quito python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='hq-bench/quitobench', repo_type='dataset', filename='README.md', local_dir='data/hf/hq-bench/quitobench'))"
conda run -n quito python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='hq-bench/quitobench', repo_type='dataset', filename='v20260315/test_hour-00001-of-00001.parquet', local_dir='data/hf/hq-bench/quitobench'))"
conda run -n quito python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='hq-bench/quitobench', repo_type='dataset', filename='v20260315/test_min-00001-of-00001.parquet', local_dir='data/hf/hq-bench/quitobench'))"
ls -lh data/hf/hq-bench/quitobench/v20260315
conda run -n quito python -c $'import pyarrow.parquet as pq\nfrom pathlib import Path\nfor name in ["hour","min"]:\n    path=Path("data/hf/hq-bench/quitobench/v20260315")/f"test_{name}-00001-of-00001.parquet"\n    pf=pq.ParquetFile(path)\n    print("===", name, "===")\n    print("rows", pf.metadata.num_rows, "row_groups", pf.metadata.num_row_groups)\n    print(pf.schema_arrow)\n    tbl=pf.read_row_group(0).slice(0,3)\n    print(tbl.to_pandas().to_string())'
sed -n '1,220p' data/hf/hq-bench/quitobench/README.md
```

## 4. 输入数据与配置

- Hugging Face benchmark dataset：`hq-bench/quitobench`
- 明确排除预训练 corpus：`hq-bench/quito-corpus`
- 本地下载目录：`data/hf/hq-bench/quitobench/`
- 数据版本目录：`v20260315/`

## 5. 实验结果

- 官方 collection：`hq-bench/quitobench-69c5f776b0782d315e2bad41`
- 官方 benchmark dataset：`hq-bench/quitobench`
- 同组织下存在预训练 corpus：`hq-bench/quito-corpus`，本阶段未使用。
- `hq-bench/quitobench` config：
  - `hour`，split：`test`
  - `min`，split：`test`
- 下载文件：
  - `data/hf/hq-bench/quitobench/README.md`
  - `data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
  - `data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`
- 文件大小：
  - `hour` parquet：约 133 MB
  - `min` parquet：约 63 MB
- parquet 元数据：
  - `hour`：7,939,052 rows，8 row groups
  - `min`：4,563,792 rows，5 row groups
- schema：
  - `item_id: int64`
  - `date_time: timestamp[ns]`
  - `ind_1` 到 `ind_5`: `double`
- README 关键说明：
  - `hour`：517 条测试序列，每条 15,356 步，测试段每条 552 步。
  - `min`：773 条测试序列，每条 5,904 步，测试段每条 3,312 步。
  - 全局 train/test cutoff：`2023-07-28 00:00:00 UTC`。
  - 1,290 条测试序列在 8 个 trend × seasonality × forecastability cell 上分层均衡，约 160 条序列/cell。

## 6. 问题与观察

- Hugging Face `datasets` builder 返回 features 为 `None`，但直接读取 parquet 元数据可以确认 schema。
- 数据集公开 config 只有 `test` split；README 说明 long series 中仍可按 `2023-07-28 00:00:00` cutoff 重建 train/test，并在 cutoff 之前再划分 train/valid。
- 当前 parquet 中没有显式 TSF cell 标签列，因此后续审计需要根据 trend、seasonality、forecastability 重新构造 cell，或在报告中把 README 的官方均衡描述作为来源说明。

## 7. 结论

QuitoBench benchmark 数据源、版本、schema 和本地路径已确认。当前可继续对 `hour` / `min` 两个 benchmark parquet 执行数据充分性统计。

## 8. 下一步计划

1. 编写并运行 QuitoBench 数据充分性审计脚本。
2. 同时输出 item 级和通道级统计，因为默认伪图像策略为通道独立。
3. 生成：
   - `outputs/data_audit/quitobench_sufficiency_report.md`
   - `outputs/data_audit/quitobench_cell_distribution.csv`
   - `outputs/data_audit/quitobench_window_counts.csv`
