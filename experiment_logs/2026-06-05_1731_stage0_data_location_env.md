# QuitoBench 数据定位与环境确认

## 1. 实验目的

确认当前工作区是否已有 QuitoBench benchmark 数据文件，并确认 `quito` conda 环境是否具备后续数据充分性审计所需的基础依赖。

## 2. 实验计划

1. 阅读交接文档和主计划，明确当前任务只做 QuitoBench 数据充分性审计，不实现 router。
2. 搜索工作区内已有 parquet、CSV、YAML、JSON 数据与配置文件。
3. 检查 Quito 数据加载和质量分析相关源码。
4. 使用 `conda run -n quito` 确认 Python 环境和关键依赖。

## 3. 执行命令

```bash
sed -n '1,240p' Doc/视觉伪图像路由项目交接.md
sed -n '1,260p' Doc/视觉伪图像路由双路线实施计划.md
sed -n '261,620p' Doc/视觉伪图像路由双路线实施计划.md
find . -maxdepth 4 \( -name '*.parquet' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) | sed 's#^./##' | head -300
rg -n "QuitoBench|benchmark|open|hour|min|item_id|quality|forecastability|seasonality|trend" quito Doc experiment_logs -g '*.md' -g '*.py' -g '*.yaml' -g '*.yml' | head -240
sed -n '1,260p' quito/README.md
sed -n '1,280p' quito/quito/utils/dataset_quality.py
sed -n '220,520p' quito/quito/datasets.py
conda run -n quito python -c "import sys; print(sys.executable); import pandas, pyarrow, scipy; print('pandas', pandas.__version__); print('pyarrow', pyarrow.__version__); print('scipy', scipy.__version__)"
conda run -n quito python -c "import importlib.util; mods=['datasets','huggingface_hub','statsmodels','arch']; print({m: importlib.util.find_spec(m) is not None for m in mods})"
```

## 4. 输入数据与配置

- 工作目录：`/home/user10/TSF/DATAPrepare`
- 交接文档：`Doc/视觉伪图像路由项目交接.md`
- 主计划：`Doc/视觉伪图像路由双路线实施计划.md`
- Quito 代码目录：`quito/`
- Conda 环境：`quito`

## 5. 实验结果

- 当前工作区没有发现 QuitoBench parquet 文件。
- `find` 仅发现：
  - `quito/configs/example_config.yaml`
  - `quito/examples/item_csv.csv`
- Quito README 指出数据需从 Hugging Face collection `hq-bench/quitobench` 下载，并放置到 `examples/datasets/cluster_data`。
- Quito 可复用模块确认：
  - `quito/quito/datasets.py` 中 `TimeSeriesDataset` 支持 parquet、`item_id`、train/valid/test 切分和滑动窗口计数。
  - `quito/quito/utils/dataset_quality.py` 提供 forecastability、seasonality strength、trend strength 等质量指标。
- `conda run -n quito` 环境确认：
  - Python：`/home/user10/miniconda3/envs/quito/bin/python`
  - pandas：`3.0.3`
  - pyarrow：`24.0.0`
  - scipy：`1.17.1`
  - `datasets`：可用
  - `huggingface_hub`：可用
  - `statsmodels`：可用
  - `arch`：不可用

## 6. 问题与观察

- 本地暂未落盘 QuitoBench benchmark 数据，因此无法直接从工作区 parquet 完成充分性统计。
- `arch` 不可用会影响 ADF 指标，但当前充分性审计核心需要的是有效长度、窗口数、forecastability、seasonality strength、trend strength 和 TSF cell 分布，暂不依赖 ADF。
- 必须继续区分 QuitoBench benchmark 与 Quito 预训练 corpus；当前只应定位和使用 benchmark 的 `hour` / `min` 子集。

## 7. 结论

当前环境具备执行 QuitoBench 数据充分性审计的基础 Python 依赖，但工作区尚未包含 QuitoBench benchmark 数据文件。下一步应通过官方 Hugging Face collection 或 datasets 接口确认 `hour` / `min` 数据集名称、schema 和可下载文件，并在确认数据来源后执行统计。

## 8. 下一步计划

1. 使用 Hugging Face 工具查询 `hq-bench/quitobench` collection 中的 QuitoBench benchmark 数据集条目。
2. 确认 `hour` / `min` 子集的 dataset id、split、字段和 row 数。
3. 若可访问，下载或流式读取 benchmark 子集，生成数据充分性审计输出：
   - `outputs/data_audit/quitobench_sufficiency_report.md`
   - `outputs/data_audit/quitobench_cell_distribution.csv`
   - `outputs/data_audit/quitobench_window_counts.csv`
