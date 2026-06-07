"""兼容入口：训练型专家缓存 runner 已迁移到 `quitobench_framework_expert_cache`。

历史日志和命令中仍可能调用本文件，因此保留转发入口。
"""

from __future__ import annotations

from tools.quitobench_framework_expert_cache import *  # noqa: F403
from tools.quitobench_framework_expert_cache import main


if __name__ == "__main__":
    main()
