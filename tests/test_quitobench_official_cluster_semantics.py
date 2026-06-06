"""Stage 0.6 官方 cluster 语义画像规则测试。"""

from __future__ import annotations

import pandas as pd

from tools.quitobench_official_cluster_semantics import (
    choose_semantic_name,
    mode_with_ratio,
    semantic_agreement_count,
)


def test_mode_with_ratio_returns_mode_ratio_and_count() -> None:
    mode, ratio, count = mode_with_ratio(pd.Series(["a", "b", "a", "a"]))

    assert mode == "a"
    assert ratio == 0.75
    assert count == 3


def test_semantic_agreement_count_compares_tsf_dimensions() -> None:
    assert semantic_agreement_count("highT_lowS_highF", "highT_highS_highF") == 2
    assert semantic_agreement_count("highT_lowS_highF", "lowT_highS_lowF") == 0


def test_choose_semantic_name_uses_high_confidence_when_stl_and_proxy_agree_strongly() -> None:
    name, confidence, note = choose_semantic_name(
        stl_mode="highT_highS_highF",
        stl_ratio=0.72,
        proxy_mode="highT_highS_highF",
        proxy_ratio=0.68,
    )

    assert name == "highT_highS_highF"
    assert confidence == "high"
    assert "一致" in note


def test_choose_semantic_name_keeps_stl_primary_when_proxy_conflicts() -> None:
    name, confidence, note = choose_semantic_name(
        stl_mode="highT_lowS_highF",
        stl_ratio=0.55,
        proxy_mode="lowT_highS_lowF",
        proxy_ratio=0.61,
    )

    assert name == "highT_lowS_highF"
    assert confidence == "low"
    assert "冲突" in note
