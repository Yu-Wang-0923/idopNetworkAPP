"""新版 GLMY / path homology 的回归测试。

覆盖四类容易回归的点：

1. 列名兼容（``from`` / ``to`` 与 ``source`` / ``target``）与自环剔除；
2. ``weight_offset`` 的自动推断与端点还原（新版约定，无穷用 ``None``）；
3. 绘图层向后兼容：新版 ``shared_bar_counts`` 签名、旧版 ``max_x`` 关键字、
   以及旧版 ``-1`` 无穷哨兵都能画；
4. 正 / 负权重拆分（4 行 × 2 列共 8 个子图）与单侧退化的情形。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from idopnetwork.analysis.glmy import (
    DIMENSION_KEYS,
    bar_counts,
    build_weighted_digraph,
    compute_glmy_homology,
    compute_glmy_homology_split,
    normalize_network,
)
from idopnetwork.analysis.plot_analysis import (
    plot_glmy_barcode,
    plot_glmy_barcode_split,
)
from idopnetwork.analysis.network_analysis import run_glmy, run_glmy_split


def _edge_table(columns=("from", "to", "weight")) -> pd.DataFrame:
    source, target, weight = columns
    return pd.DataFrame(
        [
            {source: "M0", target: "M1", weight: 1.5},
            {source: "M1", target: "M2", weight: -0.8},
            {source: "M2", target: "M0", weight: 0.4},
            {source: "M0", target: "M2", weight: -1.2},
            # 自环：必须被剔除
            {source: "M1", target: "M1", weight: 9.9},
        ]
    )


def test_normalize_accepts_both_column_conventions():
    from_style = normalize_network(_edge_table(("from", "to", "weight")))
    source_style = normalize_network(_edge_table(("source", "target", "weight")))
    assert list(from_style.columns) == ["source", "target", "weight"]
    assert from_style.equals(source_style)


def test_normalize_drops_self_loops():
    clean = normalize_network(_edge_table())
    assert not (clean["source"] == clean["target"]).any()
    assert len(clean) == 4


def test_normalize_accepts_effect_as_weight_column():
    frame = _edge_table(("From", "To", "Effect"))
    clean = normalize_network(frame)
    assert list(clean.columns) == ["source", "target", "weight"]


def test_normalize_rejects_missing_columns():
    with pytest.raises(ValueError):
        normalize_network(pd.DataFrame({"a": [1], "b": [2]}))


def test_auto_weight_offset_makes_weights_positive():
    _, weighted_edges, offset = build_weighted_digraph(_edge_table())
    assert offset == pytest.approx(1.0 - (-1.2))
    assert all(edge[2] > 0 for edge in weighted_edges)


def test_explicit_weight_offset_is_respected():
    _, _, offset = build_weighted_digraph(_edge_table(), 0.0)
    assert offset == 0.0


def test_compute_uses_none_for_infinite_bars():
    homology = compute_glmy_homology(_edge_table())
    assert sorted(homology) == sorted(DIMENSION_KEYS)
    endpoints = [v for bars in homology.values() for bar in bars for v in bar]
    assert None in endpoints
    # 新版不再产生 -1 哨兵
    assert -1 not in endpoints


def test_run_glmy_reports_resolved_offset():
    result = run_glmy(_edge_table())
    assert result["backend"] == "python"
    assert result["weight_offset"] == pytest.approx(1.0 - (-1.2))
    assert set(result["vertex_id_map"]) == {"M0", "M1", "M2"}


def test_bar_counts_covers_all_dimensions():
    counts = bar_counts(compute_glmy_homology(_edge_table()))
    assert set(counts) == set(DIMENSION_KEYS)


def test_plot_glmy_barcode_new_signature():
    homology = compute_glmy_homology(_edge_table())
    fig = plot_glmy_barcode(homology, bar_counts(homology))
    assert len(fig.axes) == len(DIMENSION_KEYS)
    plt.close(fig)


def test_plot_glmy_barcode_accepts_legacy_max_x():
    homology = compute_glmy_homology(_edge_table())
    fig = plot_glmy_barcode(homology, max_x=2.0)
    assert fig.axes[-1].get_xlim() == pytest.approx((-2.2, 2.2))
    plt.close(fig)


def test_plot_glmy_barcode_accepts_legacy_minus_one_sentinel():
    """M3 / Paper §3.2 自检路径产出的 homology 用 -1 表示无穷，也要能画。"""
    legacy = {
        key: [
            [
                -1 if birth is None else birth,
                -1 if death is None else death,
            ]
            for birth, death in bars
        ]
        for key, bars in compute_glmy_homology(_edge_table()).items()
    }
    fig = plot_glmy_barcode(legacy, max_x=1.5)
    assert len(fig.axes) == len(DIMENSION_KEYS)
    plt.close(fig)


def test_split_returns_both_signs():
    homologies = compute_glmy_homology_split(_edge_table())
    assert set(homologies) == {"positive", "negative"}
    assert set(homologies["positive"]) == set(DIMENSION_KEYS)


def test_split_plot_has_eight_panels():
    split = run_glmy_split(_edge_table())
    fig = plot_glmy_barcode_split(split["homologies"])
    assert len(fig.axes) == 2 * len(DIMENSION_KEYS)
    plt.close(fig)


def test_split_edge_counts_match_signs():
    split = run_glmy_split(_edge_table())
    assert split["edge_counts"] == {"positive": 2, "negative": 2}


def test_split_handles_single_sign_without_error():
    only_positive = _edge_table()
    only_positive = only_positive[only_positive["weight"] > 0]
    split = run_glmy_split(only_positive)
    assert split["edge_counts"] == {"positive": 2, "negative": 0}
    fig = plot_glmy_barcode_split(split["homologies"])
    assert len(fig.axes) == 2 * len(DIMENSION_KEYS)
    plt.close(fig)
