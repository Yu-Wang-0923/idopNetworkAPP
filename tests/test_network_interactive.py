"""新版交互式网络图（``idopnetwork.network.interactive``）的回归测试。

覆盖三件容易回归的事：
1. 邻接矩阵 → 边表的桥接约定（与静态 plot_network 一致，剔除自环）；
2. ``modules`` 显式传参优先于模块级全局 ``MODULES``（并发会话隔离）；
3. 端到端能写出非空 HTML。
"""

import pandas as pd

from idopnetwork.network import interactive


def _adj() -> pd.DataFrame:
    """adj.loc[source, target] = source -> target。"""
    return pd.DataFrame(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 3.0],
            [4.0, 0.0, 0.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


def test_adjacency_to_edges_keeps_only_real_edges():
    edges = interactive.network_edges_from_adjacency(_adj())
    got = {(row.source, row.target, row.weight) for row in edges.itertuples()}
    assert got == {
        ("A", "B", 1.0),
        ("B", "A", 2.0),
        ("B", "C", 3.0),
        ("C", "A", 4.0),
    }


def test_adjacency_target_node_keeps_incoming_edges_only():
    edges = interactive.network_edges_from_adjacency(_adj(), target_node="B")
    assert list(edges["source"]) == ["A"]
    assert list(edges["target"]) == ["B"]


def test_adjacency_top_edges_ranks_by_absolute_weight():
    edges = interactive.network_edges_from_adjacency(_adj(), top_edges=2)
    assert list(edges["weight"]) == [4.0, 3.0]


def test_adjacency_unknown_target_returns_empty_frame():
    edges = interactive.network_edges_from_adjacency(_adj(), target_node="ZZZ")
    assert edges.empty
    assert list(edges.columns) == ["source", "target", "weight"]


def test_modules_argument_wins_over_module_level_global():
    """并发会话隔离：显式 modules 必须优先，未传参时才回退全局。"""
    edges = interactive.network_edges_from_adjacency(_adj())
    original = interactive.MODULES
    try:
        # 故意把全局污染成无关值；改造前 make_degree 只读全局会输出 POISON
        interactive.MODULES = ["POISON"]

        degree = interactive.make_degree(edges, modules=["A", "B", "C"])
        # 隔离性：用的是传入的集合，POISON 不得出现
        assert set(degree["module"]) == {"A", "B", "C"}
        # 同时固化排序行为：make_degree 按 (out_degree, total_degree,
        # weighted_out_abs) 降序——B 出度 2 居首；A/C 出度同为 1，A 总度 3 > C 总度 2
        assert list(degree["module"]) == ["B", "A", "C"]

        layout = interactive._build_ring_layout(
            degree, "grp", modules=["A", "B", "C"],
        )
        # 注意：_build_ring_layout 会按环形分配重排节点，这里只断言"用的是传入
        # 的集合"，不约束顺序（顺序由 hub/ring 布局决定）。
        assert set(layout["module"]) == {"A", "B", "C"}

        # 不传参时保持独立脚本的旧行为
        assert list(interactive.make_degree(edges)["module"]) == ["POISON"]
    finally:
        interactive.MODULES = original


def test_plot_network_writes_non_empty_html(tmp_path):
    out = tmp_path / "network.html"
    result = interactive.plot_network(
        interactive.network_edges_from_adjacency(_adj()),
        str(out),
        group_name="demo",
        group_dir=str(tmp_path),
        modules=["A", "B", "C"],
    )
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert "<html" in out.read_text(encoding="utf-8").lower()
    # 环形布局与显示边表应一并落盘
    assert (tmp_path / "13A_ring_layout.csv").exists()
    assert (tmp_path / "13B_display_edges.csv").exists()
