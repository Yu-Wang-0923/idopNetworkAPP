"""Pure Python GLMY/path homology helpers.

This module wraps :class:`backend.analysis.digraph.Digraph` so the Streamlit page can
compute barcode data without the legacy external executable.

Algorithm inputs match the historical ``GLMY.exe`` convention: vertices as
integers ``1..n`` in sorted order, edge weights ``weight + offset`` with
default ``offset=100``. Birth/death values are then **shifted back** by
``offset`` for display (``-1`` kept as the infinity sentinel).

The ``+ offset / - offset`` pair is a numerical *shift* needed because the
underlying path-homology filtration assumes non-negative, well-separated
weights — it is **not** a normalisation of the user-supplied weights into 
``[-1, 1]`` or any other interval.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from backend.analysis.digraph import Digraph


DEFAULT_DIMENSION: int = 4
DEFAULT_WEIGHT_OFFSET: float = 100.0


def _vertex_sort_key(name: str) -> tuple[int, float | str]:
    """Sort keys like numeric node ids before lexicographic strings."""
    try:
        return (0, float(name))
    except ValueError:
        return (1, name)


def _sorted_vertex_names(from_to_df: pd.DataFrame) -> list[str]:
    names = set(from_to_df["from"].astype(str)).union(set(from_to_df["to"].astype(str)))
    return sorted(names, key=_vertex_sort_key)


def _restore_bar_endpoint(
    value: Any,
    *,
    weight_offset: float,
) -> float | int:
    """Restore a barcode endpoint to the user's weight scale.

    This is **not** a normalisation. The pipeline shifts every input weight by
    ``+ weight_offset`` before running the path-homology algorithm; this helper
    undoes that shift for the corresponding ``birth`` / ``death`` values, so
    the returned numbers live in the same scale as the original
    ``weight`` / ``Effect`` column.

    Conventions
    -----------
    * ``value == -1`` is the infinity sentinel and is passed through unchanged.
    * Any other value is returned as ``round(float(value) - weight_offset, 6)``.
    """
    if value == -1:
        return -1
    restored = float(value) - weight_offset
    return round(restored, 6)


def build_weighted_digraph(
    from_to_df: pd.DataFrame,
    *,
    weight_offset: float = DEFAULT_WEIGHT_OFFSET,
) -> tuple[list[int], list[list[Any]], dict[str, int]]:
    """Build ``Digraph`` inputs: integer vertices ``1..n`` and shifted weights.

    Parameters
    ----------
    from_to_df:
        DataFrame containing ``from``, ``to`` and ``weight`` columns.
    weight_offset:
        Added to each edge weight before calling ``Digraph`` (legacy GLMY
        requirement that inputs be safely positive and numerically separated).

    Returns
    -------
    vertices_int, weighted_edges, vertex_id_map
        ``vertices_int`` is ``[1, 2, ..., n]``. ``weighted_edges`` rows are
        ``[u, v, weight + offset]`` with integer ``u, v``. ``vertex_id_map``
        maps the string node name from the table to that integer id.
    """
    required = {"from", "to", "weight"}
    missing = required - set(from_to_df.columns)
    if missing:
        raise ValueError(f"from_to 数据缺少必需列: {sorted(missing)}")
    if from_to_df.empty:
        raise ValueError("from_to.csv 为空，无法运行 GLMY。")

    clean_df = from_to_df.copy()
    clean_df["from"] = clean_df["from"].astype(str)
    clean_df["to"] = clean_df["to"].astype(str)
    clean_df["weight"] = pd.to_numeric(clean_df["weight"], errors="coerce")
    clean_df = clean_df.dropna(subset=["weight"])
    if clean_df.empty:
        raise ValueError("from_to.csv 中没有有效的数值型 weight，无法运行 GLMY。")

    # 丢弃自环：path-homology 的 allowed paths 要求相邻顶点不同。``Digraph``
    # 内部未做过滤，自环会让 boundary set 退化为单元素并配对掉对应顶点的
    # β₀ 无穷条，同时把高维 ``P[n]`` 污染成相邻重复的 non-allowed path。
    self_loop_mask = clean_df["from"] == clean_df["to"]
    if self_loop_mask.any():
        clean_df = clean_df.loc[~self_loop_mask].reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("过滤掉自环后 from_to 没有可用边，无法运行 GLMY。")

    ordered = _sorted_vertex_names(clean_df)
    vertex_id_map = {name: idx + 1 for idx, name in enumerate(ordered)}
    vertices_int = list(range(1, len(ordered) + 1))
    weighted_edges: list[list[Any]] = [
        [
            vertex_id_map[str(row["from"])],
            vertex_id_map[str(row["to"])],
            float(row["weight"]) + float(weight_offset),
        ]
        for _, row in clean_df.iterrows()
    ]
    return vertices_int, weighted_edges, vertex_id_map


def compute_glmy_homology(
    from_to_df: pd.DataFrame,
    *,
    dim: int = DEFAULT_DIMENSION,
    weight_offset: float = DEFAULT_WEIGHT_OFFSET,
) -> tuple[dict[str, list[list[float | int]]], dict[str, int]]:
    """Compute persistent path homology with the bundled Python algorithm.

    Parameters
    ----------
    from_to_df:
        DataFrame containing ``from``, ``to`` and ``weight`` columns.
    dim:
        Compute homology dimensions ``0`` through ``dim - 1``.
    weight_offset:
        Numerical shift applied internally: edges are passed to ``Digraph`` as
        ``weight + weight_offset``, and finite barcode endpoints have the same
        offset subtracted on the way out (``-1`` unchanged). This shift is
        **not** a normalisation — output endpoints stay in the original
        ``weight`` scale.

    Returns
    -------
    homology, vertex_id_map
        ``homology`` is compatible with ``plot_glmy_barcode``.
    """
    if dim < 1:
        raise ValueError("dim 必须为正整数。")

    vertices_int, weighted_edges, vertex_id_map = build_weighted_digraph(
        from_to_df,
        weight_offset=weight_offset,
    )
    digraph = Digraph(vertices_int, weighted_edges, dim)
    digraph.get_persistence()

    homology: dict[str, list[list[float | int]]] = {}
    for dimension in range(dim):
        key = str(dimension)
        bars: list[list[float | int]] = []
        for bar in digraph.diagram.get(key, []):
            if len(bar) < 2:
                continue
            bars.append(
                [
                    _restore_bar_endpoint(bar[0], weight_offset=weight_offset),
                    _restore_bar_endpoint(bar[1], weight_offset=weight_offset),
                ]
            )
        homology[key] = bars
    return homology, vertex_id_map
