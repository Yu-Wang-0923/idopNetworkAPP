"""Pure Python GLMY/path homology helpers.

This module wraps :class:`backend.Digraph.Digraph` so the Streamlit page can
compute barcode data without the legacy external executable.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from backend.Digraph import Digraph


DEFAULT_DIMENSION: int = 4


def _normalise_bar_value(value: Any) -> float | int:
    """Return a JSON-friendly numeric barcode endpoint."""
    if value == -1:
        return -1
    return round(float(value), 6)


def build_weighted_digraph(
    from_to_df: pd.DataFrame,
) -> tuple[list[str], list[list[Any]], dict[str, int]]:
    """Build ``Digraph`` inputs from a ``from_to.csv`` DataFrame.

    Parameters
    ----------
    from_to_df:
        DataFrame containing ``from``, ``to`` and ``weight`` columns.

    Returns
    -------
    vertices, weighted_edges, vertex_id_map
        ``vertices`` keeps node names as strings. ``weighted_edges`` has
        ``[source, target, weight]`` rows accepted by ``Digraph``.
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

    vertices = sorted(set(clean_df["from"]).union(set(clean_df["to"])))
    vertex_id_map = {name: idx + 1 for idx, name in enumerate(vertices)}
    weighted_edges: list[list[Any]] = [
        [str(row["from"]), str(row["to"]), float(row["weight"])]
        for _, row in clean_df.iterrows()
    ]
    return vertices, weighted_edges, vertex_id_map


def compute_glmy_homology(
    from_to_df: pd.DataFrame,
    *,
    dim: int = DEFAULT_DIMENSION,
) -> tuple[dict[str, list[list[float | int]]], dict[str, int]]:
    """Compute persistent path homology with the bundled Python algorithm.

    Parameters
    ----------
    from_to_df:
        DataFrame containing ``from``, ``to`` and ``weight`` columns.
    dim:
        Compute homology dimensions ``0`` through ``dim - 1``.

    Returns
    -------
    homology, vertex_id_map
        ``homology`` is compatible with ``plot_glmy_barcode``.
    """
    if dim < 1:
        raise ValueError("dim 必须为正整数。")

    vertices, weighted_edges, vertex_id_map = build_weighted_digraph(from_to_df)
    digraph = Digraph(vertices, weighted_edges, dim)
    digraph.get_persistence()

    homology: dict[str, list[list[float | int]]] = {}
    for dimension in range(dim):
        key = str(dimension)
        bars: list[list[float | int]] = []
        for bar in digraph.diagram.get(key, []):
            if len(bar) < 2:
                continue
            bars.append([_normalise_bar_value(bar[0]), _normalise_bar_value(bar[1])])
        homology[key] = bars
    return homology, vertex_id_map