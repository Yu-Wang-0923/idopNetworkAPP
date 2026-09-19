"""GLMY / path homology（新版实现）。

移植自新版 ``network_glmy.py``，相对上游脚本有三处适配：

1. ``from digraph import Digraph`` → ``from idopnetwork.analysis.digraph import
   Digraph``（包内路径）。
2. **列名兼容**：上游用 ``source`` / ``target``，而页面导出并上传的
   ``from_to.csv`` 用 ``from`` / ``to``（另有 ``Effect`` 作权重）。入口统一
   归一化，两种列名都能吃。
3. **保留自环过滤**（上游没有这一步）。path homology 的 allowed path 要求相邻
   顶点不同，而 ``Digraph`` 内部不过滤自环：自环会让 boundary set 退化为单元素、
   配对掉对应顶点的 β₀ 无穷条，并把高维 ``P[n]`` 污染成相邻重复的 non-allowed
   path。

数值约定（新版）
----------------
``weight_offset`` 默认 ``None``，自动取 ``max(0.0, 1.0 - min(weight))``，保证
偏移后的权重严格为正；barcode 端点再减去同一偏移，还原回原始 weight 尺度。
**无穷区间用 ``None`` 表示**（旧版是 ``-1``）；``-1`` 在绘图层仍被识别，以兼容
自检路径。顶点编号沿用「数值优先」排序，与模块命名（M1…M12 / 0…11）一致。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from idopnetwork.analysis.digraph import Digraph

DIMENSION: int = 5

DIMENSION_KEYS: tuple[str, ...] = (
    "3",
    "2",
    "1",
    "0",
)
DIMENSION_COLORS: tuple[str, ...] = (
    "purple",
    "g",
    "r",
    "b",
)
DIMENSION_LABELS: tuple[str, ...] = (
    r"$\beta_3$",
    r"$\beta_2$",
    r"$\beta_1$",
    r"$\beta_0$",
)

Bar = list
Homology = dict

# 兼容既有导入名
DEFAULT_DIMENSION: int = DIMENSION
DEFAULT_WEIGHT_OFFSET: float | None = None

# 列名别名：统一到 source / target / weight
_COLUMN_ALIASES = {
    "from": "source",
    "source": "source",
    "to": "target",
    "target": "target",
    "weight": "weight",
    "effect": "weight",
    "w": "weight",
}


def _vertex_sort_key(name: str) -> tuple[int, float | str]:
    """数值型节点名排在字符串之前（"2" < "10"，与模块命名一致）。"""
    try:
        return (0, float(name))
    except ValueError:
        return (1, str(name))


def normalize_network(network: pd.DataFrame) -> pd.DataFrame:
    """把边表归一化成 ``source`` / ``target`` / ``weight`` 三列。

    接受 ``source`` / ``target``（上游脚本）与 ``from`` / ``to``（页面导出的
    ``from_to.csv``），权重列接受 ``weight`` / ``Effect``。丢弃非数值权重，
    并**剔除自环**（见模块 docstring 的说明）。
    """
    frame = pd.DataFrame(network).copy()
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[column] = _COLUMN_ALIASES[key]
    frame = frame.rename(columns=rename)

    required = {"source", "target", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"边表缺少必需列: {sorted(missing)}；"
            f"需要 source/target/weight 或 from/to/weight，实际列={list(frame.columns)}"
        )

    frame = frame[["source", "target", "weight"]].copy()
    frame["source"] = frame["source"].astype(str).str.strip()
    frame["target"] = frame["target"].astype(str).str.strip()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["weight"])

    if frame.empty:
        raise ValueError("边表中没有有效的数值型 weight，无法运行 GLMY。")

    self_loop_mask = frame["source"] == frame["target"]
    if self_loop_mask.any():
        frame = frame.loc[~self_loop_mask]
    if frame.empty:
        raise ValueError("过滤掉自环后没有可用边，无法运行 GLMY。")

    return frame.reset_index(drop=True)


def build_weighted_digraph(
    network: pd.DataFrame,
    weight_offset: float | None = None,
) -> tuple[list[int], list[list[float | int]], float]:
    """构造 ``Digraph`` 输入：整数顶点 ``1..n`` 与整体平移为正的边权。

    Parameters
    ----------
    network:
        含 ``source`` / ``target`` / ``weight``（或 ``from`` / ``to`` / ``weight``）
        的边表。
    weight_offset:
        加到每条边权上的偏移。为 ``None`` 时自动取
        ``max(0.0, 1.0 - min(weight))``，保证平移后权重严格为正；权重本就
        非负（例如已按符号拆分）时可显式传 ``0.0``。

    Returns
    -------
    vertices, weighted_edges, weight_offset
    """
    clean = normalize_network(network)

    node_names = sorted(
        set(clean["source"]).union(set(clean["target"])),
        key=_vertex_sort_key,
    )
    node_map = {
        name: index + 1
        for index, name in enumerate(node_names)
    }

    if weight_offset is None:
        minimum_weight = float(clean["weight"].min())
        weight_offset = max(0.0, 1.0 - minimum_weight)

    weighted_edges: list[list[float | int]] = []
    for row in clean.itertuples(index=False):
        weighted_edges.append(
            [
                node_map[row.source],
                node_map[row.target],
                float(row.weight) + float(weight_offset),
            ]
        )

    vertices = list(range(1, len(node_names) + 1))
    return vertices, weighted_edges, float(weight_offset)


def vertex_id_map(network: pd.DataFrame) -> dict[str, int]:
    """节点名 → 整数顶点编号（与 :func:`build_weighted_digraph` 同一套编号）。"""
    clean = normalize_network(network)
    node_names = sorted(
        set(clean["source"]).union(set(clean["target"])),
        key=_vertex_sort_key,
    )
    return {name: index + 1 for index, name in enumerate(node_names)}


def _restore_bar_endpoint(
    value: Any,
    *,
    weight_offset: float,
) -> float | None:
    """把 barcode 端点还原回用户权重尺度。

    ``-1`` 与 ``None`` 都视为无穷哨兵，统一归一化成 ``None``；其余值减去
    ``weight_offset``。
    """
    if value is None or value == -1:
        return None
    return float(value) - float(weight_offset)


def compute_glmy_homology(
    network: pd.DataFrame,
    weight_offset: float | None = None,
    *,
    dimension: int = DIMENSION,
) -> Homology:
    """计算 0–3 维的 path homology，返回 ``{dim: [[birth, death], ...]}``。

    无穷区间用 ``None`` 表示。``dimension`` 透传给 ``Digraph``（默认 5，即多算
    一维以保证 3 维结果正确），返回的维度数固定为 ``len(DIMENSION_KEYS)`` 个。
    """
    vertices, weighted_edges, weight_offset = build_weighted_digraph(
        network, weight_offset,
    )

    digraph = Digraph(vertices, weighted_edges, dimension)
    digraph.get_persistence()

    homology: Homology = {}
    for index, key in enumerate(DIMENSION_KEYS):
        dimension_number = int(key)
        bars: list[Bar] = []

        for birth, death in digraph.diagram.get(key, []):
            if dimension_number == 0 and birth == 0:
                restored_birth = None
            else:
                restored_birth = _restore_bar_endpoint(
                    birth, weight_offset=weight_offset,
                )

            bars.append(
                [
                    restored_birth,
                    _restore_bar_endpoint(death, weight_offset=weight_offset),
                ]
            )

        homology[key] = bars

    return homology


def _split_network_by_sign(
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按权重符号拆成正 / 负两组。

    零权边被丢弃（只会产生退化条）；负权组取绝对值。返回的两张表已经是
    ``source`` / ``target`` / ``weight`` 归一化之后的形态。
    """
    clean = normalize_network(network)

    positive = clean.loc[clean["weight"] > 0].reset_index(drop=True)
    negative = clean.loc[clean["weight"] < 0].reset_index(drop=True)
    negative = negative.copy()
    negative["weight"] = negative["weight"].abs()

    return positive, negative


def compute_glmy_homology_split(
    network: pd.DataFrame,
    *,
    dimension: int = DIMENSION,
) -> dict[str, Homology]:
    """分别对正权子图与负权子图计算 homology。

    Returns
    -------
    dict[str, Homology]
        键为 ``"positive"`` / ``"negative"``，各自可喂给
        :func:`plot_glmy_barcode_split`。某一侧无可用边时返回空条目的
        homology（而不是报错）。
    """
    positive, negative = _split_network_by_sign(network)

    homologies: dict[str, Homology] = {}
    for name, subgraph in (("positive", positive), ("negative", negative)):
        if subgraph.empty:
            homologies[name] = {key: [] for key in DIMENSION_KEYS}
        else:
            homologies[name] = compute_glmy_homology(
                subgraph,
                weight_offset=0.0,
                dimension=dimension,
            )
    return homologies


def bar_counts(homology: Homology) -> dict[str, int]:
    """单个 homology 每个维度的条数，用作共享 y 轴高度。"""
    return {key: len(homology.get(key, [])) for key in DIMENSION_KEYS}


def bar_counts_split(
    homologies: dict[str, Homology],
) -> dict[str, dict[str, int]]:
    """正 / 负两侧的条数，用于 :func:`plot_glmy_barcode_split` 的共享 y 轴。"""
    return {
        sign: bar_counts(homologies.get(sign, {}))
        for sign in ("positive", "negative")
    }
