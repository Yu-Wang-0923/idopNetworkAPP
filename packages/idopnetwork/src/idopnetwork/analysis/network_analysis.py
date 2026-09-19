"""Network Analysis backend helpers.

The GLMY barcode workflow consumes ``from_to.csv`` files exported by
``pages/3_NetRecon.py`` and computes persistent path homology with the bundled
pure Python implementation in ``backend.analysis.glmy``.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Any

import pandas as pd

from idopnetwork.analysis.glmy import (
    DEFAULT_DIMENSION,
    DEFAULT_WEIGHT_OFFSET,
    build_weighted_digraph,
    compute_glmy_homology,
    compute_glmy_homology_split,
    normalize_network,
    vertex_id_map,
)


# ── ZIP 解析 ──────────────────────────────────────────────────────────────────

def _read_zip_member_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """从 ZIP 中读取 CSV 成员（兼容 utf-8-sig BOM）。"""
    with zf.open(name) as fh:
        return pd.read_csv(io.BytesIO(fh.read()))


def list_from_to_members(zip_bytes: bytes) -> list[str]:
    """枚举 ZIP 中所有 ``from_to.csv`` 路径。

    单层导出返回 ``["from_to.csv"]``；多层导出返回形如
    ``"inter_cluster/<cond>/from_to.csv"`` 与
    ``"intra_cluster/<cond>/<cluster>/from_to.csv"`` 的列表。
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.endswith("from_to.csv")]
    names.sort()
    return names


def member_display_label(member_path: str) -> str:
    """把 ZIP 内 from_to.csv 路径转成下拉框友好的标签。"""
    if member_path == "from_to.csv":
        return "single_layer"
    if member_path.endswith("/from_to.csv"):
        return member_path[: -len("/from_to.csv")]
    return member_path


def load_from_to_from_zip(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """从 ZIP 中读取指定 ``from_to.csv``，校验列名。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        df = _read_zip_member_csv(zf, member_path)
    required = {"from", "to", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{member_path} 缺少必需列: {sorted(missing)}；实际列={list(df.columns)}"
        )
    df = df.copy()
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"]).reset_index(drop=True)
    return df


# ── GLMY 计算 ─────────────────────────────────────────────────────────────────

def run_glmy(
    from_to_df: pd.DataFrame,
    *,
    dim: int = DEFAULT_DIMENSION,
    weight_offset: float | None = DEFAULT_WEIGHT_OFFSET,
) -> dict[str, Any]:
    """Compute GLMY/path homology with the bundled Python implementation.

    Parameters
    ----------
    from_to_df:
        边表，接受 ``from`` / ``to`` / ``weight``（页面导出的 from_to.csv）
        或 ``source`` / ``target`` / ``weight``（上游脚本）。
    dim:
        透传给 ``Digraph`` 的维度（默认 5，多算一维以保证 3 维结果正确）；
        返回的维度固定为 β₃…β₀ 四个。
    weight_offset:
        加到每条边权上的数值偏移，仅用于滤波数值分离，**不是**归一化。
        为 ``None``（默认）时自动取 ``max(0, 1 - min(weight))``。

    Returns
    -------
    dict
        ``homology`` 为 ``{dim: [[birth, death], ...]}``，无穷区间用 ``None``；
        ``weight_offset`` 是本轮**实际**使用的偏移（自动模式下为推算值）。
    """
    clean = normalize_network(from_to_df)

    # 先把偏移解析出来（None → 自动值），这样返回给页面的就是实际用的数字
    _, _, resolved_offset = build_weighted_digraph(clean, weight_offset)

    homology = compute_glmy_homology(
        clean,
        resolved_offset,
        dimension=dim,
    )
    return {
        "homology": homology,
        "vertex_id_map": vertex_id_map(clean),
        "dimension": dim,
        "weight_offset": resolved_offset,
        "backend": "python",
    }


def run_glmy_split(
    from_to_df: pd.DataFrame,
    *,
    dim: int = DEFAULT_DIMENSION,
) -> dict[str, Any]:
    """分别对正权与负权子图计算 homology（新版正负拆分视图用）。

    Returns
    -------
    dict
        ``homologies`` 为 ``{"positive": Homology, "negative": Homology}``；
        另附 ``edge_counts`` 便于页面显示两侧各用了多少条边。
    """
    clean = normalize_network(from_to_df)
    homologies = compute_glmy_homology_split(clean, dimension=dim)

    positive_count = int((clean["weight"] > 0).sum())
    negative_count = int((clean["weight"] < 0).sum())

    return {
        "homologies": homologies,
        "edge_counts": {
            "positive": positive_count,
            "negative": negative_count,
        },
        "dimension": dim,
        "weight_offset": 0.0,
        "backend": "python",
    }


# ── 工具：自适应 max_x ────────────────────────────────────────────────────────

def suggest_max_x(
    from_to_df: pd.DataFrame,
    *,
    buffer_ratio: float = 0.1,
    floor: float = 1e-9,
) -> float:
    """估计 barcode 横轴右端的自适应建议值（**不**归一化 weight）。

    取 ``|weight|.max() * (1 + buffer_ratio)`` 作为 barcode 横轴右端建议值，
    barcode 数值本身不变；这只是绘图时的 ``xlim`` 自适应，不会改写
    ``from_to.csv`` 中的 ``weight``。

    历史版本曾把 ``floor`` 设为 ``1.0``，在 ``|weight|.max() < 1`` 时强制
    把横轴拉到 ``[-1.0, 1.1]``，视觉上很像把 weight 归一化到 ``[-1, 1]`` ——
    但其实数据没动，只是横轴被 floor 钉住了。这里把 ``floor`` 降到 ``1e-9``，
    仅作为**空表 / 全零兜底**，避免返回 ``0`` 让横轴退化。
    """
    if from_to_df.empty:
        return floor
    abs_max = float(from_to_df["weight"].abs().max())
    suggested = abs_max * (1.0 + max(0.0, buffer_ratio))
    return max(floor, suggested)


# ── 工具：合法文件名片段 ─────────────────────────────────────────────────────

def sanitize_name(name: str) -> str:
    """把任意字符串规范成可安全用于文件名的片段。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "glmy"
