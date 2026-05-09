"""Network Analysis backend helpers.

The GLMY barcode workflow consumes ``from_to.csv`` files exported by
``pages/3_NetRecon.py`` and computes persistent path homology with the bundled
pure Python implementation in ``backend.glmy``.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Any

import pandas as pd

from backend.glmy import DEFAULT_DIMENSION, compute_glmy_homology


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
) -> dict[str, Any]:
    """Compute GLMY/path homology with the bundled Python implementation.

    Parameters
    ----------
    from_to_df:
        DataFrame containing ``from``, ``to`` and ``weight`` columns.
    dim:
        Compute homology dimensions ``0`` through ``dim - 1``.

    Returns
    -------
    dict
        Structure consumed by ``pages/4_NetAnal.py`` and
        ``plot_glmy_barcode``.
    """
    homology, vertex_id_map = compute_glmy_homology(from_to_df, dim=dim)
    return {
        "homology": homology,
        "vertex_id_map": vertex_id_map,
        "dimension": dim,
        "backend": "python",
    }


# ── 工具：自适应 max_x ────────────────────────────────────────────────────────

def suggest_max_x(
    from_to_df: pd.DataFrame,
    *,
    buffer_ratio: float = 0.1,
    floor: float = 1.0,
) -> float:
    """根据 ``weight`` 绝对值最大值估计 barcode 横轴范围。"""
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
