"""GLMY 自检模块：用 ``backend.Digraph`` 复刻原 ``GLMY1.py`` 流程跑 ``M3.csv``。

目的
----
把"是不是 ``backend.Digraph`` 算错了"这件事从 NetRecon 数据流里隔离出来，
做一个最小可复现：

* 输入：仓库根的 ``M3.csv``（列 ``From, To, Effect``，与原 ``GLMY1.py`` 同源）。
* 算法：``backend.Digraph.Digraph`` 纯 Python 实现，**不**调外部 ``GLMY.exe``。
* 流程：等价复刻原 ``GLMY1.py``：

    1. ``vertices = sorted(set(From) | set(To))``（字符串排序）；
    2. ``V[name] = 1..n``、``input_V = list(range(1, n + 1))``；
    3. ``weight = float(Effect) + 100``；
    4. ``Digraph(V, weighted_edges, dim=4).get_persistence()``；
    5. 端点处理：``-1`` 透传，其它 ``round(value - 100, 6)``；
    6. β₀..β₃ barcode 排序与原 ``draw_unfiltered_barcode`` 一致。

* 输出：``homology`` dict，可被 ``backend.plot_network_analysis.plot_glmy_barcode``
  直接消费。

历史背景
--------
本文件原为仓库根的脚本 ``glmy_M3_test.py``，被前端 ``pages/4_NetAnal.py`` 直接
``import``。Streamlit Cloud 上 ``pages/`` 子模块解析时找不到顶层脚本，且原文件
本身存在 ``SyntaxError``（``load_m3_dataframe`` 函数体被截断、``build_digraph_inputs``
形参列表混入 ``for`` 子句），导致 ``ImportError``。现按项目规范迁入 ``backend/``，
由页面通过 ``from backend.glmy_m3_test import ...`` 引用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.Digraph import Digraph

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_M3_CSV: Path = REPO_ROOT / "M3.csv"
DEFAULT_DIM: int = 4
DEFAULT_WEIGHT_OFFSET: float = 100.0
DEFAULT_MAX_X: float = 2.0

REQUIRED_M3_COLUMNS: tuple[str, str, str] = ("From", "To", "Effect")


def load_m3_dataframe(csv_path: Path | str | None = None) -> pd.DataFrame:
    """读取 ``M3.csv`` 并校验列名（``From, To, Effect``）。

    Parameters
    ----------
    csv_path:
        CSV 路径，``None`` 表示使用仓库根 ``M3.csv``。

    Returns
    -------
    pd.DataFrame
        三列 DataFrame：``From``/``To`` 为字符串，``Effect`` 强制转为数值；
        无效 ``Effect`` 行会被丢弃。

    Raises
    ------
    FileNotFoundError
        指定路径不存在。
    ValueError
        缺少必需列、或清洗后没有任何有效数据行。
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_M3_CSV
    if not path.exists():
        raise FileNotFoundError(f"找不到 M3 CSV: {path}")

    df = pd.read_csv(path)
    missing = set(REQUIRED_M3_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} 缺少必需列: {sorted(missing)}；实际列={list(df.columns)}"
        )

    df = df.loc[:, list(REQUIRED_M3_COLUMNS)].copy()
    df["From"] = df["From"].astype(str)
    df["To"] = df["To"].astype(str)
    df["Effect"] = pd.to_numeric(df["Effect"], errors="coerce")
    df = df.dropna(subset=["Effect"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path.name} 中没有有效的数值型 Effect，无法运行 Digraph。")
    return df


def build_digraph_inputs(
    df: pd.DataFrame,
    *,
    weight_offset: float = DEFAULT_WEIGHT_OFFSET,
) -> tuple[list[int], list[list[Any]], dict[str, int]]:
    """把 ``From,To,Effect`` 表转成 ``Digraph`` 期望的 ``(V, WG, vertex_map)``。

    复刻原 ``GLMY1.py`` 的 ``generation_unfiltered_data``：顶点用字符串
    ``sorted`` 排序后映射到 ``1..n``，边权 = ``Effect + weight_offset``。

    自环（``From == To``）被静默丢弃：path-homology 的 allowed paths 要求
    相邻顶点不同；``backend.Digraph`` 内部未做过滤，自环会让 boundary 退化
    为单元素 set 并把对应顶点的 β₀ 无穷条直接配对掉。M3.csv 本身没有自环，
    这里加过滤只是为了与主流水线保持一致。
    """
    work = df.copy()
    work["From"] = work["From"].astype(str)
    work["To"] = work["To"].astype(str)
    self_loop_mask = work["From"] == work["To"]
    if self_loop_mask.any():
        work = work.loc[~self_loop_mask].reset_index(drop=True)
    if work.empty:
        raise ValueError("过滤掉自环后 M3 数据没有可用边，无法运行 Digraph。")

    vertices = sorted(set(work["From"]).union(set(work["To"])))
    vertex_map: dict[str, int] = {name: idx + 1 for idx, name in enumerate(vertices)}
    v_int: list[int] = list(range(1, len(vertices) + 1))

    weighted_edges: list[list[Any]] = []
    for _, row in work.iterrows():
        u = vertex_map[str(row["From"])]
        v = vertex_map[str(row["To"])]
        weight = float(row["Effect"]) + float(weight_offset)
        weighted_edges.append([u, v, weight])
    return v_int, weighted_edges, vertex_map


def _restore_endpoint(value: Any, *, weight_offset: float) -> float | int:
    """端点还原：``-1`` 透传（无穷条），其它减偏移并 round 到 6 位。"""
    if value == -1:
        return -1
    return round(float(value) - float(weight_offset), 6)


def run_digraph_on_m3(
    df: pd.DataFrame,
    *,
    dim: int = DEFAULT_DIM,
    weight_offset: float = DEFAULT_WEIGHT_OFFSET,
) -> tuple[dict[str, list[list[float | int]]], dict[str, int]]:
    """跑 ``backend.Digraph``，返回 ``(homology, vertex_map)``。

    ``homology`` 结构与 ``backend.plot_network_analysis.plot_glmy_barcode`` 完全
    兼容：``{"0": [[birth, death], ...], "1": [...], ...}``，``death == -1``
    表示无穷条。
    """
    if dim < 1:
        raise ValueError("dim 必须为正整数。")

    v_int, weighted_edges, vertex_map = build_digraph_inputs(
        df, weight_offset=weight_offset
    )
    dg = Digraph(v_int, weighted_edges, dim)
    dg.get_persistence()

    homology: dict[str, list[list[float | int]]] = {}
    for d in range(dim):
        key = str(d)
        bars: list[list[float | int]] = []
        for bar in dg.diagram.get(key, []):
            if len(bar) < 2:
                continue
            bars.append(
                [
                    _restore_endpoint(bar[0], weight_offset=weight_offset),
                    _restore_endpoint(bar[1], weight_offset=weight_offset),
                ]
            )
        homology[key] = bars
    return homology, vertex_map


def betti_summary(
    homology: dict[str, list[list[float | int]]],
    *,
    dim: int = DEFAULT_DIM,
) -> dict[str, int]:
    """``β_d`` = 维度 ``d`` 的 barcode 条数（与 NetAnal 页面口径一致）。"""
    return {f"β{d}": len(homology.get(str(d), [])) for d in range(dim)}


def paper_3_2_dataframe() -> pd.DataFrame:
    """返回论文 §3.2 / Figure 1 的 filtered digraph，作为 path-homology 地面真值。

    论文给出的过滤过程为 ``G0 = {1, 2, 3}``,
    ``G1 = G0 ∪ {21}``, ``G2 = G1 ∪ {12}``,
    ``G3 = G2 ∪ {23}``, ``G4 = G3 ∪ {31}``,
    ``G5 = G4 ∪ {13}``。这里的 ``Effect`` 即每条有向边首次出现的
    filtration index；``G0`` 中的顶点由边端点集合隐式恢复。
    """
    return pd.DataFrame(
        [
            {"From": "2", "To": "1", "Effect": 1},
            {"From": "1", "To": "2", "Effect": 2},
            {"From": "2", "To": "3", "Effect": 3},
            {"From": "3", "To": "1", "Effect": 4},
            {"From": "1", "To": "3", "Effect": 5},
        ],
        columns=list(REQUIRED_M3_COLUMNS),
    )


def _print_homology(homology: dict[str, list[list[float | int]]]) -> None:
    print("\n=== homology (already restored to Effect scale) ===")
    print(json.dumps(homology, indent=2, ensure_ascii=False))


def main(
    csv_path: Path | str | None = None,
    *,
    dim: int = DEFAULT_DIM,
    weight_offset: float = DEFAULT_WEIGHT_OFFSET,
    max_x: float = DEFAULT_MAX_X,
    save_dir: Path | str | None = None,
) -> dict[str, list[list[float | int]]]:
    """命令行入口：跑 M3.csv 并保存 barcode PNG/PDF。"""
    import matplotlib.pyplot as plt

    from backend.plot_network_analysis import plot_glmy_barcode

    df = load_m3_dataframe(csv_path)
    print(f"Loaded {len(df)} edges from {Path(csv_path or DEFAULT_M3_CSV).name}.")
    homology, vertex_map = run_digraph_on_m3(
        df, dim=dim, weight_offset=weight_offset
    )
    print("Vertex map:", vertex_map)
    print("Betti summary:", betti_summary(homology, dim=dim))
    _print_homology(homology)

    out_dir = Path(save_dir) if save_dir is not None else REPO_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_glmy_barcode(homology, max_x=float(max_x))
    png_path = out_dir / "M3_barcode.png"
    pdf_path = out_dir / "M3_barcode.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved barcode to:\n  {png_path}\n  {pdf_path}")
    return homology


if __name__ == "__main__":
    main()
