"""Functional Clustering 可视化模块。

当前进度：
- ``plot_initialization_grid``：给 ``FunClu._initialize`` 的初值做诊断网格图，
  支持 ``"k_by_l"``（K 行 × L 列）与 ``"l_by_k"``（L 行 × K 列）两种布局。

后续 EM/最终结果的可视化（``plot_cluster_profiles`` 等）会逐步补齐。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.figure import Figure

from backend.utils import font_prop


# 子图配色：(成员细线, KMeans 中心散点, 幂律拟合曲线)
_DEFAULT_INIT_PALETTE: List[Tuple[str, str, str]] = [
    ("#FAD7A0", "#D35400", "#A04000"),
    ("#AED6F1", "#2E86C1", "#1B4F72"),
    ("#A9DFBF", "#239B56", "#196F3D"),
    ("#F5B7B1", "#C0392B", "#922B21"),
    ("#D7BDE2", "#8E44AD", "#5B2C6F"),
    ("#A3E4D7", "#16A085", "#0E6655"),
]


def _set_chinese_axes(ax: plt.Axes) -> None:
    """让 tick 与 title 使用项目中文字体（与 plot_curve_fitting 风格一致）。"""
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font_prop)


def plot_initialization_grid(
    *,
    X_list: List[torch.Tensor],
    times_list: List[np.ndarray],
    labels: np.ndarray,
    centers_kl: List[List[np.ndarray]],
    params_mu: np.ndarray,
    condition_labels: Optional[List[str]] = None,
    cluster_label_prefix: str = "M",
    layout: str = "k_by_l",
    figsize_per_panel: Tuple[float, float] = (3.0, 2.2),
    member_alpha: float = 0.75,
    member_lw: float = 0.8,
    center_marker_size: float = 28.0,
    fit_lw: float = 2.0,
    use_semilogx: bool = False,
    use_semilogy: bool = True,
    palette: Optional[Sequence[Tuple[str, str, str]]] = None,
    show_legend: bool = True,
    dpi: int = 200,
    show_in_streamlit: bool = True,
) -> Figure:
    """绘制 KMeans 初始化的诊断网格图。

    每个子图对应一对 ``(cluster k, condition i)``，叠加三层信息：

    1. **成员细线**：被分到该簇的特征在该 condition 下的原始曲线（淡色细线）；
    2. **KMeans 子中心**：簇中心向量按时间长度切回该 condition 的部分（散点）；
    3. **幂律拟合**：``y = a · t^b`` 的密集采样曲线（粗线，``a, b`` 来自
       ``params_mu[k, i]``）。

    Args:
        X_list: 长度为 ``L`` 的列表，第 i 项形如 ``(N, n_t_i)`` 的 ``torch.Tensor``。
        times_list: 长度为 ``L`` 的列表，第 i 项为该 condition 的时间向量
            ``(n_t_i,)``。
        labels: ``(N,)`` 的整型聚类标签（``np.ndarray`` 或可转 numpy 的张量）。
        centers_kl: ``centers_kl[k][i]`` 形如 ``(n_t_i,)`` 的 KMeans 子中心。
        params_mu: ``(K, L, 2)`` 的幂律参数 ``[a, b]``。
        condition_labels: 长度 ``L`` 的列表，给每个 condition 起个标签；为 ``None``
            时用 ``"Cond 1..L"``。
        cluster_label_prefix: 簇标签前缀；默认 ``"M"`` → ``M1, M2, ...``。
        layout: ``"k_by_l"``（K 行 × L 列，每行一个 cluster）或
            ``"l_by_k"``（L 行 × K 列，每行一个 condition）。
        figsize_per_panel: 单子图英寸 ``(宽, 高)``，默认 ``(3.0, 2.2)``。
        member_alpha, member_lw: 成员线的透明度与线宽。
        center_marker_size: 子中心散点大小。
        fit_lw: 拟合曲线粗细。
        use_semilogx, use_semilogy: 坐标轴是否取对数。
        palette: 长度任意，元组顺序为
            ``(成员色, 中心散点色, 拟合线色)``；不足时循环填充。
            ``None`` 走模块默认色板。
        show_legend: 是否在图顶部画一个全局 legend。
        dpi: 分辨率。
        show_in_streamlit: 是否在内部直接 ``st.pyplot(fig)``；为 ``False`` 时仅返回
            ``Figure``，便于单独保存或嵌入。

    Returns:
        ``matplotlib.figure.Figure`` 实例（已绘制完成）。

    Raises:
        ValueError: ``layout`` 非法 / 形状不一致。
    """
    K = int(params_mu.shape[0])
    L = int(params_mu.shape[1])
    if len(X_list) != L or len(times_list) != L or len(centers_kl) != K:
        raise ValueError(
            f"形状不一致：K={K}, L={L}; len(X_list)={len(X_list)}, "
            f"len(times_list)={len(times_list)}, len(centers_kl)={len(centers_kl)}"
        )
    if layout not in ("k_by_l", "l_by_k"):
        raise ValueError(f"layout 必须为 'k_by_l' 或 'l_by_k'，当前为 {layout!r}")

    if condition_labels is None:
        condition_labels = [f"Cond {i + 1}" for i in range(L)]

    pal: List[Tuple[str, str, str]] = (
        list(palette) if palette is not None else list(_DEFAULT_INIT_PALETTE)
    )
    while len(pal) < max(K, L):
        pal.extend(_DEFAULT_INIT_PALETTE)

    n_rows, n_cols = (K, L) if layout == "k_by_l" else (L, K)
    sw, sh = figsize_per_panel
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(sw * n_cols, sh * n_rows),
        sharex=False,
        sharey=False,
        dpi=dpi,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.asarray(axes).reshape(1, n_cols)
    elif n_cols == 1:
        axes = np.asarray(axes).reshape(n_rows, 1)

    labels_np = np.asarray(labels).astype(np.int64, copy=False)
    legend_drawn = False

    for k in range(K):
        mask_k = labels_np == k
        for i in range(L):
            t = np.asarray(times_list[i], dtype=np.float64)
            X_i_np = X_list[i].detach().cpu().numpy() if isinstance(
                X_list[i], torch.Tensor
            ) else np.asarray(X_list[i], dtype=np.float64)
            members = X_i_np[mask_k] if mask_k.any() else np.empty((0, t.size))
            center = np.asarray(centers_kl[k][i], dtype=np.float64)
            a = float(params_mu[k, i, 0])
            b = float(params_mu[k, i, 1])

            row, col = (k, i) if layout == "k_by_l" else (i, k)
            ax = axes[row, col]

            # 颜色：按 condition 取色（i 维度），保证同一 condition 在不同簇里颜色一致
            mem_c, ctr_c, fit_c = pal[i % len(pal)]

            # 1) 成员细线
            for mb in members:
                ax.plot(
                    t,
                    mb,
                    "-",
                    color=mem_c,
                    linewidth=member_lw,
                    alpha=member_alpha,
                    zorder=1,
                )

            # 2) 子中心散点（zorder=4，置于拟合曲线之上，避免被覆盖）
            ax.scatter(
                t,
                center,
                s=center_marker_size,
                color=ctr_c,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
                label="KMeans center" if not legend_drawn else None,
            )

            # 3) 幂律拟合曲线
            t_min = float(t.min()) if t.size else 0.0
            t_max = float(t.max()) if t.size else 1.0
            t_lo = max(t_min, 1e-12) if use_semilogx else t_min
            t_dense = np.linspace(t_lo, t_max, 200)
            with np.errstate(over="ignore", invalid="ignore"):
                y_fit = a * np.power(t_dense, b)
            ax.plot(
                t_dense,
                y_fit,
                "-",
                color=fit_c,
                linewidth=fit_lw,
                zorder=3,
                label=r"$y=a\cdot t^{b}$" if not legend_drawn else None,
            )
            legend_drawn = True

            if use_semilogx:
                ax.set_xscale("log")
            if use_semilogy:
                ax.set_yscale("log")

            n_k = int(mask_k.sum())
            title = (
                f"{cluster_label_prefix}{k + 1} | {condition_labels[i]}\n"
                f"a={a:.2g}, b={b:.2g}, n={n_k}"
            )
            ax.set_title(title, fontsize=8, fontproperties=font_prop)
            ax.tick_params(labelsize=7)
            ax.grid(False)
            _set_chinese_axes(ax)

    if show_legend:
        handles, lbls = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                lbls,
                loc="upper center",
                ncol=2,
                fontsize=9,
                bbox_to_anchor=(0.5, 0.99),
            )

    fig.tight_layout(rect=(0, 0, 1, 0.96) if show_legend else (0, 0, 1, 1))

    if show_in_streamlit:
        st.pyplot(fig, use_container_width=True)

    return fig
