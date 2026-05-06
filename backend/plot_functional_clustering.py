"""Functional Clustering 可视化模块。

实现进度：

- ``plot_initialization_grid``：给 ``FunClu._initialize`` 的初值做诊断网格图，
  支持 ``"k_by_l"``（K 行 × L 列）与 ``"l_by_k"``（L 行 × K 列）两种布局。
- ``plot_loglik_history``：EM 收敛曲线（``log-likelihood`` vs iteration）。
- ``plot_cluster_profiles``：EM 拟合完成后，按簇绘制各 condition 的成员
  曲线 + 加权均值幂律拟合曲线 + 可选 CI 带。

绘图函数全部为模块级（不依赖 ``FunClu`` 实例方法），既符合项目分层
（``backend/plot_xxx.py`` 仅做绘图），又便于在 page 之外被复用。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.figure import Figure

from backend.curve_fitting import fit_power_loglinear
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

# 簇 profile 配色：(成员散点/细线色, 均值曲线色)
_CLUSTER_PROFILE_DEFAULT_PALETTE: List[Tuple[str, str]] = [
    ("#FAD7A0", "#D35400"),
    ("#AED6F1", "#2E86C1"),
    ("#A9DFBF", "#239B56"),
    ("#F5B7B1", "#C0392B"),
    ("#D7BDE2", "#8E44AD"),
    ("#A3E4D7", "#16A085"),
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
    center_marker_size: float = 1.0,
    fit_lw: float = 2.0,
    use_semilogx: bool = False,
    use_semilogy: bool = True,
    palette: Optional[Sequence[Tuple[str, str, str]]] = None,
    show_legend: bool = True,
    dpi: int = 200,
    show_in_streamlit: bool = True,
    share_x: bool = False,
    share_y: bool = False,
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
        sharex=share_x,
        sharey=share_y,
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
                linewidths=0.3,
                alpha=0.25,
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


def plot_loglik_history(
    history: Sequence[float],
    *,
    figsize: Tuple[float, float] = (5.0, 2.6),
    dpi: int = 200,
    show_in_streamlit: bool = True,
    title: str = "EM convergence",
    color: str = "#2E86C1",
) -> Figure:
    """绘制 EM 主循环的 ``log-likelihood`` 收敛曲线。

    Args:
        history: 长度 ``T`` 的 ``log-likelihood`` 序列（``FunClu.loglik_history``）。
        figsize: 画布尺寸（英寸）。
        dpi: 分辨率。
        show_in_streamlit: 是否直接 ``st.pyplot``。
        title: 图标题。
        color: 折线颜色。

    Returns:
        ``matplotlib.figure.Figure``。
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if not history:
        ax.text(
            0.5,
            0.5,
            "no iterations",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            fontproperties=font_prop,
        )
    else:
        xs = np.arange(1, len(history) + 1)
        ax.plot(xs, np.asarray(history, dtype=np.float64), "-o", color=color, lw=1.5, ms=3.5)
        ax.set_xlabel("iteration", fontproperties=font_prop)
        ax.set_ylabel("log-likelihood", fontproperties=font_prop)
    ax.set_title(title, fontsize=10, fontproperties=font_prop)
    ax.grid(True, ls=":", alpha=0.5)
    _set_chinese_axes(ax)
    fig.tight_layout()

    if show_in_streamlit:
        st.pyplot(fig, use_container_width=True)
    return fig


def _fit_mean_curve_power_sample(
    t_curve: np.ndarray,
    y_mean: np.ndarray,
    t_sample: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """基于 ``y = a · t^b`` 拟合均值曲线，并在 ``t_sample`` 上采样。

    与 ``FunClu`` 内部初始化所用的 :func:`backend.curve_fitting.fit_power_loglinear`
    同源（双对数线性回归），保持口径一致。仅在 ``t_curve`` 的有效域 ``[t_lo, t_hi]``
    内输出有限值，域外为 ``NaN``，便于上游做兜底。
    """
    x_arr = np.asarray(t_curve, dtype=np.float64)
    y_arr = np.asarray(y_mean, dtype=np.float64)
    xg = np.asarray(t_sample, dtype=np.float64)
    out = np.full(xg.shape, np.nan, dtype=np.float64)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return xg, out
    a, b = fit_power_loglinear(
        x_arr[mask],
        y_arr[mask],
        clip_a=(1e-12, np.inf),
        clip_b=(-10.0, 10.0),
    )
    x_lo = float(np.min(x_arr[mask]))
    x_hi = float(np.max(x_arr[mask]))
    in_range = np.isfinite(xg) & (xg >= x_lo) & (xg <= x_hi)
    with np.errstate(over="ignore", invalid="ignore"):
        out[in_range] = a * np.power(xg[in_range], b)
    return xg, out


def plot_cluster_profiles(
    *,
    data_scatter: List[pd.DataFrame],
    data_curve: Optional[List[pd.DataFrame]] = None,
    labels: np.ndarray,
    common_cols: Optional[Sequence[str]] = None,
    n_components: Optional[int] = None,
    condition_labels: Optional[Sequence[str]] = None,
    member_source: str = "qd_df",
    show_members: bool = True,
    show_mean: bool = True,
    show_mean_ci: bool = False,
    ci_alpha: float = 0.25,
    ci_z: float = 1.96,
    use_semilogy: bool = True,
    use_log_x: bool = False,
    n_cols: int = 3,
    subplot_hw: Tuple[float, float] = (4.0, 3.0),
    palette: Optional[Sequence[Tuple[str, str]]] = None,
    panel_title_prefix: str = "M",
    panel_title_fontsize: float = 11.0,
    xlabel: str = "Index",
    ylabel: Optional[str] = None,
    axis_label_fontsize: float = 12.0,
    linewidth_mean: float = 3.0,
    linewidth_member: float = 1.2,
    markersize_qd: float = 7.0,
    alpha_member_lines: float = 0.7,
    alpha_qd_marker: float = 0.9,
    x_margin: float = 0.1,
    y_margin: float = 0.2,
    show_legend: bool = True,
    legend_loc: str = "upper center",
    legend_ncol: Optional[int] = None,
    legend_fontsize: float = 11.0,
    legend_bbox: Tuple[float, float] = (0.5, 0.99),
    dpi: int = 200,
    show_in_streamlit: bool = True,
) -> Figure:
    """按簇绘制 EM 拟合结果的 profile 图。每簇一个子图，叠加多 condition。

    每个子图叠加三层信息（按 condition ``i`` 配同一对色）：

    1. **成员**（可选）：被分到该簇的特征在该 condition 下的曲线，可选两种来源：

       - ``member_source="qd_df"``：把每个时间点 × 每个成员特征作为散点（圈，仅描边）；
       - ``member_source="curve"``：把每个成员特征当作一条细线。

    2. **均值曲线**：簇内成员在 ``data_curve`` 上的逐时间点均值，再用同源的
       ``fit_power_loglinear`` 拟合 ``a·t^b`` 后在原时间网格上采样。
    3. **CI 带**（可选）：``mean ± ci_z · SE``，``SE = std/√n``，按 ``data_curve`` 的列统计。

    Args:
        data_scatter: 长度 ``L`` 的 DataFrame 列表，行索引为时间，列为特征；用于成员层。
        data_curve: 长度 ``L`` 的 DataFrame 列表，用于均值/CI；为 ``None`` 时复用
            ``data_scatter``。
        labels: ``(N,)`` 的整型聚类标签（``np.ndarray`` / ``torch.Tensor`` 均可）。
        common_cols: 训练时使用的列（``FunClu.common_cols``）；为 ``None`` 时
            按 ``len(labels)`` 截前若干列。
        n_components: 簇数 ``K``；为 ``None`` 时按 ``labels.max() + 1`` 推断。
        condition_labels: 各 condition 的图例文字；为 ``None`` 时使用 ``Cond 1..L``。
        member_source: ``"qd_df"`` 或 ``"curve"``。
        show_members / show_mean / show_mean_ci: 各层显隐。
        ci_alpha / ci_z: CI 带的填充透明度与倍数（默认近似 95%）。
        use_semilogy / use_log_x: 坐标轴是否取对数；半对数 Y 时把非正值夹到 ``1e-10``
            避免对数报错。
        n_cols: 每行子图数；行数自动按 ``ceil(K / n_cols)`` 计算。
        subplot_hw: 单子图 ``(宽, 高)`` 英寸。
        palette: 各 condition 的 ``(成员色, 均值色)``；不足时循环填充。
        panel_title_prefix / panel_title_fontsize: 子图标题样式（``"M k (n)"``）。
        xlabel / ylabel / axis_label_fontsize: 共享坐标标签；``ylabel`` 为 ``None``
            时按 ``use_semilogy`` 自动给出。
        linewidth_mean / linewidth_member / markersize_qd / alpha_*: 视觉参数。
        x_margin / y_margin: 子图 margins。
        show_legend / legend_*: 顶部全局图例。
        dpi: 分辨率。
        show_in_streamlit: 是否直接 ``st.pyplot(fig)``。

    Returns:
        ``matplotlib.figure.Figure``。

    Raises:
        ValueError: ``data_scatter`` 为空、``data_curve`` 长度不匹配，或
            ``member_source`` 非法。
    """
    if not data_scatter:
        raise ValueError("data_scatter 不能为空")
    if member_source not in ("qd_df", "curve"):
        raise ValueError(
            f"member_source 必须为 'qd_df' 或 'curve'，当前为：{member_source!r}"
        )
    if data_curve is None:
        data_curve = data_scatter
    if len(data_curve) != len(data_scatter):
        raise ValueError(
            f"data_curve / data_scatter 长度不一致："
            f"{len(data_curve)} vs {len(data_scatter)}"
        )

    L = len(data_scatter)
    labels_np = (
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor)
        else np.asarray(labels)
    ).astype(np.int64)

    if n_components is None:
        n_components = int(labels_np.max()) + 1 if labels_np.size > 0 else 1
    K = int(n_components)

    if condition_labels is None:
        condition_labels = [f"Cond {i + 1}" for i in range(L)]
    else:
        condition_labels = list(condition_labels)
        if len(condition_labels) < L:
            condition_labels = condition_labels + [
                f"Cond {i + 1}" for i in range(len(condition_labels), L)
            ]

    pal: List[Tuple[str, str]] = (
        list(palette) if palette is not None else list(_CLUSTER_PROFILE_DEFAULT_PALETTE)
    )
    while len(pal) < L:
        pal.extend(_CLUSTER_PROFILE_DEFAULT_PALETTE)

    n_rows = (K + n_cols - 1) // n_cols
    sw, sh = subplot_hw
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(sw * n_cols, sh * n_rows),
        sharex=True,
        sharey=True,
        dpi=dpi,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.asarray(axes).reshape(1, n_cols)
    elif n_cols == 1:
        axes = np.asarray(axes).reshape(n_rows, 1)
    axes_flat = axes.flat

    y_label_eff = ylabel if ylabel is not None else (
        "Value (log scale)" if use_semilogy else "Value"
    )

    def _plot_y(ax_, x_, y_, *, fmt: str = "-", **kw) -> None:
        y_arr = np.asarray(y_, dtype=np.float64)
        if use_semilogy:
            y_plot = np.maximum(y_arr, 1e-10)
            ax_.semilogy(x_, y_plot, fmt, **kw)
        else:
            ax_.plot(x_, y_arr, fmt, **kw)

    def _select_member_columns(df: pd.DataFrame) -> pd.DataFrame:
        if common_cols is not None:
            cols = [c for c in common_cols if c in df.columns]
            return df[cols]
        return df.iloc[:, : labels_np.size]

    for k in range(K):
        ax = axes_flat[k]
        mask = labels_np == k
        n_in_cluster = int(mask.sum())

        if n_in_cluster == 0:
            ax.text(
                0.5,
                0.5,
                f"{panel_title_prefix} {k + 1}\n(empty)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=panel_title_fontsize,
                fontproperties=font_prop,
            )
            ax.grid(False)
            continue

        for i in range(L):
            df_scatter = data_scatter[i]
            df_curve = data_curve[i]
            t_scatter = df_scatter.index.values.astype(np.float64)
            t_curve = df_curve.index.values.astype(np.float64)
            scatter_color, line_color = pal[i % len(pal)]

            if show_members:
                df_for_members = (
                    df_scatter if member_source == "qd_df" else df_curve
                )
                t_for_members = (
                    t_scatter if member_source == "qd_df" else t_curve
                )
                df_for_members = _select_member_columns(df_for_members)
                cluster_data = df_for_members.iloc[:, mask]

                if member_source == "qd_df":
                    x_vals = np.repeat(t_for_members, n_in_cluster)
                    y_vals = cluster_data.values.ravel()
                    if use_semilogy:
                        valid = (x_vals > 0) & (y_vals > 0)
                    else:
                        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                    x_vals = x_vals[valid]
                    y_vals = y_vals[valid]
                    if use_semilogy:
                        ax.semilogy(
                            x_vals,
                            np.maximum(y_vals, 1e-10),
                            "o",
                            markerfacecolor="none",
                            markeredgecolor=scatter_color,
                            markersize=markersize_qd,
                            alpha=alpha_qd_marker,
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            x_vals,
                            y_vals,
                            "o",
                            markerfacecolor="none",
                            markeredgecolor=scatter_color,
                            markersize=markersize_qd,
                            alpha=alpha_qd_marker,
                            zorder=1,
                        )
                else:
                    for col in cluster_data.columns:
                        _plot_y(
                            ax,
                            t_for_members,
                            cluster_data[col].values,
                            fmt="-",
                            color=scatter_color,
                            linewidth=linewidth_member,
                            alpha=alpha_member_lines,
                            zorder=1,
                        )

            df_curve_sub = _select_member_columns(df_curve)
            cluster_data_curve = df_curve_sub.iloc[:, mask]
            mean_curve_points = cluster_data_curve.mean(axis=1).values.astype(np.float64)
            mean_x, mean_curve = _fit_mean_curve_power_sample(
                t_curve, mean_curve_points, t_curve
            )
            if not np.isfinite(mean_curve).any():
                mean_x = t_curve
                mean_curve = mean_curve_points
            if use_semilogy:
                mean_curve = np.maximum(mean_curve, 1e-10)

            if show_mean_ci and n_in_cluster > 0:
                std_curve = cluster_data_curve.std(axis=1).values.astype(np.float64)
                sem = std_curve / np.sqrt(max(n_in_cluster, 1))
                lo = mean_curve - ci_z * sem
                hi = mean_curve + ci_z * sem
                if use_semilogy:
                    lo = np.maximum(lo, 1e-10)
                    hi = np.maximum(hi, 1e-10)
                ax.fill_between(
                    t_curve,
                    lo,
                    hi,
                    color=line_color,
                    alpha=ci_alpha,
                    zorder=2,
                    linewidth=0,
                )

            if show_mean:
                _plot_y(
                    ax,
                    mean_x,
                    mean_curve,
                    fmt="-",
                    color=line_color,
                    linewidth=linewidth_mean,
                    label=condition_labels[i] if k == 0 else None,
                    zorder=3,
                )

        ax.text(
            0.5,
            0.97,
            f"{panel_title_prefix} {k + 1} ({n_in_cluster})",
            transform=ax.transAxes,
            fontsize=panel_title_fontsize,
            va="top",
            ha="center",
            fontweight="bold",
            fontproperties=font_prop,
        )
        ax.margins(x=x_margin, y=y_margin)
        ax.grid(False)
        if use_log_x:
            ax.set_xscale("log")
        _set_chinese_axes(ax)

    for idx in range(K, len(axes.flat)):
        fig.delaxes(axes.flat[idx])

    if show_legend:
        ncol_leg = legend_ncol if legend_ncol is not None else min(L, 6)
        handles = [
            plt.Line2D(
                [],
                [],
                color=pal[i % len(pal)][1],
                linewidth=linewidth_mean,
                label=condition_labels[i],
            )
            for i in range(L)
        ]
        fig.legend(
            handles=handles,
            loc=legend_loc,
            ncol=ncol_leg,
            fontsize=legend_fontsize,
            bbox_to_anchor=legend_bbox,
        )

    fig.text(
        0.5,
        0.01,
        xlabel,
        ha="center",
        fontsize=axis_label_fontsize,
        fontproperties=font_prop,
    )
    fig.text(
        0.005,
        0.5,
        y_label_eff,
        va="center",
        rotation="vertical",
        fontsize=axis_label_fontsize,
        fontproperties=font_prop,
    )

    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95) if show_legend else (0.02, 0.02, 1.0, 1.0))

    if show_in_streamlit:
        st.pyplot(fig, use_container_width=True)
    return fig
