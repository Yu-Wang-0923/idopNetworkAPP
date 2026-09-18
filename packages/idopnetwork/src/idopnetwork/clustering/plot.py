"""Functional Clustering 可视化模块。

实现进度：

- ``plot_initialization_grid``：给 ``FunClu._initialize`` 的初值做诊断网格图，
  支持 ``"k_by_l"``（K 行 × L 列）与 ``"l_by_k"``（L 行 × K 列）两种布局。
- ``plot_cluster_profiles``：EM 拟合完成后，按簇绘制各 condition 的成员
  曲线 + 成员逐点均值曲线 + 可选 CI 带。

绘图函数全部为模块级（不依赖 ``FunClu`` 实例方法），既符合项目分层
（``backend/plot_xxx.py`` 仅做绘图），又便于在 page 之外被复用。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from idopnetwork.curve_fitting import fit_power_loglinear

try:
    import torch
except ImportError:
    raise ImportError(
        "idopnetwork[ml] is required for clustering plots. "
        "Install with: pip install idopnetwork[ml]"
    )

font_prop = None  # Set by application layer for CJK font support


# 子图配色：(成员细线, KMeans 中心散点, 幂律拟合曲线)
_DEFAULT_INIT_PALETTE: List[Tuple[str, str, str]] = [
    ("#FAD7A0", "#D35400", "#A04000"),
    ("#AED6F1", "#2E86C1", "#1B4F72"),
    ("#A9DFBF", "#239B56", "#196F3D"),
    ("#F5B7B1", "#C0392B", "#922B21"),
    ("#D7BDE2", "#8E44AD", "#5B2C6F"),
    ("#A3E4D7", "#16A085", "#0E6655"),
]

# 簇 profile 配色：(成员细线色, 均值曲线色) —— 与 funclu_v4 参考实现对齐。
# 每对为「淡色成员线 + 同色系深色均值线」，三色循环；淡色承载成员云，
# 深色均值线压在最上层，保证跨簇一眼可比。
_CLUSTER_PROFILE_DEFAULT_PALETTE: List[Tuple[str, str]] = [
    ("#FFB3A6", "#D94E3D"),
    ("#A8D8EA", "#2B7A9E"),
    ("#C1E1C1", "#2E8B57"),
]


def _font_at(size: float, weight: Optional[str] = None) -> FontProperties:
    """返回带指定字号（可选字重）的项目中文字体。

    ``font_prop`` 由 app 层用 ``FontProperties(fname=...)`` 构造，**只带字体
    文件、字号是默认的 10**。把它直接交给 ``fontproperties=`` 会连同 ``fontsize=``
    与 ``fontweight=`` 一起被它覆盖回默认值。所以凡是要控制字号/字重的地方，
    都必须先写进 FontProperties 再传，不能指望 ``fontsize=``/``fontweight=`` 生效。
    """
    if font_prop is None:
        prop = FontProperties(size=size)
    else:
        prop = font_prop.copy()
        prop.set_size(size)
    if weight is not None:
        prop.set_weight(weight)
    return prop


def _legend_font(size: float) -> object:
    """返回指定字号的项目中文字体，用于 legend。"""
    return _font_at(size)


def _set_chinese_axes(ax: plt.Axes, labelsize: Optional[float] = None) -> None:
    """让 tick 使用项目中文字体。

    ``labelsize`` 为 ``None`` 时沿用 ``font_prop`` 自带字号（保持既有行为）；
    显式传入时按该字号渲染——否则 ``font_prop`` 会把 ``tick_params`` 的
    ``labelsize`` 覆盖回默认的 10。
    """
    prop = _font_at(labelsize) if labelsize is not None else font_prop
    if prop is None:
        return
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(prop)


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
    show_in_streamlit: bool = False,
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
                prop=_legend_font(9.0),
                bbox_to_anchor=(0.5, 0.99),
            )

    fig.tight_layout(rect=(0, 0, 1, 0.96) if show_legend else (0, 0, 1, 1))

    if show_in_streamlit:
        import streamlit as st
        st.pyplot(fig, use_container_width=True)

    return fig


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
    layout: str = "combined",
    n_cols: int = 3,
    subplot_hw: Tuple[float, float] = (4.0, 3.6),
    panel_box_aspect: Optional[float] = 0.75,
    palette: Optional[Sequence[Tuple[str, str]]] = None,
    title: str = "Cluster profiles",
    panel_title_prefix: str = "M",
    panel_title_fontsize: float = 18.0,
    tick_labelsize: float = 12.0,
    show_grid: bool = False,
    xlabel: str = "Index",
    ylabel: Optional[str] = None,
    axis_label_fontsize: float = 12.0,
    linewidth_mean: float = 3.5,
    linewidth_member: float = 1.5,
    markersize_qd: float = 7.0,
    alpha_member_lines: float = 0.85,
    alpha_qd_marker: float = 0.9,
    x_margin: float = 0.1,
    y_margin: float = 0.2,
    show_legend: bool = True,
    legend_loc: str = "upper center",
    legend_ncol: Optional[int] = None,
    legend_fontsize: float = 11.0,
    legend_bbox: Tuple[float, float] = (0.5, 0.96),
    dpi: int = 300,
    show_in_streamlit: bool = False,
) -> Figure:
    """按簇绘制 EM 拟合结果的 profile 图。

    ``layout="combined"`` 表示每个 cluster 一个子图，多 condition 叠加；
    ``"k_by_l"`` 表示 K 行 × L 列；``"l_by_k"`` 表示 L 行 × K 列。

    每个子图的**绘图区**默认锁成 **4:3**（``panel_box_aspect=0.75``），
    因此子图比例不受字体、图例、共享轴影响；整张大图的宽高比则随
    ``subplot_hw`` 与行列数自然变化。
    """
    if not data_scatter:
        raise ValueError("data_scatter 不能为空")
    if member_source not in ("qd_df", "curve"):
        raise ValueError(
            f"member_source 必须为 'qd_df' 或 'curve'，当前为：{member_source!r}"
        )
    if layout not in ("combined", "k_by_l", "l_by_k"):
        raise ValueError(
            "layout 必须为 'combined'、'k_by_l' 或 'l_by_k'，"
            f"当前为：{layout!r}"
        )
    if data_curve is None:
        data_curve = data_scatter
    if len(data_curve) != len(data_scatter):
        raise ValueError(
            f"data_curve / data_scatter 长度不一致："
            f"{len(data_curve)} vs {len(data_scatter)}"
        )

    n_conditions = len(data_scatter)
    labels_np = (
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor)
        else np.asarray(labels)
    ).astype(np.int64)

    if n_components is None:
        n_components = int(labels_np.max()) + 1 if labels_np.size > 0 else 1
    n_clusters = int(n_components)

    if condition_labels is None:
        condition_labels = [f"Cond {i + 1}" for i in range(n_conditions)]
    else:
        condition_labels = list(condition_labels)
        if len(condition_labels) < n_conditions:
            condition_labels = condition_labels + [
                f"Cond {i + 1}"
                for i in range(len(condition_labels), n_conditions)
            ]

    pal: List[Tuple[str, str]] = (
        list(palette) if palette is not None else list(_CLUSTER_PROFILE_DEFAULT_PALETTE)
    )
    while len(pal) < n_conditions:
        pal.extend(_CLUSTER_PROFILE_DEFAULT_PALETTE)

    n_cols = max(1, int(n_cols))
    if layout == "combined":
        n_cols_eff = min(n_cols, max(n_clusters, 1))
        n_rows = (n_clusters + n_cols_eff - 1) // n_cols_eff
    elif layout == "k_by_l":
        n_rows = n_clusters
        n_cols_eff = n_conditions
    else:
        n_rows = n_conditions
        n_cols_eff = n_clusters

    panel_w, panel_h = subplot_hw
    fig, axes = plt.subplots(
        n_rows,
        n_cols_eff,
        figsize=(panel_w * n_cols_eff, panel_h * n_rows),
        sharex=True,
        sharey=True,
        dpi=dpi,
    )
    if n_rows == 1 and n_cols_eff == 1:
        axes_arr = np.array([[axes]])
    elif n_rows == 1:
        axes_arr = np.asarray(axes).reshape(1, n_cols_eff)
    elif n_cols_eff == 1:
        axes_arr = np.asarray(axes).reshape(n_rows, 1)
    else:
        axes_arr = np.asarray(axes)

    y_label_eff = ylabel if ylabel is not None else (
        "Value (log scale)" if use_semilogy else "Value"
    )

    def _select_member_columns(df: pd.DataFrame) -> pd.DataFrame:
        if common_cols is not None:
            cols = [c for c in common_cols if c in df.columns]
            return df[cols]
        return df.iloc[:, : labels_np.size]

    def _plot_y(ax_: plt.Axes, x_, y_, *, fmt: str = "-", **kwargs) -> None:
        x_arr = np.asarray(x_, dtype=np.float64)
        y_arr = np.asarray(y_, dtype=np.float64)
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if use_log_x:
            valid &= x_arr > 0
        if use_semilogy:
            valid &= y_arr > 0
        if not valid.any():
            return

        x_plot = x_arr[valid]
        y_plot = y_arr[valid]
        if use_semilogy:
            ax_.semilogy(x_plot, np.maximum(y_plot, 1e-10), fmt, **kwargs)
        else:
            ax_.plot(x_plot, y_plot, fmt, **kwargs)

    def _style_axis(ax_: plt.Axes) -> None:
        if use_log_x:
            ax_.set_xscale("log")
        # 绘图区锁定宽高比（默认 3/4 → 4:3）。显式锁定后，子图比例不再受
        # 字体度量、图例/总标题留白、共享 y 轴等影响，与 funclu_v4 一致。
        if panel_box_aspect is not None:
            ax_.set_box_aspect(panel_box_aspect)
        # funclu_v4 风格：默认无网格、保留完整边框、刻度字号偏大
        # 注意：不能写成 grid(show_grid, linestyle=...)，matplotlib 在
        # 第一参数为 False 且带线型属性时会「反过来」打开网格并告警。
        if show_grid:
            ax_.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)
        else:
            ax_.grid(False)
        ax_.tick_params(labelsize=tick_labelsize)
        _set_chinese_axes(ax_, tick_labelsize)

    def _draw_cluster_condition(
        ax_: plt.Axes,
        *,
        cluster_idx: int,
        condition_idx: int,
        label_condition: bool,
    ) -> None:
        df_scatter = data_scatter[condition_idx]
        df_curve = data_curve[condition_idx]
        t_scatter = df_scatter.index.values.astype(np.float64)
        t_curve = df_curve.index.values.astype(np.float64)
        scatter_color, line_color = pal[condition_idx % len(pal)]
        mask = labels_np == cluster_idx
        n_in_cluster = int(mask.sum())

        if show_members:
            df_for_members = df_scatter if member_source == "qd_df" else df_curve
            t_for_members = t_scatter if member_source == "qd_df" else t_curve
            df_for_members = _select_member_columns(df_for_members)
            cluster_data = df_for_members.iloc[:, mask]

            if member_source == "qd_df":
                x_vals = np.repeat(t_for_members, n_in_cluster)
                y_vals = cluster_data.values.ravel()
                valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                if use_log_x:
                    valid &= x_vals > 0
                if use_semilogy:
                    valid &= y_vals > 0
                if valid.any():
                    x_vals = x_vals[valid]
                    y_vals = y_vals[valid]
                    if use_semilogy:
                        ax_.semilogy(
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
                        ax_.plot(
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
                        ax_,
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
        mean_points = cluster_data_curve.mean(axis=1).values.astype(np.float64)
        mean_x = t_curve
        mean_curve = mean_points

        if show_mean_ci and n_in_cluster > 0:
            std_curve = cluster_data_curve.std(axis=1).values.astype(np.float64)
            sem = std_curve / np.sqrt(max(n_in_cluster, 1))
            lo = mean_curve - ci_z * sem
            hi = mean_curve + ci_z * sem
            valid = np.isfinite(t_curve) & np.isfinite(lo) & np.isfinite(hi)
            if use_log_x:
                valid &= t_curve > 0
            if use_semilogy:
                valid &= hi > 0
                lo = np.maximum(lo, 1e-10)
                hi = np.maximum(hi, 1e-10)
            if valid.any():
                ax_.fill_between(
                    t_curve[valid],
                    lo[valid],
                    hi[valid],
                    color=line_color,
                    alpha=ci_alpha,
                    zorder=2,
                    linewidth=0,
                )

        if show_mean:
            _plot_y(
                ax_,
                mean_x,
                mean_curve,
                fmt="-",
                color=line_color,
                linewidth=linewidth_mean,
                label=condition_labels[condition_idx] if label_condition else None,
                zorder=10,
            )

    def _draw_panel(
        ax_: plt.Axes,
        *,
        cluster_idx: int,
        condition_indexes: Sequence[int],
        title_suffix: str = "",
        show_panel_title: bool = True,
    ) -> None:
        n_in_cluster = int((labels_np == cluster_idx).sum())
        if n_in_cluster == 0:
            empty_title = f"{panel_title_prefix}{cluster_idx + 1}\n(empty)"
            if title_suffix:
                empty_title = f"{empty_title}\n{title_suffix}"
            ax_.text(
                0.5,
                0.5,
                empty_title,
                transform=ax_.transAxes,
                ha="center",
                va="center",
                fontproperties=_font_at(panel_title_fontsize),
            )
            _style_axis(ax_)
            return

        for condition_idx in condition_indexes:
            _draw_cluster_condition(
                ax_,
                cluster_idx=cluster_idx,
                condition_idx=condition_idx,
                label_condition=(layout == "combined" and cluster_idx == 0),
            )

        if show_panel_title:
            panel_title = f"{panel_title_prefix}{cluster_idx + 1}({n_in_cluster})"
            if title_suffix:
                panel_title = f"{panel_title} | {title_suffix}"
            ax_.set_title(
                panel_title,
                fontproperties=_font_at(panel_title_fontsize),
            )
        ax_.margins(x=x_margin, y=y_margin)
        _style_axis(ax_)

    if layout == "combined":
        axes_flat = list(axes_arr.flat)
        for cluster_idx in range(n_clusters):
            _draw_panel(
                axes_flat[cluster_idx],
                cluster_idx=cluster_idx,
                condition_indexes=range(n_conditions),
            )
        for idx in range(n_clusters, len(axes_flat)):
            fig.delaxes(axes_flat[idx])
    elif layout == "k_by_l":
        for cluster_idx in range(n_clusters):
            for condition_idx in range(n_conditions):
                _draw_panel(
                    axes_arr[cluster_idx, condition_idx],
                    cluster_idx=cluster_idx,
                    condition_indexes=[condition_idx],
                    show_panel_title=False,
                )
        for condition_idx, condition_label in enumerate(condition_labels):
            axes_arr[0, condition_idx].set_title(
                str(condition_label),
                fontproperties=_font_at(panel_title_fontsize),
            )
        for cluster_idx in range(n_clusters):
            n_in_cluster = int((labels_np == cluster_idx).sum())
            axes_arr[cluster_idx, 0].set_ylabel(
                f"{panel_title_prefix}{cluster_idx + 1}\n({n_in_cluster})",
                fontproperties=_font_at(axis_label_fontsize),
            )
        for condition_idx in range(n_conditions):
            axes_arr[-1, condition_idx].set_xlabel(
                xlabel, fontproperties=font_prop
            )
    else:
        for condition_idx in range(n_conditions):
            for cluster_idx in range(n_clusters):
                _draw_panel(
                    axes_arr[condition_idx, cluster_idx],
                    cluster_idx=cluster_idx,
                    condition_indexes=[condition_idx],
                    show_panel_title=False,
                )
        for cluster_idx in range(n_clusters):
            n_in_cluster = int((labels_np == cluster_idx).sum())
            axes_arr[0, cluster_idx].set_title(
                f"{panel_title_prefix}{cluster_idx + 1}({n_in_cluster})",
                fontproperties=_font_at(panel_title_fontsize),
            )
            axes_arr[-1, cluster_idx].set_xlabel(
                xlabel, fontproperties=font_prop
            )
        for condition_idx, condition_label in enumerate(condition_labels):
            axes_arr[condition_idx, 0].set_ylabel(
                f"{condition_label}\n{y_label_eff}",
                fontproperties=_font_at(axis_label_fontsize),
            )

    if title:
        fig.suptitle(
            title,
            fontproperties=_font_at(axis_label_fontsize + 2, "bold"),
            y=0.995,
        )

    if show_legend:
        ncol_leg = legend_ncol if legend_ncol is not None else min(n_conditions, 6)
        handles = [
            plt.Line2D(
                [],
                [],
                color=pal[i % len(pal)][1],
                linewidth=linewidth_mean,
                label=condition_labels[i],
            )
            for i in range(n_conditions)
        ]
        fig.legend(
            handles=handles,
            loc=legend_loc,
            ncol=ncol_leg,
            prop=_legend_font(legend_fontsize),
            bbox_to_anchor=legend_bbox,
        )

    if layout == "combined":
        fig.text(
            0.5,
            0.01,
            xlabel,
            ha="center",
            fontproperties=_font_at(axis_label_fontsize),
        )
        fig.text(
            0.005,
            0.5,
            y_label_eff,
            va="center",
            rotation="vertical",
            fontproperties=_font_at(axis_label_fontsize),
        )

    # 边距只按「真正会渲染的内容」预留：v4 用的是裸 tight_layout()，不做任何
    # 固定预留。之前无论有没有图例/总标题都固定留 0.89/0.93，会把面板压扁。
    if layout == "combined":
        # combined 用 fig.text 画共用 x/y 标签，需要底部与左侧边距
        left, bottom = 0.035, 0.045
    else:
        # k_by_l / l_by_k 的标签挂在各自 axes 上，无需额外预留
        left, bottom = 0.015, 0.02
    if show_legend:
        top = 0.89
    elif title:
        top = 0.93
    else:
        top = 0.995
    fig.tight_layout(rect=(left, bottom, 0.995, top))

    if show_in_streamlit:
        import streamlit as st
        st.pyplot(fig, use_container_width=True)
    return fig


def plot_cluster_profiles_per_cluster(
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
    use_semilogy: bool = False,
    use_log_x: bool = False,
    cluster_ids: Optional[Sequence[int]] = None,
    max_clusters: Optional[int] = None,
    palette: Optional[Sequence[Tuple[str, str]]] = None,
    panel_prefix: str = "M",
    panel_figsize: Tuple[float, float] = (4.0, 3.4),
    panel_box_aspect: Optional[float] = 0.75,
    title_fontsize: float = 18.0,
    tick_labelsize: float = 12.0,
    show_grid: bool = False,
    linewidth_mean: float = 3.5,
    linewidth_member: float = 1.5,
    markersize_qd: float = 7.0,
    alpha_member_lines: float = 0.85,
    alpha_qd_marker: float = 0.9,
    x_margin: float = 0.1,
    share_y: bool = True,
    share_y_pad: float = 0.05,
    dpi: int = 300,
    save_dir: Optional[str | Path] = None,
    show_legend: bool = False,
    legend_fontsize: float = 11.0,
    show_in_streamlit: bool = False,
) -> List[Figure]:
    """每个 cluster 单独输出一张 ``panel_figsize`` 图，复刻 funclu_v4 的排版。

    与 :func:`plot_cluster_profiles` 的网格版不同，这里是 **一簇一图**：

    - 每张图固定 ``panel_figsize``（默认 ``(4, 3)``，与 v4 的 ``figsize=(4, 3)``
      一致），因此绘图区比例天然与 v4 对齐；
    - ``share_y=True`` 时先扫描**全部簇**的成员曲线与均值曲线，取统一 y 范围
      （``share_y_pad`` 比例留白，默认 5%），使各张图可以直接横向比较——这正是
      v4 ``plot_cluster(share_y=True)`` 的行为；
    - 绘制顺序复刻 v4：先画所有 condition 的成员曲线（``zorder=1``），最后统一
      画均值曲线（``zorder=10``），保证均值始终压在最上层；
    - 标题为 v4 的紧凑写法 ``M1(23)``，默认无网格、无图例、无轴标签。

    Args:
        data_scatter, data_curve, labels, common_cols, n_components,
        condition_labels, member_source, show_members, show_mean,
        show_mean_ci, ci_alpha, ci_z, use_semilogy, use_log_x:
            含义与 :func:`plot_cluster_profiles` 完全一致。
        cluster_ids: 只画这些簇编号（0 基）；为 ``None`` 时画全部。
        max_clusters: 在 ``cluster_ids`` 之后再做一次截断，只画前 N 个簇。
        palette: ``(成员线色, 均值线色)`` 序列，默认用 v4 的三组配色。
        panel_prefix: 簇标签前缀，默认 ``"M"`` → ``M1``、``M2``…
        panel_figsize: 单张图的英寸尺寸，默认 ``(4, 3.4)``。
        panel_box_aspect: 绘图区高宽比（height/width），默认 ``0.75`` 即 **4:3**。
            显式锁定后子图比例不再受字体度量、图例/总标题留白、共享 y 轴影响；
            置 ``None`` 则交回 matplotlib 自动排布。
        title_fontsize, tick_labelsize: 标题与刻度字号，默认 18 / 12（同 v4）。
        show_grid: 是否显示网格，默认关闭（同 v4）。
        linewidth_mean, linewidth_member, alpha_member_lines:
            均值线宽 / 成员线宽 / 成员线透明度，默认 3.5 / 1.5 / 0.85（同 v4）。
        x_margin: x 方向留白比例。
        share_y: 是否让所有簇共用同一 y 范围（默认 ``True``）。
        share_y_pad: 共享 y 范围上下各留白的比例，默认 0.05。
        dpi: 图片分辨率，默认 300（同 v4）。
        save_dir: 给定则把每张图存成 ``{panel_prefix}{k}.png``（v4 的落盘行为）。
        show_legend: 是否在坐标轴内画图例，默认关闭。
        show_in_streamlit: 为 ``True`` 时逐张 ``st.pyplot``。

    Returns:
        每簇一个 :class:`~matplotlib.figure.Figure` 的列表，顺序与簇编号一致。
        调用方负责在渲染后 ``plt.close(fig)``。
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

    n_conditions = len(data_scatter)
    labels_np = (
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor)
        else np.asarray(labels)
    ).astype(np.int64)
    if n_components is None:
        n_components = int(labels_np.max()) + 1 if labels_np.size > 0 else 1
    n_clusters = int(n_components)

    if condition_labels is None:
        condition_labels = [f"Cond {i + 1}" for i in range(n_conditions)]
    else:
        condition_labels = list(condition_labels)
        while len(condition_labels) < n_conditions:
            condition_labels.append(f"Cond {len(condition_labels) + 1}")

    pal: List[Tuple[str, str]] = (
        list(palette) if palette is not None else list(_CLUSTER_PROFILE_DEFAULT_PALETTE)
    )
    if not pal:
        pal = list(_CLUSTER_PROFILE_DEFAULT_PALETTE)
    while len(pal) < n_conditions:
        pal.extend(_CLUSTER_PROFILE_DEFAULT_PALETTE)

    def _member_frame(condition_idx: int) -> pd.DataFrame:
        frame = (
            data_scatter[condition_idx]
            if member_source == "qd_df"
            else data_curve[condition_idx]
        )
        if common_cols is not None:
            cols = [c for c in common_cols if c in frame.columns]
            return frame[cols]
        return frame.iloc[:, : labels_np.size]

    def _finite(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if use_semilogy:
            vals = vals[vals > 0.0]
        return vals

    # 共享 y 范围：与 v4 一样扫描【全部】簇（而非仅选中的簇），
    # 这样即使只导出部分图，y 轴口径仍然一致、可跨图比较。
    global_ylim: Optional[Tuple[float, float]] = None
    if share_y:
        vmin, vmax = np.inf, -np.inf
        for k in range(n_clusters):
            mask_k = labels_np == k
            if not mask_k.any():
                continue
            for condition_idx in range(n_conditions):
                sub = _member_frame(condition_idx).iloc[:, mask_k]
                if sub.shape[1] == 0:
                    continue
                members = _finite(sub.to_numpy(float))
                if members.size:
                    vmin = min(vmin, float(members.min()))
                    vmax = max(vmax, float(members.max()))
                means = _finite(sub.mean(axis=1).to_numpy(float))
                if means.size:
                    vmin = min(vmin, float(means.min()))
                    vmax = max(vmax, float(means.max()))
        if vmin == np.inf:
            vmin, vmax = 0.0, 1.0
        span = vmax - vmin
        pad = span * share_y_pad if span > 0 else 1.0
        lo, hi = vmin - pad, vmax + pad
        if use_semilogy and lo <= 0:
            lo = vmin / (1.0 + share_y_pad) if vmin > 0 else 1e-12
        global_ylim = (lo, hi)

    if cluster_ids is not None:
        wanted = {int(c) for c in cluster_ids}
        selected = [k for k in range(n_clusters) if k in wanted]
    else:
        selected = list(range(n_clusters))
    if max_clusters is not None:
        selected = selected[: max(0, int(max_clusters))]

    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    figures: List[Figure] = []
    for k in selected:
        mask = labels_np == k
        n_members = int(mask.sum())
        fig, ax = plt.subplots(figsize=panel_figsize, dpi=dpi)
        if use_log_x:
            ax.set_xscale("log")
        if use_semilogy:
            ax.set_yscale("log")

        if n_members == 0:
            ax.text(
                0.5, 0.5,
                f"{panel_prefix}{k + 1}\n(empty)",
                transform=ax.transAxes, ha="center", va="center",
                fontproperties=_font_at(title_fontsize),
            )
        else:
            # 1) 先画所有 condition 的成员曲线（v4：成员在下）
            if show_members:
                for condition_idx in range(n_conditions):
                    member_color = pal[condition_idx % len(pal)][0]
                    frame = _member_frame(condition_idx)
                    sub = frame.iloc[:, mask]
                    if sub.shape[1] == 0:
                        continue
                    if member_source == "qd_df":
                        times = frame.index.to_numpy(float)
                        x_vals = np.repeat(times, n_members)
                        y_vals = sub.to_numpy(float).ravel()
                        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                        if use_semilogy:
                            valid &= y_vals > 0.0
                        if use_log_x:
                            valid &= x_vals > 0.0
                        if valid.any():
                            ax.plot(
                                x_vals[valid], y_vals[valid], "o",
                                markerfacecolor="none",
                                markeredgecolor=member_color,
                                markersize=markersize_qd,
                                alpha=alpha_qd_marker,
                                linestyle="none",
                                zorder=1,
                            )
                    else:
                        times = frame.index.to_numpy(float)
                        for col in sub.columns:
                            y_vals = sub[col].to_numpy(float)
                            valid = np.isfinite(times) & np.isfinite(y_vals)
                            if use_semilogy:
                                valid &= y_vals > 0.0
                            if use_log_x:
                                valid &= times > 0.0
                            if not valid.any():
                                continue
                            ax.plot(
                                times[valid], y_vals[valid],
                                color=member_color,
                                linewidth=linewidth_member,
                                alpha=alpha_member_lines,
                                zorder=1,
                            )

            # 2) 再统一画均值曲线，保证压在成员之上（v4：均值 zorder=10）
            for condition_idx in range(n_conditions):
                mean_color = pal[condition_idx % len(pal)][1]
                frame = _member_frame(condition_idx)
                sub = frame.iloc[:, mask]
                if sub.shape[1] == 0:
                    continue
                times = frame.index.to_numpy(float)
                mean_vals = sub.mean(axis=1).to_numpy(float)

                if show_mean_ci:
                    sem = sub.std(axis=1).to_numpy(float) / np.sqrt(max(n_members, 1))
                    lo_band, hi_band = mean_vals - ci_z * sem, mean_vals + ci_z * sem
                    valid = np.isfinite(times) & np.isfinite(lo_band) & np.isfinite(hi_band)
                    if use_semilogy:
                        valid &= hi_band > 0.0
                        lo_band = np.maximum(lo_band, 1e-10)
                        hi_band = np.maximum(hi_band, 1e-10)
                    if use_log_x:
                        valid &= times > 0.0
                    if valid.any():
                        ax.fill_between(
                            times[valid], lo_band[valid], hi_band[valid],
                            color=mean_color, alpha=ci_alpha, zorder=2, linewidth=0,
                        )

                if show_mean:
                    valid = np.isfinite(times) & np.isfinite(mean_vals)
                    if use_semilogy:
                        valid &= mean_vals > 0.0
                    if use_log_x:
                        valid &= times > 0.0
                    if valid.any():
                        ax.plot(
                            times[valid], mean_vals[valid],
                            color=mean_color,
                            linewidth=linewidth_mean,
                            label=condition_labels[condition_idx],
                            zorder=10,
                        )

        if global_ylim is not None:
            ax.set_ylim(*global_ylim)
            ax.margins(x=x_margin)
        else:
            ax.margins(x=x_margin, y=0.2)
        if panel_box_aspect is not None:
            ax.set_box_aspect(panel_box_aspect)

        ax.set_title(
            f"{panel_prefix}{k + 1}({n_members})",
            fontproperties=_font_at(title_fontsize),
        )
        if show_grid:
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)
        else:
            ax.grid(False)
        ax.tick_params(labelsize=tick_labelsize)
        _set_chinese_axes(ax, tick_labelsize)
        if show_legend:
            ax.legend(prop=_legend_font(legend_fontsize), frameon=False)

        fig.tight_layout()
        if out_dir is not None:
            fig.savefig(
                out_dir / f"{panel_prefix}{k + 1}.png",
                dpi=dpi, bbox_inches="tight",
            )
        if show_in_streamlit:
            import streamlit as st
            st.pyplot(fig, use_container_width=True)
        figures.append(fig)

    return figures


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


def plot_bic_elbow(
    *,
    bic_results,  # pd.DataFrame
    best_K: Optional[int] = None,
    candidate_Ks: Optional[Sequence[int]] = None,
    auto_select_best: bool = True,
    figsize: Tuple[float, float] = (8.0, 5.0),
    dpi: int = 200,
    show_in_streamlit: bool = False,
    title: str = "BIC Elbow Plot",
    xlabel: str = "Number of clusters (K)",
    ylabel: str = "BIC",
    best_k_color: str = "#e74c3c",
    line_color: str = "#2c3e50",
    marker_color: str = "#2c3e50",
) -> Figure:
    """BIC elbow plot for choosing the optimal number of clusters.

    Plots BIC vs K as a connected line with circular markers.  The best K
    (lowest BIC) is highlighted with a distinct marker and annotation.

    Args:
        bic_results: DataFrame with columns ``K`` and ``BIC`` (from
            :func:`~idopnetwork.clustering.funclu.compute_bic_scores`).
        best_K: K value to highlight.  If *None*, auto-selected from
            the minimum BIC among converged rows.
        candidate_Ks: Optional K values to mark as checkmark-like candidates.
        auto_select_best: If False, do not auto-highlight a best K when
            ``best_K`` is None.
        figsize: Figure size in inches.
        dpi: Figure resolution.
        show_in_streamlit: If *True*, render via ``st.pyplot`` and also
            return the Figure.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        best_k_color: Color for the best-K marker and annotation.
        line_color: Color for the BIC line.
        marker_color: Fill color for the regular BIC markers.

    Returns:
        Matplotlib Figure.
    """
    df = bic_results
    df_valid = df.dropna(subset=["BIC"]).copy()
    if df_valid.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No valid BIC results", ha="center", va="center",
                transform=ax.transAxes, fontproperties=font_prop)
        if show_in_streamlit:
            import streamlit as st
            st.pyplot(fig, use_container_width=True)
        return fig

    ks = df_valid["K"].values.astype(int)
    bics = df_valid["BIC"].values

    if best_K is None and auto_select_best:
        converged = df_valid[df_valid["converged"] == True]
        if not converged.empty:
            best_K = int(converged.loc[converged["BIC"].idxmin(), "K"])
        else:
            best_K = int(ks[bics.argmin()])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(ks, bics, "-o", color=line_color, markerfacecolor=marker_color,
            markersize=8, linewidth=2, zorder=2, label="BIC")

    candidate_set = {int(k) for k in (candidate_Ks or [])}
    candidate_x = [int(k) for k in ks if int(k) in candidate_set]
    candidate_y = [bics[ks.tolist().index(k)] for k in candidate_x]
    if candidate_x:
        ax.plot(
            candidate_x,
            candidate_y,
            "o",
            color=best_k_color,
            markersize=11,
            zorder=3,
            label="Checkmark candidates",
        )

    if best_K is not None and int(best_K) in ks.tolist():
        best_k_int = int(best_K)
        best_bic = bics[ks.tolist().index(best_k_int)]
        ax.plot([best_k_int], [best_bic], "o", color=best_k_color,
                markersize=14, zorder=4)
        ax.annotate(
            f"Best K={best_k_int}",
            xy=(best_k_int, best_bic),
            xytext=(best_k_int + 1.2, best_bic),
            fontsize=11,
            fontweight="bold",
            color=best_k_color,
            fontproperties=font_prop,
            arrowprops=dict(arrowstyle="->", color=best_k_color, lw=1.5),
        )

    ax.set_xlabel(xlabel, fontproperties=font_prop)
    ax.set_ylabel(ylabel, fontproperties=font_prop)
    ax.set_title(title, fontweight="bold", fontproperties=font_prop)
    ax.set_xticks(ks)
    ax.grid(True, linestyle=":", alpha=0.5)
    _set_chinese_axes(ax)

    fig.tight_layout()

    if show_in_streamlit:
        import streamlit as st
        st.pyplot(fig, use_container_width=True)
    return fig
