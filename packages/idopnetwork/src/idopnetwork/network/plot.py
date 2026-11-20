"""Network Construction 绘图模块。

Direction rule
--------------
当前 APP 统一使用：

    adj_df.loc[source, target] = source -> target

也就是：行节点指向列节点。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font_prop = None  # Set by application layer for CJK font support


# =============================================================================
# 1. 通用工具：邻接矩阵校验与标准化
# =============================================================================

def _normalize_adjacency_matrix(adj_df: pd.DataFrame) -> pd.DataFrame:
    """标准化邻接矩阵：index/columns 转字符串，并保证方阵同序。"""
    if not isinstance(adj_df, pd.DataFrame):
        raise TypeError("adj_df must be a pandas DataFrame.")

    if adj_df.shape[0] != adj_df.shape[1]:
        raise ValueError("adj_df must be a square matrix.")

    out = adj_df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)

    if set(out.index) != set(out.columns):
        raise ValueError("adj_df.index and adj_df.columns must contain the same node names.")

    # 保持 index 顺序，columns 按 index 对齐
    out = out.loc[list(out.index), list(out.index)]
    out = out.apply(pd.to_numeric, errors="raise")

    if out.isna().values.any():
        raise ValueError("adj_df contains missing values.")

    return out


# =============================================================================
# 2. 绘图函数：邻接矩阵热图与效应分解图
# =============================================================================
def plot_adjusted_matrix_heatmap(
    adj_df: pd.DataFrame,
    *,
    title: str = "Adjusted Matrix Heatmap",
    show_values: bool = False,
    zero_eps: float = 1e-12,
    normalize: bool = True,
) -> None:
    """绘制邻接矩阵热图。

    Direction rule:
        adj_df.loc[source, target] = source -> target

    显示规则：
    - 非对角线：边权
        正值 = 红色
        负值 = 蓝色
        绝对值越大颜色越深
        0 = 白色 / 空白
        normalize=True 时按非对角线最大绝对值缩放到 [-1, 1]
    - 对角线：自身效应绝对值
        先取 abs
        绝对值越大越黑
        绝对值越小越白
        normalize=True 时按对角线最大绝对值缩放到 [0, 1]
    """
    try:
        adj_df = _normalize_adjacency_matrix(adj_df)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
        ax.text(
            0.5,
            0.5,
            f"邻接矩阵错误：{e}",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            transform=ax.transAxes,
            fontproperties=font_prop,
        )
        ax.axis("off")
        return fig

    raw = adj_df.to_numpy(dtype=float)
    n = raw.shape[0]

    diag_mask = np.eye(n, dtype=bool)
    offdiag_mask = ~diag_mask

    # =====================================================
    # 1. 非对角线：边权，正红负蓝
    # =====================================================
    offdiag_raw = np.where(offdiag_mask, raw, np.nan)
    offdiag_raw = np.where(np.abs(offdiag_raw) > zero_eps, offdiag_raw, np.nan)

    if np.isfinite(offdiag_raw).any():
        offdiag_max_abs = float(np.nanmax(np.abs(offdiag_raw)))
        if offdiag_max_abs < 1e-12:
            offdiag_max_abs = 1.0
    else:
        offdiag_max_abs = 1.0

    if normalize:
        offdiag_plot = offdiag_raw / offdiag_max_abs
        offdiag_vmin = -1.0
        offdiag_vmax = 1.0
    else:
        offdiag_plot = offdiag_raw
        offdiag_vmin = -offdiag_max_abs
        offdiag_vmax = offdiag_max_abs

    offdiag_ma = np.ma.masked_invalid(offdiag_plot)

    cmap_off = plt.get_cmap("bwr").copy()  # blue-white-red
    cmap_off.set_bad(color=(1, 1, 1, 0))

    # =====================================================
    # 2. 对角线：自身效应绝对值，灰度
    # =====================================================
    diag_abs = np.where(diag_mask, np.abs(raw), np.nan)
    diag_abs = np.where(np.abs(diag_abs) > zero_eps, diag_abs, np.nan)

    if np.isfinite(diag_abs).any():
        diag_max = float(np.nanmax(diag_abs))
        if diag_max < 1e-12:
            diag_max = 1.0
    else:
        diag_max = 1.0

    if normalize:
        diag_plot = diag_abs / diag_max
        diag_vmin = 0.0
        diag_vmax = 1.0
    else:
        diag_plot = diag_abs
        diag_vmin = 0.0
        diag_vmax = diag_max

    diag_ma = np.ma.masked_invalid(diag_plot)

    cmap_diag = plt.get_cmap("Greys").copy()
    cmap_diag.set_bad(color=(1, 1, 1, 0))

    # =====================================================
    # 3. 画图
    # =====================================================
    fig_size = max(6.5, min(14.0, 0.42 * n + 4.2))
    fig = plt.figure(
        figsize=(fig_size + 2.2, fig_size),
        dpi=220,
        facecolor="white",
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[18, 0.7, 0.7],
        wspace=0.30,
    )

    ax = fig.add_subplot(gs[0, 0])
    cax_off = fig.add_subplot(gs[0, 1])
    cax_diag = fig.add_subplot(gs[0, 2])

    ax.set_facecolor("#FAFAFA")

    # 先画非对角线边权
    im_off = ax.imshow(
        offdiag_ma,
        cmap=cmap_off,
        vmin=offdiag_vmin,
        vmax=offdiag_vmax,
        aspect="equal",
        interpolation="none",
    )

    # 再覆盖对角线自身效应
    im_diag = ax.imshow(
        diag_ma,
        cmap=cmap_diag,
        vmin=diag_vmin,
        vmax=diag_vmax,
        aspect="equal",
        interpolation="none",
    )

    source_names = adj_df.index.astype(str).tolist()
    target_names = adj_df.columns.astype(str).tolist()

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    label_fs = 9 if n <= 15 else 7 if n <= 35 else 6

    ax.set_xticklabels(
        target_names,
        rotation=90,
        fontsize=label_fs,
        fontproperties=font_prop,
    )
    ax.set_yticklabels(
        source_names,
        fontsize=label_fs,
        fontproperties=font_prop,
    )

    ax.set_xlabel("Target node", fontsize=11, fontproperties=font_prop)
    ax.set_ylabel("Source node", fontsize=11, fontproperties=font_prop)

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=12,
        fontproperties=font_prop,
    )

    # 网格线
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # =====================================================
    # 4. 可选显示原始数值
    # =====================================================
    if show_values and n <= 20:
        for i in range(n):
            for j in range(n):
                val = raw[i, j]
                if abs(val) <= zero_eps:
                    continue

                if i == j:
                    norm_v = abs(val) / diag_max if diag_max > 0 else 0.0
                else:
                    norm_v = abs(val) / offdiag_max_abs if offdiag_max_abs > 0 else 0.0

                txt_color = "white" if norm_v > 0.55 else "black"

                ax.text(
                    j,
                    i,
                    f"{val:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                    fontproperties=font_prop,
                )

    # =====================================================
    # 5. Colorbar
    # =====================================================
    cb_off = fig.colorbar(im_off, cax=cax_off)
    if normalize:
        cb_off.set_label(
            f"Edge weight / {offdiag_max_abs:.2g}",
            fontsize=9,
            fontproperties=font_prop,
        )
    else:
        cb_off.set_label(
            "Edge weight",
            fontsize=9,
            fontproperties=font_prop,
        )
    for lab in cb_off.ax.get_yticklabels():
        lab.set_fontproperties(font_prop)

    cb_diag = fig.colorbar(im_diag, cax=cax_diag)
    if normalize:
        cb_diag.set_label(
            f"|Self effect| / {diag_max:.2g}",
            fontsize=9,
            fontproperties=font_prop,
        )
    else:
        cb_diag.set_label(
            "|Self effect|",
            fontsize=9,
            fontproperties=font_prop,
        )
    for lab in cb_diag.ax.get_yticklabels():
        lab.set_fontproperties(font_prop)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CFCFCF")
        spine.set_linewidth(1.0)

    fig.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.22)

    return fig


def plot_effect(
    quasi_dynamic_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    effect_df_list: list[pd.DataFrame],
    intercept: pd.Series | None = None,
    log_log: bool = False,
    plot_ncols: int = 4,
    plot_max_vars: int = 0,
) -> None:
    """
    在子图网格上叠加 effect。

    - 原始 quasi-dynamic 数据：橙色散点
    - 模型预测：蓝色粗线
    - 自效应：红色线，并加截距
    - 非自身效应：每个 source 使用不同颜色（且不使用红色）
    - 全局图例中逐一列出每个非自身效应 source，对应颜色与图中一致
    """
    
    ncols = max(1, int(plot_ncols))

    cols = list(quasi_dynamic_df.columns)
    if plot_max_vars > 0:
        cols = cols[:plot_max_vars]

    n = len(cols)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.2 * nrows),
        dpi=300,
    )
    axs = np.array(axs).reshape(-1)

    # 固定主颜色
    color_self = "#D62728"   # 自效应专用红色
    color_pred = "#1F77B4"   # 模型预测蓝色
    color_data = "#F59E0B"   # 数据橙色

    # 非自身效应颜色池：刻意避开红色
    nonself_palette = [
        "#2CA02C",  # 绿
        "#9467BD",  # 紫
        "#17BECF",  # 青
        "#8C564B",  # 棕
        "#7F7F7F",  # 灰
        "#BCBD22",  # 橄榄
        "#1B9E77",  # 深青绿
        "#66A61E",  # 草绿
        "#6A3D9A",  # 深紫
        "#A6761D",  # 褐黄
        "#4D908E",  # 蓝绿
        "#577590",  # 灰蓝
    ]

    # 为每个 source 固定一种非红色
    all_sources = list(quasi_dynamic_df.columns)
    cross_color_map = {
        src: nonself_palette[i % len(nonself_palette)]
        for i, src in enumerate(all_sources)
    }

    # 全局图例句柄
    base_handles: list[tuple] = []
    used_cross_sources: list[str] = []
    used_cross_source_set: set[str] = set()

    for idx, c in enumerate(cols):
        ax = axs[idx]

        # 原始数据
        h_scatter = ax.scatter(
            quasi_dynamic_df.index,
            quasi_dynamic_df[c],
            color=color_data,
            alpha=0.72,
            s=42,
            zorder=1,
            label="数据",
        )

        # 模型预测
        h_pred = ax.plot(
            curve_df.index,
            curve_df[c],
            color=color_pred,
            linestyle="-",
            linewidth=2.5,
            zorder=3,
            label="模型预测（Σ效应）",
        )[0]

        eff_df = effect_df_list[idx]
        icpt = (
            float(intercept.loc[c])
            if intercept is not None and c in intercept.index
            else 0.0
        )

        h_self_line = None

        for src in eff_df.columns:
            y = eff_df[src].to_numpy(dtype=float)

            # 自效应：唯一红线
            if src == c:
                y = y + icpt
                h_self_line = ax.plot(
                    eff_df.index,
                    y,
                    color=color_self,
                    linewidth=2.8,
                    alpha=0.95,
                    zorder=4,
                    label=f"自效应：{src}",
                )[0]
            else:
                if np.allclose(y, 0.0, rtol=0.0, atol=1e-12):
                    continue

                cross_color = cross_color_map.get(src, "#2CA02C")

                ax.plot(
                    eff_df.index,
                    y,
                    color=cross_color,
                    linewidth=1.8,
                    alpha=0.88,
                    zorder=2,
                    label=f"{src} → {c}",
                )

                # 记录全局图例中真正出现过的非自身效应 source
                if src not in used_cross_source_set:
                    used_cross_source_set.add(src)
                    used_cross_sources.append(src)

        ax.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
            zorder=0,
        )

        if log_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_title(c, fontproperties=font_prop, fontsize=10)
        ax.tick_params(labelsize=8)

        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontproperties(font_prop)

        # 基础图例句柄只收一次
        if idx == 0:
            base_handles = [
                (h_scatter, "数据"),
                (h_pred, "模型预测（Σ效应）"),
            ]
            if h_self_line is not None:
                base_handles.append((h_self_line, "自效应（+截距）"))

    # 多余空白子图关闭
    for ax in axs[n:]:
        ax.axis("off")

    plt.tight_layout()

    # 全局图例：基础项 + 每一个非自身效应 source
    legend_handles = [h for h, _ in base_handles]
    legend_labels = [lab for _, lab in base_handles]

    from matplotlib.lines import Line2D

    for src in used_cross_sources:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=cross_color_map[src],
                linewidth=2.0,
            )
        )
        legend_labels.append(f"非自身效应：{src}")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(5, len(legend_labels)),
        fontsize=8,
        frameon=True,
        prop=font_prop,
    )

    return fig
