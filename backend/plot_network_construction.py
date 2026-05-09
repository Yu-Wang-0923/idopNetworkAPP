"""Network Construction 绘图模块。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D

from backend.utils import font_prop


def plot_effect(
    quasi_dynamic_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    effect_df_list: list[pd.DataFrame],
    intercept: pd.Series | None = None,
    log_log: bool = False,
    plot_ncols: int = 4,
    plot_max_vars: int = 0,
) -> None:
    """在子图网格上叠加 effect：截距并入自效应，自效应红线，其余交互绿线。""" # 
    ncols = max(1, plot_ncols)
    cols = list(quasi_dynamic_df.columns)
    if plot_max_vars > 0:
        cols = cols[:plot_max_vars]
    n = len(cols)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), dpi=300)
    axs = np.array(axs).reshape(-1)
    color_self = "red"
    color_cross = "green"

    global_handles: list = []
    for idx, c in enumerate(cols):
        ax = axs[idx]
        h_scatter = ax.scatter(
            quasi_dynamic_df.index,
            quasi_dynamic_df[c],
            color="orange",
            alpha=0.7,
            s=50,
            zorder=1,
        )
        h_pred = ax.plot(
            curve_df.index,
            curve_df[c],
            color="blue",
            linestyle="-",
            linewidth=2.5,
        )[0]
        eff_df = effect_df_list[idx]
        icpt = float(intercept.loc[c]) if intercept is not None and c in intercept.index else 0.0
        h_self_line = None
        h_cross_line = None

        for src in eff_df.columns:
            y = eff_df[src].to_numpy(dtype=float)
            if src == c:
                y = y + icpt
                h_self = ax.plot(
                    eff_df.index, y, color=color_self, linewidth=2.8, alpha=0.95
                )[0]
                h_self_line = h_self
            else:
                if np.allclose(y, 0.0, rtol=0.0, atol=1e-12):
                    continue
                h_c = ax.plot(
                    eff_df.index, y, color=color_cross, linewidth=2.2, alpha=0.85
                )[0]
                if h_cross_line is None:
                    h_cross_line = h_c

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=0)
        if log_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(c, fontproperties=font_prop, fontsize=10)

        if idx == 0:
            global_handles = [(h_scatter, "数据"), (h_pred, "模型预测（Σ效应）")]
            if h_self_line is not None:
                global_handles.append((h_self_line, "自效应（+截距）"))
            if h_cross_line is not None:
                global_handles.append((h_cross_line, "交互效应"))

    for ax in axs[n:]:
        ax.axis("off")

    plt.tight_layout()
    if global_handles:
        handles, labels = zip(*global_handles)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(len(labels), 4),
            fontsize=8,
            frameon=True,
            prop=font_prop,
        )
    st.pyplot(fig)
    plt.close(fig)


def plot_network(adj_df: pd.DataFrame, target_node: str = "") -> None:
    """绘制有向加权网络图。"""
    nodes = list(adj_df.columns)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edges = []
    weights = []
    for source in adj_df.index:
        for target in adj_df.columns:
            if source == target:
                continue
            w = float(adj_df.loc[target, source])
            if abs(w) < 1e-12:
                continue
            if target_node and target != target_node:
                continue
            edges.append((source, target))
            weights.append(w)

    if not edges:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.text(
            0.5,
            0.5,
            "无符合条件的边",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
            fontproperties=font_prop,
        )
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        return

    G.add_edges_from(edges)
    if target_node:
        relevant = {target_node} | {s for s, _ in edges}
        G = G.subgraph(relevant).copy()

    pos = nx.circular_layout(G)
    abs_weights = [abs(w) for w in weights]
    max_w = max(abs_weights) if abs_weights else 1.0
    widths = [0.5 + 4.5 * (aw / max_w) for aw in abs_weights]
    edge_colors = ["#dc2626" if w > 0 else "#2563eb" for w in weights]
    alphas = [0.4 + 0.6 * (aw / max_w) for aw in abs_weights]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=800,
        node_color="#f8fafc",
        edgecolors="#475569",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold", font_family="sans-serif")

    for (u, v), w, width, color, alpha in zip(edges, weights, widths, edge_colors, alphas):
        if u not in G or v not in G:
            continue
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            width=width,
            edge_color=[color],
            alpha=alpha,
            arrows=True,
            arrowsize=15,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12",
            min_source_margin=18,
            min_target_margin=18,
        )

    title = f"指向 {target_node} 的交互网络" if target_node else "有向交互网络"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, fontproperties=font_prop)
    ax.axis("off")
    legend_elements = [
        Line2D([0], [0], color="#dc2626", linewidth=2.5, label="正（促进）"),
        Line2D([0], [0], color="#2563eb", linewidth=2.5, label="负（抑制）"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=8,
        frameon=True,
        prop=font_prop,
    )
    st.pyplot(fig)
    plt.close(fig)
