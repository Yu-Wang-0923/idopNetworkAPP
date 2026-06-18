"""Visualization helpers for machine-learning Hub analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


def _empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=13,
        color="#64748B",
        transform=ax.transAxes,
    )
    ax.axis("off")
    return fig


def plot_hub_network(
    edges_df: pd.DataFrame,
    hub_scores: pd.DataFrame,
    *,
    top_nodes: int = 40,
    top_edges: int = 120,
    label_top_n: int = 12,
    rank_metric: str = "hub_score",
    random_state: int = 123,
):
    """Plot the strongest signed ML edges around the highest-ranked hubs."""
    if edges_df.empty:
        return _empty_figure("No ML edges passed the coefficient threshold.")

    top_nodes = max(2, int(top_nodes))
    top_edges = max(1, int(top_edges))
    label_top_n = max(0, int(label_top_n))

    hub_table = hub_scores.copy()
    hub_table["variable"] = hub_table["variable"].astype(str)
    if rank_metric not in hub_table.columns:
        rank_metric = "hub_score"
    tie_cols = []
    for col in [rank_metric, "out_degree", "out_strength", "hub_score"]:
        if col in hub_table.columns and col not in tie_cols:
            tie_cols.append(col)
    hub_table = hub_table.sort_values(tie_cols, ascending=False)
    keep_nodes = set(hub_table["variable"].head(top_nodes).tolist())

    plot_edges = edges_df[
        edges_df["source"].astype(str).isin(keep_nodes)
        | edges_df["target"].astype(str).isin(keep_nodes)
    ].copy()
    if plot_edges.empty:
        plot_edges = edges_df.copy()

    plot_edges = plot_edges.sort_values("abs_weight", ascending=False).head(top_edges)
    nodes = sorted(set(plot_edges["source"].astype(str)).union(set(plot_edges["target"].astype(str))))

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row in plot_edges.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            weight=float(row.coefficient),
            abs_weight=float(row.abs_weight),
            effect=str(row.effect),
        )

    if graph.number_of_nodes() == 0:
        return _empty_figure("No nodes to display.")

    if graph.number_of_nodes() <= 2:
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(
            graph,
            seed=int(random_state),
            weight="abs_weight",
            k=1.25 / np.sqrt(max(1, graph.number_of_nodes())),
        )

    size_metric = rank_metric if rank_metric in hub_table.columns else "hub_score"
    hub_lookup = hub_table.set_index("variable")[size_metric].to_dict()
    score_values = np.array([float(hub_lookup.get(node, 0.0)) for node in graph.nodes()], dtype=float)
    max_score = float(score_values.max()) if len(score_values) else 0.0
    if max_score > 0:
        node_sizes = 500 + 2100 * score_values / max_score
    else:
        node_sizes = np.repeat(900, graph.number_of_nodes())

    top_hub_set = set(hub_table["variable"].head(3).tolist())
    node_colors = ["#FDE68A" if node in top_hub_set else "#DBEAFE" for node in graph.nodes()]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=180, facecolor="#F8FAFC")
    ax.set_facecolor("white")

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#334155",
        linewidths=1.0,
        ax=ax,
    )

    edge_rows = list(graph.edges(data=True))
    if edge_rows:
        max_weight = max(float(data["abs_weight"]) for _, _, data in edge_rows)
        max_weight = max(max_weight, 1e-12)
        for source, target, data in edge_rows:
            weight = float(data["weight"])
            width = 0.8 + 4.5 * float(data["abs_weight"]) / max_weight
            color = "#DC2626" if weight > 0 else "#2563EB"
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(source, target)],
                width=width,
                edge_color=color,
                alpha=0.78,
                arrows=True,
                arrowsize=16,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.08",
                min_source_margin=18,
                min_target_margin=18,
                ax=ax,
            )

    label_nodes = set(hub_table["variable"].head(label_top_n).tolist())
    if graph.number_of_nodes() <= label_top_n:
        label_nodes = set(graph.nodes())
    labels = {node: node for node in graph.nodes() if node in label_nodes}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8,
        font_color="#0F172A",
        ax=ax,
    )

    ax.set_title(
        "ML-inferred signed Hub network",
        fontsize=15,
        fontweight="bold",
        pad=14,
        color="#0F172A",
    )
    ax.axis("off")

    legend = [
        Line2D([0], [0], color="#DC2626", linewidth=3, label="Promote / positive coefficient"),
        Line2D([0], [0], color="#2563EB", linewidth=3, label="Inhibit / negative coefficient"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FDE68A", markeredgecolor="#334155", markersize=10, label="Top Hub"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True)
    fig.tight_layout()
    return fig
