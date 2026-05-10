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
import networkx as nx
import streamlit as st
from matplotlib.lines import Line2D

from backend.utils import font_prop


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


def _degree_table(
    adj_df: pd.DataFrame,
    *,
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
    sort_by: str = "total_degree",
    ascending: bool = False,
) -> pd.DataFrame:
    """计算节点出入度表。

    Direction rule:
        adj_df.loc[source, target] = source -> target

    因此：
        outdegree = 行非零数
        indegree  = 列非零数
    """
    adj_df = _normalize_adjacency_matrix(adj_df)
    edge_arr = adj_df.to_numpy(dtype=float, copy=True)

    if not include_self_edges:
        np.fill_diagonal(edge_arr, 0.0)

    edge_mat = pd.DataFrame(
        edge_arr,
        index=adj_df.index,
        columns=adj_df.columns,
    )

    active = edge_mat.abs() > float(edge_threshold)

    outdegree = active.sum(axis=1).astype(int)
    indegree = active.sum(axis=0).astype(int)

    weighted_outdegree = edge_mat.abs().sum(axis=1)
    weighted_indegree = edge_mat.abs().sum(axis=0)

    degree_df = pd.DataFrame(
        {
            "node": list(adj_df.index),
            "outdegree": outdegree.reindex(adj_df.index).values,
            "indegree": indegree.reindex(adj_df.index).values,
            "total_degree": (
                outdegree.reindex(adj_df.index).values
                + indegree.reindex(adj_df.index).values
            ),
            "weighted_outdegree": weighted_outdegree.reindex(adj_df.index).values,
            "weighted_indegree": weighted_indegree.reindex(adj_df.index).values,
            "weighted_total_degree": (
                weighted_outdegree.reindex(adj_df.index).values
                + weighted_indegree.reindex(adj_df.index).values
            ),
            "node_weight": np.diag(adj_df.values),
        }
    )

    allowed = [
        "node",
        "outdegree",
        "indegree",
        "total_degree",
        "weighted_outdegree",
        "weighted_indegree",
        "weighted_total_degree",
        "node_weight",
    ]

    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of {allowed}")

    degree_df = degree_df.sort_values(
        sort_by,
        ascending=ascending,
    ).reset_index(drop=True)

    return degree_df


def _get_node_order(
    adj_df: pd.DataFrame,
    *,
    order_by: str = "outdegree",
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
) -> list[str]:
    """根据节点度生成节点顺序。"""
    adj_df = _normalize_adjacency_matrix(adj_df)

    if order_by == "original":
        return list(adj_df.index.astype(str))

    degree_df = _degree_table(
        adj_df,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
        sort_by=order_by,
        ascending=False,
    )

    return degree_df["node"].astype(str).tolist()


def _select_hub_nodes(
    adj_df: pd.DataFrame,
    *,
    hub_n: int = 3,
    hub_by: str = "outdegree",
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
) -> list[str]:
    """选择 hub 节点。"""
    adj_df = _normalize_adjacency_matrix(adj_df)

    hub_n = int(hub_n)
    if hub_n <= 0:
        return []

    degree_df = _degree_table(
        adj_df,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
        sort_by=hub_by,
        ascending=False,
    )

    return degree_df["node"].astype(str).head(min(hub_n, len(degree_df))).tolist()


# =============================================================================
# 2. 网络构建与布局
# =============================================================================

def _build_digraph(
    adj_df: pd.DataFrame,
    *,
    edge_threshold: float = 0.0,
    top_edges: int | None = None,
    include_self_edges: bool = False,
) -> nx.DiGraph:
    """根据邻接矩阵构建 NetworkX 有向图。

    Direction rule:
        adj_df.loc[source, target] = source -> target
    """
    adj_df = _normalize_adjacency_matrix(adj_df)

    G = nx.DiGraph()

    for node in adj_df.index:
        node_str = str(node)
        G.add_node(node_str, node_weight=float(adj_df.loc[node, node]))

    edges: list[tuple[str, str, float]] = []

    for source in adj_df.index:
        for target in adj_df.columns:
            if source == target and not include_self_edges:
                continue

            w = float(adj_df.loc[source, target])

            if abs(w) <= float(edge_threshold):
                continue

            edges.append((str(source), str(target), w))

    if top_edges is not None:
        top_edges = int(top_edges)
        if top_edges < 0:
            raise ValueError("top_edges must be non-negative or None.")
        edges = sorted(edges, key=lambda x: abs(x[2]), reverse=True)[:top_edges]

    for source, target, w in edges:
        G.add_edge(
            source,
            target,
            weight=w,
            abs_weight=abs(w),
            sign=1 if w > 0 else -1,
        )

    return G


def _fixed_circular_layout(
    node_order: list[str],
    *,
    radius: float = 1.0,
    start_angle: float = np.pi / 2,
    clockwise: bool = True,
) -> dict[str, np.ndarray]:
    """固定圆形布局：第一个节点在 12 点钟方向。"""
    node_order = [str(x) for x in node_order]
    n = len(node_order)

    if n == 0:
        return {}

    pos: dict[str, np.ndarray] = {}
    direction = -1 if clockwise else 1

    for i, node in enumerate(node_order):
        theta = start_angle + direction * 2 * np.pi * i / n
        pos[node] = np.array([radius * np.cos(theta), radius * np.sin(theta)])

    return pos


def _hub_circular_layout(
    *,
    node_order: list[str],
    hub_nodes: list[str],
    layout_scale: float = 1.0,
    hub_inner_radius: float = 0.28,
    hub_outer_radius: float = 1.0,
    start_angle: float = np.pi / 2,
    clockwise: bool = True,
) -> dict[str, np.ndarray]:
    """hub 节点在中心，非 hub 节点外圈环绕。"""
    node_order = [str(x) for x in node_order]
    hub_nodes = [str(x) for x in hub_nodes if str(x) in node_order]

    hub_set = set(hub_nodes)
    outer_nodes = [node for node in node_order if node not in hub_set]

    pos: dict[str, np.ndarray] = {}
    direction = -1 if clockwise else 1

    hub_count = len(hub_nodes)
    outer_count = len(outer_nodes)

    if hub_count == 1:
        pos[hub_nodes[0]] = np.array([0.0, 0.0])

    elif hub_count > 1:
        for i, node in enumerate(hub_nodes):
            theta = start_angle + direction * 2 * np.pi * i / hub_count
            pos[node] = (
                np.array(
                    [
                        hub_inner_radius * np.cos(theta),
                        hub_inner_radius * np.sin(theta),
                    ]
                )
                * layout_scale
            )

    if outer_count > 0:
        for i, node in enumerate(outer_nodes):
            theta = start_angle + direction * 2 * np.pi * i / outer_count
            pos[node] = (
                np.array(
                    [
                        hub_outer_radius * np.cos(theta),
                        hub_outer_radius * np.sin(theta),
                    ]
                )
                * layout_scale
            )

    return pos


def _get_layout(
    G: nx.DiGraph,
    *,
    layout: str = "hub_circular",
    seed: int = 123,
    layout_scale: float = 1.0,
    node_order: list[str] | None = None,
    hub_nodes: list[str] | None = None,
    hub_inner_radius: float = 0.28,
    hub_outer_radius: float = 1.0,
    start_angle: float = np.pi / 2,
    clockwise: bool = True,
) -> dict[str, np.ndarray]:
    """生成网络布局。"""
    layout = str(layout).lower()

    nodes = [str(x) for x in G.nodes()]
    if not nodes:
        return {}

    if node_order is None:
        node_order = nodes
    else:
        node_order = [str(x) for x in node_order if str(x) in nodes]
        node_order += [x for x in nodes if x not in node_order]

    if hub_nodes is None:
        hub_nodes = []
    else:
        hub_nodes = [str(x) for x in hub_nodes if str(x) in nodes]

    if layout == "fixed_circular":
        return _fixed_circular_layout(
            node_order=node_order,
            radius=layout_scale,
            start_angle=start_angle,
            clockwise=clockwise,
        )

    if layout == "hub_circular":
        return _hub_circular_layout(
            node_order=node_order,
            hub_nodes=hub_nodes,
            layout_scale=layout_scale,
            hub_inner_radius=hub_inner_radius,
            hub_outer_radius=hub_outer_radius,
            start_angle=start_angle,
            clockwise=clockwise,
        )

    if layout == "circular":
        return nx.circular_layout(G, scale=layout_scale)

    if layout == "spring":
        return nx.spring_layout(G, seed=seed, scale=layout_scale)

    if layout == "shell":
        return nx.shell_layout(G, scale=layout_scale)

    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G, scale=layout_scale)

    if layout == "spectral":
        return nx.spectral_layout(G, scale=layout_scale)

    raise ValueError(
        "layout must be one of: "
        "'hub_circular', 'fixed_circular', 'circular', 'spring', "
        "'shell', 'kamada_kawai', 'spectral'."
    )


# =============================================================================
# 3. 绘图辅助函数
# =============================================================================

def _compute_node_sizes(
    adj_df: pd.DataFrame,
    nodes: list[str],
    *,
    uniform_node_size: bool = True,
    node_size: float = 850,
    node_size_min: float = 450,
    node_size_max: float = 1800,
    node_size_scale: float = 1.0,
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
) -> dict[str, float]:
    """节点大小。默认所有节点大小一致；可按 outdegree 改变节点大小。"""
    adj_df = _normalize_adjacency_matrix(adj_df)
    nodes = [str(x) for x in nodes]

    if uniform_node_size:
        return {node: float(node_size) * float(node_size_scale) for node in nodes}

    degree_df = _degree_table(
        adj_df,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
        sort_by="node",
        ascending=True,
    ).set_index("node")

    out_values = np.array(
        [degree_df.loc[node, "outdegree"] if node in degree_df.index else 0 for node in nodes],
        dtype=float,
    )

    if len(out_values) > 0 and out_values.max() > out_values.min():
        sizes = node_size_min + (
            (out_values - out_values.min())
            / (out_values.max() - out_values.min())
        ) * (node_size_max - node_size_min)
    else:
        sizes = np.repeat(node_size, len(nodes))

    sizes = sizes * float(node_size_scale)

    return {node: float(size) for node, size in zip(nodes, sizes)}


def _draw_labels_with_font(
    ax: plt.Axes,
    pos: dict[str, np.ndarray],
    nodes: list[str],
    *,
    font_size: int = 9,
    font_color: str = "black",
    font_weight: str = "normal",
) -> None:
    """用 font_prop 手动绘制节点标签，避免中文乱码。"""
    for node in nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=font_size,
            color=font_color,
            fontweight=font_weight,
            fontproperties=font_prop,
            zorder=5,
        )

def _center_network_axis(
    ax: plt.Axes,
    pos: dict[str, np.ndarray],
    *,
    pad_ratio: float = 0.30,
) -> None:
    """让网络图在坐标轴中居中显示，并保证完整显示。"""
    if not pos:
        return

    xy = np.array(list(pos.values()), dtype=float)
    xs = xy[:, 0]
    ys = xy[:, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    half_span = max((xmax - xmin) / 2.0, (ymax - ymin) / 2.0, 0.8)
    half_span = half_span * (1.0 + pad_ratio)

    ax.set_xlim(cx - half_span, cx + half_span)
    ax.set_ylim(cy - half_span, cy + half_span)

    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    
def _draw_signed_edges(
    G: nx.DiGraph,
    pos: dict[str, np.ndarray],
    ax: plt.Axes,
    *,
    positive_edge_color: str = "#E74C3C",
    negative_edge_color: str = "#2F7ED8",
    arrowsize: int = 18,
    edge_alpha: float = 0.92,
    edge_width_min: float = 1.2,
    edge_width_max_add: float = 4.5,
    reciprocal_rad: float = 0.40,
    single_edge_rad: float = 0.08,
) -> None:
    """绘制有向边。

    - 正边红色，负边蓝色
    - 若 A->B 和 B->A 同时存在，则使用相反曲率，避免重合
    - 若只有单向边，则使用 single_edge_rad
    """
    edges = list(G.edges(data=True))
    if not edges:
        return

    max_abs_w = max(abs(float(d.get("weight", 1.0))) for _, _, d in edges)
    max_abs_w = max(max_abs_w, 1e-12)

    edge_set = {(str(u), str(v)) for u, v, _ in edges}

    for u, v, d in edges:
        u = str(u)
        v = str(v)

        w = float(d.get("weight", 1.0))
        aw = abs(w)

        width = edge_width_min + edge_width_max_add * aw / max_abs_w
        color = positive_edge_color if w >= 0 else negative_edge_color

        # 双向边：两条边用相反曲率
        if u != v and (v, u) in edge_set:
            rad = float(reciprocal_rad) if u < v else -float(reciprocal_rad)
        else:
            rad = float(single_edge_rad)

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=color,
            alpha=edge_alpha,
            arrows=True,
            arrowsize=arrowsize,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=22,
            min_target_margin=22,
            ax=ax,
        )


def _mirror_barh_degree(
    ax: plt.Axes,
    degree_df: pd.DataFrame,
    *,
    node_order: list[str] | None = None,
    title: str = "In-degree / Out-degree",
    bg_color: str = "white",
    bar_height: float = 0.72,
) -> None:
    """右侧镜像出入度图：左入度，右出度。"""
    df = degree_df.copy()

    if node_order is not None:
        node_order = [str(x) for x in node_order]
        df["node"] = df["node"].astype(str)
        df["node"] = pd.Categorical(df["node"], categories=node_order, ordered=True)
        df = df.sort_values("node").reset_index(drop=True)

    y = np.arange(len(df))

    # 左侧：入度
    ax.barh(
        y,
        -df["indegree"].values,
        height=bar_height,
        label="In-degree",
        color="#7EA6C8",
        alpha=0.95,
    )

    # 右侧：出度
    ax.barh(
        y,
        df["outdegree"].values,
        height=bar_height,
        label="Out-degree",
        color="#D98C70",
        alpha=0.95,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        df["node"].astype(str).tolist(),
        fontsize=9,
        fontproperties=font_prop,
    )

    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2)

    xmax = max(
        int(df["indegree"].max()) if len(df) else 0,
        int(df["outdegree"].max()) if len(df) else 0,
        1,
    )

    # 左右对称扩张
    ax.set_xlim(-xmax - 0.8, xmax + 0.8)

    ax.set_xlabel("Degree", fontsize=11, fontproperties=font_prop)
    ax.set_title(title, fontsize=13, fontweight="bold", fontproperties=font_prop)

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right", prop=font_prop)
    ax.invert_yaxis()
    ax.set_facecolor(bg_color)

    # 让图框更像一张独立的小图
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D9D9D9")
        spine.set_linewidth(1.0)


# =============================================================================
# 4. 效应分解图
# =============================================================================

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
    st.caption("DEBUG: plot_effect new version loaded")
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

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# 5. 网络图：NetRecon 页面直接调用这个函数
# =============================================================================

def plot_network(
    adj_df: pd.DataFrame,
    target_node: str = "",
    *,
    layout: str = "hub_circular",
    node_order_by: str = "outdegree",
    hub_center: bool = True,
    hub_n: int = 3,
    hub_by: str = "outdegree",
    edge_threshold: float = 0.0,
    top_edges: int | None = None,
    include_self_edges: bool = False,
    uniform_node_size: bool = True,
    node_size: float = 850,
    node_facecolor: str = "#EADFB4",
    node_edge_color: str = "#2C4A6B",
    node_linewidth: float = 1.1,
    positive_edge_color: str = "#E74C3C",
    negative_edge_color: str = "#2F7ED8",
    show_degree_panel: bool = True,
    reciprocal_rad: float = 0.30,
    single_edge_rad: float = 0.00,
) -> None:
    """绘制有向加权网络图。

    Direction rule:
        adj_df.loc[source, target] = source -> target

    Parameters
    ----------
    target_node:
        如果非空，则只展示“指向该 target_node 的边”。
    """
    st.caption("DEBUG: plot_network new version loaded")
    try:
        adj_df = _normalize_adjacency_matrix(adj_df)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    # target_node 筛选：只保留 source -> target_node 的边
    if target_node:
        target_node = str(target_node)

        if target_node not in adj_df.columns:
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
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            return

        source_nodes = [
            source
            for source in adj_df.index
            if source != target_node
            and abs(float(adj_df.loc[source, target_node])) > float(edge_threshold)
        ]

        if not source_nodes:
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
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            return

        keep_nodes = source_nodes + [target_node]
        plot_adj = pd.DataFrame(
            0.0,
            index=keep_nodes,
            columns=keep_nodes,
        )

        for source in source_nodes:
            plot_adj.loc[source, target_node] = float(adj_df.loc[source, target_node])

        title = f"指向 {target_node} 的交互网络"

    else:
        plot_adj = adj_df
        title = "有向交互网络"

    # 过滤后如果没有边
    tmp_arr = plot_adj.to_numpy(dtype=float, copy=True)

    if not include_self_edges:
        np.fill_diagonal(tmp_arr, 0.0)

    if not (np.abs(tmp_arr) > float(edge_threshold)).any():
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    node_order = _get_node_order(
        plot_adj,
        order_by=node_order_by,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
    )

    hub_nodes = (
        _select_hub_nodes(
            plot_adj,
            hub_n=min(int(hub_n), plot_adj.shape[0]),
            hub_by=hub_by,
            edge_threshold=edge_threshold,
            include_self_edges=include_self_edges,
        )
        if hub_center or layout == "hub_circular"
        else []
    )

    layout_to_use = "hub_circular" if hub_center else layout

    G = _build_digraph(
        plot_adj,
        edge_threshold=edge_threshold,
        top_edges=top_edges,
        include_self_edges=include_self_edges,
    )

    # 保证孤立节点也显示
    for node in plot_adj.index:
        if str(node) not in G:
            G.add_node(str(node), node_weight=float(plot_adj.loc[node, node]))

    pos = _get_layout(
        G,
        layout="hub_circular",
        seed=123,
        layout_scale=1.0,
        node_order=node_order,
        hub_nodes=hub_nodes,
        hub_inner_radius=0.45,
        hub_outer_radius=1.35,
        clockwise=True,
    )

    nodes = [str(x) for x in G.nodes()]
    draw_nodes = [node for node in node_order if node in nodes]
    draw_nodes += [node for node in nodes if node not in draw_nodes]

    size_dict = _compute_node_sizes(
        plot_adj,
        nodes=draw_nodes,
        uniform_node_size=uniform_node_size,
        node_size=node_size,
        node_size_min=450,
        node_size_max=1800,
        node_size_scale=1.0,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
    )

    n_nodes = len(draw_nodes)

    # 根据节点数动态调整图高
    fig_height = max(7.5, 0.55 * n_nodes + 3.5)

    if show_degree_panel:
        fig = plt.figure(figsize=(15.5, fig_height), dpi=180, facecolor="#F6F6F6")
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[2.25, 0.18, 1.20],  # 中间一列专门留白
            wspace=0.00,
        )

        ax_net = fig.add_subplot(gs[0, 0])
        ax_gap = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[0, 2])

        ax_gap.axis("off")
    else:
        fig, ax_net = plt.subplots(figsize=(9.0, fig_height), dpi=200, facecolor="#F6F6F6")
        ax_bar = None

    ax_net.set_facecolor("white")

    if ax_bar is not None:
        ax_bar.set_facecolor("white")

    # 给网络图 panel 加一个边框
    net_border = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_net.transAxes,
        fill=False,
        edgecolor="#D9D9D9",
        linewidth=1.0,
        zorder=10,
        clip_on=False,
    )
    ax_net.add_patch(net_border)

    # 给出入度图 panel 加一个边框
    if ax_bar is not None:
        bar_border = plt.Rectangle(
            (0, 0), 1, 1,
            transform=ax_bar.transAxes,
            fill=False,
            edgecolor="#D9D9D9",
            linewidth=1.0,
            zorder=10,
            clip_on=False,
        )
        ax_bar.add_patch(bar_border)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=draw_nodes,
        node_size=[size_dict[node] for node in draw_nodes],
        node_color=node_facecolor,
        edgecolors=node_edge_color,
        linewidths=node_linewidth,
        ax=ax_net,
    )

    _draw_signed_edges(
        G,
        pos,
        ax_net,
        positive_edge_color=positive_edge_color,
        negative_edge_color=negative_edge_color,
        arrowsize=18,
        edge_alpha=0.92,
        reciprocal_rad=reciprocal_rad,
        single_edge_rad=single_edge_rad,
    )

    _draw_labels_with_font(
        ax_net,
        pos,
        draw_nodes,
        font_size=9,
        font_color="black",
        font_weight="normal",
    )

    ax_net.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=16,
        fontproperties=font_prop,
    )
    ax_net.axis("off")
    _center_network_axis(ax_net, pos, pad_ratio=0.32)

    legend_elements = [
        Line2D([0], [0], color=positive_edge_color, linewidth=2.8, label="正（促进）"),
        Line2D([0], [0], color=negative_edge_color, linewidth=2.8, label="负（抑制）"),
    ]

    ax_net.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=8,
        frameon=True,
        prop=font_prop,
    )

    if show_degree_panel and ax_bar is not None:
        degree_df = _degree_table(
            plot_adj,
            edge_threshold=edge_threshold,
            include_self_edges=include_self_edges,
            sort_by="node",
            ascending=True,
        )

        _mirror_barh_degree(
            ax_bar,
            degree_df,
            node_order=node_order,
            title="In-degree / Out-degree",
            bg_color="white",
        )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.10)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)