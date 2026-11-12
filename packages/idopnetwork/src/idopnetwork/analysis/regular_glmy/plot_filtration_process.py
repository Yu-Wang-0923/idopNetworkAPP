"""Show a weighted digraph reappearing along its GLMY filtration.

Click Run: load src/network_data.py, compute the barcode once, and save a
300-DPI figure with threshold guides above fixed-layout network snapshots.
Every edge enters when abs(weight) <= t; no shortest-path edges are added.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from numbers import Integral
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import pandas as pd

if __package__:
    from .plot_barcode import _load_python_edgelist, _prepare_edges, compute_barcode
    from .network_plot_style import MidpointArrowStyle, SIGN_COLORS, _signed_edge_weights, _weight_sign
else:
    from plot_barcode import _load_python_edgelist, _prepare_edges, compute_barcode
    from network_plot_style import MidpointArrowStyle, SIGN_COLORS, _signed_edge_weights, _weight_sign


# Change these defaults here when using an editor's Run button.
N_STAGES = 4
FILTRATION_VALUES = None  # e.g. [0.0, 0.3, 0.6, 1.0]; None selects edge quantiles.
EDGE_SIGN = "all"        # "all", "positive", or "negative"


@dataclass(frozen=True)
class FiltrationSnapshot:
    """A full-vertex subgraph and its unreduced F_2 Betti numbers at a threshold."""

    number: int
    threshold: float
    edges: tuple[tuple[Hashable, Hashable], ...]
    betti: tuple[int, int, int]


def _select_edges(edgelist, nodes, edge_sign):
    if edge_sign not in ("all", "positive", "negative"):
        raise ValueError("edge_sign must be 'all', 'positive' or 'negative'")
    if isinstance(edgelist, (str, Path)):
        edgelist = pd.read_csv(edgelist)
    # Validate before selecting, and retain the full vertex set even if sign
    # selection removes every incident edge of a vertex.
    labels, _ = _prepare_edges(edgelist, nodes)
    weights = np.asarray(edgelist["weight"], dtype=float)
    mask = np.ones(len(weights), dtype=bool)
    if edge_sign == "positive":
        mask = weights > 0
    elif edge_sign == "negative":
        mask = weights < 0
    selected = edgelist.loc[mask].copy()
    selected.attrs["nodes"] = labels
    _, edges = _prepare_edges(selected, labels)
    return selected, labels, edges


def _thresholds(edges, filtration_values, n_stages):
    if isinstance(n_stages, bool) or not isinstance(n_stages, Integral) or n_stages < 2:
        raise ValueError("n_stages must be an integer of at least 2")
    if filtration_values is not None:
        try:
            values = np.asarray(list(filtration_values), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("filtration_values must be finite nonnegative numbers") from exc
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("filtration_values must be a nonempty list of finite nonnegative numbers")
        if (np.diff(values) <= 0).any():
            raise ValueError("filtration_values must be strictly increasing")
        return values.tolist()
    levels = sorted(set(edges.values()))
    if not levels:
        return [0.0]
    # Pick actual edge-entry times. Include zero and the final graph; deduplicate
    # tied levels instead of manufacturing different-looking identical stages.
    quantiles = np.linspace(0, 1, n_stages)[1:]
    picked = np.quantile(levels, quantiles, method="nearest")
    return sorted({0.0, *(float(t) for t in picked)})


def _compute_process(edgelist, nodes, edge_sign, filtration_values, n_stages, max_paths):
    selected, labels, edges = _select_edges(edgelist, nodes, edge_sign)
    values = _thresholds(edges, filtration_values, n_stages)
    barcode = compute_barcode(selected, max_dim=2, nodes=labels, max_paths=max_paths)
    snapshots = []
    for number, threshold in enumerate(values, start=1):
        active = (barcode.birth <= threshold) & (threshold < barcode.death)
        betti = tuple(int((active & (barcode.dimension == d)).sum()) for d in range(3))
        stage_edges = tuple(
            (labels[u], labels[v]) for (u, v), entry in edges.items() if entry <= threshold
        )
        snapshots.append(FiltrationSnapshot(number, threshold, stage_edges, betti))
    return barcode, snapshots, labels, edges, selected


def compute_filtration_snapshots(
    edgelist: pd.DataFrame | str | Path,
    *,
    nodes: Iterable[Hashable] | None = None,
    edge_sign: str = "all",
    filtration_values: Sequence[float] | None = None,
    n_stages: int = 4,
    max_paths: int = 100_000,
) -> tuple[pd.DataFrame, list[FiltrationSnapshot]]:
    """Return the full H0-H2 barcode and snapshots of G_t=(V, {|w|<=t}).

    By default, select zero followed by nearest edge-entry quantiles, ending at
    the largest absolute edge weight. Ties may yield fewer distinct stages.
    With no edges the only default stage is zero. Explicit filtration_values
    override quantile selection and must be finite, nonnegative and increasing.

    Sign selection uses the signed weight, not the optional sign text column.
    It retains all vertices from the unfiltered input, including isolates.
    Duplicate directed observations merge at their earliest entry time.
    Birth/death ties follow the half-open interval convention [birth, death).
    """
    if nodes is not None:
        nodes = list(nodes)
    barcode, snapshots, _, _, _ = _compute_process(
        edgelist, nodes, edge_sign, filtration_values, n_stages, max_paths,
    )
    return barcode, snapshots


def _draw_barcode(ax, barcode, snapshots, last, edge_sign, n_vertices, n_edges):
    colors = {0: "red", 1: "green", 2: "blue"}
    span = last if last > 0 else 1.0
    infinity_end = last + 0.06 * span
    offset = 0
    tick_positions, tick_labels = [], []
    for degree in range(3):
        group = barcode.loc[barcode.dimension == degree].iloc[::-1]
        count = len(group)
        for row, (_, birth, death) in enumerate(group.itertuples(index=False, name=None)):
            y = offset + row
            end = infinity_end if np.isinf(death) else death
            ax.hlines(y, birth, end, color=colors[degree], linewidth=1.5)
            ax.plot(birth, y, "o", color="black", markerfacecolor="white", markersize=2.3)
            if np.isinf(death):
                ax.plot(end, y, ">", color="black", markersize=3)
                ax.annotate(r"$+\infty$", (end, y), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7)
            else:
                ax.plot(end, y, "s", color="black", markersize=2.3)
        tick_positions.append(offset + (max(1, count) - 1) / 2)
        tick_labels.append(rf"$\beta_{{{degree}}}$ ({count})")
        offset += max(1, count) + 2
    ax.set_yticks(tick_positions, labels=tick_labels, fontsize=9)
    for degree, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(colors[degree] if degree else "black")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1, offset + 1)
    ax.set_xlim(-0.02 * span, last + 0.20 * span)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.set_xlabel("filtration", fontsize=9)
    ax.grid(False, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    # These are explicit stage guides, not background grid lines.
    for stage in snapshots:
        ax.axvline(stage.threshold, color="0.55", linewidth=0.65,
                   linestyle=(0, (3, 3)), zorder=0)
        ax.annotate(
            f"[{stage.number}]", xy=(stage.threshold, 1),
            xycoords=("data", "axes fraction"), xytext=(0, 3),
            textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )
    selection = {"all": "All edges", "positive": "Positive edges", "negative": "Negative edges"}[edge_sign]
    ax.set_title(f"{selection}: {n_vertices} vertices, {n_edges} directed edges",
                 fontsize=11, pad=20)


def _draw_snapshot(ax, snapshot, graph, positions, signed_weights=None):
    support = set(snapshot.edges)
    reciprocal = [edge for edge in snapshot.edges if edge[::-1] in support]
    straight = [edge for edge in snapshot.edges if edge[::-1] not in support]
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=24, node_color="black")
    for edges, curvature in ((straight, 0.0), (reciprocal, 0.15)):
        colors = ("0.30" if signed_weights is None else
                  [SIGN_COLORS[_weight_sign(signed_weights[edge])] for edge in edges])
        nx.draw_networkx_edges(
            graph, positions, edgelist=edges, ax=ax, width=0.85,
            edge_color=colors, alpha=0.85, arrows=True, arrowstyle=MidpointArrowStyle(),
            arrowsize=10, node_size=24, connectionstyle=f"arc3,rad={curvature}",
        )
    label_positions = {node: point * 1.13 for node, point in positions.items()}
    nx.draw_networkx_labels(
        graph, label_positions, ax=ax, font_size=7, font_color="black",
    )
    beta = ", ".join(str(value) for value in snapshot.betti)
    ax.set_title(
        rf"[{snapshot.number}]  $t={snapshot.threshold:.4g}$"
        + "\n" + rf"$\beta=({beta})$; {len(snapshot.edges)} edges",
        fontsize=9, pad=9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.65")
        spine.set_linewidth(0.6)


def _export_tables(barcode, snapshots, labels, edges, positions, signed_weights, output):
    barcode.to_csv(output.with_name(output.stem + "_barcode.csv"), index=False)
    pd.DataFrame([
        {"stage": s.number, "threshold": s.threshold, "n_vertices": len(labels),
         "n_edges": len(s.edges), "beta0": s.betti[0], "beta1": s.betti[1], "beta2": s.betti[2]}
        for s in snapshots
    ]).to_csv(output.with_name(output.stem + "_snapshots.csv"), index=False)
    entry_times = {(labels[u], labels[v]): t for (u, v), t in edges.items()}
    pd.DataFrame([
        {"stage": s.number, "threshold": s.threshold,
         "source": u, "target": v, "entry_time": entry_times[u, v],
         "weight": signed_weights[u, v], "sign": _weight_sign(signed_weights[u, v])}
        for s in snapshots for u, v in s.edges
    ], columns=["stage", "threshold", "source", "target", "entry_time", "weight", "sign"]).to_csv(
        output.with_name(output.stem + "_edges.csv"), index=False,
    )
    pd.DataFrame([
        {"node": node, "x": positions[node][0], "y": positions[node][1]}
        for node in labels
    ], columns=["node", "x", "y"]).to_csv(
        output.with_name(output.stem + "_nodes.csv"), index=False,
    )


def plot_filtration_process(
    edgelist: pd.DataFrame | str | Path,
    *,
    nodes: Iterable[Hashable] | None = None,
    edge_sign: str = "all",
    color_by_sign: bool = False,
    filtration_values: Sequence[float] | None = None,
    n_stages: int = 4,
    columns: int | None = None,
    max_paths: int = 100_000,
    save_path: str | Path | None = None,
    export_vector: bool = False,
    show: bool = False,
) -> tuple[pd.DataFrame, list[FiltrationSnapshot], plt.Figure]:
    """Draw barcode guides and networks progressively recovered by |w| <= t.

    Return (barcode, snapshots, figure). All snapshots include every vertex,
    share a circular layout, preserve edge directions, and show exactly the
    edges present at their threshold. Betti labels are computed over F_2 from
    this same run's H0-H2 barcode, not from the drawing. The barcode's group
    counts are interval counts; the snapshot's beta tuple is the actual Betti
    number at that threshold. Zero-weight edges already appear at t=0.
    All stages appear in a single row by default (columns=None). Snapshots use
    the representative gallery's black nodes, external labels and gray frames.
    color_by_sign=True colors positive edges red and negative edges blue; zero
    weights are gray. This only affects the network drawing, not homology.
    Duplicate edges use the signed weight of the earliest observation, with
    ties resolved by input order. CSV edge tables retain this weight and sign.

    save_path writes a 300-DPI PNG and four associated CSV tables. Set
    export_vector=True for additional editable SVG/PDF exports. show=False
    avoids opening a window. Close the returned figure after use.
    """
    if columns is not None and (isinstance(columns, bool) or not isinstance(columns, Integral) or columns < 1):
        raise ValueError("columns must be a positive integer")
    if nodes is not None:
        nodes = list(nodes)
    barcode, snapshots, labels, edges, selected = _compute_process(
        edgelist, nodes, edge_sign, filtration_values, n_stages, max_paths,
    )
    signed_weights = _signed_edge_weights(selected)
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)
    graph.add_edges_from((labels[u], labels[v]) for u, v in edges)
    positions = nx.circular_layout(graph)
    n_columns = len(snapshots) if columns is None else min(columns, len(snapshots))
    n_rows = ceil(len(snapshots) / n_columns)
    barcode_height = max(1.7, (len(barcode) + 10) * 0.10)
    last = max(barcode.attrs["final_threshold"], snapshots[-1].threshold)
    with mpl.rc_context({
        "font.family": "sans-serif", "font.size": 9,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    }):
        figure = plt.figure(
            figsize=(2.3 * n_columns + 0.4, barcode_height + 2.7 * n_rows + 0.5),
            layout="constrained",
        )
        grid = figure.add_gridspec(n_rows + 1, n_columns,
                                  height_ratios=[barcode_height] + [2.5] * n_rows)
        barcode_ax = figure.add_subplot(grid[0, :])
        _draw_barcode(barcode_ax, barcode, snapshots, last, edge_sign, len(labels), len(edges))
        for index, stage in enumerate(snapshots):
            ax = figure.add_subplot(grid[1 + index // n_columns, index % n_columns])
            _draw_snapshot(ax, stage, graph, positions,
                           signed_weights if color_by_sign else None)
            if index == 0 and color_by_sign:
                legend_signs = ["positive", "negative"] if edge_sign == "all" else [edge_sign]
                if any(weight == 0 for weight in signed_weights.values()):
                    legend_signs.append("zero")
                sign_labels = {"positive": "Positive (w > 0)", "negative": "Negative (w < 0)",
                               "zero": "Zero (w = 0)"}
                ax.legend(
                    handles=[Line2D([], [], color=SIGN_COLORS[sign], label=sign_labels[sign])
                             for sign in legend_signs],
                    title="Network edges", title_fontsize=7, fontsize=7,
                    loc="upper center", bbox_to_anchor=(0.5, -0.03), frameon=False,
                )
        if save_path is not None:
            output = Path(save_path)
            if output.suffix.lower() != ".png":
                raise ValueError("save_path must end in .png")
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=300, bbox_inches="tight")
            if export_vector:
                figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
                figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
            _export_tables(barcode, snapshots, labels, edges, positions, signed_weights, output)
        if show:
            plt.show()
    return barcode, snapshots, figure


def main(argv: list[str] | None = None) -> None:
    """IDE Run entry point; no preceding scripts need to be run."""
    import argparse

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edgelist", nargs="?", type=Path, default=root / "src" / "network_data.py")
    parser.add_argument("--output", type=Path, default=root / "output" / "filtration_process.png")
    parser.add_argument("--thresholds", nargs="+", type=float, default=FILTRATION_VALUES)
    parser.add_argument("--n-stages", type=int, default=N_STAGES)
    parser.add_argument("--edge-sign", choices=["all", "positive", "negative"], default=EDGE_SIGN)
    parser.add_argument("--color-by-sign", action="store_true",
                        help="Color network edges red for positive and blue for negative weights")
    parser.add_argument("--n-nodes", type=int)
    parser.add_argument("--columns", type=int, default=None,
                        help="Stages per row; default shows all stages in one row")
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--vector", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.n_nodes is not None and args.n_nodes < 0:
        parser.error("--n-nodes must be nonnegative")
    source = args.edgelist.resolve()
    edgelist = _load_python_edgelist(source) if source.suffix.lower() == ".py" else source
    _, snapshots, figure = plot_filtration_process(
        edgelist, nodes=range(args.n_nodes) if args.n_nodes is not None else None,
        edge_sign=args.edge_sign, color_by_sign=args.color_by_sign,
        filtration_values=args.thresholds, n_stages=args.n_stages,
        columns=args.columns, max_paths=args.max_paths, save_path=args.output,
        export_vector=args.vector, show=args.show,
    )
    print(f"Input: {source}")
    for stage in snapshots:
        print(f"[{stage.number}] t={stage.threshold:.6g}; edges={len(stage.edges)}; beta={stage.betti}")
    print(f"Figure: {args.output.resolve()}")
    plt.close(figure)


if __name__ == "__main__":
    main()
