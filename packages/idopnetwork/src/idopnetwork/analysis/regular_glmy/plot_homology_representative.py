"""Plot one GLMY cycle representative per nonzero H_1/H_2 barcode interval.

Click Run to use src/network_data.py and save 300-DPI PNGs plus exact chain
tables. Each panel shows the directed EDGE SUPPORT of an F_2 cycle. In degree
two, that drawing is a projection of a sum of 2-paths, not a filled triangle
complex. The path table, rather than the edge drawing alone, defines the cycle.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
import json
from math import ceil
from numbers import Integral
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

if __package__:
    from .network_plot_style import MidpointArrowStyle, SIGN_COLORS, _signed_edge_weights, _weight_sign
    from .plot_barcode import (
        HomologyRepresentative, _load_python_edgelist, _prepare_edges,
        compute_barcode, plot_barcode,
    )
else:
    from network_plot_style import MidpointArrowStyle, SIGN_COLORS, _signed_edge_weights, _weight_sign
    from plot_barcode import (
        HomologyRepresentative, _load_python_edgelist, _prepare_edges,
        compute_barcode, plot_barcode,
    )


def compute_homology_representatives(
    edgelist: pd.DataFrame | str | Path,
    *,
    nodes: Iterable[Hashable] | None = None,
    max_paths: int = 100_000,
) -> tuple[pd.DataFrame, list[HomologyRepresentative]]:
    """Return the barcode and one exact F_2 cycle per H_1/H_2 interval.

    Each representative has dimension, number, birth, death and paths fields.
    The cycle is the XOR sum of its elementary paths (coefficient 1 modulo 2).
    Numbering agrees with plot_barcode's bottom-to-top numbering, including
    intervals whose tick labels are hidden to keep that plot uncluttered.

    Finite representatives exist at birth and are boundaries at death. Essential
    representatives remain nonboundaries in the final graph. The set of alive
    representatives at a threshold is a homology basis. No minimality or
    uniqueness is claimed. Signs are ignored; filtration is abs(weight) <= t.
    """
    if nodes is not None:
        nodes = list(nodes)
    barcode, representatives = compute_barcode(
        edgelist, max_dim=2, nodes=nodes, max_paths=max_paths,
        return_representatives=True,
    )
    return barcode, [rep for rep in representatives if rep.dimension in (1, 2)]


def _support_edges(rep: HomologyRepresentative):
    # Set UNION, not parity: multiple 2-path terms may use the same edge, and
    # cancelling their edge projections would destroy the support drawing.
    return list(dict.fromkeys(
        edge for path in rep.paths for edge in zip(path, path[1:])
    ))


def _json_label(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _save_chain_tables(representatives, labels, positions, signed_weights, output_dir):
    rows, path_rows, edge_rows = [], [], []
    for rep in representatives:
        support = _support_edges(rep)
        rows.append({
            "dimension": rep.dimension, "number": rep.number,
            "birth": rep.birth, "death": rep.death,
            "n_paths": len(rep.paths), "n_support_edges": len(support),
            "representative_threshold": rep.birth,
        })
        for u, v in support:
            weight = signed_weights[u, v]
            edge_rows.append({
                "dimension": rep.dimension, "number": rep.number,
                "source": u, "target": v, "weight": weight,
                "sign": _weight_sign(weight), "entry_time": abs(weight),
            })
        for term, path in enumerate(rep.paths, start=1):
            path_rows.append({
                "dimension": rep.dimension, "number": rep.number,
                "term": term, "coefficient_F2": 1,
                "path": json.dumps(list(path), ensure_ascii=False, default=_json_label),
            })
    pd.DataFrame(rows, columns=[
        "dimension", "number", "birth", "death", "n_paths", "n_support_edges",
        "representative_threshold",
    ]).to_csv(output_dir / "homology_representative_intervals.csv", index=False)
    pd.DataFrame(path_rows, columns=[
        "dimension", "number", "term", "coefficient_F2", "path",
    ]).to_csv(output_dir / "homology_representative_paths.csv", index=False)
    pd.DataFrame(edge_rows, columns=[
        "dimension", "number", "source", "target", "weight", "sign", "entry_time",
    ]).to_csv(output_dir / "homology_representative_edges.csv", index=False)
    pd.DataFrame([
        {"node": label, "x": positions[label][0], "y": positions[label][1]}
        for label in labels
    ], columns=["node", "x", "y"]).to_csv(
        output_dir / "homology_representative_nodes.csv", index=False,
    )


def _draw_representative(ax, rep, graph, positions, color, show_node_labels, signed_weights=None):
    support = _support_edges(rep)
    reciprocal = [(u, v) for u, v in support if (v, u) in support]
    straight = [(u, v) for u, v in support if (v, u) not in support]
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=24, node_color="black")
    for edges, radius in ((straight, 0.0), (reciprocal, 0.15)):
        edge_colors = (color if signed_weights is None else
                       [SIGN_COLORS[_weight_sign(signed_weights[edge])] for edge in edges])
        nx.draw_networkx_edges(
            graph, positions, ax=ax, edgelist=edges, edge_color=edge_colors,
            width=1.5, arrows=True, arrowstyle=MidpointArrowStyle(), arrowsize=10,
            node_size=24, connectionstyle=f"arc3,rad={radius}",
        )
    if show_node_labels:
        label_positions = {node: point * 1.13 for node, point in positions.items()}
        nx.draw_networkx_labels(
            graph, label_positions, ax=ax, font_size=7, font_color="black",
        )
    death = r"\infty" if np.isinf(rep.death) else f"{rep.death:.3g}"
    ax.set_title(
        rf"#{rep.number}   $[{rep.birth:.3g},\,{death})$",
        color=color, fontsize=9, pad=6,
    )
    # Number of terms explicitly distinguishes a 2-chain from its edge support.
    ax.set_xlabel(f"{len(rep.paths)} {'edges' if rep.dimension == 1 else '2-paths'}",
                  fontsize=7, labelpad=6)
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_on()  # NetworkX disables axes; keep the small-panel frame.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.65")
        spine.set_linewidth(0.6)


def plot_homology_representative(
    edgelist: pd.DataFrame | str | Path,
    *,
    nodes: Iterable[Hashable] | None = None,
    max_paths: int = 100_000,
    max_columns: int = 5,
    show_node_labels: bool = True,
    color_by_sign: bool = False,
    output_dir: str | Path | None = None,
    export_vector: bool = False,
    show: bool = False,
) -> tuple[list[HomologyRepresentative], dict[int, plt.Figure]]:
    """Draw H_1/H_2 galleries using a shared circular node layout.

    Return (representatives, figures), with figures keyed by degree 1 and 2.
    Every nonzero interval gets exactly one panel, in #1, #2, ... order from
    left to right, then top to bottom. All graph vertices, including isolates,
    appear at identical positions in every panel. Only edges in the actual
    representative's path support are drawn. H_1 is green and H_2 is blue.
    With color_by_sign=True the network edges instead use red for positive,
    blue for negative and gray for zero weight; degree titles keep their colors.
    Edge signs affect drawing only. To analyze a sign subgraph, pass its selected
    edge table and the complete original node list.

    Output includes 300-DPI PNGs and interval, path-term and node-layout CSVs.
    Set export_vector=True for additional editable SVG/PDF copies.
    Empty dimensions get a labeled placeholder, not a fabricated cycle. Set
    output_dir=None to compute/draw without writing files. Close returned figures
    with plt.close() after use. The default CLI saves to project/output.

    The panel count is the number of positive-length BARCODE INTERVALS in the
    degree, not the final Betti number. A representative is valid throughout
    its own [birth, death) interval; different panels can belong to different
    filtration stages. Each 2D plot is only an edge-support visualization.
    """
    if isinstance(max_columns, bool) or not isinstance(max_columns, Integral) or max_columns < 1:
        raise ValueError("max_columns must be a positive integer")
    if nodes is not None:
        nodes = list(nodes)
    if isinstance(edgelist, (str, Path)):
        edgelist = pd.read_csv(edgelist)
    labels, edges = _prepare_edges(edgelist, nodes)
    signed_weights = _signed_edge_weights(edgelist)
    barcode, representatives = compute_homology_representatives(
        edgelist, nodes=nodes, max_paths=max_paths,
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)
    graph.add_edges_from((labels[u], labels[v]) for u, v in edges)
    positions = nx.circular_layout(graph)
    figures = {}
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context({
        "font.family": "sans-serif", "font.size": 9,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    }):
        for degree, color in ((1, "green"), (2, "blue")):
            group = [rep for rep in representatives if rep.dimension == degree]
            columns = min(max_columns, max(1, len(group)))
            rows = ceil(max(1, len(group)) / columns)
            figure, axes = plt.subplots(
                rows, columns, figsize=(2.3 * columns, 2.5 * rows + 0.35),
                squeeze=False, layout="constrained",
            )
            figures[degree] = figure
            for ax, rep in zip(axes.flat, group):
                _draw_representative(ax, rep, graph, positions, color, show_node_labels,
                                     signed_weights if color_by_sign else None)
            if color_by_sign and group:
                present_signs = {_weight_sign(signed_weights[edge])
                                 for rep in group for edge in _support_edges(rep)}
                sign_labels = {"positive": "Positive (w > 0)", "negative": "Negative (w < 0)",
                               "zero": "Zero (w = 0)"}
                figure.legend(
                    handles=[Line2D([], [], color=SIGN_COLORS[sign], label=sign_labels[sign])
                             for sign in ("positive", "negative", "zero") if sign in present_signs],
                    title="Network edges", title_fontsize=7, fontsize=7,
                    loc="outside lower center", frameon=False,
                )
            for ax in list(axes.flat)[len(group):]:
                ax.set_axis_off()
            if not group:
                axes[0, 0].text(
                    0.5, 0.5, "No nonzero intervals", ha="center", va="center",
                    transform=axes[0, 0].transAxes, fontsize=9,
                )
            noun = "representative" if len(group) == 1 else "representatives"
            figure.suptitle(
                rf"$\beta_{{{degree}}}$: {len(group)} {noun}",
                fontsize=11, color=color,
            )
            if destination is not None:
                figure.savefig(
                    destination / f"homology_representative_beta{degree}.png",
                    dpi=300, bbox_inches="tight",
                )
                if export_vector:
                    figure.savefig(destination / f"homology_representative_beta{degree}.svg",
                                   bbox_inches="tight")
                    figure.savefig(destination / f"homology_representative_beta{degree}.pdf",
                                   bbox_inches="tight")
        if destination is not None:
            # Save the exact barcode used for numbering, from this same run.
            barcode.to_csv(destination / "homology_representative_barcode.csv", index=False)
            _save_chain_tables(representatives, labels, positions, signed_weights, destination)
        if show:
            plt.show()
    return representatives, figures


def main(argv: list[str] | None = None) -> None:
    """IDE Run entry point; no arguments are needed."""
    import argparse

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edgelist", nargs="?", type=Path,
                        default=project_root / "src" / "network_data.py")
    parser.add_argument("--output-dir", type=Path, default=project_root / "output")
    parser.add_argument("--n-nodes", type=int, help="Include vertices 0,...,n-1")
    parser.add_argument("--max-columns", type=int, default=5)
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--color-by-sign", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--vector", action="store_true", help="Also save SVG and PDF copies")
    args = parser.parse_args(argv)
    if args.n_nodes is not None and args.n_nodes < 0:
        parser.error("--n-nodes must be nonnegative")
    source = args.edgelist.resolve()
    edgelist = _load_python_edgelist(source) if source.suffix.lower() == ".py" else source
    representatives, figures = plot_homology_representative(
        edgelist, nodes=range(args.n_nodes) if args.n_nodes is not None else None,
        max_paths=args.max_paths, max_columns=args.max_columns,
        show_node_labels=not args.no_labels, output_dir=args.output_dir,
        color_by_sign=args.color_by_sign,
        export_vector=args.vector, show=args.show,
    )
    # Use the SAME generated frame, even when seed=None, to refresh the barcode
    # image and keep its numbering synchronized with these representatives.
    barcode, barcode_figure, _ = plot_barcode(
        edgelist, max_dim=2,
        nodes=range(args.n_nodes) if args.n_nodes is not None else None,
        max_paths=args.max_paths, save_path=args.output_dir / "barcode.png",
    )
    barcode.to_csv(args.output_dir / "barcode.csv", index=False)
    plt.close(barcode_figure)
    print(f"Input: {source}")
    for degree, figure in figures.items():
        count = sum(rep.dimension == degree for rep in representatives)
        print(f"H_{degree}: {count} interval representatives")
        print((args.output_dir / f"homology_representative_beta{degree}.png").resolve())
        plt.close(figure)
    print("Exact F_2 path chains: " + str((args.output_dir / "homology_representative_paths.csv").resolve()))


if __name__ == "__main__":
    main()
