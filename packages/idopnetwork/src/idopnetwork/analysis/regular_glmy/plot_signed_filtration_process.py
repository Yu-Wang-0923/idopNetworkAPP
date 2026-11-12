"""Compare sign coloring and separate positive/negative GLMY filtrations.

Click Run to load src/network_data.py once and save recovery-process figures
and H1/H2 representative galleries (300 DPI) in output/sign_comparison.
All three views share vertices, layout and recovery thresholds.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from .plot_barcode import _load_python_edgelist
    from .plot_filtration_process import (
        FiltrationSnapshot, _select_edges, _thresholds, plot_filtration_process,
    )
    from .plot_homology_representative import plot_homology_representative
else:
    from plot_barcode import _load_python_edgelist
    from plot_filtration_process import (
        FiltrationSnapshot, _select_edges, _thresholds, plot_filtration_process,
    )
    from plot_homology_representative import plot_homology_representative


# Defaults for the editor's Run button.
N_STAGES = 4
FILTRATION_VALUES = None  # Or e.g. [0.0, 0.3, 0.6, 1.0], shared by all views.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "sign_comparison"
VIEW_FOLDERS = {
    "all": Path("01_all_edges_colored"),
    "positive": Path("02_separate_signs") / "positive",
    "negative": Path("02_separate_signs") / "negative",
}


def plot_signed_filtration_process(
    edgelist: pd.DataFrame | str | Path,
    *,
    nodes: Iterable[Hashable] | None = None,
    filtration_values: Sequence[float] | None = None,
    n_stages: int = 4,
    columns: int | None = None,
    max_columns: int = 5,
    max_paths: int = 100_000,
    output_dir: str | Path = OUTPUT_DIR,
    export_vector: bool = False,
    show: bool = False,
) -> dict[str, tuple[pd.DataFrame, list[FiltrationSnapshot], plt.Figure]]:
    """Save a paired comparison using a single signed edge list.

    The all-edge view only changes colors. The positive and negative views
    each recompute ordinary F_2 GLMY homology on their selected subgraph.
    Every view uses abs(weight) <= t and retains all original vertices.
    Thresholds are chosen once from entry times across the three views, so even
    an empty sign subgraph has the same stages and node positions as the others.
    Each subdirectory also receives H1/H2 representative PNGs and exact chain
    tables, computed on that view's selected edge table and complete vertex set.
    Network edges use sign colors in both recovery and representative figures.
    Empty degrees receive a labeled placeholder. max_columns controls gallery
    wrapping; columns controls the recovery stages only.

    Return {"all": (barcode, snapshots, figure), "positive": ..., "negative": ...}.
    Close the returned recovery figures after use. Gallery figures are saved
    and closed internally (after display if show=True). The input is not modified.
    """
    if nodes is not None:
        nodes = list(nodes)
    frame, labels, _ = _select_edges(edgelist, nodes, "all")
    selected_views, entry_times = {}, {}
    for selection in VIEW_FOLDERS:
        selected, _, selected_edges = _select_edges(frame, labels, selection)
        selected_views[selection] = (selected, selected_edges)
        # Opposite-sign parallel observations can enter later in a sign-only
        # graph than in the merged all-edge graph. Include those times too.
        entry_times.update({(selection, edge): t for edge, t in selected_edges.items()})
    values = _thresholds(entry_times, filtration_values, n_stages)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "input_edges.csv", index=False)
    pd.DataFrame({"node": labels}).to_csv(output / "input_nodes.csv", index=False)
    results = {}
    representative_figures = []
    stage_rows, selection_rows = [], []
    try:
        for selection, folder in VIEW_FOLDERS.items():
            destination = output / folder
            barcode, snapshots, figure = plot_filtration_process(
                frame, nodes=labels, edge_sign=selection, color_by_sign=True,
                filtration_values=values, n_stages=n_stages, columns=columns,
                max_paths=max_paths, save_path=destination / "filtration_process.png",
                export_vector=export_vector,
            )
            results[selection] = (barcode, snapshots, figure)
            selected, selected_edges = selected_views[selection]
            selected.to_csv(destination / "network.csv", index=False)
            representatives, galleries = plot_homology_representative(
                selected, nodes=labels, max_paths=max_paths, max_columns=max_columns,
                color_by_sign=True, output_dir=destination, export_vector=export_vector,
            )
            representative_figures.extend(galleries.values())
            selection_rows.append({
                "selection": selection, "n_vertices": len(labels),
                "input_rows": len(selected), "n_directed_edges": len(selected_edges),
                "positive_rows": int((selected.weight.astype(float) > 0).sum()),
                "negative_rows": int((selected.weight.astype(float) < 0).sum()),
                "zero_rows": int((selected.weight.astype(float) == 0).sum()),
                **{f"beta{degree}_intervals": int((barcode.dimension == degree).sum())
                   for degree in range(3)},
                **{f"beta{degree}_representatives": sum(rep.dimension == degree for rep in representatives)
                   for degree in (1, 2)},
            })
            stage_rows.extend({
                "selection": selection, "stage": stage.number, "threshold": stage.threshold,
                "n_vertices": len(labels), "n_edges": len(stage.edges),
                "beta0": stage.betti[0], "beta1": stage.betti[1], "beta2": stage.betti[2],
            } for stage in snapshots)
        pd.DataFrame(selection_rows).to_csv(output / "selection_summary.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(output / "comparison_snapshots.csv", index=False)
        if show:
            plt.show()
    except Exception:
        for _, _, figure in results.values():
            plt.close(figure)
        raise
    finally:
        for figure in representative_figures:
            plt.close(figure)
    return results


def main(argv: list[str] | None = None) -> None:
    """Run both methods once; no preceding scripts need to be run."""
    import argparse

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edgelist", nargs="?", type=Path, default=root / "src" / "network_data.py")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--thresholds", nargs="+", type=float, default=FILTRATION_VALUES)
    parser.add_argument("--n-stages", type=int, default=N_STAGES)
    parser.add_argument("--n-nodes", type=int)
    parser.add_argument("--columns", type=int, default=None)
    parser.add_argument("--max-columns", type=int, default=5,
                        help="Maximum panels per row in representative galleries")
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--vector", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.n_nodes is not None and args.n_nodes < 0:
        parser.error("--n-nodes must be nonnegative")
    source = args.edgelist.resolve()
    # Generate only once, including when the user's generator uses seed=None.
    frame = _load_python_edgelist(source) if source.suffix.lower() == ".py" else source
    results = plot_signed_filtration_process(
        frame, nodes=range(args.n_nodes) if args.n_nodes is not None else None,
        filtration_values=args.thresholds, n_stages=args.n_stages, columns=args.columns,
        max_columns=args.max_columns,
        max_paths=args.max_paths, output_dir=args.output_dir,
        export_vector=args.vector, show=args.show,
    )
    print(f"Input: {source}")
    for selection, (barcode, snapshots, figure) in results.items():
        print(f"{selection}: " + "; ".join(
            f"t={s.threshold:.6g}, edges={len(s.edges)}, beta={s.betti}" for s in snapshots
        ))
        print(f"Figure: {(args.output_dir / VIEW_FOLDERS[selection] / 'filtration_process.png').resolve()}")
        for degree in (1, 2):
            count = int((barcode.dimension == degree).sum())
            destination = args.output_dir / VIEW_FOLDERS[selection] / f"homology_representative_beta{degree}.png"
            print(f"H_{degree}: {count} representatives; {destination.resolve()}")
        plt.close(figure)


if __name__ == "__main__":
    main()
