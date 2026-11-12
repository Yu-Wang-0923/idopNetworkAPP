"""Regular GLMY persistent path homology over F_2 for weighted digraphs.

The filtration is E_t = {(u, v): abs(weight(u, v)) <= t}, t >= 0.
All vertices are born at zero. This is ordinary (unreduced) path homology.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HomologyRepresentative:
    """One interval's exact F_2 cycle, as an XOR sum of elementary paths.

    ``number`` matches the barcode's bottom-to-top numbering in this degree.
    Finite intervals use the reduced death column; essential intervals use a
    zero-column transformation cycle. Representatives are not minimal/unique.
    """

    dimension: int
    number: int
    birth: float
    death: float
    paths: tuple[tuple[Hashable, ...], ...]


def _bits(vector: int):
    """Indices of nonzero entries in an F_2 vector packed into an integer."""
    while vector:
        bit = vector & -vector
        yield bit.bit_length() - 1
        vector ^= bit


class _Basis:
    """Echelon basis with coordinates in the original, filtered basis."""

    def __init__(self):
        self.rows: dict[int, tuple[int, int]] = {}

    def reduce(self, vector: int) -> tuple[int, int]:
        coordinates = 0
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in self.rows:
                break
            row, tags = self.rows[pivot]
            vector ^= row
            coordinates ^= tags
        return vector, coordinates

    def add(self, vector: int, index: int) -> bool:
        remainder, coordinates = self.reduce(vector)
        if not remainder:
            return False
        self.rows[remainder.bit_length() - 1] = (
            remainder, coordinates ^ (1 << index)
        )
        return True

    def coordinates(self, vector: int) -> int:
        remainder, coordinates = self.reduce(vector)
        if remainder:
            raise RuntimeError("Boundary is outside the preceding Omega space")
        return coordinates


def _kernel(columns: Iterable[tuple[int, int]]):
    """Kernel vectors, tagged by the ambient index of each input column."""
    basis = _Basis()
    for index, column in columns:
        remainder, coordinates = basis.reduce(column)
        tags = coordinates ^ (1 << index)
        if remainder:
            basis.rows[remainder.bit_length() - 1] = (remainder, tags)
        else:
            yield tags


def _regular_faces(path: tuple[int, ...]):
    for i in range(len(path)):
        face = path[:i] + path[i + 1:]
        # Nonregular faces vanish in the regular path quotient. Repeated
        # NONADJACENT vertices are allowed, e.g. (u, v, u).
        if all(a != b for a, b in zip(face, face[1:])):
            yield face


def _prepare_edges(edgelist, nodes):
    if isinstance(edgelist, (str, Path)):
        edgelist = pd.read_csv(edgelist)
    if not isinstance(edgelist, pd.DataFrame):
        raise TypeError("edgelist must be a pandas DataFrame or a CSV path")
    required = ["source", "target", "weight"]
    if not edgelist.columns.is_unique or not set(required) <= set(edgelist.columns):
        raise ValueError("edgelist needs unique source, target and weight columns")
    if edgelist[required].isna().any().any():
        raise ValueError("source, target and weight cannot contain missing values")
    try:
        weights = np.asarray(edgelist["weight"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be finite real numbers") from exc
    if not np.isfinite(weights).all():
        raise ValueError("weights must be finite real numbers")

    node_ids = {}

    def add_node(node):
        if not isinstance(node, Hashable):
            raise ValueError("node labels must be hashable")
        missing = pd.isna(node)
        if np.ndim(missing) == 0 and bool(missing):
            raise ValueError("node labels cannot be missing")
        if node not in node_ids:
            node_ids[node] = len(node_ids)
        return node_ids[node]

    # DataFrame metadata preserves isolated vertices from our generator.
    # CSV does not preserve attrs; supply nodes explicitly after CSV loading.
    for node in (nodes if nodes is not None else edgelist.attrs.get("nodes", [])):
        add_node(node)
    edges = {}
    for (source, target), weight in zip(
        edgelist[["source", "target"]].itertuples(index=False, name=None), weights
    ):
        u, v = add_node(source), add_node(target)
        if u == v:
            raise ValueError("Self-loops are not supported; use a loopless digraph")
        edge = (u, v)
        # Parallel observations enter at the earliest absolute weight.
        edges[edge] = min(edges.get(edge, np.inf), abs(float(weight)))
    return list(node_ids), edges


def compute_barcode(
    edgelist: pd.DataFrame | str | Path,
    max_dim: int = 2,
    *,
    nodes: Iterable[Hashable] | None = None,
    max_paths: int = 100_000,
    return_representatives: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[HomologyRepresentative]]:
    """Compute GLMY persistence intervals [birth, death) over F_2.

    Parameters
    ----------
    edgelist : DataFrame or CSV path
        Columns ``source``, ``target``, ``weight``. Edge directions are kept;
        absolute weights determine entry times. The ``sign`` column is ignored.
        Duplicate directed edges are merged at their minimum absolute weight.
        Self-loops and nonfinite weights are rejected.
    max_dim : int, default 2
        Return H_0 through H_max_dim. Paths through degree max_dim + 1 are
        constructed so that deaths in the highest requested degree are included.
    nodes : iterable, optional
        Additional/full vertex set, including isolated vertices. Otherwise use
        DataFrame.attrs['nodes'] if present, together with all edge endpoints.
    max_paths : int, default 100000
        Limit on total allowed elementary paths across all constructed degrees.
        Raises ValueError instead of returning a truncated approximation.
    return_representatives : bool, default False
        Also track cycle representatives and return (barcode, representatives).
        This uses additional memory for the column transformation matrix.

    Returns
    -------
    pandas.DataFrame
        Columns ``dimension``, ``birth``, ``death``; infinite deaths are np.inf.
        Zero-length intervals are omitted. Vertices are born at t=0.

    Notes
    -----
    At each edge threshold, construct Omega_p = {c in A_p: boundary(c) in
    A_{p-1}} as the kernel of the forbidden-face boundary matrix. Extend the
    previous Omega basis, express boundaries in this filtration-compatible
    basis, and apply ordinary persistence column reduction. All arithmetic is
    exact modulo two. This reference implementation targets small sparse graphs;
    path counts grow rapidly with degree and graph density.
    """
    if isinstance(max_dim, bool) or not isinstance(max_dim, Integral) or max_dim < 0:
        raise ValueError("max_dim must be a nonnegative integer")
    if isinstance(max_paths, bool) or not isinstance(max_paths, Integral) or max_paths < 1:
        raise ValueError("max_paths must be a positive integer")
    node_labels, edges = _prepare_edges(edgelist, nodes)
    n = len(node_labels)
    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)

    paths = [[(u,) for u in range(n)]]
    times = [[0.0] * n]
    path_count = n
    if path_count > max_paths:
        raise ValueError("max_paths exceeded by the vertex set")
    for degree in range(1, max_dim + 2):
        new_paths, new_times = [], []
        for path, birth in zip(paths[-1], times[-1]):
            for v in adjacency[path[-1]]:
                path_count += 1
                if path_count > max_paths:
                    raise ValueError("max_paths exceeded; reduce max_dim or graph size")
                new_paths.append(path + (v,))
                new_times.append(max(birth, edges[path[-1], v]))
        paths.append(new_paths)
        times.append(new_times)

    # Allowed paths come first in each ambient space; add forbidden regular
    # faces as extra rows so cancellations between paths are accounted for.
    ambient = [{path: i for i, path in enumerate(group)} for group in paths]
    boundaries = [[0] * n]
    for degree in range(1, max_dim + 2):
        lower = ambient[degree - 1]
        columns = []
        for path in paths[degree]:
            column = 0
            for face in _regular_faces(path):
                if face not in lower:
                    lower[face] = len(lower)
                column ^= 1 << lower[face]
            columns.append(column)
        boundaries.append(columns)

    omega = [_Basis() for _ in paths]
    reducers: dict[int, int] = {}
    births: dict[int, tuple[int, float]] = {}
    intervals = []
    representative_vectors = []
    chain_vectors = {}
    reducer_transforms = {}
    birth_cycles = {}

    def ambient_cycle(coordinates):
        vector = 0
        for index in _bits(coordinates):
            vector ^= chain_vectors[index]
        return vector

    chain_index = 0
    for threshold in sorted({0.0, *edges.values()}):
        allowed = [
            sum(1 << i for i, birth in enumerate(group) if birth <= threshold)
            for group in times
        ]
        for degree in range(max_dim + 2):
            if degree == 0:
                candidates = (1 << i for i in range(n))
            else:
                forbidden = ((1 << len(ambient[degree - 1])) - 1) ^ allowed[degree - 1]
                candidates = _kernel(
                    (i, boundaries[degree][i] & forbidden)
                    for i in _bits(allowed[degree])
                )
            for vector in candidates:
                if not omega[degree].add(vector, chain_index):
                    continue
                if return_representatives:
                    chain_vectors[chain_index] = vector
                transform = (1 << chain_index) if return_representatives else 0
                boundary = 0
                for i in _bits(vector):
                    boundary ^= boundaries[degree][i]
                column = omega[degree - 1].coordinates(boundary) if degree else 0
                while column:
                    pivot = column.bit_length() - 1
                    if pivot not in reducers:
                        break
                    column ^= reducers[pivot]
                    if return_representatives:
                        transform ^= reducer_transforms[pivot]
                if column:
                    pivot = column.bit_length() - 1
                    reducers[pivot] = column
                    if return_representatives:
                        reducer_transforms[pivot] = transform
                    birth_degree, birth_time = births.pop(pivot)
                    if birth_degree <= max_dim and birth_time < threshold:
                        intervals.append((birth_degree, birth_time, threshold))
                        if return_representatives:
                            # This cycle is born at the pivot and becomes an
                            # actual boundary at this death, unlike an arbitrary
                            # birth cycle which can merge into an older class.
                            representative_vectors.append(ambient_cycle(column))
                    if return_representatives:
                        birth_cycles.pop(pivot, None)
                else:
                    births[chain_index] = (degree, threshold)
                    if return_representatives:
                        birth_cycles[chain_index] = transform
                chain_index += 1

    for index, (degree, birth) in births.items():
        if degree <= max_dim:
            intervals.append((degree, birth, np.inf))
            if return_representatives:
                representative_vectors.append(ambient_cycle(birth_cycles[index]))
    result = pd.DataFrame(intervals, columns=["dimension", "birth", "death"])
    result = result.astype({"dimension": int, "birth": float, "death": float})
    result = result.sort_values(["dimension", "birth", "death"], kind="stable")
    representatives = []
    if return_representatives:
        for degree in range(max_dim + 1):
            group = result.loc[result.dimension == degree].iloc[::-1]
            for number, (original_index, row) in enumerate(group.iterrows(), start=1):
                cycle = representative_vectors[original_index]
                labelled_paths = tuple(
                    tuple(node_labels[v] for v in paths[degree][i])
                    for i in _bits(cycle)
                )
                representatives.append(HomologyRepresentative(
                    degree, number, float(row.birth), float(row.death), labelled_paths,
                ))
    result = result.reset_index(drop=True)
    result.attrs.update(
        coefficient_field="F_2", filtration="abs(weight) <= t",
        vertex_birth=0.0, max_dim=int(max_dim),
        final_threshold=max(edges.values(), default=0.0),
    )
    if return_representatives:
        return result, representatives
    return result


def plot_barcode(
    edgelist: pd.DataFrame | str | Path,
    max_dim: int = 2,
    *,
    nodes: Iterable[Hashable] | None = None,
    max_paths: int = 100_000,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[pd.DataFrame, plt.Figure, np.ndarray]:
    """Compute and plot GLMY barcodes; return (barcode, figure, axes).

    See compute_barcode for mathematical conventions and input handling.
    Each row of the returned DataFrame is an interval [birth, death).
    Infinite intervals end with an arrow labeled +infinity. ``axes`` is always a
    one-dimensional array, including when max_dim=0. No GUI opens unless show=True.
    Subplots and returned axes run from max_dim at the top to zero at the bottom,
    with math-rendered beta labels. Finite deaths use filled square markers.
    Bars are drawn in reverse table order, from bottom to top.
    Y ticks number intervals #1, #2, ... from bottom to top, not vertex IDs.
    H_0 bars are red with black ticks; H_1 bars and ticks are green;
    H_2 bars and ticks are blue. Default output includes H_0 through H_2.
    Births use black hollow circles, deaths black filled squares or arrows.
    Each subplot's plotting box has a fixed width:height ratio of 4:2,
    and adjacent plotting boxes share a border without vertical whitespace.
    """
    barcode = compute_barcode(edgelist, max_dim, nodes=nodes, max_paths=max_paths)
    # Match the available grid height to the fixed box aspect, so Matplotlib
    # does not shrink the axes inside taller cells and leave gaps between them.
    figure_width = 4.5
    left_margin, right_margin = 0.7, 0.2  # inches
    bottom_margin, top_margin = 0.45, 0.12
    panel_height = (figure_width - left_margin - right_margin) * (2 / 4)
    figure_height = panel_height * (max_dim + 1) + bottom_margin + top_margin
    figure, axes_grid = plt.subplots(
        max_dim + 1, 1, figsize=(figure_width, figure_height),
        sharex=True, squeeze=False, layout="none",
    )
    figure.subplots_adjust(
        left=left_margin / figure_width,
        right=1 - right_margin / figure_width,
        bottom=bottom_margin / figure_height,
        top=1 - top_margin / figure_height,
        hspace=0,
    )
    axes = axes_grid[:, 0]
    last = barcode.attrs["final_threshold"]
    span = last if last > 0 else 1.0
    infinity_end = last + 0.13 * span
    for degree, ax in zip(range(max_dim, -1, -1), axes):
        ax.set_box_aspect(2 / 4)  # Width:height = 4:2; API takes height / width.
        bars = barcode.loc[barcode["dimension"] == degree]
        color = {0: "red", 1: "green", 2: "blue"}.get(degree, f"C{degree % 10}")
        tick_color = {0: "black", 1: "green", 2: "blue"}.get(degree, "black")
        for row, (_, birth, death) in enumerate(bars.iloc[::-1].itertuples(index=False, name=None)):
            end = infinity_end if np.isinf(death) else death
            ax.hlines(row, birth, end, color=color, linewidth=2)
            ax.plot(
                birth, row, "o", color="black", markerfacecolor="white",
                markersize=4,
            )
            if np.isinf(death):
                ax.plot(end, row, ">", color="black", markersize=6)
                ax.annotate(
                    r"$+\infty$", xy=(end, row),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=12, color="black",
                )
            else:
                ax.plot(end, row, "s", color="black", markersize=4)
        if bars.empty:
            ax.text(0.5, 0.5, "No nonzero intervals", ha="center", transform=ax.transAxes)
        ax.set_ylim(-1, max(1, len(bars)))
        tick_rows = np.linspace(0, len(bars) - 1, min(4, len(bars)), dtype=int)
        ax.set_yticks(tick_rows, labels=[f"#{i + 1}" for i in tick_rows])
        ax.tick_params(axis="y", colors=tick_color, labelsize=8)
        ax.set_ylabel(rf"$\beta_{{{degree}}}$", labelpad=8)
        ax.grid(False, which="both", axis="both")
        ax.minorticks_off()
        ax.set_xlim(-0.09 * span, last + 0.50 * span)
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    axes[-1].set_xlabel("filtration")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    # figure.suptitle(r"GLMY regular path homology over $\mathbb{F}_2$", fontsize=11)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return barcode, figure, axes


def _load_python_edgelist(source: Path) -> pd.DataFrame:
    """Accept main() returning a frame or calling generate_network_data().

    Capture the generator's returned frame during the actual main() call so
    scripts that only save CSV also work, with their parameters and node attrs.
    No stale CSV is read, and the generator is not called a second time.
    """
    import runpy
    from functools import wraps

    namespace = runpy.run_path(str(source))
    generator_main = namespace.get("main")
    if not callable(generator_main):
        raise ValueError(f"{source} must define main()")
    scope = generator_main.__globals__
    generator = scope.get("generate_network_data")
    generated_frames = []

    if callable(generator):
        @wraps(generator)
        def capture(*args, **kwargs):
            frame = generator(*args, **kwargs)
            if isinstance(frame, pd.DataFrame):
                generated_frames.append(frame)
            return frame

        scope["generate_network_data"] = capture
    try:
        result = generator_main()
    finally:
        if callable(generator):
            scope["generate_network_data"] = generator
    if isinstance(result, pd.DataFrame):
        return result
    if result is None and len(generated_frames) == 1:
        return generated_frames[0]
    raise ValueError(
        f"main() in {source} must return an edge-list DataFrame or call "
        "generate_network_data() exactly once to produce one"
    )


def main(argv: list[str] | None = None) -> None:
    """Run directly in an IDE, or accept an optional Python/CSV data source."""
    import argparse

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "edgelist", nargs="?", type=Path,
        default=Path(__file__).with_name("network_data.py"),
        help="CSV edge list or Python script with main() generating an edge list; "
             "defaults to src/network_data.py",
    )
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--n-nodes", type=int, help="Include vertices 0,...,n-1 (including isolates)")
    parser.add_argument("--output", type=Path, default=project_root / "output" / "barcode.png")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true", help="Also open a plot window")
    display.add_argument("--no-show", dest="show", action="store_false", help="Save without opening a plot window (default)")
    parser.set_defaults(show=False)
    args = parser.parse_args(argv)
    if args.n_nodes is not None and args.n_nodes < 0:
        parser.error("--n-nodes must be nonnegative")
    source = args.edgelist.resolve()
    if source.suffix.lower() == ".py":
        try:
            edgelist = _load_python_edgelist(source)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        edgelist = source
    intervals, fig, _ = plot_barcode(
        edgelist, max_dim=args.max_dim,
        nodes=range(args.n_nodes) if args.n_nodes is not None else None,
        save_path=args.output,
    )
    intervals.to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Input: {source}")
    print(intervals.to_string(index=False))
    print(f"Barcode figure: {args.output.resolve()}")
    print(f"Barcode intervals: {args.output.with_suffix('.csv').resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
