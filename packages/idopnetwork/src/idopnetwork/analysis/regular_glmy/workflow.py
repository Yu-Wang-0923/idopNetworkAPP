"""Run the supplied GLMY reference implementation on NetRecon edge tables."""
from __future__ import annotations

import io
from pathlib import Path
import tempfile
import threading
import zipfile

import matplotlib.pyplot as plt
import pandas as pd

from .plot_barcode import plot_barcode
from .plot_filtration_process import plot_filtration_process
from .plot_homology_representative import plot_homology_representative
from .plot_signed_filtration_process import plot_signed_filtration_process

DEFAULT_MODE = "all_edges_colored"
MODES = (DEFAULT_MODE, "separate_signs")
_PLOT_LOCK = threading.RLock()


def run_regular_glmy(
    frame: pd.DataFrame, *, mode: str = DEFAULT_MODE, n_stages: int = 4,
    max_paths: int = 100_000, max_columns: int = 5,
    thresholds: list[float] | None = None, export_vector: bool = False,
) -> dict:
    """Return image/table bytes and a reproducible ZIP; never persist user inputs."""
    if mode not in MODES:
        raise ValueError(f"Unknown GLMY mode: {mode}")
    edges = frame.rename(columns={"from": "source", "to": "target", "Effect": "weight"}).copy()
    if not {"source", "target", "weight"} <= set(edges.columns):
        raise ValueError("需要 from/to/weight 或 source/target/weight 三列。")
    # NetRecon adjacency exports may include diagonal entries. Retain their
    # vertices as isolates, but exclude self-loops from the regular path input.
    nodes = list(dict.fromkeys(edges.source.tolist() + edges.target.tolist()))
    loops = edges.source.eq(edges.target)
    removed_loops = int(loops.sum())
    edges = edges.loc[~loops, ["source", "target", "weight"]].copy()
    edges.attrs["nodes"] = nodes
    with _PLOT_LOCK, tempfile.TemporaryDirectory(prefix="idop_glmy_") as folder:
        destination = Path(folder)
        before = set(plt.get_fignums())
        try:
            if mode == DEFAULT_MODE:
                output = destination / "01_all_edges_colored"
                output.mkdir()
                edges.to_csv(output / "network.csv", index=False)
                plot_filtration_process(
                    edges, nodes=nodes, color_by_sign=True, n_stages=n_stages,
                    filtration_values=thresholds, max_paths=max_paths,
                    save_path=output / "filtration_process.png", export_vector=export_vector,
                )
                plot_homology_representative(
                    edges, nodes=nodes, color_by_sign=True, max_paths=max_paths,
                    max_columns=max_columns, output_dir=output, export_vector=export_vector,
                )
            else:
                plot_signed_filtration_process(
                    edges, nodes=nodes, n_stages=n_stages, filtration_values=thresholds,
                    max_paths=max_paths, max_columns=max_columns,
                    output_dir=destination, export_vector=export_vector,
                )
            for output in sorted(destination.rglob('network.csv')):
                # Use the original labels rather than CSV's inferred label type.
                sign = output.parent.name
                selected = edges if sign not in {"positive", "negative"} else edges.loc[
                    edges.weight.gt(0) if sign == "positive" else edges.weight.lt(0)]
                barcode, figure, _ = plot_barcode(selected, nodes=nodes, max_paths=max_paths,
                                            save_path=output.parent / "barcode.png")
                barcode.to_csv(output.parent / "barcode.csv", index=False)
                if export_vector:
                    for suffix in ('svg', 'pdf'):
                        figure.savefig(output.parent / f"barcode.{suffix}", bbox_inches="tight")
            artifacts = {p.relative_to(destination).as_posix(): p.read_bytes()
                         for p in sorted(destination.rglob('*')) if p.is_file()}
        finally:
            for number in set(plt.get_fignums()) - before:
                plt.close(number)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for name, content in artifacts.items():
            output.writestr(name, content)
    return {"mode": mode, "artifacts": artifacts, "zip": archive.getvalue(),
            "removed_self_loops": removed_loops, "n_nodes": len(nodes)}
