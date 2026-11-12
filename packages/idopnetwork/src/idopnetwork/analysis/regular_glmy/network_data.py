# -*- coding: utf-8 -*-
"""
@author: Yu Wang (yuwang@bimsa.cn)
Generate signed, weighted, directed, sparse network data.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def generate_network_data(
    n: int,
    p: float,
    positive_probability: float = 0.5,
    weight_min: float = 0.1,
    weight_max: float = 1.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate signed, weighted, directed network data.

    Notes
    -----
    Generate a sparse directed Erdős-Rényi graph and independently
    assign a signed random weight to each edge.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of creating each directed edge.
    positive_probability : float, default=0.5
        Probability that an edge has a positive weight.
    weight_min : float, default=0.1
        Minimum absolute edge weight.
    weight_max : float, default=1.0
        Maximum absolute edge weight.
    seed : int or None, default=42
        Random seed.

    Returns
    -------
    edgelist : pandas.DataFrame
        Edge list containing the ``source``, ``target``, and
        signed ``weight`` and ``sign`` columns. The ``sign``
        values are ``positive`` or ``negative``.

    """
    if n < 1:
        raise ValueError("n must be at least 1")

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")

    if not 0.0 <= positive_probability <= 1.0:
        raise ValueError(
            "positive_probability must be between 0 and 1"
        )

    if weight_min <= 0.0:
        raise ValueError("weight_min must be greater than 0")

    if weight_min >= weight_max:
        raise ValueError(
            "weight_min must be less than weight_max"
        )

    graph = nx.fast_gnp_random_graph(
        n=n,
        p=p,
        seed=seed,
        directed=True,
    )

    edgelist = nx.to_pandas_edgelist(
        graph,
        source="source",
        target="target",
    )

    rng = np.random.default_rng(seed)
    n_edges = len(edgelist)

    # Generate absolute weight values.
    magnitudes = rng.uniform(
        low=weight_min,
        high=weight_max,
        size=n_edges,
    )

    # Assign positive or negative signs.
    signs = rng.choice(
        [-1.0, 1.0],
        size=n_edges,
        p=[
            1.0 - positive_probability,
            positive_probability,
        ],
    )

    edgelist["weight"] = magnitudes * signs
    edgelist["sign"] = np.where(
        signs > 0.0,
        "positive",
        "negative",
    )

    edgelist = edgelist[
        ["source", "target", "weight", "sign"]
    ]
    # Keep isolated vertices available to plot_barcode for in-memory input.
    # CSV does not preserve this metadata; pass nodes=range(n) when reading CSV.
    edgelist.attrs["nodes"] = list(graph.nodes)
    return edgelist


def main() -> None:
    edgelist = generate_network_data(
        n=12,
        p=0.7,
        positive_probability=0.5,
        weight_min=0.1,
        weight_max=1.0,
        seed=42,
    )

    output_path = Path.cwd() / "output" / "network.csv"
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    edgelist.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
