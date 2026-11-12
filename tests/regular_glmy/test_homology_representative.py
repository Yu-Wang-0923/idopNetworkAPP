"""Validate exact cycles, lifespan, independence and drawing/export behavior."""

import itertools
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from PIL import Image

from idopnetwork.analysis.regular_glmy.plot_barcode import compute_barcode
from idopnetwork.analysis.regular_glmy.plot_homology_representative import (
    compute_homology_representatives, plot_homology_representative,
)
from idopnetwork.analysis.regular_glmy.network_plot_style import SIGN_COLORS
from .test_plot_barcode import dense_rank, dense_spaces, edge_frame


def boundary(chain):
    """Independent regular boundary of an XOR path sum."""
    terms = set()
    for path in chain:
        for i in range(len(path)):
            face = path[:i] + path[i + 1:]
            if all(a != b for a, b in zip(face, face[1:])):
                if face in terms:
                    terms.remove(face)
                else:
                    terms.add(face)
    return terms


class RepresentativeTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_cycle_and_shortcut_have_exact_representatives(self):
        rows = [("a", "b", 1), ("b", "c", 1), ("c", "a", 1)]
        _, reps = compute_homology_representatives(edge_frame(rows))
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].number, 1)
        self.assertEqual(set(reps[0].paths), {("a", "b"), ("b", "c"), ("c", "a")})
        self.assertTrue(np.isinf(reps[0].death))
        rows.append(("a", "c", 2))
        _, reps = compute_homology_representatives(edge_frame(rows))
        self.assertEqual((reps[0].birth, reps[0].death), (1, 2))
        self.assertFalse(boundary(reps[0].paths))

    def test_nonzero_h2_and_death_under_coning(self):
        # Three layers of two vertices: the XOR of all 8 length-2 paths is
        # a nonboundary 2-cycle. Coning the graph at time 2 kills it.
        layers = [("a0", "a1"), ("b0", "b1"), ("c0", "c1")]
        rows = [(u, v, 1) for left, right in zip(layers, layers[1:])
                for u in left for v in right]
        _, reps = compute_homology_representatives(edge_frame(rows))
        h2 = [rep for rep in reps if rep.dimension == 2]
        self.assertEqual(len(h2), 1)
        self.assertEqual(len(h2[0].paths), 8)
        self.assertFalse(boundary(h2[0].paths))
        self.assertTrue(np.isinf(h2[0].death))
        rows += [(v, "tip", 2) for layer in layers for v in layer]
        _, reps = compute_homology_representatives(edge_frame(rows))
        h2 = [rep for rep in reps if rep.dimension == 2]
        self.assertEqual(len(h2), 1)
        self.assertEqual((h2[0].birth, h2[0].death), (1, 2))
        self.assertFalse(boundary(h2[0].paths))
        # An explicit 3-chain bounding the representative at death.
        self.assertEqual(boundary(tuple(path + ("tip",) for path in h2[0].paths)),
                         set(h2[0].paths))

    def test_random_cycles_lifespan_basis_and_barcode_numbering(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            rows = [(u, v, int(rng.integers(1, 4)))
                    for u in range(4) for v in range(4) if u != v and rng.random() < 0.7]
            barcode, reps = compute_homology_representatives(edge_frame(rows), nodes=range(4))
            pd.testing.assert_frame_equal(barcode, compute_barcode(edge_frame(rows), nodes=range(4)))
            paths = {d: [p for p in itertools.product(range(4), repeat=d + 1)
                         if all(a != b for a, b in zip(p, p[1:]))] for d in (1, 2)}
            indices = {d: {p: i for i, p in enumerate(group)} for d, group in paths.items()}
            vectors = {}
            for rep in reps:
                self.assertFalse(boundary(rep.paths), (seed, rep))
                self.assertTrue(rep.paths)
                vector = np.zeros((len(paths[rep.dimension]), 1), dtype=np.uint8)
                for path in rep.paths:
                    vector[indices[rep.dimension][path], 0] ^= 1
                    for u, v in zip(path, path[1:]):
                        self.assertTrue(any(a == u and b == v and abs(w) <= rep.birth
                                            for a, b, w in rows))
                vectors[rep.dimension, rep.number] = vector
            for degree in (1, 2):
                expected = barcode.loc[barcode.dimension == degree].iloc[::-1]
                group = [rep for rep in reps if rep.dimension == degree]
                self.assertEqual(len(expected), len(group))
                for number, (rep, (_, row)) in enumerate(zip(group, expected.iterrows()), start=1):
                    self.assertEqual((rep.number, rep.birth, rep.death),
                                     (number, row.birth, row.death))
            for t in range(4):
                cycles, images = dense_spaces(4, rows, t, 2)
                for degree in (1, 2):
                    image = images[degree]
                    rank = dense_rank(image)
                    alive = [rep for rep in reps if rep.dimension == degree and rep.birth <= t < rep.death]
                    combined = np.column_stack([image] + [vectors[degree, rep.number] for rep in alive])
                    self.assertEqual(dense_rank(combined) - rank, len(alive), (seed, t, degree))
                    self.assertEqual(len(alive), dense_rank(cycles[degree]) - rank)
                    # Finite chains actually become boundaries at the claimed death.
                    for rep in reps:
                        if rep.dimension == degree and rep.death <= t:
                            self.assertEqual(dense_rank(np.column_stack([image, vectors[degree, rep.number]])), rank)

    def test_gallery_export_and_empty_dimension(self):
        frame = edge_frame([(0, 1, 1), (1, 2, 1), (2, 0, 1)])
        frame.attrs["nodes"] = [0, 1, 2, 3]
        with tempfile.TemporaryDirectory() as folder:
            reps, figures = plot_homology_representative(frame, output_dir=folder)
            self.assertEqual(len(reps), 1)
            self.assertEqual(set(figures), {1, 2})
            for degree in (1, 2):
                with Image.open(Path(folder) / f"homology_representative_beta{degree}.png") as image:
                    self.assertAlmostEqual(image.info["dpi"][0], 300, places=2)
            self.assertEqual(len(pd.read_csv(Path(folder) / "homology_representative_paths.csv")), 3)
            self.assertEqual(len(pd.read_csv(Path(folder) / "homology_representative_nodes.csv")), 4)
            self.assertEqual(len(pd.read_csv(Path(folder) / "homology_representative_intervals.csv")), 1)

    def test_no_intervals_and_invalid_columns(self):
        frame = edge_frame([])
        reps, figures = plot_homology_representative(frame, nodes=["isolated"])
        self.assertEqual(reps, [])
        self.assertEqual(len(figures), 2)
        with self.assertRaises(ValueError):
            plot_homology_representative(frame, max_columns=0)

    def test_signed_h2_coloring_preserves_exact_chain_and_exports_support(self):
        layers = [(0, 1), (2, 3), (4, 5)]
        edges = [(u, v) for left, right in zip(layers, layers[1:]) for u in left for v in right]
        frame = edge_frame([(u, v, 1 if index % 2 else -1) for index, (u, v) in enumerate(edges)])
        _, expected = compute_homology_representatives(frame, nodes=range(7))
        with tempfile.TemporaryDirectory() as folder:
            reps, figures = plot_homology_representative(
                frame, nodes=range(7), color_by_sign=True, output_dir=folder,
            )
            self.assertEqual(reps, expected)
            h2 = [rep for rep in reps if rep.dimension == 2]
            self.assertEqual(len(h2), 1)
            self.assertFalse(boundary(h2[0].paths))
            colors = {tuple(patch.get_edgecolor()) for patch in figures[2].axes[0].patches}
            self.assertEqual(colors, {to_rgba(SIGN_COLORS[sign]) for sign in ("positive", "negative")})
            support = pd.read_csv(Path(folder) / "homology_representative_edges.csv")
            support = support.loc[support.dimension == 2]
            self.assertEqual(set(zip(support.source, support.target)), set(edges))
            self.assertEqual(set(support.sign), {"positive", "negative"})
            np.testing.assert_array_equal(support.entry_time, support.weight.abs())


if __name__ == "__main__":
    unittest.main()
