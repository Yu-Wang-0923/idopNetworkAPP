"""Analytical examples and independent dense persistence-rank checks."""

import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idopnetwork.analysis.regular_glmy.network_data import generate_network_data
from idopnetwork.analysis.regular_glmy.plot_barcode import _load_python_edgelist, compute_barcode, plot_barcode


def edge_frame(rows):
    return pd.DataFrame(rows, columns=["source", "target", "weight"])


def dense_rref(matrix):
    matrix = matrix.copy() % 2
    pivots = []
    for column in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[len(pivots):, column])
        if not candidates.size:
            continue
        row = len(pivots)
        swap = row + candidates[0]
        matrix[[row, swap]] = matrix[[swap, row]]
        for other in range(matrix.shape[0]):
            if other != row and matrix[other, column]:
                matrix[other] ^= matrix[row]
        pivots.append(column)
    return matrix, pivots


def dense_kernel(matrix):
    reduced, pivots = dense_rref(matrix)
    free = [i for i in range(matrix.shape[1]) if i not in pivots]
    result = np.zeros((matrix.shape[1], len(free)), dtype=np.uint8)
    for k, column in enumerate(free):
        result[column, k] = 1
        for row, pivot in enumerate(pivots):
            result[pivot, k] = reduced[row, column]
    return result


def dense_rank(matrix):
    return len(dense_rref(matrix)[1])


def dense_spaces(n, rows, threshold, max_dim):
    """Independent reference: enumerate ALL regular paths, even forbidden ones."""
    edges = {(u, v) for u, v, w in rows if abs(w) <= threshold}
    paths = [
        [p for p in itertools.product(range(n), repeat=d + 1)
         if all(a != b for a, b in zip(p, p[1:]))]
        for d in range(max_dim + 2)
    ]
    allowed = [
        np.array([all((a, b) in edges for a, b in zip(p, p[1:])) for p in group])
        for group in paths
    ]
    boundary = [np.zeros((0, n), dtype=np.uint8)]
    omega = [np.eye(n, dtype=np.uint8)]
    for d in range(1, max_dim + 2):
        lower = {p: i for i, p in enumerate(paths[d - 1])}
        matrix = np.zeros((len(lower), len(paths[d])), dtype=np.uint8)
        for j, path in enumerate(paths[d]):
            for k in range(len(path)):
                face = path[:k] + path[k + 1:]
                if face in lower:
                    matrix[lower[face], j] ^= 1
        boundary.append(matrix)
        admissible = np.flatnonzero(allowed[d])
        kernel = dense_kernel(matrix[np.ix_(~allowed[d - 1], admissible)])
        basis = np.zeros((len(paths[d]), kernel.shape[1]), dtype=np.uint8)
        basis[admissible] = kernel
        omega.append(basis)
    cycles = [(omega[d] @ dense_kernel(boundary[d] @ omega[d] % 2)) % 2
              for d in range(max_dim + 1)]
    images = [boundary[d + 1] @ omega[d + 1] % 2 for d in range(max_dim + 1)]
    return cycles, images


class BarcodeTests(unittest.TestCase):
    def bars(self, rows, degree=1, **kwargs):
        result = compute_barcode(edge_frame(rows), max_dim=max(1, degree), **kwargs)
        return list(result.loc[result.dimension == degree, ["birth", "death"]]
                    .itertuples(index=False, name=None))

    def test_directed_cycle_has_essential_h1(self):
        self.assertEqual(self.bars([(0, 1, 1), (1, 2, 1), (2, 0, 1)]), [(1, np.inf)])

    def test_shortcut_kills_cycle(self):
        rows = [(0, 1, 1), (1, 2, 1), (2, 0, 1), (0, 2, 2)]
        self.assertEqual(self.bars(rows), [(1, 2)])

    def test_transitive_triangle_is_filled(self):
        self.assertEqual(self.bars([(0, 1, 1), (1, 2, 1), (0, 2, 1)]), [])

    def test_diamond_requires_combination_of_paths(self):
        # Neither 012 nor 032 has an allowed boundary individually, but their
        # sum does. A directed flag complex would incorrectly leave an H1 bar.
        self.assertEqual(self.bars([(0, 1, 1), (1, 2, 1), (0, 3, 1), (3, 2, 1)]), [])

    def test_reciprocal_edges_use_regular_backtracking_path(self):
        self.assertEqual(self.bars([(0, 1, 1), (1, 0, 2)]), [])

    def test_h0_components_and_isolated_vertex(self):
        self.assertEqual(self.bars([(0, 1, 1), (1, 2, 2)], degree=0, nodes=range(4)),
                         [(0, 1), (0, 2), (0, np.inf), (0, np.inf)])

    def test_signs_duplicates_and_order(self):
        rows = [(0, 1, -1), (1, 2, -1), (2, 0, 1), (0, 2, -2), (0, 2, 3)]
        self.assertEqual(self.bars(rows), [(1, 2)])
        pd.testing.assert_frame_equal(compute_barcode(edge_frame(rows)),
                                      compute_barcode(edge_frame(rows[::-1])))

    def test_empty_graph_and_generator_metadata(self):
        empty = generate_network_data(n=4, p=0)
        self.assertEqual(len(compute_barcode(empty)), 4)
        self.assertTrue(compute_barcode(edge_frame([])).empty)
        self.assertEqual(self.bars([], degree=0, nodes=["a", "b"]),
                         [(0, np.inf), (0, np.inf)])

    def test_zero_weight_and_h0_only(self):
        result = compute_barcode(edge_frame([("a", "b", 0)]), max_dim=0)
        self.assertEqual(result.values.tolist(), [[0, 0, np.inf]])

    def test_invalid_inputs(self):
        for rows in [[(0, 0, 1)], [(0, 1, np.inf)], [(0, 1, np.nan)], [(None, 1, 1)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                compute_barcode(edge_frame(rows))
        for dimension in [-1, 0.5, True]:
            with self.assertRaises(ValueError):
                compute_barcode(edge_frame([]), max_dim=dimension)
        with self.assertRaises(ValueError):
            compute_barcode(edge_frame([(0, 1, 1)]), max_paths=2)
        with self.assertRaises(ValueError):
            compute_barcode(pd.DataFrame({"source": [0]}))

    def test_csv_and_plot(self):
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "network.csv"
            png = Path(folder) / "nested" / "barcode.png"
            edge_frame([(0, 1, 1), (1, 2, 1), (2, 0, 1)]).to_csv(csv, index=False)
            result, fig, axes = plot_barcode(csv, nodes=range(4), save_path=png)
            self.assertEqual(axes.shape, (3,))
            self.assertTrue(png.stat().st_size > 1000)
            self.assertEqual(len(result[result.dimension == 1]), 1)
            plt.close(fig)
            _, fig, axes = plot_barcode(edge_frame([]), max_dim=0)
            self.assertEqual(axes.shape, (1,))
            plt.close(fig)

    def test_random_persistence_ranks_including_h2(self):
        # Compare every inclusion-induced rank, not merely Betti curves.
        for seed in range(12):
            rng = np.random.default_rng(seed)
            rows = [(u, v, int(rng.integers(1, 4)))
                    for u in range(4) for v in range(4) if u != v and rng.random() < 0.65]
            barcode = compute_barcode(edge_frame(rows), max_dim=2, nodes=range(4))
            spaces = [dense_spaces(4, rows, t, 2) for t in range(4)]
            for start in range(4):
                for end in range(start, 4):
                    for degree in range(3):
                        cycles = spaces[start][0][degree]
                        boundaries = spaces[end][1][degree]
                        expected = dense_rank(np.column_stack([cycles, boundaries])) - dense_rank(boundaries)
                        observed = ((barcode.dimension == degree) &
                                    (barcode.birth <= start) & (barcode.death > end)).sum()
                        self.assertEqual(expected, observed, (seed, start, end, degree))

    def test_run_without_input_from_another_working_directory(self):
        project_root = Path(__import__("idopnetwork.analysis.regular_glmy.plot_barcode", fromlist=["__file__"]).__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "barcode.png"
            completed = subprocess.run(
                [sys.executable, str(project_root / "plot_barcode.py"),
                 "--output", str(output)],
                cwd=folder, env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(str(project_root / "network_data.py"), completed.stdout)
            self.assertGreater(output.stat().st_size, 1000)
            # Follow the current data script's settings, including isolated
            # nodes, rather than assuming its original 12-node example.
            generated = _load_python_edgelist(project_root / "network_data.py")
            self.assertIsInstance(generated, pd.DataFrame)
            expected = compute_barcode(generated)
            pd.testing.assert_frame_equal(pd.read_csv(output.with_suffix(".csv")), expected)

    def test_python_main_without_return_keeps_isolates_and_runs_once(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "data.py"
            source.write_text(
                "import pandas as pd\n"
                "calls = 0\n"
                "def generate_network_data(n):\n"
                "    global calls\n"
                "    calls += 1\n"
                "    assert calls == 1\n"
                "    frame = pd.DataFrame([(0, 1, 0.5)], columns=['source', 'target', 'weight'])\n"
                "    frame.attrs['nodes'] = list(range(n))\n"
                "    return frame\n"
                "def main():\n"
                "    edgelist = generate_network_data(n=4)\n",
                encoding="utf-8",
            )
            generated = _load_python_edgelist(source)
            barcode = compute_barcode(generated)
            self.assertEqual(len(barcode[barcode.dimension == 0]), 4)
            self.assertEqual(sum(np.isinf(barcode.death)), 3)

    def test_python_main_can_return_modified_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "data.py"
            source.write_text(
                "import pandas as pd\n"
                "def generate_network_data():\n"
                "    return pd.DataFrame([(0, 1, 1.)], columns=['source', 'target', 'weight'])\n"
                "def main():\n"
                "    frame = generate_network_data()\n"
                "    frame['weight'] = 0.25\n"
                "    return frame\n",
                encoding="utf-8",
            )
            generated = _load_python_edgelist(source)
            self.assertEqual(generated.weight.tolist(), [0.25])


if __name__ == "__main__":
    unittest.main()
