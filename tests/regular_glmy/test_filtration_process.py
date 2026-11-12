"""Sublevel inclusions, sign selection, exact stage Betti numbers and export."""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from idopnetwork.analysis.regular_glmy.plot_filtration_process import (
    compute_filtration_snapshots, plot_filtration_process,
)
from .test_plot_barcode import edge_frame


class FiltrationProcessTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_sublevel_stages_and_death_tie(self):
        frame = edge_frame([(0, 1, 1), (1, 2, -1), (2, 0, 1), (0, 2, -2)])
        _, stages = compute_filtration_snapshots(
            frame, nodes=range(4), filtration_values=[0, 1, 2],
        )
        self.assertEqual([len(s.edges) for s in stages], [0, 3, 4])
        self.assertEqual([s.betti for s in stages], [(4, 0, 0), (2, 1, 0), (2, 0, 0)])
        self.assertEqual(set(stages[1].edges), {(0, 1), (1, 2), (2, 0)})
        for earlier, later in zip(stages, stages[1:]):
            self.assertLessEqual(set(earlier.edges), set(later.edges))

    def test_sign_selection_retains_all_vertices(self):
        frame = edge_frame([("a", "b", 0.2), ("b", "c", -0.3)])
        frame.attrs["nodes"] = ["a", "b", "c", "d"]
        _, positive = compute_filtration_snapshots(frame, edge_sign="positive")
        _, negative = compute_filtration_snapshots(frame, edge_sign="negative")
        self.assertEqual(set(positive[-1].edges), {("a", "b")})
        self.assertEqual(set(negative[-1].edges), {("b", "c")})
        self.assertEqual(positive[0].betti, (4, 0, 0))
        self.assertEqual(positive[-1].betti, (3, 0, 0))
        self.assertEqual(negative[-1].betti, (3, 0, 0))

    def test_default_initial_and_complete_network_with_duplicates(self):
        frame = edge_frame([(0, 1, -0.2), (0, 1, 0.8), (1, 2, 0.4),
                            (2, 3, -0.6), (3, 0, 1.0)])
        _, stages = compute_filtration_snapshots(frame)
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[0].threshold, 0)
        self.assertEqual(stages[-1].threshold, 1)
        self.assertEqual(len(stages[-1].edges), 4)
        self.assertEqual(len(stages[-1].edges), len(set(stages[-1].edges)))
        self.assertIn(stages[1].threshold, [0.2, 0.4, 0.6, 1])

    def test_zero_weight_and_empty_graph(self):
        _, stages = compute_filtration_snapshots(edge_frame([(0, 1, 0)]))
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].edges, ((0, 1),))
        self.assertEqual(stages[0].betti, (1, 0, 0))
        _, stages = compute_filtration_snapshots(edge_frame([]), nodes=["a", "b"])
        self.assertEqual(stages[0].betti, (2, 0, 0))

    def test_h2_stage_labels(self):
        layers = [(0, 1), (2, 3), (4, 5)]
        rows = [(u, v, 1) for left, right in zip(layers, layers[1:])
                for u in left for v in right]
        rows += [(v, 6, 2) for layer in layers for v in layer]
        _, stages = compute_filtration_snapshots(edge_frame(rows), filtration_values=[0, 1, 2])
        self.assertEqual(stages[0].betti, (7, 0, 0))
        self.assertEqual(stages[1].betti, (2, 0, 1))
        self.assertEqual(stages[2].betti, (1, 0, 0))

    def test_export_and_shared_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "process.png"
            frame = edge_frame([(0, 1, 0.5), (1, 2, 1)])
            barcode, stages, figure = plot_filtration_process(
                frame, nodes=range(4), save_path=output, filtration_values=[0, 0.5, 1],
            )
            self.assertEqual(len(figure.axes), 4)
            # Node collections occur last, after edge-arrow patches.
            positions = [ax.collections[-1].get_offsets() for ax in figure.axes[1:]]
            for other in positions[1:]:
                np.testing.assert_array_equal(positions[0], other)
            with Image.open(output) as image:
                self.assertAlmostEqual(image.info["dpi"][0], 300, places=2)
            table = pd.read_csv(output.with_name("process_snapshots.csv"))
            self.assertEqual(table.n_edges.tolist(), [0, 1, 2])
            self.assertEqual(table.n_vertices.tolist(), [4, 4, 4])
            self.assertEqual(len(pd.read_csv(output.with_name("process_nodes.csv"))), 4)
            pd.testing.assert_frame_equal(pd.read_csv(output.with_name("process_barcode.csv")), barcode)
            self.assertEqual(len(pd.read_csv(output.with_name("process_edges.csv"))), 3)

    def test_invalid_parameters(self):
        empty = edge_frame([])
        for values in ([], [1, 0], [0, 0], [-1], [np.inf], [[0, 1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compute_filtration_snapshots(empty, filtration_values=values)
        with self.assertRaises(ValueError):
            compute_filtration_snapshots(empty, n_stages=1)
        with self.assertRaises(ValueError):
            compute_filtration_snapshots(empty, edge_sign="invalid")
        with self.assertRaises(ValueError):
            plot_filtration_process(empty, columns=0)


if __name__ == "__main__":
    unittest.main()
