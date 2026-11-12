"""Paired signed views retain source data, thresholds and exact homology."""

from pathlib import Path
import json
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
from idopnetwork.analysis.regular_glmy.plot_filtration_process import SIGN_COLORS
from idopnetwork.analysis.regular_glmy.plot_signed_filtration_process import VIEW_FOLDERS, plot_signed_filtration_process
from .test_plot_barcode import edge_frame


class SignedFiltrationProcessTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_paired_views_preserve_data_and_compute_selected_homology(self):
        frame = edge_frame([(0, 1, 0.2), (1, 2, -0.4), (2, 0, 0.6), (0, 2, -0.8)])
        frame.attrs["nodes"] = [0, 1, 2, 3]
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            results = plot_signed_filtration_process(
                frame, filtration_values=[0, 0.6, 0.8], output_dir=output,
            )
            pd.testing.assert_frame_equal(frame, original)
            pd.testing.assert_frame_equal(results["all"][0], compute_barcode(frame))
            self.assertEqual(results["all"][1][1].betti, (2, 1, 0))
            self.assertEqual(results["positive"][1][1].betti, (2, 0, 0))
            self.assertEqual(results["negative"][1][1].betti, (3, 0, 0))
            reference_positions = results["all"][2].axes[1].collections[-1].get_offsets()
            for selection, (barcode, snapshots, figure) in results.items():
                self.assertEqual([s.threshold for s in snapshots], [0, 0.6, 0.8])
                self.assertEqual(snapshots[0].betti, (4, 0, 0))
                for ax in figure.axes[1:]:
                    np.testing.assert_array_equal(ax.collections[-1].get_offsets(), reference_positions)
                folder = output / VIEW_FOLDERS[selection]
                selected = pd.read_csv(folder / "network.csv")
                pd.testing.assert_frame_equal(barcode, compute_barcode(selected, nodes=range(4)))
                pd.testing.assert_frame_equal(
                    pd.read_csv(folder / "homology_representative_barcode.csv"), barcode,
                )
                pd.testing.assert_frame_equal(
                    pd.read_csv(folder / "homology_representative_nodes.csv"),
                    pd.read_csv(folder / "filtration_process_nodes.csv"),
                )
                intervals = pd.read_csv(folder / "homology_representative_intervals.csv")
                paths = pd.read_csv(folder / "homology_representative_paths.csv")
                support = pd.read_csv(folder / "homology_representative_edges.csv")
                selected_edges = set(zip(selected.source, selected.target))
                for path in paths.path:
                    vertices = json.loads(path)
                    self.assertLessEqual(set(zip(vertices, vertices[1:])), selected_edges)
                if selection != "all":
                    self.assertTrue((support.sign == selection).all())
                for degree in (1, 2):
                    expected = barcode.loc[barcode.dimension == degree].iloc[::-1]
                    group = intervals.loc[intervals.dimension == degree]
                    self.assertEqual(group.number.tolist(), list(range(1, len(expected) + 1)))
                    np.testing.assert_array_equal(group[["birth", "death"]], expected[["birth", "death"]])
                    with Image.open(folder / f"homology_representative_beta{degree}.png") as image:
                        self.assertAlmostEqual(image.info["dpi"][0], 300, places=2)
                with Image.open(folder / "filtration_process.png") as image:
                    self.assertAlmostEqual(image.info["dpi"][0], 300, places=2)
                rows = pd.read_csv(folder / "filtration_process_edges.csv")
                np.testing.assert_array_equal(rows.entry_time, rows.weight.abs())
                if selection != "all":
                    self.assertTrue((rows.sign == selection).all())
            # Verify the actual rendered arrows use both signed colors.
            colors = {tuple(patch.get_edgecolor()) for patch in results["all"][2].axes[-1].patches}
            self.assertEqual(colors, {to_rgba(SIGN_COLORS[s], alpha=0.85)
                                      for s in ("positive", "negative")})
            summary = pd.read_csv(output / "selection_summary.csv")
            self.assertEqual(summary.input_rows.tolist(), [4, 2, 2])
            self.assertEqual(len(pd.read_csv(output / "comparison_snapshots.csv")), 9)

    def test_duplicates_and_zero_weights_keep_sign_specific_entry_times(self):
        frame = edge_frame([(0, 1, -0.2), (0, 1, 0.9), (1, 2, 0)])
        with tempfile.TemporaryDirectory() as directory:
            results = plot_signed_filtration_process(frame, nodes=range(4), output_dir=directory)
            for _, snapshots, _ in results.values():
                self.assertEqual(snapshots[-1].threshold, 0.9)
            self.assertEqual(results["positive"][1][-1].edges, ((0, 1),))
            self.assertEqual(results["negative"][1][-1].edges, ((0, 1),))
            self.assertEqual(results["all"][1][0].edges, ((1, 2),))
            all_edges = pd.read_csv(Path(directory) / VIEW_FOLDERS["all"] / "filtration_process_edges.csv")
            self.assertEqual(set(all_edges.weight), {-0.2, 0.0})
            self.assertEqual(set(all_edges.sign), {"negative", "zero"})

    def test_empty_sign_subgraph_still_has_all_stages_and_vertices(self):
        with tempfile.TemporaryDirectory() as directory:
            results = plot_signed_filtration_process(
                edge_frame([("a", "b", 0.5)]), nodes=["a", "b", "isolated"], output_dir=directory,
            )
            negative = results["negative"][1]
            self.assertEqual([s.threshold for s in negative], [0, 0.5])
            self.assertTrue(all(s.edges == () and s.betti == (3, 0, 0) for s in negative))


if __name__ == "__main__":
    unittest.main()
