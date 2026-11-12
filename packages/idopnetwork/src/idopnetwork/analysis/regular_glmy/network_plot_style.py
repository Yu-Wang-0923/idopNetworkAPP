"""Shared midpoint arrows and signed-edge colors for network figures."""

import numpy as np
from matplotlib.patches import ArrowStyle
from matplotlib.path import Path


SIGN_COLORS = {"positive": "#d62728", "negative": "#1f77b4", "zero": "0.45"}


def _weight_sign(weight):
    return "positive" if weight > 0 else "negative" if weight < 0 else "zero"


def _signed_edge_weights(edgelist):
    """Match earliest-entry merging; equal absolute weights keep the first row.

    The caller validates the edge table before using this drawing helper.
    """
    weights = {}
    for u, v, value in edgelist[["source", "target", "weight"]].itertuples(index=False, name=None):
        value = float(value)
        if (u, v) not in weights or abs(value) < abs(weights[u, v]):
            weights[u, v] = value
    return weights


class MidpointArrowStyle(ArrowStyle._Base):
    """Keep the full edge curve and put a filled arrowhead at its midpoint.

    The path and tangent are evaluated at draw time in display coordinates,
    so straight/reciprocal curved edges work at any figure size and export DPI.
    This style uses the quadratic Bezier paths produced by ConnectionStyle.Arc3.
    """

    def __init__(self, head_length=0.45, head_width=0.30):
        self.head_length = head_length
        self.head_width = head_width
        super().__init__()

    def transmute(self, path, mutation_size, linewidth):
        x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)
        start, control, end = np.array([[x0, y0], [x1, y1], [x2, y2]])
        midpoint = 0.25 * start + 0.5 * control + 0.25 * end
        tangent = end - start  # Quadratic Bezier derivative at t=0.5.
        length = np.linalg.norm(tangent)
        if length == 0:
            return path, False
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        head_length = min(self.head_length * mutation_size, length * 0.3)
        half_width = min(self.head_width * mutation_size / 2, length * 0.15)
        base = midpoint - head_length * tangent
        head = Path(
            [midpoint, base + half_width * normal, base - half_width * normal, midpoint],
            [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
        )
        return [path, head], [False, True]
