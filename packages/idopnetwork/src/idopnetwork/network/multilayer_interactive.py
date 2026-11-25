"""Build a self-contained, editable overview of a two-layer IDOP network."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files

import pandas as pd

from . import interactive as idop
from .multilayer_plot import (
    DEFAULT_EDGE_CURVATURE_SCALE,
    HEADER_COLOR,
    LABEL_COLOR,
    PANEL_BORDER,
    _panel_node_diameters,
    _prepare_matrix,
)


def build_multilayer_html(
    inter_cluster: Mapping[str, pd.DataFrame],
    intra_cluster: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    conditions: Sequence[str] | None = None,
    modules: Sequence[str] | None = None,
    max_edges_per_panel: int | None = None,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render the same panel data as the static overview in an editable HTML matrix."""
    columns, rows = _prepare_matrix(inter_cluster, intra_cluster, conditions, modules, max_edges_per_panel)
    payload = {
        "columns": columns,
        "rows": [
            {
                "module": row.module,
                "radius": row.radius,
                "maxWeight": max(
                    (float(panel[2]["weight"].abs().max()) for panel in row.panels
                     if panel is not None and not panel[2].empty),
                    default=0.0,
                ),
                "panels": [_panel_data(panel, row.width_levels) for panel in row.panels],
            }
            for row in rows
        ],
        "colors": {
            "border": PANEL_BORDER,
            "header": HEADER_COLOR,
            "heading": LABEL_COLOR,
            "positive": idop.POSITIVE_EDGE_COLOR,
            "negative": idop.NEGATIVE_EDGE_COLOR,
            "nodeOutline": idop.NODE_EDGE_COLOR,
            "label": idop.TEXT_COLOR,
        },
        "labels": dict(labels or {}),
    }
    # Escape the HTML raw-text delimiter in uploaded condition, module and node names.
    content = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    template = files(__package__).joinpath("multilayer_matrix.html").read_text(encoding="utf-8")
    return template.replace("__MATRIX_DATA__", content)


def _panel_data(
    panel: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    width_levels: tuple[float, float],
) -> dict | None:
    """Serialize one panel's initial positions, visual encodings and directed edges."""
    if panel is None:
        return None

    layout, display, _ = panel
    diameters = _panel_node_diameters(layout)
    pairs = set(zip(display["source"], display["target"]))
    return {
        "nodes": [
            {
                "name": node.module,
                "x": float(node.x),
                "y": float(node.y),
                "radius": float(diameters[node.module] * 2 / 3),  # points of diameter to pixels of radius
                "color": idop._ring_color(node.ring_number),
            }
            for node in layout.itertuples(index=False)
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": float(edge.weight),
                "width": 0.8 * idop._edge_width(abs(edge.weight), *width_levels),
                "curve": DEFAULT_EDGE_CURVATURE_SCALE * idop._edge_curvature(
                    edge.source, edge.target, (edge.target, edge.source) in pairs,
                ),
            }
            for edge in display.iloc[::-1].itertuples(index=False)
        ],
    }
