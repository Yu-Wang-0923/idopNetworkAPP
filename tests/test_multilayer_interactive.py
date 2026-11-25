"""Checks for the matrix data embedded in the interactive HTML overview."""

import json
import re

import pandas as pd

from idopnetwork.network.multilayer_interactive import build_multilayer_html


def test_multilayer_html_preserves_selection_missing_panels_and_signed_edges() -> None:
    """Keep each network in its requested cell and retain isolates and edge directions."""
    names = ["A", "B", "Isolated"]
    adjacency = pd.DataFrame([[0, 0.5, 0], [-1, 0, 0], [0, 0, 0]], index=names, columns=names)

    html = build_multilayer_html(
        {"Site A": adjacency},
        {"Site A": {"M1": adjacency}, "Site B": {"M2": adjacency}},
        conditions=["Site B", "Site A"],
        modules=["M2"],
    )
    data = json.loads(re.search(r'<script id="matrix-data"[^>]*>(.*?)</script>', html)[1])

    assert data["columns"] == ["Site B", "Site A"]
    inter, intra = data["rows"]
    assert inter["module"] is None and inter["panels"][0] is None
    assert intra["module"] == "M2" and intra["panels"][1] is None
    panel = inter["panels"][1]
    assert {node["name"] for node in panel["nodes"]} == set(names)
    assert {(edge["source"], edge["target"], edge["weight"]) for edge in panel["edges"]} == {
        ("A", "B", 0.5), ("B", "A", -1.0),
    }


def test_multilayer_html_escapes_script_delimiters_in_input_names() -> None:
    """Keep uploaded names intact without letting them close the HTML data tag."""
    name = "</script><b>Site</b>"
    adjacency = pd.DataFrame([[0]], index=[name], columns=[name])

    html = build_multilayer_html({name: adjacency}, {})
    data = json.loads(re.search(r'<script id="matrix-data"[^>]*>(.*?)</script>', html)[1])

    assert name not in html
    assert data["columns"] == [name]
    assert data["rows"][0]["panels"][0]["nodes"][0]["name"] == name
