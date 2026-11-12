"""Renderer revisions must invalidate cached network HTML."""
import ast
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st


def test_updated_renderer_invalidates_identical_network_cache():
    page = Path(__file__).resolve().parents[1] / 'packages/idopnetwork-app/src/idopnetwork_app/pages/3_NetRecon.py'
    tree = ast.parse(page.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_build_interactive_network_html')
    namespace = {'st': st, 'pd': pd, '__name__': __name__}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(page), 'exec'), namespace)
    build = namespace['_build_interactive_network_html']
    build.clear()
    edges = pd.DataFrame([('a', 'b', 1)], columns=['source', 'target', 'weight'])
    generated = []

    def draw(edges, output, **kwargs):
        generated.append(len(generated) + 1)
        Path(output).write_text(f'html version {generated[-1]}', encoding='utf-8')

    with patch('idopnetwork.network.interactive.plot_network', side_effect=draw):
        first = build(edges, ('a', 'b'), 'Single network', 'old-source')
        assert build(edges, ('a', 'b'), 'Single network', 'old-source') == first
        assert len(generated) == 1
        assert build(edges, ('a', 'b'), 'Single network', 'new-source') != first
        assert len(generated) == 2
    build.clear()
