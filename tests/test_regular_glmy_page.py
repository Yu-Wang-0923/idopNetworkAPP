"""Exercise new GLMY defaults, result export, and stale-input handling."""
from streamlit.testing.v1 import AppTest
from pathlib import Path
import io
from unittest.mock import patch
import zipfile


def analysis_page():
    import pandas as pd
    import streamlit as st
    from idopnetwork_app.glmy_workflow import render_glmy_analysis

    edges = pd.DataFrame([('a', 'b', -1.), ('b', 'c', 2.), ('c', 'a', 3.)],
                         columns=['from', 'to', 'weight'])
    if st.session_state.get('changed'):
        edges['weight'] *= 2
    render_glmy_analysis(edges, 'from_to.csv')


def test_default_mode_generates_download_and_hides_stale_results():
    app = AppTest.from_function(analysis_page, default_timeout=90).run()
    assert not app.exception
    assert app.selectbox[0].value == 'all_edges_colored'
    assert not app.get('download_button')
    app.button[0].click().run()
    assert not app.exception
    assert app.success
    assert len(app.get('download_button')) == 6
    assert app.session_state['regular_glmy_result'][1]['mode'] == 'all_edges_colored'
    app.session_state['changed'] = True
    app.run()
    assert not app.exception
    assert not app.get('download_button')


def test_netanal_reads_existing_netrecon_zip_into_new_glmy():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('from_to.csv', 'from,to,weight\na,b,-1\nb,c,2\nc,a,3\n')
    buffer.name = 'single_layer_idopnetwork_export.zip'
    page = Path(__file__).resolve().parents[1] / 'packages/idopnetwork-app/src/idopnetwork_app/pages/4_NetAnal.py'
    app = AppTest.from_file(str(page), default_timeout=90)
    app.session_state['logged_in'] = True
    with patch('streamlit.file_uploader', side_effect=lambda *args, **kw: buffer if kw.get('key') == 'netanal_glmy_zip_upload' else None), \
         patch('idopnetwork_app.utils.setup_sidebar'):
        app.run()
    assert not app.exception
    assert app.selectbox(key='regular_glmy_mode').value == 'all_edges_colored'
