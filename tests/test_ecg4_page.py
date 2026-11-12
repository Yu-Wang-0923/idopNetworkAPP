"""Dynamic UI selects idopECG(4) and exports native numerical results."""
import ast
import io
from pathlib import Path
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from idopnetwork_app.curve_fitting_workflow import build_fitting_export, fit_uploaded_csv
from idopnetwork.network.dynamic_idop import SOLVER_VERSION


def test_netrecon_dynamic_full_record_and_native_export():
    t = np.arange(128) / 100
    frame = pd.DataFrame({'a': 1+np.sin(2*np.pi*2*t), 'b': np.sin(2*np.pi*2*t+.4),
                          'c': np.cos(2*np.pi*3*t)}, index=pd.Index(t, name='time'))
    tables = fit_uploaded_csv(frame.to_csv().encode(), data_mode='dynamic')
    upload = io.BytesIO(build_fitting_export({'example.csv': tables}))
    upload.name = 'curve_fitting_export.zip'
    root = Path(__file__).resolve().parents[1]
    page = root / 'packages/idopnetwork-app/src/idopnetwork_app/pages/3_NetRecon.py'
    app = AppTest.from_file(str(page), default_timeout=90)
    app.session_state['logged_in'] = True
    with patch('streamlit.file_uploader', side_effect=lambda *args, **kw: upload if kw.get('key') == 'netrecon_single_upload' else None), \
         patch('idopnetwork_app.utils.setup_sidebar'):
        app.run()
        assert not app.exception
        # The real upload key is verified from the page rather than seeding model results.
        if not app.radio:
            raise AssertionError('NetRecon upload did not reach construction controls')
        app.radio(key='netrecon_idop_algorithm').set_value('dynamic_idop').run()
        assert not app.exception
        assert app.number_input(key='netrecon_ecg4_n_fourier').value == 0
        assert app.number_input(key='netrecon_ecg4_alpha_ratio').value == .1
        assert app.number_input(key='netrecon_ecg4_threshold').value == .9
        next(button for button in app.button if button.label == 'Run IdopNetwork').click().run()
        assert not app.exception
        assert not app.error
    result = app.session_state['netrecon_result']
    assert result['model'].solver_version_ == SOLVER_VERSION
    assert len(result['predicted_df']) == len(frame)
    # Exercise the actual export helpers from the page without executing UI code.
    names = {'_adjacency_to_from_to','_params_to_df','_to_csv_utf8_sig_bytes',
             '_collect_network_export_artifacts','_build_singlelayer_export_zip'}
    nodes = [node for node in ast.parse(page.read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'pd':pd, 'np':np, 'io':io, 'zipfile':zipfile}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(page), 'exec'), namespace)
    archive = namespace['_build_singlelayer_export_zip'](result)
    with zipfile.ZipFile(io.BytesIO(archive)) as exported:
        assert 'dynamic/network.csv' in exported.namelist()
        assert 'dynamic/reconstruction.csv' in exported.namelist()
        raw = pd.read_csv(io.BytesIO(exported.read('dynamic/network.csv')))
        pd.testing.assert_frame_equal(raw, result['model'].network_, check_dtype=False)
