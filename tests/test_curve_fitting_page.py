"""Exercise the actual Streamlit page, with uploads supplied at the I/O boundary."""
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / 'packages/idopnetwork-app/src/idopnetwork_app/pages/1_Curve Fitting.py'


def upload(name, frame):
    file = io.BytesIO(frame.to_csv().encode())
    file.name = name
    return file


class CurveFittingPageTests(unittest.TestCase):
    def test_upload_results_replacement_comparison_and_clear(self):
        frame = pd.DataFrame({'a': range(1, 21), 'b': [x * x for x in range(1, 21)]})
        files = [upload('A.csv', frame), upload('B.csv', frame * 2)]
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['logged_in'] = True
        with patch('streamlit.file_uploader', side_effect=lambda *a, **kw: files), \
             patch('idopnetwork_app.utils.setup_sidebar'):
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 3)
            self.assertEqual(app.dataframe[1].value.shape[0], 30)
            self.assertNotIn('Dynamic Data', [tab.label for tab in app.tabs])
            self.assertEqual(len(app.get('download_button')), 4)
            old = app.dataframe[0].value.copy()
            app.checkbox[0].check().run()
            self.assertFalse(app.exception)
            # Same filename, different bytes: cached fits must not leak old results.
            files[:] = [upload('A.csv', frame.assign(a=frame.a ** 3))]
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(old.equals(app.dataframe[0].value))
            files[:] = []
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('download_button')), 0)
            self.assertEqual(len(app.dataframe), 0)

    def test_partial_failure_is_visible_and_does_not_block_valid_files(self):
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['logged_in'] = True
        files = [upload('bad.csv', pd.DataFrame({'a': [1, 1]})),
                 upload('good.csv', pd.DataFrame({'a': [1, 2, 3]}))]
        with patch('streamlit.file_uploader', return_value=files), \
             patch('idopnetwork_app.utils.setup_sidebar'):
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertTrue(any('1 / 2' in w.value for w in app.warning))
        self.assertEqual(len(app.get('download_button')), 4)
