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
    def test_many_features_default_to_six_and_can_be_selected_or_cleared(self):
        from idopnetwork.curve_fitting.plot import plot_curve_fitting

        frame = pd.DataFrame({f'feature_{i}': [x * (i + 1) for x in range(1, 21)]
                              for i in range(12)})
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['logged_in'] = True
        with patch('streamlit.file_uploader', return_value=[upload('many.csv', frame)]), \
             patch('idopnetwork_app.utils.setup_sidebar'), \
             patch('idopnetwork.curve_fitting.plot.plot_curve_fitting', wraps=plot_curve_fitting) as draw:
            app.run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.multiselect[0].value, list(frame.columns[:6]))
            self.assertEqual(list(draw.call_args.kwargs['df_scatter'].columns), list(frame.columns[:6]))
            app.multiselect[0].set_value(['feature_10', 'feature_11']).run()
            self.assertFalse(app.exception)
            self.assertEqual(list(draw.call_args.kwargs['df_scatter'].columns), ['feature_10', 'feature_11'])
            draw.reset_mock()
            app.multiselect[0].set_value([]).run()
            self.assertFalse(app.exception)
            draw.assert_not_called()
            self.assertEqual(len(app.dataframe), 3)

    def test_upload_results_replacement_comparison_and_clear(self):
        frame = pd.DataFrame({'a': range(1, 21), 'b': [x * x for x in range(1, 21)]})
        files = [upload('A.csv', frame), upload('B.csv', frame * 2)]
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['logged_in'] = True
        with patch('streamlit.file_uploader', side_effect=lambda *a, **kw: files), \
             patch('idopnetwork_app.utils.setup_sidebar'):
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 0)
            self.assertEqual(len(app.get('download_button')), 1)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 3)
            self.assertEqual(app.dataframe[1].value.shape[0], 30)
            self.assertNotIn('Dynamic Data', [tab.label for tab in app.tabs])
            self.assertEqual(len(app.get('download_button')), 5)
            old = app.dataframe[0].value.copy()
            app.checkbox[0].check().run()
            self.assertFalse(app.exception)
            # Same filename, different bytes: cached fits must not leak old results.
            files[:] = [upload('A.csv', frame.assign(a=frame.a ** 3))]
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 0)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(old.equals(app.dataframe[0].value))
            app.number_input(key='fit_n_samples').set_value(12)
            app.number_input(key='fit_trim_percent').set_value(0.0)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[1].value), 12)
            self.assertEqual(app.metric[2].value, '12')
            files[:] = []
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('download_button')), 1)
            self.assertEqual(len(app.dataframe), 0)

    def test_partial_failure_is_visible_and_does_not_block_valid_files(self):
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['logged_in'] = True
        files = [upload('bad.csv', pd.DataFrame({'a': [1, 1]})),
                 upload('good.csv', pd.DataFrame({'a': [1, 2, 3]}))]
        with patch('streamlit.file_uploader', return_value=files), \
             patch('idopnetwork_app.utils.setup_sidebar'):
            app.run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertTrue(any('1 / 2' in w.value for w in app.warning))
        self.assertEqual(len(app.get('download_button')), 5)
