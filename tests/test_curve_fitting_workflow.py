import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from idopnetwork.curve_fitting import power_fitting
from idopnetwork_app.curve_fitting_workflow import fit_uploaded_csv, build_fitting_export


class WorkflowTests(unittest.TestCase):
    def test_matches_v4_example_pipeline_and_export(self):
        raw = pd.DataFrame({'a': np.arange(120.), 'b': np.arange(120.) ** 2},
                           index=['duplicate'] * 120)
        result = fit_uploaded_csv(raw.to_csv().encode())
        transformed = np.log10(1 + (raw - raw.min() + 1))
        sums = transformed.sum(axis=1).to_numpy()
        order = np.argsort(sums, kind='stable')
        expected = transformed.iloc[order].copy()
        expected.index = pd.Index(np.log1p(sums[order]), name='quasi time')
        expected = expected[expected.index > 0]
        expected = expected.iloc[int(.01 * len(expected)):]
        pd.testing.assert_frame_equal(result['quasi_dynamic'], expected)
        params, samples = power_fitting(expected, n_samples=30)
        pd.testing.assert_frame_equal(result['curve_params'], params)
        pd.testing.assert_frame_equal(result['curve_sample'], samples)
        with zipfile.ZipFile(io.BytesIO(build_fitting_export({'input.csv': result}))) as archive:
            self.assertEqual(set(archive.namelist()), {
                'input.csv/quasi_dynamic.csv', 'input.csv/curve_sample.csv', 'input.csv/curve_params.csv'})
            exported = pd.read_csv(archive.open('input.csv/curve_sample.csv'), index_col=0)
            np.testing.assert_allclose(exported, samples)

    def test_custom_parameters(self):
        raw = pd.DataFrame({'a': np.arange(1., 101.), 'b': np.arange(1., 101.) ** 2})
        result = fit_uploaded_csv(raw.to_csv().encode(), first_transform="None",
                                  second_transform="None", n_samples=12, trim_percent=10)
        self.assertEqual(len(result['curve_sample']), 12)
        self.assertEqual(len(result['quasi_dynamic']), 90)
        np.testing.assert_allclose(result['quasi_dynamic'].to_numpy(), raw.iloc[10:].to_numpy())
        with self.assertRaises(ValueError):
            fit_uploaded_csv(raw.to_csv().encode(), trim_percent=100)

    def test_invalid_csv_and_name_collisions(self):
        with self.assertRaises(ValueError):
            fit_uploaded_csv(b'id,a\nx,1\ny,1\n')
        result = fit_uploaded_csv(b'id,a,b\nx,1,2\ny,2,3\nz,3,6\n')
        with self.assertRaises(ValueError):
            build_fitting_export({'a/b': result, 'a_b': result})
