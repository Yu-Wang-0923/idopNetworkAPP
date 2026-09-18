"""Regression coverage for the v4 kernel and the application compatibility boundary."""
import ast
import io
from pathlib import Path
import unittest
import zipfile

import numpy as np
import pandas as pd
import torch

from idopnetwork.curve_fitting import power_fitting, get_power_function_params
from idopnetwork.clustering.funclu import FunClu, compute_bic_scores
from idopnetwork.clustering._v4 import FunClu as NativeFunClu

ROOT = Path(__file__).resolve().parents[1]


def datasets():
    rng = np.random.default_rng(14)
    raw, params, samples = [], [], []
    for i, count in enumerate((19, 25)):
        t = np.linspace(1, 8 + i, count)
        a = np.r_[np.full(8, 2.), np.full(8, 7.)] * rng.uniform(.95, 1.05, 16)
        b = np.r_[np.full(8, .7), np.full(8, -.4)]
        frame = pd.DataFrame(a * t[:, None] ** b, index=t,
                             columns=[f'f{j}' for j in range(16)])
        p, s = power_fitting(frame, n_samples=12 + i)
        raw.append(frame); params.append(p); samples.append(s)
    return raw, params, samples


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.raw, cls.params, cls.samples = datasets()

    def test_power_fit_and_samples(self):
        p, s = power_fitting(self.raw[0])
        self.assertEqual(s.shape, (30, 16))
        np.testing.assert_allclose(p.b, [.7] * 8 + [-.4] * 8, atol=1e-12)
        np.testing.assert_allclose(s, p.a.to_numpy() * s.index.to_numpy()[:, None] ** p.b.to_numpy())
        self.assertTrue({'c', 'beta', 'a', 'b', 'n_positive'} <= set(p.columns))
        pd.testing.assert_frame_equal(p, get_power_function_params(self.raw[0]))

    def test_degenerate_time_and_invalid_features(self):
        with self.assertRaisesRegex(ValueError, '时间点'):
            power_fitting(pd.DataFrame({'a': [1., 2.]}, index=[1., 1.]))
        frame = self.raw[0].assign(invalid=0.)
        p, s = power_fitting(frame)
        self.assertTrue(np.isnan(p.loc['invalid', 'c']))
        model = FunClu(2, max_iter=10, use_minibatch_kmeans=False).fit([s], parameter_data=[p])
        self.assertNotIn('invalid', model.common_cols)
        self.assertIn('invalid', model.excluded_features_)
        self.assertEqual(len(model.labels), 16)
        with self.assertRaisesRegex(ValueError, '有效共同特征'):
            FunClu(17).fit([s], parameter_data=[p])

    def test_adapter_matches_native_kernel(self):
        native = NativeFunClu(2, max_iter=20, random_state=42,
                              compute_device='cpu', use_mini_batch=False, progress=False).fit(
            dict(enumerate(self.samples)), dict(enumerate(self.params)))
        model = FunClu(2, max_iter=20, use_minibatch_kmeans=False).fit(
            self.samples, parameter_data=self.params)
        np.testing.assert_allclose(model.kernel_.responsibilities_, native.responsibilities_)
        self.assertAlmostEqual(model.bic, native.bic())
        self.assertEqual(model.n_params, 13)
        for i in range(2):
            t, curves = model.get_cluster_curves(i)
            a, b = model.params_mu[:, i].numpy().T
            np.testing.assert_allclose(curves, a[:, None] * t[None, :] ** b[:, None], rtol=1e-12)
        np.testing.assert_array_equal(model.predict(self.samples), model.labels)
        self.assertEqual(model.common_cols, list(self.samples[0]))

    def test_bic_uses_v4_and_reports_failures(self):
        table = compute_bic_scores(self.samples, k_min=2, k_max=3,
                                   max_iter=10, parameter_data=self.params)
        self.assertTrue(np.isfinite(table.BIC).all())
        self.assertEqual(table.n_params.tolist(), [13, 18])
        self.assertEqual(table.fit_successes.tolist(), [1, 1])

    def test_empty_initial_clusters_and_common_features(self):
        t = np.linspace(1, 4, 10)
        same = pd.DataFrame(np.tile(t[:, None], (1, 4)), index=t, columns=list('abcd'))
        model = FunClu(2, max_iter=5, use_minibatch_kmeans=False).fit([same])
        self.assertTrue(np.isfinite(model.bic))
        self.assertTrue(np.isfinite(model.get_cluster_curves()[1]).all())
        second = self.samples[1].drop(columns='f0').iloc[:, ::-1]
        model = FunClu(2, max_iter=5).fit([self.samples[0], second])
        self.assertEqual(model.common_cols, [f'f{i}' for i in range(1, 16)])
        failed = compute_bic_scores([same * np.nan], k_min=2, k_max=2, max_iter=5)
        self.assertEqual(failed.fit_successes.iloc[0], 0)
        self.assertTrue(failed.error.iloc[0])

    def test_export_for_network_reconstruction(self):
        model = FunClu(2, max_iter=10, use_minibatch_kmeans=False).fit(
            self.samples, parameter_data=self.params)
        page = ROOT / 'packages/idopnetwork-app/src/idopnetwork_app/pages/2_FunClu.py'
        tree = ast.parse(page.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_build_funclu_k_export_zip')
        scope = dict(FunClu=FunClu, pd=pd, np=np, io=io, zipfile=zipfile)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(page), 'exec'), scope)
        payload = scope[function.name](model=model, cond_names=['A', 'B'],
                    curve_sample_dict=dict(zip(['A', 'B'], self.samples)),
                    quasi_dynamic_dict=dict(zip(['A', 'B'], self.raw)))
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            labels = pd.read_csv(zf.open('labels.csv'))
            self.assertEqual(len(labels), 16)
            centers = pd.read_csv(zf.open('cluster_centers/A/cluster_center_curve_sample.csv'), index_col=0)
            np.testing.assert_allclose(centers.to_numpy().T, model.get_cluster_curves(0)[1])


if __name__ == '__main__':
    unittest.main()
