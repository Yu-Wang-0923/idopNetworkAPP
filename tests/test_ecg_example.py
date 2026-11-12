"""The downloadable ECG CSV must retain all leads and enter the new solver."""
from importlib.resources import files
import numpy as np

from idopnetwork_app.curve_fitting_workflow import fit_uploaded_csv
from idopnetwork.network.dynamic_idop import DynamicIDOPRegressor, SOLVER_VERSION


def test_downloadable_ecg_preserves_time_and_twelve_leads():
    content = (files('idopnetwork_app') / 'data' / 'ecg_100hz.csv').read_bytes()
    result = fit_uploaded_csv(content, data_mode='dynamic')
    fitted = result['curve_sample']
    assert fitted.shape == (1000,12)
    assert list(fitted.columns) == ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    assert fitted.index.name == 'time'
    np.testing.assert_allclose(fitted.index, np.arange(1000)/100.)
    model = DynamicIDOPRegressor().fit(fitted)
    assert model.solver_version_ == SOLVER_VERSION
    assert model.predict().shape == (1000,12)
    np.testing.assert_allclose(model.predict().index, fitted.index)
