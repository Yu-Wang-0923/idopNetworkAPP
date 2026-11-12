"""Verify full/zoom panels and display-only effect windows preserve native data."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from idopnetwork.curve_fitting.ecg_plot import plot_fitting
from idopnetwork.network.ecg_effect_plot import plot_effect_decomposition
from idopnetwork.network.dynamic_idop import DynamicIDOPRegressor
from idopnetwork_app.ecg_visualization import fitting_artifacts, effect_artifacts


def test_native_fitting_has_full_record_and_first_250_panels(tmp_path):
    time = 2 + np.arange(400)/100
    original = pd.DataFrame({'time': time, 'I': np.sin(time)})
    fitted = original.assign(I=lambda frame: frame.I * .8)
    close = plt.close
    with patch('matplotlib.pyplot.close'):
        plot_fitting(original, fitted, tmp_path)
        figure = plt.gcf()
        assert len(figure.axes) == 2
        for axis, count in zip(figure.axes, [400,250]):
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), time[:count])
            np.testing.assert_allclose(axis.lines[0].get_ydata(), original.I.iloc[:count])
            np.testing.assert_allclose(axis.lines[1].get_ydata(), fitted.I.iloc[:count])
        assert figure.axes[0].lines[1].get_label() == 'SWT fit'
    close(figure)


def test_native_effect_window_keeps_full_record_initial_value(tmp_path):
    time = 3 + np.arange(128)/100
    frame = pd.DataFrame({'a': 2+np.sin(10*time), 'b': np.cos(10*time+.2)},
                          index=pd.Index(time,name='time'))
    model = DynamicIDOPRegressor(lasso_threshold=0.).fit(frame)
    expected = model.effects_['reconstruction'].query("target == 'a'")
    mask = (expected.time >= 3.2-1e-10) & (expected.time <= 3.6+1e-10)
    close = plt.close
    with patch('matplotlib.pyplot.close'):
        plot_effect_decomposition(model.effects_, tmp_path, start_time=.2, duration=.4,
                                  selected_leads=['a'], formats=('png',))
        figure = plt.gcf()
        overview = figure.axes[0]
        np.testing.assert_allclose(overview.get_xlim(), [3.2,3.6])
        np.testing.assert_allclose(overview.lines[0].get_ydata(), expected.smoothed.loc[mask])
        np.testing.assert_allclose(overview.lines[1].get_ydata(), expected.reconstructed.loc[mask])
        np.testing.assert_allclose(overview.lines[2].get_ydata(), expected.self_trajectory.loc[mask])
        assert len(figure.axes) == 2 + len(model.effects_['network'].query("target == 'a'"))
    close(figure)


def test_artifacts_include_native_png_pdf_svg_and_close_figures():
    time = np.arange(64)/100
    frame = pd.DataFrame({'a': 1+np.sin(12*time), 'b': np.cos(12*time+.2)},
                          index=pd.Index(time,name='time'))
    before = set(plt.get_fignums())
    fitting = fitting_artifacts(frame[['a']], frame[['a']], renderer_revision='test')
    model = DynamicIDOPRegressor().fit(frame)
    effects = effect_artifacts(model.effects_, lead='a', renderer_revision='test')
    for artifacts in (fitting,effects):
        assert artifacts['png'].startswith(b'\x89PNG')
        assert artifacts['pdf'].startswith(b'%PDF')
        assert b'<svg' in artifacts['svg']
    assert set(plt.get_fignums()) == before
