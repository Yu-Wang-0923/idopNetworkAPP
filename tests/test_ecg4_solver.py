"""Full-record, Shapley-allocation, time, sign and export contracts."""
import numpy as np
import pandas as pd
import pytest

from idopnetwork.network.dynamic_idop import DynamicIDOPRegressor, fit_dynamic_idop_network, observed_for_plot
from idopnetwork.network.ecg_solver import get_basis, get_effects, get_support_sets


def signals(n=120):
    t = np.arange(n) / 100
    return pd.DataFrame({'a': 1 + np.sin(2*np.pi*2*t),
                         'b': -2 + .7*np.sin(2*np.pi*2*t+.3) + .1*np.cos(2*np.pi*5*t),
                         'c': .4*np.cos(2*np.pi*3*t) + .2*np.sin(2*np.pi*2*t)})


def fitted():
    return DynamicIDOPRegressor(lasso_threshold=0.0).fit(signals())


def test_complete_record_and_additive_identity():
    model = fitted()
    observed, prediction = model.observed(), model.predict()
    assert prediction.shape == signals().shape
    np.testing.assert_allclose(observed.to_numpy(), signals().to_numpy())
    for target, effect in zip(prediction.columns, model.effect()):
        np.testing.assert_allclose(effect.sum(axis=1) + model.coef_.loc['intercept', target],
                                   prediction[target], atol=1e-10)
        np.testing.assert_allclose(effect.iloc[0], 0., atol=1e-12)
    assert model.n_fourier == int(np.ceil(10 * (len(observed)-1)/100))
    assert prediction.index.equals(model._design().index)
    pd.testing.assert_frame_equal(observed_for_plot(model, signals()), observed)


def test_shapley_allocation_is_source_order_invariant():
    data = signals()
    supports = pd.DataFrame({'target': ['a','b','c'], 'source': ['{b,c}', '{a,c}', '{}']})
    reverse = supports.copy()
    reverse['source'] = ['{c,b}', '{c,a}', '{}']
    first = get_effects(data, supports)
    second = get_effects(data, reverse)
    keys = ['target','source','time']
    a = first['interaction_effects'].sort_values(keys).reset_index(drop=True)
    b = second['interaction_effects'].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, atol=1e-10, rtol=1e-10)


def test_signed_adjacency_uses_prescribed_energy_share_and_exports_raw_sign():
    model = fitted()
    adjacency = model.adjacency_matrix()
    for row in model.network_.itertuples():
        assert adjacency.loc[row.source, row.target] == pytest.approx(row.weight * (-1 if row.sign == '-' else 1))
    totals = model.network_.groupby('target').weight.sum()
    diagnostics = model.diagnostics_.set_index('target')
    for target, total in totals.items():
        assert total == pytest.approx(diagnostics.loc[target, 'edge_energy_share'])
    assert {'coefficients','self_effects','interaction_effects','reconstruction','diagnostics',
            'network','settings','orthogonalization','support_set','edge_frequency'} <= model.ecg_exports_.keys()


def test_time_column_is_retained_and_not_selected_as_lead():
    data = signals()
    data.insert(0, 'time', 3 + np.arange(len(data)) / 100)
    model = DynamicIDOPRegressor().fit(data)
    assert list(model.predict().columns) == ['a','b','c']
    np.testing.assert_allclose(model.predict().index, data.time)
    assert model.coef_.loc['intercept','a'] == data.a.iloc[0]
    indexed = data.set_index('time')
    pd.testing.assert_frame_equal(DynamicIDOPRegressor().fit(indexed).predict(), model.predict())


def test_no_support_and_constant_channels_are_valid():
    data = pd.DataFrame({'a': np.ones(64), 'b': np.zeros(64)})
    model = DynamicIDOPRegressor().fit(data)
    assert model.network_.empty
    assert not model.adjacency_matrix().to_numpy().any()
    np.testing.assert_allclose(model.predict(), data)


@pytest.mark.parametrize('change', ['nan', 'short', 'duplicate', 'bad_time'])
def test_invalid_inputs_are_rejected(change):
    data = signals()
    if change == 'nan': data.iloc[0,0] = np.nan
    if change == 'short': data = data.iloc[:3]
    if change == 'duplicate': data.columns = ['a','a','c']
    if change == 'bad_time': data['time'] = np.zeros(len(data))
    with pytest.raises(ValueError):
        DynamicIDOPRegressor().fit(data)


def test_selector_uses_sliding_window_frequency_and_validates_ratio():
    supports = get_support_sets(signals(), window_size=30, stride=10)
    frequency = supports.attrs['edge_frequency']
    assert frequency.shape == (3,3)
    assert np.isnan(np.diag(frequency)).all()
    values = frequency.stack().dropna()
    assert np.all((values >= 0) & (values <= 1))
    with pytest.raises(ValueError): get_support_sets(signals(), alpha_ratio=1.1)


def test_page_result_contract_and_full_sample_count():
    result = fit_dynamic_idop_network(signals())
    assert set(result) == {'model','quasi_dynamic_df','curve_sample_df','design_X',
                          'response_Y','predicted_df','effect_df_list','adj_df','adjacency_aggregation'}
    assert len(result['predicted_df']) == 120
    assert result['adjacency_aggregation'] == 'energy_share'
    assert len(result['effect_df_list']) == 3


def test_basis_uses_source_signal_not_time_modulation():
    data = signals()
    supports = pd.DataFrame({'target':['a','b','c'], 'source':['{b}','{}','{}']})
    basis = get_basis(data, supports, legendre_order=3)
    assert basis['sources']['b']['integrals'].shape == (120,3)
    np.testing.assert_allclose(basis['sources']['b']['integrals'][-1], 0, atol=1e-12)
