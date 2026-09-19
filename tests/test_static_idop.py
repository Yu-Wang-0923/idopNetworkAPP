"""静态数据版 idopNetwork（LASSO 选边 + cvxpy 弱形式 ODE）的回归测试。"""

import numpy as np
import pandas as pd
import pytest

from idopnetwork.network.static_idop import (
    StaticIDOPModel,
    fit_power_params,
    fit_static_idop_network,
    sample_power_function,
    select_edges_lasso,
    to_quasi_dynamic,
    transform_static_data,
)


def _static_data(n_samples: int = 24, n_features: int = 4, seed: int = 3):
    """构造一份有内部关联的静态数据（样本 × 特征）。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(n_samples, 1))
    columns = {}
    for index in range(n_features):
        coupling = 0.6 if index % 2 else -0.4
        columns[f"M{index}"] = (
            base[:, 0] * coupling + rng.normal(0.0, 0.3, size=n_samples) + index + 2.0
        )
    return pd.DataFrame(
        columns, index=[f"s{i}" for i in range(n_samples)]
    ).clip(lower=0.05)


def _network_inputs(n_samples: int = 24, n_features: int = 4):
    static = _static_data(n_samples, n_features)
    quasi = to_quasi_dynamic(static)
    quasi.index = pd.Index(np.log1p(quasi.index.to_numpy(dtype=float)), name="quasi time")
    quasi = quasi.iloc[int(0.05 * len(quasi)):]
    params = fit_power_params(quasi)
    samples = sample_power_function(params, quasi.index, n_samples=30)
    return quasi, samples


# ── 变换 / 拟动态 ────────────────────────────────────────────────────────────

def test_transform_shift_min_makes_min_equal_one():
    static = _static_data()
    shifted = transform_static_data(static, "Shift_min")
    assert np.allclose(shifted.min(axis=0).to_numpy(), 1.0)


def test_transform_unknown_method_raises():
    with pytest.raises(ValueError):
        transform_static_data(_static_data(), "Nope")


def test_quasi_dynamic_is_sorted_by_row_sum():
    static = _static_data()
    quasi = to_quasi_dynamic(static)
    row_sums = quasi.sum(axis=1).to_numpy(dtype=float)
    assert np.all(np.diff(row_sums) >= -1e-9)
    assert quasi.index.name == "quasi time"


# ── 选边器 ───────────────────────────────────────────────────────────────────

def test_edge_select_returns_target_source_frame():
    supports = select_edges_lasso(_static_data(), alpha=0.05, k=1, threshold=0.5)
    assert list(supports.columns) == ["target", "source"]
    assert set(supports["target"]) == {"M0", "M1", "M2", "M3"}


def test_edge_select_rejects_bad_parameters():
    with pytest.raises(ValueError):
        select_edges_lasso(_static_data(), alpha=0.0)
    with pytest.raises(ValueError):
        select_edges_lasso(_static_data(), threshold=1.5)


# ── 端到端建网 ───────────────────────────────────────────────────────────────

def test_fit_static_idop_network_matches_page_contract():
    quasi, samples = _network_inputs()
    network = fit_static_idop_network(
        samples, quasi, basis_order=0, alpha=0.05, windows=1, threshold=0.5,
    )

    assert set(network) == {
        "model",
        "quasi_dynamic_df",
        "curve_sample_df",
        "design_X",
        "response_Y",
        "predicted_df",
        "effect_df_list",
        "adj_df",
        "adjacency_aggregation",
    }

    adj_df = network["adj_df"]
    features = list(samples.columns)
    assert list(adj_df.index) == features
    assert list(adj_df.columns) == features

    predicted = network["predicted_df"]
    assert predicted.shape == (len(samples), len(features))
    assert list(predicted.columns) == features

    effects = network["effect_df_list"]
    assert len(effects) == len(features)
    for effect in effects:
        assert effect.shape == (len(samples), len(features))
        assert list(effect.columns) == features

    # _design 的索引必须覆盖 response_Y 的重建（页面用它对齐）
    assert network["design_X"].index.equals(samples.index)
    assert list(network["response_Y"].columns) == features


def test_model_exposes_attributes_used_by_export():
    quasi, samples = _network_inputs()
    network = fit_static_idop_network(
        samples, quasi, basis_order=0, alpha=0.05, windows=1, threshold=0.5,
    )
    model = network["model"]

    # 导出逻辑会读这些属性
    assert isinstance(model.max_order, int)
    assert isinstance(model.alpha, float)
    assert isinstance(model.mix, float)
    assert isinstance(model.nonneg_self, bool)
    assert isinstance(model.max_interactions, int)
    assert isinstance(model.basis_type, str)
    assert isinstance(model.ebic_gamma, float)
    assert isinstance(model.enforce_effect_constraints, bool)
    assert isinstance(model.mse_, float) and np.isfinite(model.mse_)

    # 效应分解绘图读 coef_.loc["intercept"]
    intercept = model.coef_.loc["intercept"]
    assert list(intercept.index) == list(samples.columns)

    # coef_ 的设计列必须能在 _design 里找到
    design_columns = set(model._design(samples).columns) - {"intercept"}
    assert design_columns.issubset(set(model.coef_.index))


def test_adjacency_matches_effect_curves():
    """adj_df.loc[source, target] 必须对应 effect_df_list[target][source] 的聚合。"""
    quasi, samples = _network_inputs()
    network = fit_static_idop_network(
        samples, quasi, basis_order=0, alpha=0.05, windows=1, threshold=0.5,
        adjacency_aggregation="mean",
    )
    adj_df = network["adj_df"]
    features = list(samples.columns)
    for target_index, target in enumerate(features):
        effect = network["effect_df_list"][target_index]
        for source in features:
            if source == target:
                continue
            expected = float(effect[source].mean())
            assert adj_df.loc[source, target] == pytest.approx(expected, abs=1e-9)


def test_integral_aggregation_differs_from_mean():
    quasi, samples = _network_inputs()
    mean_network = fit_static_idop_network(
        samples, quasi, basis_order=0, alpha=0.05, adjacency_aggregation="mean",
    )
    integral_network = fit_static_idop_network(
        samples, quasi, basis_order=0, alpha=0.05, adjacency_aggregation="integral",
    )
    assert not np.allclose(
        mean_network["adj_df"].to_numpy(dtype=float),
        integral_network["adj_df"].to_numpy(dtype=float),
    )


def test_basis_order_one_still_solves():
    quasi, samples = _network_inputs()
    network = fit_static_idop_network(
        samples, quasi, basis_order=1, alpha=0.05, windows=1, threshold=0.5,
    )
    model = network["model"]
    # intercept + 每个源 (basis_order+1) 列
    assert len(model.coef_.index) == 1 + len(samples.columns) * 2
