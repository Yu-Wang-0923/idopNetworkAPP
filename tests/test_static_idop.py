"""可选建网算法（新版选边器）的回归测试。

重点覆盖：

1. ``select_edges_lasso`` 的行为与参数校验；
2. ``StaticIDOPRegressor`` 只换选边器——**效应分解、预测、邻接矩阵的格式与
   仓库 ``IDOPRegressor`` 完全一致**，并且 ``intercept + Σ effects == predicted``；
3. ``fit_static_idop_network`` 产出页面约定的 network 字典。
"""

import numpy as np
import pandas as pd
import pytest

from idopnetwork.curve_fitting import get_power_function_params, get_power_function_sample
from idopnetwork.network.construction import IDOPRegressor, _build_groups
from idopnetwork.network.static_idop import (
    StaticIDOPRegressor,
    fit_static_idop_network,
    select_edges_lasso,
)


def _static_data(n_samples: int = 26, n_features: int = 5, seed: int = 7):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(n_samples, 1))
    return pd.DataFrame(
        {
            f"M{i}": base[:, 0] * (0.7 if i % 2 else -0.5)
            + rng.normal(0.0, 0.25, size=n_samples) + i + 2.0
            for i in range(n_features)
        },
        index=[f"s{i}" for i in range(n_samples)],
    ).clip(lower=0.05)


def _network_inputs(n_samples: int = 26, n_features: int = 5):
    static = _static_data(n_samples, n_features)
    row_sum = static.sum(axis=1)
    order = np.argsort(row_sum.to_numpy(), kind="stable")
    quasi = static.iloc[order].copy()
    quasi.index = pd.Index(
        np.log1p(row_sum.to_numpy(dtype=float)[order]), name="quasi time"
    )
    quasi = quasi.iloc[int(0.05 * len(quasi)):]

    params = get_power_function_params(quasi)
    samples = get_power_function_sample(quasi, n_samples=30)
    return quasi, samples, params


def _repo_model(max_order: int = 1) -> IDOPRegressor:
    return IDOPRegressor(
        max_order=max_order, mix=0.5, fix_mix=False, nonneg_self=True,
        max_interactions=0, adaptive_weights=False,
    )


def _static_model(max_order: int = 1) -> StaticIDOPRegressor:
    return StaticIDOPRegressor(
        max_order=max_order, mix=0.5, fix_mix=False, nonneg_self=True,
        max_interactions=0, adaptive_weights=False,
        lasso_alpha=0.05, lasso_windows=1, lasso_threshold=0.5,
    )


# ── 选边器 ───────────────────────────────────────────────────────────────────

def test_select_edges_lasso_returns_target_source_frame():
    supports = select_edges_lasso(_static_data(), alpha=0.05, k=1, threshold=0.5)
    assert list(supports.columns) == ["target", "source"]
    assert set(supports["target"]) == {"M0", "M1", "M2", "M3", "M4"}


def test_select_edges_lasso_rejects_bad_parameters():
    with pytest.raises(ValueError):
        select_edges_lasso(_static_data(), alpha=0.0)
    with pytest.raises(ValueError):
        select_edges_lasso(_static_data(), threshold=1.5)
    with pytest.raises(ValueError):
        select_edges_lasso(_static_data(), k=0)


def test_higher_threshold_selects_fewer_or_equal_sources():
    data = _static_data(n_samples=60)
    loose = select_edges_lasso(data, alpha=0.05, k=3, threshold=0.0)
    strict = select_edges_lasso(data, alpha=0.05, k=3, threshold=0.99)
    def _count(frame):
        return sum(
            len([s for s in str(row["source"]).strip("{}").split(",") if s.strip()])
            for _, row in frame.iterrows()
        )
    assert _count(strict) <= _count(loose)


# ── 与仓库路线的一致性（本文件的核心） ────────────────────────────────────────

def test_effect_decomposition_format_matches_repo():
    """效应分解的索引/列/形状必须与仓库路线一致 —— 这是「像之前一样」的契约。"""
    quasi, samples, params = _network_inputs()
    repo = _repo_model().fit(samples, quasi, power_function_params=params)
    static = _static_model().fit(samples, quasi, power_function_params=params)

    repo_effects = repo.effect(samples)
    static_effects = static.effect(samples)

    assert len(repo_effects) == len(static_effects)
    for expected, actual in zip(repo_effects, static_effects):
        assert list(actual.columns) == list(expected.columns)
        assert actual.index.equals(expected.index)
        assert actual.shape == expected.shape


def test_effect_identity_holds_for_both_routes():
    """``intercept + Σ effects == predicted`` 两条路线都必须成立。"""
    quasi, samples, params = _network_inputs()
    for model in (
        _repo_model().fit(samples, quasi, power_function_params=params),
        _static_model().fit(samples, quasi, power_function_params=params),
    ):
        effects = model.effect(samples)
        predicted = model.predict(samples).to_numpy(dtype=float)
        intercept = model.coef_.loc["intercept"].to_numpy(dtype=float)
        total = np.column_stack([
            intercept[j]
            + sum(effects[j][c].to_numpy(dtype=float) for c in effects[j].columns)
            for j in range(len(effects))
        ])
        assert np.allclose(total, predicted, atol=1e-8)


def test_effect_curves_start_at_zero_like_repo():
    """基函数首点为 0，因此每条效应曲线都应从 0 起步（两条路线一致）。"""
    quasi, samples, params = _network_inputs()
    for model in (
        _repo_model().fit(samples, quasi, power_function_params=params),
        _static_model().fit(samples, quasi, power_function_params=params),
    ):
        for effect in model.effect(samples):
            assert np.allclose(effect.to_numpy(dtype=float)[0, :], 0.0, atol=1e-12)


def test_adjacency_matrix_format_matches_repo():
    quasi, samples, params = _network_inputs()
    repo = _repo_model().fit(samples, quasi, power_function_params=params)
    static = _static_model().fit(samples, quasi, power_function_params=params)

    repo_adj = repo.adjacency_matrix(samples)
    static_adj = static.adjacency_matrix(samples)
    assert list(static_adj.index) == list(repo_adj.index)
    assert list(static_adj.columns) == list(repo_adj.columns)
    assert static_adj.shape == repo_adj.shape


def test_cross_support_comes_from_new_selector():
    """覆写生效：交叉边支撑集必须来自 select_edges_lasso，而不是父类的 alpha 路径。"""
    quasi, samples, params = _network_inputs()
    model = _static_model().fit(samples, quasi, power_function_params=params)

    assert model.edge_supports_ is not None
    basis = model._design(samples).values[:, 1:]
    groups = _build_groups(len(samples.columns), model.max_order)
    cross_ids = model._select_lasso_cross_support(
        basis, np.zeros(len(samples)), groups, 0
    )

    target = model._target_names_[0]
    expected = {
        index + 1
        for index, name in enumerate(model._feature_names_)
        if name != target and name in model._support_names_[target]
    }
    assert set(cross_ids) == expected


def test_selector_override_actually_changes_results():
    """两套选边器在同一份数据上应给出不同的支撑集（否则这个可选项没有意义）。"""
    quasi, samples, params = _network_inputs()
    repo = _repo_model().fit(samples, quasi, power_function_params=params)
    static = _static_model().fit(samples, quasi, power_function_params=params)
    repo_nnz = int((repo.adjacency_matrix(samples).to_numpy(dtype=float) != 0).sum())
    static_nnz = int((static.adjacency_matrix(samples).to_numpy(dtype=float) != 0).sum())
    assert repo_nnz != static_nnz


# ── 页面契约 ─────────────────────────────────────────────────────────────────

def test_fit_static_idop_network_matches_page_contract():
    quasi, samples, params = _network_inputs()
    network = fit_static_idop_network(
        samples, quasi,
        max_order=1,
        nonneg_self=True,
        max_interactions=0,
        adjacency_aggregation="mean",
        power_function_params=params,
        lasso_alpha=0.05,
    )

    assert set(network) == {
        "model", "quasi_dynamic_df", "curve_sample_df", "design_X",
        "response_Y", "predicted_df", "effect_df_list", "adj_df",
        "adjacency_aggregation",
    }
    features = list(samples.columns)
    assert list(network["adj_df"].index) == features
    assert list(network["adj_df"].columns) == features
    assert network["predicted_df"].shape == (len(samples), len(features))
    assert len(network["effect_df_list"]) == len(features)
    assert network["design_X"].index.equals(samples.index)

    model = network["model"]
    assert isinstance(model, StaticIDOPRegressor)
    # 导出逻辑会读这些属性
    assert isinstance(model.mse_, float) and np.isfinite(model.mse_)
    assert list(model.coef_.loc["intercept"].index) == features
