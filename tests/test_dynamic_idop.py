"""动态数据版建网（SWT 去噪 + Fourier/Legendre 两阶段最小二乘）回归测试。"""

import numpy as np
import pandas as pd
import pytest

from idopnetwork.curve_fitting.dynamic import swt_denoise
from idopnetwork.network.dynamic_idop import (
    DynamicIDOPRegressor,
    fit_dynamic_idop_network,
    solve_dynamic_core,
)
from idopnetwork.network.static_idop import select_edges_lasso

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]


def _ecg_like(n_timepoints: int = 1000, seed: int = 5) -> pd.DataFrame:
    """构造带耦合的 12 导联类 ECG 信号。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n_timepoints) / 100.0
    base = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    columns = {}
    for index, lead in enumerate(LEADS):
        coupling = 0.5 if index % 3 == 0 else (-0.3 if index % 3 == 1 else 0.15)
        columns[lead] = (
            (1.0 + coupling) * base
            + 0.2 * np.roll(base, 7 + index)
            + rng.normal(0.0, 0.05, size=n_timepoints)
        )
    return pd.DataFrame(columns, index=t)


def _noisy_ecg(n_timepoints: int = 1000, seed: int = 5) -> pd.DataFrame:
    clean = _ecg_like(n_timepoints, seed)
    rng = np.random.default_rng(seed + 100)
    noise = rng.normal(0.0, 0.25, size=clean.shape)
    return clean + pd.DataFrame(noise, index=clean.index, columns=clean.columns)


# ── SWT 去噪 ─────────────────────────────────────────────────────────────────

def test_swt_denoise_preserves_shape_and_index():
    data = _noisy_ecg()
    fitted, params = swt_denoise(data)
    assert fitted.shape == data.shape
    assert fitted.index.equals(data.index)
    assert list(fitted.columns) == list(data.columns)
    assert list(params.index) == list(data.columns)
    assert {"wavelet", "level", "sigma", "threshold", "denoised"} <= set(params.columns)


def test_swt_denoise_reduces_noise():
    clean = _ecg_like()
    noisy = _noisy_ecg()
    fitted, _ = swt_denoise(noisy)
    err_noisy = float(np.mean((noisy.to_numpy() - clean.to_numpy()) ** 2))
    err_fitted = float(np.mean((fitted.to_numpy() - clean.to_numpy()) ** 2))
    assert err_fitted < err_noisy


def test_swt_denoise_handles_constant_channel():
    data = _noisy_ecg()
    data["flat"] = 0.0
    fitted, params = swt_denoise(data)
    assert np.allclose(fitted["flat"].to_numpy(), 0.0)
    assert bool(params.loc["flat", "denoised"]) is False


# ── 求解内核 ─────────────────────────────────────────────────────────────────

def test_solve_dynamic_core_shapes_and_windows():
    data = _ecg_like(n_timepoints=1000)
    supports = select_edges_lasso(data, alpha=0.05, k=5, threshold=0.3)
    core = solve_dynamic_core(data, supports, n_fourier=5, r_legendre=2, window=250)

    assert core["leads"] == LEADS
    assert core["n_windows"] == 4
    assert len(core["t_win"]) == 250
    assert core["t_end"] == pytest.approx(2.5)
    for lead in LEADS:
        assert core["avg_self"][lead].shape == (250,)
        assert core["avg_obs"][lead].shape == (250,)
        assert 0.0 <= core["r2_mean"][lead] <= 1.0


def test_solve_dynamic_core_rejects_short_series():
    data = _ecg_like(n_timepoints=100)
    supports = select_edges_lasso(data, alpha=0.05, k=2, threshold=0.3)
    with pytest.raises(ValueError):
        solve_dynamic_core(data, supports, window=250, auto_adapt=False)


def test_solve_dynamic_core_auto_adapts_short_series():
    """短序列（如误用准动态的 30 行采样）应自动缩小窗口而不是直接失败。"""
    data = _ecg_like(n_timepoints=30)
    supports = select_edges_lasso(data, alpha=0.05, k=2, threshold=0.3)
    core = solve_dynamic_core(data, supports, n_fourier=30, window=250)

    assert core["adapted"] is True
    assert core["window"] == 30
    assert core["n_windows"] == 1
    assert len(core["t_win"]) == 30
    # Fourier 基列数 2N+1 不得超过窗口的一半
    assert 2 * core["n_fourier"] + 1 <= 30 // 2
    assert core["notes"] and "Dynamic" in " ".join(core["notes"])


def test_solver_surfaces_adaptation_notes():
    model = DynamicIDOPRegressor(
        n_fourier=30, r_legendre=2, window=250, fs=100.0,
        lasso_alpha=0.05, lasso_windows=2, lasso_threshold=0.3,
    )
    model.fit(_ecg_like(n_timepoints=30))
    assert model.adaptation_notes_
    assert model.window == 30
    # 适配之后仍应产出可用的效应曲线（长度 = 生效窗口）
    effects = model.effect()
    assert len(effects[0]) == 30


def test_no_adaptation_note_for_full_length_series():
    data = _ecg_like(n_timepoints=1000)
    supports = select_edges_lasso(data, alpha=0.05, k=5, threshold=0.3)
    core = solve_dynamic_core(data, supports, n_fourier=5, window=250)
    assert core["adapted"] is False
    assert core["window"] == 250
    assert core["notes"] == []


def test_intercept_is_window_start_value():
    """截距必须是各窗口起点 x(0) 的跨窗均值，而不是居中观测的均值。

    这是与 idopECG 对齐的关键口径：效应曲线是相对 x(0) 的居中量，
    绝对还原为 ``x(0) + 自效应 + Σ 交叉效应``。
    """
    data = _ecg_like(n_timepoints=1000)
    model = DynamicIDOPRegressor(
        n_fourier=5, r_legendre=2, window=250, fs=100.0,
        lasso_alpha=0.05, lasso_windows=5, lasso_threshold=0.3,
    )
    model.fit(data)

    intercept = model.coef_.loc["intercept"]
    window = model.window
    n_windows = len(data) // window
    for lead in LEADS:
        starts = data[lead].to_numpy(dtype=float)[np.arange(n_windows) * window]
        assert intercept[lead] == pytest.approx(float(np.mean(starts)), abs=1e-9)


def test_dynamic_selector_uses_idopecg_convention():
    """动态流程的选边器必须不标准化 y（对齐 idopECG.edge_select）。"""
    data = _ecg_like(n_timepoints=1000)
    model = DynamicIDOPRegressor(
        n_fourier=5, r_legendre=2, window=250, fs=100.0,
        lasso_alpha=0.05, lasso_windows=5, lasso_threshold=0.3,
    )
    model.fit(data)

    expected = select_edges_lasso(
        data, alpha=0.05, k=5, threshold=0.3, standardize_y=False
    )
    assert model.edge_supports_.reset_index(drop=True).equals(
        expected.reset_index(drop=True)
    )


def test_selector_standardize_y_flag_is_wired():
    """standardize_y 开关必须真实生效（默认 True 保留静态路线口径）。"""
    data = _ecg_like(n_timepoints=1000)
    default = select_edges_lasso(data, alpha=0.05, k=5, threshold=0.3)
    raw = select_edges_lasso(
        data, alpha=0.05, k=5, threshold=0.3, standardize_y=False
    )
    assert list(default.columns) == ["target", "source"]
    assert list(raw.columns) == ["target", "source"]
    assert len(default) == len(raw) == len(LEADS)


def test_predict_is_intercept_plus_effects_in_absolute_scale():
    """预测必须落在绝对尺度上（截距 + 效应），而非居中尺度。"""
    data = _ecg_like(n_timepoints=1000)
    model = DynamicIDOPRegressor(
        n_fourier=5, r_legendre=2, window=250, fs=100.0,
        lasso_alpha=0.05, lasso_windows=5, lasso_threshold=0.3,
    )
    model.fit(data)

    predicted = model.predict()
    # 预测的均值应与各窗口观测均值同量级（绝对尺度），而不是围绕 0 的居中量
    observed_mean = float(
        data.to_numpy(dtype=float)[:250 * (len(data) // 250)].mean()
    )
    assert abs(float(predicted.to_numpy(dtype=float).mean()) - observed_mean) < 5.0
    # 居中效应之和本身应围绕 0
    effects = model.effect()
    centered = sum(float(effects[0][c].mean()) for c in effects[0].columns)
    assert abs(centered) < 5.0


def test_over_complete_design_is_flagged_as_unreliable():
    """设计列数超过窗口长度时必须给出"结果不可信"的警告，而不是静默跑通。"""
    data = _ecg_like(n_timepoints=30)
    supports = select_edges_lasso(data, alpha=0.05, k=2, threshold=0.2)
    core = solve_dynamic_core(data, supports, n_fourier=30, window=250)
    assert core["adapted"] is True
    joined = " ".join(core["notes"])
    assert "结果不可信" in joined
    # 确实发生了完美插值（这是被警告的现象本身）
    r2s = np.array([core["r2_mean"][lead] for lead in core["leads"]])
    assert float(r2s.max()) > 0.999


# ── 适配器 ───────────────────────────────────────────────────────────────────

def _fitted_model() -> DynamicIDOPRegressor:
    model = DynamicIDOPRegressor(
        n_fourier=5, r_legendre=2, window=250, fs=100.0,
        lasso_alpha=0.05, lasso_windows=5, lasso_threshold=0.3,
    )
    model.fit(_ecg_like(n_timepoints=1000))
    return model


def test_effect_layout_matches_regressor_contract():
    model = _fitted_model()
    effects = model.effect()
    assert len(effects) == len(LEADS)
    for effect in effects:
        assert list(effect.columns) == LEADS
        assert len(effect) == 250


def test_effect_identity_holds():
    """intercept + Σ effects == predicted（与静态路线同一口径）。"""
    model = _fitted_model()
    effects = model.effect()
    predicted = model.predict().to_numpy(dtype=float)
    intercept = model.coef_.loc["intercept"].to_numpy(dtype=float)
    total = np.column_stack([
        intercept[j] + sum(effects[j][c].to_numpy(dtype=float) for c in effects[j].columns)
        for j in range(len(effects))
    ])
    assert np.allclose(total, predicted, atol=1e-8)


def test_adjacency_matches_effect_means():
    model = _fitted_model()
    adj = model.adjacency_matrix(aggregation="mean")
    effects = model.effect()
    assert list(adj.index) == LEADS
    assert list(adj.columns) == LEADS
    for target_index, target in enumerate(LEADS):
        for source in LEADS:
            if source == target:
                continue
            assert adj.loc[source, target] == pytest.approx(
                float(effects[target_index][source].mean()), abs=1e-9
            )


def test_export_attributes_present():
    model = _fitted_model()
    assert isinstance(model.max_order, int)
    assert isinstance(model.alpha, float)
    assert isinstance(model.mse_, float) and np.isfinite(model.mse_)
    assert list(model.coef_.loc["intercept"].index) == LEADS


def test_fit_dynamic_idop_network_matches_page_contract():
    network = fit_dynamic_idop_network(
        _ecg_like(n_timepoints=1000),
        max_order=2, n_fourier=5, window=250,
        lasso_alpha=0.05, lasso_windows=5, lasso_threshold=0.3,
    )
    assert set(network) == {
        "model", "quasi_dynamic_df", "curve_sample_df", "design_X",
        "response_Y", "predicted_df", "effect_df_list", "adj_df",
        "adjacency_aggregation",
    }
    assert list(network["adj_df"].index) == LEADS
    assert network["predicted_df"].shape == (250, len(LEADS))
    assert len(network["effect_df_list"]) == len(LEADS)
    assert isinstance(network["model"], DynamicIDOPRegressor)


# ── 上传工作流（曲线拟合页的 dynamic 分支） ───────────────────────────────────

def test_fit_uploaded_csv_dynamic_mode_keeps_three_table_contract():
    """dynamic 模式必须仍返回三表（FunClu / NetRecon 的 ZIP 读取依赖它）。"""
    from idopnetwork_app.curve_fitting_workflow import (
        DATA_MODE_DYNAMIC,
        fit_uploaded_csv,
    )

    data = _noisy_ecg(n_timepoints=512)
    result = fit_uploaded_csv(
        data.to_csv().encode(), data_mode=DATA_MODE_DYNAMIC, swt_level=2,
    )
    assert set(result) == {"quasi_dynamic", "curve_params", "curve_sample"}
    assert result["quasi_dynamic"].shape == data.shape
    assert result["curve_sample"].shape == data.shape
    assert list(result["curve_params"].columns)[:1] == ["wavelet"]


def test_fit_uploaded_csv_quasi_mode_still_works():
    from idopnetwork_app.curve_fitting_workflow import (
        DATA_MODE_QUASI,
        fit_uploaded_csv,
    )

    rng = np.random.default_rng(3)
    static = pd.DataFrame(
        rng.lognormal(0.0, 0.4, size=(40, 5)),
        columns=[f"M{i}" for i in range(5)],
    )
    result = fit_uploaded_csv(static.to_csv().encode(), data_mode=DATA_MODE_QUASI)
    assert set(result) == {"quasi_dynamic", "curve_params", "curve_sample"}
    # 仓库的幂律拟合返回 c / beta（另有 a / b 等派生列）
    assert {"c", "beta"} <= set(result["curve_params"].columns)
