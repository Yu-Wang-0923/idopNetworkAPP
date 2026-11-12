"""Full-record idopECG(4) adapter: interaction-first Shapley decomposition."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from .ecg_solver import get_basis, get_effects, get_support_sets

DEFAULT_FS = 100.0
DEFAULT_WINDOW = 0  # Automatic support-selection window, not fitting windows.
DEFAULT_N_FOURIER = 0  # Automatic order from 10 Hz bandwidth.
DEFAULT_R_LEGENDRE = 3
SOLVER_VERSION = "idopECG4_interaction_first_shapley"


def solve_dynamic_core(data_fit: pd.DataFrame, edge_supports: pd.DataFrame, *,
                       n_fourier: int | None = None, r_legendre: int = 3,
                       fs: float = 100.0, ridge_alpha: float = 1e-6,
                       fourier_max_frequency: float = 10.0) -> dict:
    """Run the supplied full-record numerical kernel without filesystem IO."""
    options = dict(sampling_rate=fs, fourier_order=None if n_fourier in (None, 0) else n_fourier,
                   fourier_max_frequency=fourier_max_frequency, legendre_order=r_legendre)
    return {"basis": get_basis(data_fit, edge_supports, **options),
            "effects": get_effects(data_fit, edge_supports, ridge_alpha=ridge_alpha, **options)}


class DynamicIDOPRegressor:
    """Preserve page interfaces while using the idopECG(4) algorithm.

    window/lasso_windows name the support-selection window size/stride;
    0 or None requests the upstream defaults. Fitting always uses all samples.
    Signed adjacency is sign × energy share. ecg_exports_ preserves the raw
    nonnegative weight and its separate sign, diagnostics and path-independent
    full-record reconstruction.
    """
    def __init__(self, *, n_fourier: int | None = None, r_legendre: int = 3,
                 window: int | None = None, fs: float = 100.0,
                 lasso_alpha: float = 0.1, lasso_windows: int | None = None,
                 lasso_threshold: float = 0.9, ridge_alpha: float = 1e-6,
                 fourier_max_frequency: float = 10.0, mix: float = 0.5,
                 nonneg_self: bool = False, max_interactions: int = 0,
                 basis_type: str = "idopECG4_source_signal_legendre",
                 ebic_gamma: float = 0.0, enforce_effect_constraints: bool = False):
        self.n_fourier = None if n_fourier in (None, 0) else int(n_fourier)
        self._requested_n_fourier = self.n_fourier
        self.r_legendre = self.max_order = int(r_legendre)
        self.window = None if window in (None, 0) else int(window)
        self.fs = float(fs)
        self.lasso_alpha = self.alpha = float(lasso_alpha)
        self.lasso_windows = None if lasso_windows in (None, 0) else int(lasso_windows)
        self.lasso_threshold = float(lasso_threshold)
        self.ridge_alpha, self.fourier_max_frequency = float(ridge_alpha), float(fourier_max_frequency)
        self.mix, self.nonneg_self = float(mix), bool(nonneg_self)
        self.max_interactions, self.basis_type = int(max_interactions), basis_type
        self.ebic_gamma = float(ebic_gamma)
        self.enforce_effect_constraints = bool(enforce_effect_constraints)
        self.solver_version_ = SOLVER_VERSION
        self._core, self.mse_, self.coef_ = None, None, None
        self.adaptation_notes_ = []

    def fit(self, power_function_sample_df: pd.DataFrame,
            quasi_dynamic_df: pd.DataFrame | None = None, *,
            power_function_params: pd.DataFrame | None = None) -> "DynamicIDOPRegressor":
        """Fit SWT-smoothed data without smoothing again or truncating samples."""
        data = power_function_sample_df.copy()
        data.columns = data.columns.map(str)
        if not data.columns.is_unique or len(data) < 4:
            raise ValueError("动态求解要求至少 4 个采样点及唯一通道名。")
        data = data.apply(pd.to_numeric, errors="raise")
        if "time" not in data.columns and str(data.index.name).lower() == "time":
            data.insert(0, "time", data.index.to_numpy(dtype=float))
        self.leads_ = [column for column in data.columns if column != "time"]
        self.edge_supports_ = get_support_sets(
            data, alpha_ratio=self.lasso_alpha, window_size=self.window,
            stride=self.lasso_windows, threshold=self.lasso_threshold)
        self.edge_frequency_ = self.edge_supports_.attrs["edge_frequency"].copy()
        self._core = solve_dynamic_core(
            data, self.edge_supports_, n_fourier=self._requested_n_fourier, r_legendre=self.r_legendre,
            fs=self.fs, ridge_alpha=self.ridge_alpha, fourier_max_frequency=self.fourier_max_frequency)
        self.effects_ = self._core["effects"]
        self.settings_ = self.effects_["settings"].iloc[0].to_dict()
        self.n_fourier = int(self.settings_["fourier_order"])
        self._time = pd.Index(self._core["basis"]["time"], name="time")
        self._observed = data.loc[:, self.leads_].set_axis(self._time)
        self.diagnostics_ = self.effects_["diagnostics"]
        self.mse_ = float(np.mean(self.effects_["reconstruction"].residual.to_numpy() ** 2))
        self.network_ = self.effects_["network"].copy()
        rows = {"intercept": self._observed.iloc[0].to_dict()}
        own = self.effects_["coefficients"].query("effect_type == 'self'")
        for (name, order), group in own.groupby(["basis", "order"], sort=False):
            rows[f"self::{name}{order}"] = group.set_index("target").coefficient.to_dict()
        self.coef_ = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=self.leads_)
        self.ecg_exports_ = {**self.effects_, "support_set": self.edge_supports_.copy(),
                             "edge_frequency": self.edge_frequency_.rename_axis('target').reset_index()}
        deficient = int(self.diagnostics_.rank_deficient.sum())
        self.adaptation_notes_ = ([f"{deficient} 个目标的设计矩阵秩亏；详见 dynamic/diagnostics.csv。"]
                                  if deficient else [])
        return self

    def _require_fit(self) -> None:
        if self._core is None:
            raise RuntimeError("call fit before requesting results")

    def _design(self, power_function_sample_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Full-record reference basis; target fits use orthogonalized bases."""
        self._require_fit()
        basis = self._core["basis"]
        data = {"intercept": np.ones(len(self._time))}
        for index, (kind, order) in enumerate(basis["self"]["labels"]):
            data[f"self::{kind}{order}"] = basis["self"]["integrals"][:, index]
        for source, block in basis["sources"].items():
            for index in range(block["integrals"].shape[1]):
                data[f"{source}::P{index + 1}"] = block["integrals"][:, index]
        return pd.DataFrame(data, index=self._time)

    def observed(self) -> pd.DataFrame:
        """Full-record observations, on the same physical time grid as prediction."""
        self._require_fit()
        return self._observed.copy()

    def predict(self, power_function_sample_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Upstream reconstruction with the original initial value."""
        self._require_fit()
        return self.effects_["reconstruction"].pivot(index="time", columns="target", values="reconstructed").reindex(index=self._time, columns=self.leads_).rename_axis(columns=None)

    def effect(self, power_function_sample_df: pd.DataFrame | None = None) -> list[pd.DataFrame]:
        """Centered cumulative self/cross contributions, one table per target."""
        self._require_fit()
        results = []
        for target in self.leads_:
            frame = pd.DataFrame(0.0, index=self._time, columns=self.leads_)
            own = self.effects_["self_effects"]
            frame[target] = own.loc[own.target.eq(target), "cumulative_effect"].to_numpy()
            cross = self.effects_["interaction_effects"]
            for source, group in cross.loc[cross.target.eq(target)].groupby("source", sort=False):
                frame[source] = group.cumulative_effect.to_numpy()
            results.append(frame)
        return results

    def adjacency_matrix(self, power_function_sample_df: pd.DataFrame | None = None,
                         aggregation: str = "energy_share") -> pd.DataFrame:
        """source→target = energy share × independently fitted sign.

        Legacy mean/integral inputs are accepted but never override the new
        solver's prescribed edge-weight definition.
        """
        self._require_fit()
        if aggregation not in {"energy_share", "mean", "integral"}:
            raise ValueError("动态邻接矩阵使用 energy_share。")
        adjacency = pd.DataFrame(0.0, index=self.leads_, columns=self.leads_)
        for row in self.network_.itertuples():
            adjacency.loc[row.source, row.target] = float(row.weight) * (-1 if row.sign == '-' else 1)
        return adjacency


def observed_for_plot(model: Any, fallback: pd.DataFrame) -> pd.DataFrame:
    """Keep static input; use dynamic observations on their exact prediction grid."""
    return model.observed() if isinstance(model, DynamicIDOPRegressor) else fallback


def fit_dynamic_idop_network(curve_sample_df: pd.DataFrame,
                             response_df: pd.DataFrame | None = None, *,
                             max_order: int = DEFAULT_R_LEGENDRE,
                             nonneg_self: bool = False, max_interactions: int = 0,
                             adjacency_aggregation: str = "energy_share",
                             power_function_params: pd.DataFrame | None = None,
                             **options) -> dict:
    """Build the existing nine-field result using idopECG(4)."""
    model = DynamicIDOPRegressor(r_legendre=max_order, nonneg_self=nonneg_self,
                                 max_interactions=max_interactions, **options).fit(curve_sample_df)
    response = model.observed()
    return {"model": model, "quasi_dynamic_df": response,
            "curve_sample_df": curve_sample_df, "design_X": model._design(),
            "response_Y": response, "predicted_df": model.predict(),
            "effect_df_list": model.effect(), "adj_df": model.adjacency_matrix(),
            "adjacency_aggregation": "energy_share"}
