"""静态数据版 idopNetwork：LASSO 选边 + cvxpy 约束弱形式 ODE 求解。

移植自新版 ``idop.py``（@author: Yu Wang），作为 :class:`IDOPRegressor`
（ASGL + BIC 路线）之外的**可选建网算法**。

流程
----
1. :func:`transform_static_data`：列方向缩放 / 变换；
2. :func:`to_quasi_dynamic`：按样本行和升序排成拟动态序列，行和作拟时间；
3. :func:`fit_power_params` / :func:`sample_power_function`：逐特征拟合
   ``y = a·x^b`` 并重采样，作为基展开输入；
4. :func:`select_edges_lasso`：把样本切成 ``k`` 个窗口，逐窗口对每个 target
   做 LASSO，**出现频率超过 ``threshold`` 的 source 才进入支撑集**；
5. :func:`_solve_ode_decomposition`：积分 Legendre 基 + cvxpy 约束凸优化，
   解 ``dy_j/dt = Q_{j←j} + Σ_k Q_{j←k}`` 的弱形式；
6. :func:`solve_ode_edgelist`：把作用曲线折算成带符号边表。

与原脚本的差异
--------------
- ``from plot_fit / plot_effect_decomposition / plot_network import ...`` 是原脚本的
  绘图依赖，本模块**不做任何绘图**，只保留计算部分。
- 新增 :class:`StaticIDOPModel`，把新算法包装成与 :class:`IDOPRegressor` 相同的
  下游接口（``fit`` / ``predict`` / ``effect`` / ``adjacency_matrix`` / ``_design``
  以及导出诊断要读的若干属性），使页面可以把两套算法当同一个东西使用。
- ``_solve_ode_decomposition`` 额外返回 ``coefficients``（每个 target 的 θ 向量）
  与 ``design_columns``，供适配器构造 ``coef_``。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Lasso

__all__ = [
    "StaticIDOPModel",
    "fit_static_idop_network",
    "transform_static_data",
    "to_quasi_dynamic",
    "fit_power_params",
    "sample_power_function",
    "select_edges_lasso",
    "solve_ode_edgelist",
]


# ── 变换器 ────────────────────────────────────────────────────────────────────

def transform_static_data(data: pd.DataFrame, method: str = "None") -> pd.DataFrame:
    """列方向缩放 / 变换（逐特征）。

    支持 ``"None"`` / ``"Minmax_0_1"`` / ``"Minmax_neg1_1"`` /
    ``"Zscore_shift"`` / ``"Log10_1p"`` / ``"Shift_min"``。
    """
    if method == "None":
        return data

    if method in {"Minmax_0_1", "Minmax_neg1_1"}:
        minimum = data.min(axis=0)
        span = data.max(axis=0) - minimum
        safe_span = span.mask(span == 0.0, 1.0)
        unit = (data - minimum) / safe_span
        return unit if method == "Minmax_0_1" else 2.0 * unit - 1.0

    if method == "Zscore_shift":
        mean = data.mean(axis=0)
        standard_deviation = data.std(axis=0, ddof=0)
        safe_standard_deviation = standard_deviation.mask(
            standard_deviation == 0.0, 1.0
        )
        zscore = (data - mean) / safe_standard_deviation
        return zscore - zscore.min(axis=0) + 1.0

    if method == "Log10_1p":
        if (data <= -1.0).any(axis=None):
            row, column = np.argwhere(data.to_numpy() <= -1.0)[0]
            raise ValueError(
                "Log10_1p 要求所有值大于 -1；非法值位于 "
                f"row {data.index[row]!r}, feature {data.columns[column]!r}。"
            )
        return np.log10(1.0 + data)

    if method == "Shift_min":
        # 纯逐列平移，保证 min == 1 (> 0)，且不改变行和顺序（拟时间不变）
        return data - data.min(axis=0) + 1.0

    raise ValueError(f"Unsupported method: {method}.")


# ── 拟动态器 ──────────────────────────────────────────────────────────────────

def to_quasi_dynamic(
    data: pd.DataFrame,
    index_log1p: bool = False,
) -> pd.DataFrame:
    """静态表 → 拟动态表：按样本**行和升序**重排，行和作为拟时间索引。"""
    row_sum = data.sum(axis=1)
    sort_pos = np.argsort(row_sum.to_numpy(), kind="stable")
    quasi_dynamic_data = data.iloc[sort_pos].copy()
    quasi_time = row_sum.to_numpy(dtype=float)[sort_pos]

    if index_log1p:
        if np.any(quasi_time <= 0):
            raise ValueError("Quasi-time values must be positive.")
        quasi_time = np.log1p(quasi_time)

    quasi_dynamic_data.index = pd.Index(quasi_time, name="quasi time")
    return quasi_dynamic_data


# ── 幂律拟合 ──────────────────────────────────────────────────────────────────

def _power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """``y = a·x^b``。"""
    return a * np.power(x, b)


def fit_power_params(
    data: pd.DataFrame,
    include_r2: bool = False,
) -> pd.DataFrame:
    """逐特征拟合 ``y = a·x^b``，返回 ``["a", "b"(, "R²")]``（index = feature）。"""
    quasi_time = data.index.to_numpy(dtype=float)

    if not np.isfinite(quasi_time).all():
        raise ValueError("quasi-time must contain only finite values.")
    if np.any(quasi_time <= 0):
        raise ValueError(
            "quasi-time must be strictly positive for stable power-law fitting."
        )

    results = []
    for feature, series in data.items():
        y = series.to_numpy(dtype=float)
        (a, b), _ = curve_fit(
            _power_law, quasi_time, y, p0=(1.0, 0.5), maxfev=10_000,
        )
        result = {"feature": feature, "a": a, "b": b}
        if include_r2:
            predicted = _power_law(quasi_time, a, b)
            ss_res = float(np.sum((y - predicted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            result["R²"] = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        results.append(result)

    return pd.DataFrame(results).set_index("feature")


def sample_power_function(
    params: pd.DataFrame,
    quasi_time: pd.Index | np.ndarray,
    n_samples: int = 100,
) -> pd.DataFrame:
    """在 ``quasi_time`` 范围内等距重采样幂律拟合曲线。"""
    x = np.asarray(quasi_time, dtype=float)
    sample_x = np.linspace(x.min(), x.max(), n_samples)

    fitted = {
        feature: _power_law(sample_x, params.at[feature, "a"], params.at[feature, "b"])
        for feature in params.index
    }

    return pd.DataFrame(
        fitted,
        index=pd.Index(sample_x, name=getattr(quasi_time, "name", None)),
    )


# ── 选边器 ────────────────────────────────────────────────────────────────────

def select_edges_lasso(
    data_ecg: pd.DataFrame,
    alpha: float = 0.1,
    k: int = 10,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """用「多窗口 LASSO 出现频率」恢复稀疏有向支撑集。

    把样本切成 ``k`` 个窗口；每个窗口内对每个 target 做一次逐列 z-score 后
    的 LASSO；系数还原回原始尺度后，**在超过 ``threshold`` 比例的窗口里都非零**
    的 source 才进入该 target 的支撑集。

    Returns
    -------
    pd.DataFrame
        ``["target", "source"]``，``source`` 形如 ``"{a,b}"``。
    """
    lead_columns = [column for column in data_ecg.columns if column != "time"]
    if len(lead_columns) < 2:
        raise ValueError("data_ecg must contain at least two lead columns")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(data_ecg):
        raise ValueError("k cannot exceed the number of time points")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    windows = np.array_split(data_ecg[lead_columns].to_numpy(dtype=float), k, axis=0)
    edge_counts = {
        target: {source: 0 for source in lead_columns if source != target}
        for target in lead_columns
    }

    for window in windows:
        for target_idx, target in enumerate(lead_columns):
            source_indices = [i for i in range(len(lead_columns)) if i != target_idx]
            x = window[:, source_indices]
            y = window[:, target_idx]

            # 标准化 x(逐列) 与 y，让 LASSO 目标尺度 O(1)，改善收敛
            x_mean = x.mean(axis=0)
            x_scale = x.std(axis=0)
            x_scale[x_scale == 0] = 1.0
            x_std = (x - x_mean) / x_scale

            y_mean = float(y.mean())
            y_scale = float(y.std())
            if y_scale == 0.0:
                y_scale = 1.0
            y_std = (y - y_mean) / y_scale

            model = Lasso(
                alpha=alpha, fit_intercept=True, max_iter=100_000, tol=1e-5,
            )
            model.fit(x_std, y_std)

            # 系数还原到原始尺度: beta_orig = beta_std * y_scale / x_scale
            coefficients = model.coef_ * (y_scale / x_scale)
            for local_idx, coefficient in enumerate(coefficients):
                if coefficient != 0.0:
                    source = lead_columns[source_indices[local_idx]]
                    edge_counts[target][source] += 1

    rows = []
    for target in lead_columns:
        selected = [
            source
            for source in lead_columns
            if source != target and edge_counts[target][source] / k > threshold
        ]
        rows.append({"target": target, "source": "{" + ",".join(selected) + "}"})

    return pd.DataFrame(rows, columns=["target", "source"])


def _parse_support_sets(
    edge_supports: pd.DataFrame,
    features: list[str],
) -> dict[str, list[str]]:
    """``{"a,b"}`` → ``{target: [source, ...]}``（只保留已知特征）。"""
    support_sets: dict[str, list[str]] = {}
    for _, row in edge_supports.iterrows():
        target = str(row["target"])
        inner = str(row["source"]).strip().strip("{}")
        sources = [s.strip() for s in inner.split(",") if s.strip()]
        support_sets[target] = [s for s in sources if s in features]
    return support_sets


# ── 积分 Legendre 基 ──────────────────────────────────────────────────────────

def integral_legendre_basis(
    sample_states: np.ndarray,
    basis_order: int,
) -> np.ndarray:
    """积分 Legendre 基：``basis[source, t, r] = ∫_0^t P_r(x_k(s)) dx_k(s)``。

    Returns
    -------
    np.ndarray
        形状 ``(n_features, n_t, basis_order + 1)``。
    """
    from scipy.special import eval_legendre

    # 注意：``sample_states`` 的形状是 (n_timepoints, n_features)，
    # 不要按 (n_features, n_timepoints) 解包。
    n_t, n_features = sample_states.shape
    max_order = basis_order + 1
    basis = np.zeros((n_features, n_t, max_order), dtype=float)

    for source in range(n_features):
        source_values = sample_states[:, source]
        minimum = float(np.min(source_values))
        maximum = float(np.max(source_values))
        if maximum > minimum:
            scaled = -1.0 + 2.0 * (source_values - minimum) / (maximum - minimum)
        else:
            scaled = np.zeros_like(source_values)
        increments = np.diff(scaled)
        for r in range(max_order):
            polynomial = eval_legendre(r, scaled)
            if n_t > 1:
                trapezoids = 0.5 * (polynomial[1:] + polynomial[:-1]) * increments
                basis[source, 1:, r] = np.cumsum(trapezoids)

    return basis


# ── 求解器内核 ────────────────────────────────────────────────────────────────

def _solve_ode_decomposition(
    quasi_dynamic_data: pd.DataFrame,
    samples: pd.DataFrame,
    edge_supports: pd.DataFrame,
    basis_order: int,
    ridge: float = 1e-6,
    cross_l1_scale: float = 5e-4,
    gap_min: float = 1e-6,
    y_pad: float = 0.1,
    y_soft_scale: float = 100.0,
    effect_cap: float | None = 2.0,
    effect_soft_scale: float = 50.0,
) -> dict:
    """弱形式 ODE 求解：积分 Legendre 基 + cvxpy 约束凸优化。

    每个 target 解一个 cvxpy 问题：

    - 目标：``||y - intercept - Dθ||² + ridge·||θ||²
      + cross_l1_scale·||θ_cross||₁``；
    - 约束：``intercept + D_self·θ_self ≥ 0``（自身动态保持非负）；
    - 约束：跨源累积效应同号且绝对值不小于 ``gap_min``（两个方向各试一次）；
    - 软惩罚：重构 y 超出观测区间 ``± y_pad·span`` 的部分，权重 ``y_soft_scale``；
    - 软惩罚：单个源累积效应超过 ``effect_cap·span`` 的部分（防止大正大负抵消
      造成的虚高单项），``effect_cap=None`` 关闭。
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("ode_solver optimization requires cvxpy.") from exc

    if not isinstance(quasi_dynamic_data, pd.DataFrame):
        raise TypeError("quasi_dynamic_data must be a pandas DataFrame.")
    if not isinstance(samples, pd.DataFrame):
        raise TypeError("samples must be a pandas DataFrame.")
    if not isinstance(edge_supports, pd.DataFrame):
        raise TypeError("edge_supports must be a pandas DataFrame.")
    if basis_order < 0:
        raise ValueError("basis_order must be non-negative.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if y_pad < 0:
        raise ValueError("y_pad must be non-negative.")
    if y_soft_scale < 0:
        raise ValueError("y_soft_scale must be non-negative.")
    if effect_cap is not None and effect_cap <= 0:
        raise ValueError("effect_cap must be positive or None.")
    if effect_soft_scale < 0:
        raise ValueError("effect_soft_scale must be non-negative.")
    if not {"target", "source"}.issubset(edge_supports.columns):
        raise ValueError("edge_supports must contain columns ['target', 'source'].")

    features = list(samples.columns)
    missing = [f for f in quasi_dynamic_data.columns if f not in features]
    if missing:
        raise ValueError(
            f"quasi_dynamic_data has features missing from samples: {missing}."
        )

    support_sets = _parse_support_sets(edge_supports, features)

    # 把 quasi_dynamic_data 插值到 samples 的时间网格，作为弱形式响应
    sample_tau = samples.index.to_numpy(dtype=float)
    observed_tau = quasi_dynamic_data.index.to_numpy(dtype=float)
    observed_values = quasi_dynamic_data[features].to_numpy(dtype=float)
    order = np.argsort(observed_tau, kind="stable")
    observed_tau, observed_values = observed_tau[order], observed_values[order]
    unique_tau, unique_idx = np.unique(observed_tau, return_index=True)
    n_features = len(features)
    response = np.column_stack(
        [
            np.interp(sample_tau, unique_tau, observed_values[unique_idx, j])
            for j in range(n_features)
        ]
    )
    n_t = response.shape[0]
    max_order = basis_order + 1

    sample_states = samples[features].to_numpy(dtype=float)
    if not np.isfinite(sample_states).all():
        raise ValueError("samples must contain only finite values.")

    basis = integral_legendre_basis(sample_states, basis_order)
    intercepts = sample_states[0].copy()
    interaction_functions: dict[tuple[str, str], np.ndarray] = {}
    predicted_states = np.empty_like(response)
    coefficients: dict[str, np.ndarray] = {}
    design_columns: dict[str, list[str]] = {}

    for target_idx, target in enumerate(features):
        block_sources = [target]
        for source in support_sets.get(target, []):
            if source not in block_sources:
                block_sources.append(source)
        source_indices = [features.index(s) for s in block_sources]
        design = np.column_stack([basis[i] for i in source_indices])

        adjusted_response = response[:, target_idx] - intercepts[target_idx]
        self_positions = np.arange(max_order)
        cross_positions = np.arange(max_order, design.shape[1])

        directions = (
            ["self_above_total", "self_below_total"]
            if len(cross_positions)
            else ["self_only"]
        )
        obs_min = float(np.min(response[:, target_idx]))
        obs_max = float(np.max(response[:, target_idx]))
        span = max(obs_max - obs_min, 1e-12)
        y_lower = obs_min - y_pad * span
        y_upper = obs_max + y_pad * span

        best_theta = None
        best_objective = np.inf

        for direction in directions:
            theta_variable = cp.Variable(design.shape[1])
            total_dynamic = design @ theta_variable
            constraints = [
                intercepts[target_idx]
                + design[:, self_positions] @ theta_variable[self_positions]
                >= 0.0,
            ]
            if len(cross_positions):
                cross_dynamic = (
                    design[:, cross_positions] @ theta_variable[cross_positions]
                )
                constrained_cross = cross_dynamic[1:] if n_t > 1 else cross_dynamic
                if direction == "self_above_total":
                    constraints.extend([
                        constrained_cross <= -gap_min,
                        cp.sum(-constrained_cross) / max(n_t - 1, 1) >= gap_min,
                    ])
                else:
                    constraints.extend([
                        constrained_cross >= gap_min,
                        cp.sum(constrained_cross) / max(n_t - 1, 1) >= gap_min,
                    ])
            regularization = ridge * cp.sum_squares(theta_variable)
            if len(cross_positions):
                regularization += cross_l1_scale * cp.norm1(
                    theta_variable[cross_positions]
                )
            excess_y = cp.pos(
                intercepts[target_idx] + total_dynamic - y_upper
            ) + cp.pos(y_lower - intercepts[target_idx] - total_dynamic)
            regularization += y_soft_scale * cp.sum_squares(excess_y)
            if effect_cap is not None:
                cap = effect_cap * span
                for block_index in range(len(block_sources)):
                    block_slice = slice(
                        block_index * max_order, (block_index + 1) * max_order,
                    )
                    block_effect = (
                        design[:, block_slice] @ theta_variable[block_slice]
                    )
                    excess = cp.pos(block_effect - cap) + cp.pos(-cap - block_effect)
                    regularization += effect_soft_scale * cp.sum_squares(excess)

            problem = cp.Problem(
                cp.Minimize(
                    cp.sum_squares(adjusted_response - total_dynamic) + regularization
                ),
                constraints,
            )
            try:
                problem.solve(solver=cp.CLARABEL, verbose=False)
            except cp.error.SolverError:
                problem.solve(solver=cp.SCS, verbose=False)
            if theta_variable.value is None or problem.status not in {
                cp.OPTIMAL, cp.OPTIMAL_INACCURATE,
            }:
                continue

            theta_value = np.asarray(theta_variable.value, dtype=float).reshape(-1)
            prediction = intercepts[target_idx] + design @ theta_value
            residual = float(np.sum(np.square(response[:, target_idx] - prediction)))
            if residual < best_objective:
                best_objective = residual
                best_theta = theta_value

        if best_theta is None:
            raise ValueError(
                f"No feasible constrained ODE solution for target {target!r}. "
                "Try a smaller basis_order, a larger ridge, or a looser edge support."
            )

        total_effect = intercepts[target_idx] * np.ones(n_t)
        for block_index, source in enumerate(block_sources):
            block = best_theta[block_index * max_order:(block_index + 1) * max_order]
            effect = basis[features.index(source)] @ block
            interaction_functions[(target, source)] = effect
            total_effect = total_effect + effect
        predicted_states[:, target_idx] = total_effect

        coefficients[target] = best_theta
        design_columns[target] = [
            f"{source}::P{r}"
            for source in block_sources
            for r in range(max_order)
        ]

    return {
        "features": features,
        "sample_tau": sample_tau,
        "intercepts": intercepts,
        "response": response,
        "predicted_states": predicted_states,
        "interaction_functions": interaction_functions,
        "support_sets": support_sets,
        # 以下两项为适配器（构造 coef_ / _design）新增
        "coefficients": coefficients,
        "design_columns": design_columns,
    }


def solve_ode_edgelist(
    quasi_dynamic_data: pd.DataFrame,
    samples: pd.DataFrame,
    edge_supports: pd.DataFrame,
    basis_order: int,
    ridge: float = 1e-6,
    cross_l1_scale: float = 5e-4,
    gap_min: float = 1e-6,
    y_pad: float = 0.1,
    y_soft_scale: float = 100.0,
    effect_cap: float | None = 2.0,
    effect_soft_scale: float = 50.0,
) -> pd.DataFrame:
    """解 ODE 系统并返回带符号边表。

    ``weight = sign(mean(effect)) · sqrt(mean(effect²))``，``effect`` 为该边重构
    出来的作用曲线（与 :func:`_solve_ode_decomposition` 同源，保证一致）。
    """
    decomposition = _solve_ode_decomposition(
        quasi_dynamic_data, samples, edge_supports, basis_order, ridge,
        cross_l1_scale, gap_min, y_pad, y_soft_scale, effect_cap,
        effect_soft_scale,
    )
    features = decomposition["features"]
    interaction_functions = decomposition["interaction_functions"]
    support_sets = decomposition["support_sets"]

    rows = []
    for target in features:
        for source in support_sets.get(target, []):
            effect = interaction_functions[(target, source)]
            weight = float(np.sqrt(float(np.mean(np.square(effect)))))
            if float(np.mean(effect)) < 0.0:
                weight = -weight
            rows.append({"source": source, "target": target, "weight": weight})

    return pd.DataFrame(rows, columns=["source", "target", "weight"])


# ── 适配器：伪装成 IDOPRegressor 的下游接口 ───────────────────────────────────

class StaticIDOPModel:
    """把「LASSO 选边 + cvxpy 弱形式 ODE」包装成与 :class:`IDOPRegressor`
    相同的下游接口。

    页面与导出逻辑会读取 ``max_order`` / ``alpha`` / ``mix`` / ``nonneg_self`` /
    ``max_interactions`` / ``basis_type`` / ``ebic_gamma`` /
    ``enforce_effect_constraints`` / ``mse_`` / ``coef_``，并调用
    ``fit`` / ``predict`` / ``effect`` / ``adjacency_matrix`` / ``_design``。
    其中 ``mix``、``nonneg_self``、``max_interactions``、``basis_type``、
    ``ebic_gamma``、``enforce_effect_constraints`` 只用于**导出时的参数留痕**，
    本算法并不使用，因此原样保留传入值以便溯源。
    """

    def __init__(
        self,
        *,
        basis_order: int = 0,
        alpha: float = 0.1,
        windows: int = 1,
        threshold: float = 0.5,
        ridge: float = 1e-6,
        cross_l1_scale: float = 5e-4,
        gap_min: float = 1e-6,
        y_pad: float = 0.1,
        y_soft_scale: float = 100.0,
        effect_cap: float | None = 2.0,
        effect_soft_scale: float = 50.0,
        # 仅用于参数留痕（与 IDOPRegressor 对齐）
        mix: float = 0.5,
        nonneg_self: bool = True,
        max_interactions: int = 0,
        basis_type: str = "integral",
        ebic_gamma: float = 0.0,
        enforce_effect_constraints: bool = True,
    ):
        self.max_order = int(basis_order)
        self.basis_order = int(basis_order)
        self.alpha = float(alpha)
        self.windows = int(windows)
        self.threshold = float(threshold)
        self.ridge = float(ridge)
        self.cross_l1_scale = float(cross_l1_scale)
        self.gap_min = float(gap_min)
        self.y_pad = float(y_pad)
        self.y_soft_scale = float(y_soft_scale)
        self.effect_cap = effect_cap
        self.effect_soft_scale = float(effect_soft_scale)

        self.mix = float(mix)
        self.nonneg_self = bool(nonneg_self)
        self.max_interactions = int(max_interactions)
        self.basis_type = str(basis_type)
        self.ebic_gamma = float(ebic_gamma)
        self.enforce_effect_constraints = bool(enforce_effect_constraints)

        self.coef_: pd.DataFrame | None = None
        self.mse_: float | None = None
        self.edge_supports_: pd.DataFrame | None = None
        self._decomposition: dict | None = None
        self._sample_index: pd.Index | None = None
        self._features: list[str] | None = None

    # -- 拟合 ---------------------------------------------------------------
    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        response_df: pd.DataFrame,
        *,
        power_function_params: pd.DataFrame | None = None,
    ) -> "StaticIDOPModel":
        """``samples`` = 幂律重采样，``response`` = 拟动态数据。"""
        samples = power_function_sample_df
        features = [str(c) for c in samples.columns]
        response = response_df.loc[:, features]

        self.edge_supports_ = select_edges_lasso(
            response if "time" in response.columns else response.assign(
                time=np.arange(len(response), dtype=float)
            ),
            alpha=self.alpha,
            k=self.windows,
            threshold=self.threshold,
        )
        # 上面为了兼容 select_edges_lasso 期望的 "time" 列做了一次附加；
        # 该列会被 lead_columns 过滤掉，不影响结果。

        self._decomposition = _solve_ode_decomposition(
            response,
            samples,
            self.edge_supports_,
            self.basis_order,
            self.ridge,
            self.cross_l1_scale,
            self.gap_min,
            self.y_pad,
            self.y_soft_scale,
            self.effect_cap,
            self.effect_soft_scale,
        )

        decomposition = self._decomposition
        self._sample_index = samples.index
        self._features = decomposition["features"]

        residual = decomposition["response"] - decomposition["predicted_states"]
        self.mse_ = float(np.mean(np.square(residual)))

        # coef_：index = ["intercept"] + 设计列名，columns = target
        rows: dict[str, dict[str, float]] = {}
        rows["intercept"] = dict(
            zip(
                decomposition["features"],
                (float(v) for v in decomposition["intercepts"]),
            )
        )
        for target in decomposition["features"]:
            theta = decomposition["coefficients"][target]
            columns = decomposition["design_columns"][target]
            for name, value in zip(columns, theta):
                rows.setdefault(name, {})[target] = float(value)
        self.coef_ = pd.DataFrame(rows).T.reindex(
            columns=decomposition["features"]
        )
        return self

    # -- 下游接口 -----------------------------------------------------------
    def _design(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        """设计矩阵：``intercept`` 列 + 每个源的积分 Legendre 基列。

        列名与 ``coef_`` 的 index（除 ``intercept``）对应。
        """
        features = [str(c) for c in power_function_sample_df.columns]
        states = power_function_sample_df[features].to_numpy(dtype=float)
        basis = integral_legendre_basis(states, self.basis_order)

        data: dict[str, np.ndarray] = {
            "intercept": np.ones(states.shape[0], dtype=float)
        }
        for source_index, source in enumerate(features):
            for r in range(self.basis_order + 1):
                data[f"{source}::P{r}"] = basis[source_index, :, r]

        return pd.DataFrame(data, index=power_function_sample_df.index)

    def predict(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        if self._decomposition is None:
            raise RuntimeError("call fit before predict")
        return pd.DataFrame(
            self._decomposition["predicted_states"],
            index=power_function_sample_df.index,
            columns=self._decomposition["features"],
        )

    def effect(self, power_function_sample_df: pd.DataFrame) -> list[pd.DataFrame]:
        """每个 target 一个 DataFrame：index = 采样网格，columns = 全部特征。"""
        if self._decomposition is None:
            raise RuntimeError("call fit before effect")
        decomposition = self._decomposition
        features = decomposition["features"]
        interaction = decomposition["interaction_functions"]

        effect_df_list: list[pd.DataFrame] = []
        for target in features:
            columns = {
                source: interaction.get(
                    (target, source), np.zeros(len(decomposition["sample_tau"]))
                )
                for source in features
            }
            effect_df_list.append(
                pd.DataFrame(
                    columns,
                    index=power_function_sample_df.index,
                )
            )
        return effect_df_list

    def adjacency_matrix(
        self,
        power_function_sample_df: pd.DataFrame,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """邻接矩阵，``adj.loc[source, target]`` = ``source -> target`` 的权重。

        ``aggregation`` 与 ASGL 版保持同名接口：``"mean"`` 取作用曲线均值，
        ``"integral"`` 沿采样网格做梯形积分。
        """
        if aggregation not in ("mean", "integral"):
            raise ValueError("aggregation 必须为 'mean' 或 'integral'")
        if self._decomposition is None:
            raise RuntimeError("call fit before adjacency_matrix")

        decomposition = self._decomposition
        features = decomposition["features"]
        interaction = decomposition["interaction_functions"]
        tau = np.asarray(decomposition["sample_tau"], dtype=float)

        matrix = np.zeros((len(features), len(features)), dtype=float)
        for target_index, target in enumerate(features):
            for source in decomposition["support_sets"].get(target, []):
                curve = interaction[(target, source)]
                if aggregation == "integral":
                    matrix[features.index(source), target_index] = float(
                        np.trapezoid(curve, x=tau)
                    )
                else:
                    matrix[features.index(source), target_index] = float(
                        np.mean(curve)
                    )

        return pd.DataFrame(matrix, index=features, columns=features)


def fit_static_idop_network(
    curve_sample_df: pd.DataFrame,
    response_df: pd.DataFrame,
    *,
    basis_order: int = 0,
    alpha: float = 0.1,
    windows: int = 1,
    threshold: float = 0.5,
    adjacency_aggregation: str = "mean",
    **model_kwargs: Any,
) -> dict:
    """按页面 ``_fit_idop_network_from_curve_sample`` 的约定产出 network 字典。

    返回结构与 ASGL 路线完全一致，因此下游（导出、效应分解、绘图）无需分支。
    """
    features = [str(c) for c in curve_sample_df.columns]
    response_df = response_df.loc[:, features]

    model = StaticIDOPModel(
        basis_order=int(basis_order),
        alpha=float(alpha),
        windows=int(windows),
        threshold=float(threshold),
        **model_kwargs,
    )
    model.fit(curve_sample_df, response_df)

    design_X = model._design(curve_sample_df)
    return {
        "model": model,
        "quasi_dynamic_df": response_df,
        "curve_sample_df": curve_sample_df,
        "design_X": design_X,
        "response_Y": response_df.reindex(design_X.index),
        "predicted_df": model.predict(curve_sample_df),
        "effect_df_list": model.effect(curve_sample_df),
        "adj_df": model.adjacency_matrix(
            curve_sample_df, aggregation=str(adjacency_aggregation)
        ),
        "adjacency_aggregation": str(adjacency_aggregation),
    }
