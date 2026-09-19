"""动态数据版建网求解器：自效应 Fourier 基 + 交互 Legendre 基。

移植自新版 ``idopECG.py`` 的 ``solve_core`` / ``_solve_window_two_stage``。与仓库
既有的两条路线**都不同**：

- 不做 cvxpy 约束凸优化，而是**两阶段正交化最小二乘**：
  Stage 1 ``β_self = lstsq(Ψ_self, y)``，残差 ``r = y − Ψ_self β_self``；
  Stage 2 先把 cross 基对 self 的 QR 空间正交化，
  ``β_cross = lstsq(Ψ_cross⊥, r)``。自效应因此天然优先，cross 只解释剩余部分。
- 自效应基是**积分 Fourier 基**（常数 + cos/sin 谐波，频率 ``n / t_end`` Hz），
  cross 基是 ``[x_k(s)·P_r(τ(s))]`` 的**累积梯形积分**（Legendre）。
- 按 ``window`` 个采样点切窗，逐窗独立拟合，再跨窗平均得到效应曲线与边权。

用于动态数据（如 12 导联 ECG）——「曲线拟合」页选 Dynamic 模式做完 SWT 去噪后，
把去噪波形喂给本求解器。选边器复用 :func:`idopnetwork.network.static_idop.
select_edges_lasso`（多窗口 LASSO + 频率阈值）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_legendre

from idopnetwork.network.static_idop import select_edges_lasso

__all__ = [
    "DynamicIDOPRegressor",
    "fit_dynamic_idop_network",
    "solve_dynamic_core",
]

# ECG 采样率：window 默认 2.5 s @ 100 Hz = 250 点
DEFAULT_FS = 100.0
DEFAULT_WINDOW = 250
DEFAULT_N_FOURIER = 30
DEFAULT_R_LEGENDRE = 3


# ── 基函数 ────────────────────────────────────────────────────────────────────

def _fourier_basis(t: np.ndarray, n_fourier: int, t_end: float) -> np.ndarray:
    """Fourier 基（归一化到 ``[-1, 1]``），频率为 ``n / t_end`` Hz。

    第 0 列为常数项 ``0.5``，其后按 cos / sin 交替排列。
    """
    tau = 2.0 * t / t_end - 1.0
    phi = np.zeros((len(t), 2 * n_fourier + 1))
    phi[:, 0] = 0.5
    for n in range(1, n_fourier + 1):
        phi[:, 2 * n - 1] = np.cos(n * np.pi * tau)
        phi[:, 2 * n] = np.sin(n * np.pi * tau)
    return phi


def _legendre_basis(t: np.ndarray, r_max: int, t_end: float) -> np.ndarray:
    """``P_0 .. P_{r_max}`` 作用在归一化时间 ``τ = 2t/t_end − 1`` 上。"""
    tau = 2.0 * t / t_end - 1.0
    return np.column_stack([eval_legendre(r, tau) for r in range(r_max + 1)])


def _integrate_columns(matrix: np.ndarray, t: np.ndarray) -> np.ndarray:
    """逐列累积梯形积分（每列从 0 开始），使拟合无需数值求导。"""
    matrix_int = np.zeros_like(matrix, dtype=float)
    for col in range(matrix.shape[1]):
        matrix_int[:, col] = cumulative_trapezoid(
            matrix[:, col].astype(float), t, initial=0.0
        )
    return matrix_int


def _r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 ``R² = 1 − SS_res / SS_tot``。"""
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _solve_window_two_stage(
    y: np.ndarray,
    psi_self: np.ndarray,
    cross_cols: list[np.ndarray],
) -> dict:
    """单窗口两阶段正交化拟合。

    Stage 1: ``β_self = lstsq(Ψ_self, y)``，``r = y − Ψ_self β_self``；
    Stage 2: ``Q = QR(Ψ_self)``，``Ψ_cross⊥ = Ψ_cross − Q Qᵀ Ψ_cross``，
    ``β_cross = lstsq(Ψ_cross⊥, r)``。
    """
    beta_self, *_ = np.linalg.lstsq(psi_self, y, rcond=None)
    y_self = psi_self @ beta_self
    residual = y - y_self

    if cross_cols:
        psi_cross = np.hstack(cross_cols)
        q_self, _ = np.linalg.qr(psi_self)
        psi_cross_perp = psi_cross - q_self @ (q_self.T @ psi_cross)
        beta_cross, *_ = np.linalg.lstsq(psi_cross_perp, residual, rcond=None)
    else:
        psi_cross_perp = np.zeros((len(y), 0))
        beta_cross = np.zeros(0)

    y_cross_list, offset = [], 0
    for block in cross_cols:
        n_cols = block.shape[1]
        y_cross_list.append(
            psi_cross_perp[:, offset:offset + n_cols]
            @ beta_cross[offset:offset + n_cols]
        )
        offset += n_cols

    y_pred = y_self + (sum(y_cross_list) if y_cross_list else 0.0)
    return {
        "y_self": y_self,
        "y_cross_list": y_cross_list,
        "y_pred": y_pred,
        "beta_self": np.asarray(beta_self, dtype=float),
        "r2": _r2_score(y, y_pred),
    }


def _parse_sources(source_str: Any) -> list[str]:
    """``'{II,aVR}'`` → ``['II', 'aVR']``。"""
    if not isinstance(source_str, str):
        return []
    return [s.strip() for s in source_str.strip("{}").split(",") if s.strip()]


# ── 求解内核 ──────────────────────────────────────────────────────────────────

def solve_dynamic_core(
    data_fit: pd.DataFrame,
    edge_supports: pd.DataFrame,
    *,
    n_fourier: int = DEFAULT_N_FOURIER,
    r_legendre: int = DEFAULT_R_LEGENDRE,
    window: int = DEFAULT_WINDOW,
    fs: float = DEFAULT_FS,
) -> dict:
    """动态数据的 ODE 求解内核（纯计算，不落盘不绘图）。

    Parameters
    ----------
    data_fit:
        去噪后的波形，``(n_timepoints, n_channels)``。
    edge_supports:
        选边器输出，列 ``["target", "source"]``，``source`` 形如 ``"{II,aVR}"``。
    n_fourier / r_legendre:
        自效应 Fourier 谐波阶数 / 交叉效应 Legendre 阶数。
    window / fs:
        窗口采样点数 / 采样率（决定窗口时长 ``window / fs`` 秒）。

    Returns
    -------
    dict
        ``leads / support_sets / t_win / t_end / n_windows / n_edges /
        window_self / window_cross / window_r2 / window_beta_self /
        avg_self / avg_cross / avg_obs / edge_rows / r2_mean``。
    """
    leads = [str(c) for c in data_fit.columns]
    x_all = data_fit[leads].to_numpy(dtype=float)
    n_samples = x_all.shape[0]
    window = int(window)
    if window < 2:
        raise ValueError("window 至少为 2。")
    if n_samples < window:
        raise ValueError(
            f"时间点数 {n_samples} 少于一个窗口 {window}，无法切窗求解。"
        )

    t_win = np.arange(window) / float(fs)
    t_end = window / float(fs)

    support_sets = {
        str(row["target"]): _parse_sources(row["source"])
        for _, row in edge_supports.iterrows()
    }
    n_edges = sum(len(v) for v in support_sets.values())

    n_windows = n_samples // window
    windows = [x_all[i * window:(i + 1) * window] for i in range(n_windows)]

    # 基函数在所有窗口上共用（窗口内时间网格固定）
    psi_self_int = _integrate_columns(_fourier_basis(t_win, n_fourier, t_end), t_win)
    legendre_t = _legendre_basis(t_win, r_legendre, t_end)

    window_self: dict[str, list[np.ndarray]] = {c: [] for c in leads}
    window_cross: dict[str, list[dict[str, np.ndarray]]] = {c: [] for c in leads}
    window_r2: dict[str, list[float]] = {c: [] for c in leads}
    window_beta_self: dict[str, list[np.ndarray]] = {c: [] for c in leads}
    window_obs: dict[str, list[np.ndarray]] = {c: [] for c in leads}

    for x_w in windows:
        x0 = x_w[0]
        for j, lead_j in enumerate(leads):
            y = x_w[:, j] - x0[j]
            cross_cols, cross_names = [], []
            for k_name in support_sets.get(lead_j, []):
                if k_name in leads:
                    k_idx = leads.index(k_name)
                    cols = x_w[:, k_idx:k_idx + 1] * legendre_t
                    cross_cols.append(_integrate_columns(cols, t_win))
                    cross_names.append(k_name)
            result = _solve_window_two_stage(y, psi_self_int, cross_cols)
            window_self[lead_j].append(result["y_self"])
            window_cross[lead_j].append(
                {k: result["y_cross_list"][i] for i, k in enumerate(cross_names)}
            )
            window_r2[lead_j].append(result["r2"])
            window_beta_self[lead_j].append(result["beta_self"])
            window_obs[lead_j].append(y)

    # ---- 聚合：跨窗平均效应曲线 + 网络边权 ----
    avg_self: dict[str, np.ndarray] = {}
    avg_cross: dict[str, dict[str, np.ndarray]] = {c: {} for c in leads}
    avg_obs: dict[str, np.ndarray] = {}
    r2_mean: dict[str, float] = {}
    edge_rows: list[dict[str, Any]] = []

    for lead_j in leads:
        avg_self[lead_j] = np.stack(window_self[lead_j]).mean(axis=0)
        avg_obs[lead_j] = np.stack(window_obs[lead_j]).mean(axis=0)
        r2_mean[lead_j] = float(np.mean(np.array(window_r2[lead_j])))

        for k_name in support_sets.get(lead_j, []):
            if k_name not in window_cross[lead_j][0]:
                continue
            curves = np.stack([d[k_name] for d in window_cross[lead_j]])
            avg_cross[lead_j][k_name] = curves.mean(axis=0)
            weight = float(curves.mean())
            edge_rows.append({
                "source": k_name,
                "target": lead_j,
                "weight": round(weight, 8),
                "sign": "positive" if weight >= 0.0 else "negative",
            })

    return {
        "leads": leads,
        "support_sets": support_sets,
        "t_win": t_win,
        "t_end": t_end,
        "n_windows": n_windows,
        "n_edges": n_edges,
        "window_self": window_self,
        "window_cross": window_cross,
        "window_r2": window_r2,
        "window_beta_self": window_beta_self,
        "avg_self": avg_self,
        "avg_cross": avg_cross,
        "avg_obs": avg_obs,
        "edge_rows": edge_rows,
        "r2_mean": r2_mean,
    }


# ── 适配器：与 IDOPRegressor 相同的下游接口 ───────────────────────────────────

class DynamicIDOPRegressor:
    """动态数据求解器，暴露与 :class:`IDOPRegressor` 相同的下游接口。

    设计输入（``power_function_sample_df``）在动态流程里就是 **SWT 去噪后的波形**，
    响应同源 —— 与 ``idopECG.solve_core`` 一致。效应曲线落在**窗口网格**
    （``0 .. window/fs`` 秒）上，而非完整时间轴。

    导出逻辑会读的 ``max_order`` / ``alpha`` / ``mix`` / ``nonneg_self`` /
    ``max_interactions`` / ``basis_type`` / ``ebic_gamma`` /
    ``enforce_effect_constraints`` 中，``max_order`` 映射为 Legendre 阶数、
    ``alpha`` 映射为 LASSO 强度；其余仅为参数留痕。
    """

    def __init__(
        self,
        *,
        n_fourier: int = DEFAULT_N_FOURIER,
        r_legendre: int = DEFAULT_R_LEGENDRE,
        window: int = DEFAULT_WINDOW,
        fs: float = DEFAULT_FS,
        lasso_alpha: float = 0.01,
        lasso_windows: int = 10,
        lasso_threshold: float = 0.4,
        mix: float = 0.5,
        nonneg_self: bool = True,
        max_interactions: int = 0,
        basis_type: str = "fourier_legendre",
        ebic_gamma: float = 0.0,
        enforce_effect_constraints: bool = False,
    ):
        self.n_fourier = int(n_fourier)
        self.r_legendre = int(r_legendre)
        self.window = int(window)
        self.fs = float(fs)
        self.lasso_alpha = float(lasso_alpha)
        self.lasso_windows = int(lasso_windows)
        self.lasso_threshold = float(lasso_threshold)

        # 供导出 / 诊断读取的兼容属性
        self.max_order = int(r_legendre)
        self.alpha = float(lasso_alpha)
        self.mix = float(mix)
        self.nonneg_self = bool(nonneg_self)
        self.max_interactions = int(max_interactions)
        self.basis_type = str(basis_type)
        self.ebic_gamma = float(ebic_gamma)
        self.enforce_effect_constraints = bool(enforce_effect_constraints)

        self.coef_: pd.DataFrame | None = None
        self.mse_: float | None = None
        self.edge_supports_: pd.DataFrame | None = None
        self._core: dict | None = None
        self._intercepts: np.ndarray | None = None

    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame | None = None,
        *,
        power_function_params: pd.DataFrame | None = None,
    ) -> "DynamicIDOPRegressor":
        signal = power_function_sample_df.apply(pd.to_numeric, errors="coerce")
        leads = [str(c) for c in signal.columns]

        self.edge_supports_ = select_edges_lasso(
            signal,
            alpha=self.lasso_alpha,
            k=self.lasso_windows,
            threshold=self.lasso_threshold,
        )
        core = solve_dynamic_core(
            signal,
            self.edge_supports_,
            n_fourier=self.n_fourier,
            r_legendre=self.r_legendre,
            window=self.window,
            fs=self.fs,
        )
        self._core = core
        self._intercepts = np.array(
            [float(np.mean(core["avg_obs"][lead])) for lead in leads]
        )

        # mse：跨窗平均观测 vs 平均预测（预测 = 自效应 + 交叉效应之和）
        errors = []
        for lead in leads:
            prediction = core["avg_self"][lead] + (
                sum(core["avg_cross"][lead].values())
                if core["avg_cross"][lead] else 0.0
            )
            errors.append(float(np.mean((core["avg_obs"][lead] - prediction) ** 2)))
        self.mse_ = float(np.mean(errors)) if errors else float("nan")

        # coef_：intercept 行（跨窗平均的 x(0)）+ 自效应 Fourier 系数的窗均值。
        # 交叉效应是逐目标、逐 source 的曲线，体现在 effect() 而非系数矩阵里。
        rows: dict[str, dict[str, float]] = {
            "intercept": {
                lead: float(self._intercepts[j]) for j, lead in enumerate(leads)
            }
        }
        beta_stack = {
            lead: np.stack(core["window_beta_self"][lead]).mean(axis=0)
            for lead in leads
        }
        n_basis = len(next(iter(beta_stack.values())))
        for i in range(n_basis):
            name = f"self::F{i}"
            rows[name] = {lead: float(beta_stack[lead][i]) for lead in leads}
        self.coef_ = pd.DataFrame(rows).T.reindex(columns=leads)
        return self

    # -- 下游接口 -----------------------------------------------------------
    def _design(self, power_function_sample_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """窗口网格上的设计矩阵：``intercept`` + 积分 Fourier 基 + Legendre 基。

        交叉基在这里只给出行参照（各 target 的支撑集不同，逐目标设计在
        ``solve_dynamic_core`` 内部构造）。
        """
        if self._core is None:
            raise RuntimeError("call fit before _design")
        core = self._core
        t_win, t_end = core["t_win"], core["t_end"]

        data: dict[str, np.ndarray] = {"intercept": np.ones(len(t_win))}
        self_basis = _integrate_columns(
            _fourier_basis(t_win, self.n_fourier, t_end), t_win
        )
        for i in range(self_basis.shape[1]):
            data[f"self::F{i}"] = self_basis[:, i]
        legendre = _legendre_basis(t_win, self.r_legendre, t_end)
        for i in range(legendre.shape[1]):
            data[f"cross::P{i}"] = legendre[:, i]
        return pd.DataFrame(data, index=pd.Index(t_win, name="time"))

    def predict(self, power_function_sample_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """绝对预测（``x(0) + 自效应 + 交叉效应``），索引为窗口时间网格。"""
        if self._core is None:
            raise RuntimeError("call fit before predict")
        core = self._core
        leads = core["leads"]
        t_win = core["t_win"]
        columns = {}
        for j, lead in enumerate(leads):
            cross = (
                sum(core["avg_cross"][lead].values())
                if core["avg_cross"][lead] else 0.0
            )
            columns[lead] = self._intercepts[j] + core["avg_self"][lead] + cross
        return pd.DataFrame(columns, index=pd.Index(t_win, name="time"))

    def effect(self, power_function_sample_df: pd.DataFrame | None = None) -> list[pd.DataFrame]:
        """每个 target 一个 DataFrame：index = 窗口网格，columns = 全部通道。

        自身列为自效应（Fourier），其余为交叉效应（Legendre），未入选的为 0 ——
        与仓库 ``IDOPRegressor.effect`` 的排布一致，且同为**居中**贡献
        （截距由 ``coef_.loc["intercept"]`` 承载）。
        """
        if self._core is None:
            raise RuntimeError("call fit before effect")
        core = self._core
        leads = core["leads"]
        index = pd.Index(core["t_win"], name="time")

        effect_df_list: list[pd.DataFrame] = []
        for lead_j in leads:
            columns = {}
            for name in leads:
                if name == lead_j:
                    columns[name] = core["avg_self"][lead_j]
                elif name in core["avg_cross"][lead_j]:
                    columns[name] = core["avg_cross"][lead_j][name]
                else:
                    columns[name] = np.zeros(len(core["t_win"]))
            effect_df_list.append(pd.DataFrame(columns, index=index))
        return effect_df_list

    def adjacency_matrix(
        self,
        power_function_sample_df: pd.DataFrame | None = None,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """邻接矩阵，``adj.loc[source, target]`` = ``source -> target`` 的权重。"""
        if aggregation not in ("mean", "integral"):
            raise ValueError("aggregation 必须为 'mean' 或 'integral'")
        if self._core is None:
            raise RuntimeError("call fit before adjacency_matrix")
        core = self._core
        leads = core["leads"]
        t_win = core["t_win"]

        matrix = np.zeros((len(leads), len(leads)), dtype=float)
        for target in leads:
            for source, curve in core["avg_cross"][target].items():
                if aggregation == "integral":
                    value = float(np.trapezoid(curve, x=t_win))
                else:
                    value = float(np.mean(curve))
                matrix[leads.index(source), leads.index(target)] = value
        return pd.DataFrame(matrix, index=leads, columns=leads)


def fit_dynamic_idop_network(
    curve_sample_df: pd.DataFrame,
    response_df: pd.DataFrame | None = None,
    *,
    max_order: int = DEFAULT_R_LEGENDRE,
    nonneg_self: bool = True,
    max_interactions: int = 0,
    adjacency_aggregation: str = "mean",
    n_fourier: int = DEFAULT_N_FOURIER,
    window: int = DEFAULT_WINDOW,
    fs: float = DEFAULT_FS,
    lasso_alpha: float = 0.01,
    lasso_windows: int = 10,
    lasso_threshold: float = 0.4,
    power_function_params: pd.DataFrame | None = None,
) -> dict:
    """按页面 ``_fit_idop_network_from_curve_sample`` 的约定产出 network 字典。

    与静态路线一样返回 9 个键，因此下游导出、效应分解、绘图都无需分支。
    """
    from idopnetwork.network.construction import align_response_to_design

    model = DynamicIDOPRegressor(
        n_fourier=int(n_fourier),
        r_legendre=int(max_order),
        window=int(window),
        fs=float(fs),
        lasso_alpha=float(lasso_alpha),
        lasso_windows=int(lasso_windows),
        lasso_threshold=float(lasso_threshold),
        mix=0.5,
        nonneg_self=bool(nonneg_self),
        max_interactions=int(max_interactions),
    )
    model.fit(curve_sample_df)

    design_X = model._design()
    response = (
        response_df if response_df is not None else curve_sample_df
    )
    return {
        "model": model,
        "quasi_dynamic_df": response,
        "curve_sample_df": curve_sample_df,
        "design_X": design_X,
        "response_Y": align_response_to_design(response, design_X.index),
        "predicted_df": model.predict(),
        "effect_df_list": model.effect(),
        "adj_df": model.adjacency_matrix(aggregation=str(adjacency_aggregation)),
        "adjacency_aggregation": str(adjacency_aggregation),
    }
