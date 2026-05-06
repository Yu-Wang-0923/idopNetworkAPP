"""Network Construction 后端计算模块。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_laguerre, eval_legendre

SUPPORTED_BASIS_KINDS = ("legendre", "laguerre", "polynomial")
SUPPORTED_SOLVERS = ("ols", "lasso", "asgl")


def polynomial_basis_expansion(
    data: pd.DataFrame,
    max_order: int,
    kind: str = "legendre",
) -> pd.DataFrame:
    """对每个特征做基展开（1..max_order 列），可选 legendre / laguerre / polynomial。

    - legendre：scipy.eval_legendre，输入域期望为 [-1, 1]。
    - laguerre：scipy.eval_laguerre，输入域期望为 -1 to 1。
    - polynomial：普通幂基，列为 x, x^2, ..., x^max_order。
    """
    if kind not in SUPPORTED_BASIS_KINDS:
        raise ValueError(
            f"kind 必须为 {SUPPORTED_BASIS_KINDS} 之一，实际收到 {kind!r}"
        )
    values = data.values.astype(float)
    n_samples, n_features = values.shape

    if kind == "legendre":
        per_order = [eval_legendre(order, values) for order in range(1, max_order + 1)]
        basis_arr = np.stack(per_order, axis=0).transpose(1, 2, 0)
    elif kind == "laguerre":
        per_order = [eval_laguerre(order, values) for order in range(1, max_order + 1)]
        basis_arr = np.stack(per_order, axis=0).transpose(1, 2, 0)
    else:
        per_order = [
            np.power(values, power) for power in range(1, max_order + 1)
        ]
        basis_arr = np.stack(per_order, axis=0).transpose(1, 2, 0)

    columns = [
        f"{data.columns[i]}_o({order + 1})"
        for i in range(n_features)
        for order in range(max_order)
    ]
    flat = basis_arr.reshape(n_samples, n_features * max_order)
    return pd.DataFrame(flat, index=data.index, columns=columns)


def polynomial_basis_expansion_integral(basis_expansion: pd.DataFrame) -> pd.DataFrame:
    """对 basis_expansion 按 index（时间）进行数值积分。"""
    t = basis_expansion.index.values.astype(float)
    integral_values = cumulative_trapezoid(
        basis_expansion.values, t, initial=0, axis=0
    )
    new_columns = [col + "_inte" for col in basis_expansion.columns]
    return pd.DataFrame(
        integral_values, index=basis_expansion.index, columns=new_columns
    )


def _project_nonneg_self(
    w: np.ndarray, Xv: np.ndarray, intercept_idx: int, self_group: np.ndarray
) -> None:
    """在原地将截距上移使自效应 (intercept + Phi_self @ theta_self) >= 0 处处成立。"""
    e = Xv[:, self_group] @ w[self_group] + w[intercept_idx]
    min_val = float(e.min())
    if min_val < 0:
        w[intercept_idx] -= min_val


def _project_nonneg_self_keep_intercept(
    w: np.ndarray,
    Xv: np.ndarray,
    intercept_idx: int,
    self_group: np.ndarray,
    pinv: np.ndarray | None = None,
) -> None:
    """调整自效应系数（不动截距）使 β + Φ_self @ θ_self >= 0 处处成立。"""
    e = Xv[:, self_group] @ w[self_group] + w[intercept_idx]
    if float(e.min()) >= 0:
        return
    e_clamped = np.maximum(e, 0.0)
    e_self_target = e_clamped - w[intercept_idx]
    if pinv is not None:
        w[self_group] = pinv @ e_self_target
    else:
        w[self_group] = np.linalg.lstsq(Xv[:, self_group], e_self_target, rcond=None)[0]


def _precompute_pinvs(Xv: np.ndarray, groups: list[np.ndarray]) -> dict[int, np.ndarray]:
    """预计算每个非截距组的伪逆，供约束投影复用。"""
    pinvs: dict[int, np.ndarray] = {}
    for gi, g in enumerate(groups):
        if gi == 0:
            continue
        pinvs[gi] = np.linalg.pinv(Xv[:, g])
    return pinvs


def _asgl_col(
    Xv: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    mix: float,
    coef_weights: np.ndarray,
    group_weights: np.ndarray,
    protected_ids: set[int] | None = None,
    target_idx: int = -1,
    nonneg_self: bool = False,
    max_iter: int = 2000,
    tol: float = 1e-5,
    w_init: np.ndarray | None = None,
    pinvs: dict[int, np.ndarray] | None = None,
    XtX: np.ndarray | None = None,
    Xty: np.ndarray | None = None,
    L: float | None = None,
    fixed_intercept: float | None = None,
) -> np.ndarray:
    """Adaptive Sparse Group Lasso (ASGL) 近端梯度下降。"""
    if protected_ids is None:
        protected_ids = set()
    n, p = Xv.shape
    if XtX is None:
        XtX = Xv.T @ Xv
    if Xty is None:
        Xty = Xv.T @ y
    if L is None:
        L = float(np.linalg.norm(XtX, ord=2))
    if L == 0:
        return np.zeros(p)
    step = 1.0 / L

    self_pinv = pinvs.get(target_idx) if pinvs else None
    w = w_init.copy() if w_init is not None else np.zeros(p)
    if fixed_intercept is not None:
        w[0] = fixed_intercept

    for it in range(max_iter):
        w_prev = w.copy()
        grad = XtX @ w - Xty
        u = w - step * grad
        if fixed_intercept is not None:
            u[0] = fixed_intercept

        for gi, g in enumerate(groups):
            if len(g) == 0:
                continue
            if gi == 0 or gi in protected_ids:
                w[g] = u[g]
                continue
            ug = u[g].copy()
            if mix > 0:
                thresh_l1 = lam * mix * coef_weights[g] * step
                ug = np.sign(ug) * np.maximum(np.abs(ug) - thresh_l1, 0.0)
            norm_ug = np.linalg.norm(ug)
            if norm_ug == 0.0:
                w[g] = 0.0
            elif mix < 1:
                pg = float(len(g))
                thresh_grp = (
                    lam * (1.0 - mix) * float(group_weights[gi]) * np.sqrt(pg) * step
                )
                if norm_ug > thresh_grp:
                    w[g] = (1.0 - thresh_grp / norm_ug) * ug
                else:
                    w[g] = 0.0
            else:
                w[g] = ug

        if (it % 10 == 0 or it < 10) and nonneg_self and target_idx > 0:
            if fixed_intercept is not None:
                _project_nonneg_self_keep_intercept(
                    w, Xv, 0, groups[target_idx], pinv=self_pinv
                )
            else:
                _project_nonneg_self(w, Xv, 0, groups[target_idx])

        if np.linalg.norm(w - w_prev) < tol:
            break

    if nonneg_self and target_idx > 0:
        if fixed_intercept is not None:
            _project_nonneg_self_keep_intercept(
                w, Xv, 0, groups[target_idx], pinv=self_pinv
            )
        else:
            _project_nonneg_self(w, Xv, 0, groups[target_idx])
    return w


def _asgl_multi_task(
    Xv: np.ndarray,
    Y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    mix: float,
    coef_weights: np.ndarray,
    group_weights: np.ndarray,
    nonneg_self: bool = False,
    max_iter: int = 2000,
    tol: float = 1e-5,
    W_init: np.ndarray | None = None,
    pinvs: dict[int, np.ndarray] | None = None,
    XtX: np.ndarray | None = None,
    XtY: np.ndarray | None = None,
    L: float | None = None,
    fixed_intercepts: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-task ASGL：跨目标 Frobenius 范数组惩罚 + 逐元素 L1。"""
    n, p = Xv.shape
    n_targets = Y.shape[1]
    if XtX is None:
        XtX = Xv.T @ Xv
    if XtY is None:
        XtY = Xv.T @ Y
    if L is None:
        L = float(np.linalg.norm(XtX, ord=2))
    if L == 0:
        return np.zeros((p, n_targets))
    step = 1.0 / L

    W = W_init.copy() if W_init is not None else np.zeros((p, n_targets))
    if fixed_intercepts is not None:
        W[0, :] = fixed_intercepts

    all_targets = set(range(n_targets))
    for it in range(max_iter):
        W_prev = W.copy()
        grad = XtX @ W - XtY
        U = W - step * grad
        if fixed_intercepts is not None:
            U[0, :] = fixed_intercepts

        for gi, g in enumerate(groups):
            if len(g) == 0:
                continue
            if gi == 0:
                W[g, :] = U[g, :]
                continue

            self_col = gi - 1
            protected = {self_col} if (nonneg_self and 0 <= self_col < n_targets) else set()
            active = sorted(all_targets - protected)
            if not active:
                W[g, :] = U[g, :]
                continue

            active_arr = np.array(active)
            Ug_active = U[np.ix_(g, active_arr)].copy()
            if mix > 0:
                thresh_l1 = lam * mix * coef_weights[np.ix_(g, active_arr)] * step
                Ug_active = np.sign(Ug_active) * np.maximum(np.abs(Ug_active) - thresh_l1, 0.0)

            frob = np.linalg.norm(Ug_active)
            if frob == 0.0:
                W[np.ix_(g, active_arr)] = 0.0
            elif mix < 1:
                pg = float(len(g))
                thresh_grp = (
                    lam * (1.0 - mix) * float(group_weights[gi]) * np.sqrt(pg) * step
                )
                if frob > thresh_grp:
                    W[np.ix_(g, active_arr)] = (1.0 - thresh_grp / frob) * Ug_active
                else:
                    W[np.ix_(g, active_arr)] = 0.0
            else:
                W[np.ix_(g, active_arr)] = Ug_active

            for j in protected:
                W[g, j] = U[g, j]

        if (it % 10 == 0 or it < 10) and nonneg_self:
            for j in range(n_targets):
                sg = j + 1
                if sg >= len(groups):
                    continue
                sp = pinvs.get(sg) if pinvs else None
                if fixed_intercepts is not None:
                    _project_nonneg_self_keep_intercept(W[:, j], Xv, 0, groups[sg], pinv=sp)
                else:
                    _project_nonneg_self(W[:, j], Xv, 0, groups[sg])

        if np.linalg.norm(W - W_prev) < tol:
            break

    if nonneg_self:
        for j in range(n_targets):
            sg = j + 1
            if sg >= len(groups):
                continue
            sp = pinvs.get(sg) if pinvs else None
            if fixed_intercepts is not None:
                _project_nonneg_self_keep_intercept(W[:, j], Xv, 0, groups[sg], pinv=sp)
            else:
                _project_nonneg_self(W[:, j], Xv, 0, groups[sg])
    return W


def _build_groups(n_features: int, max_order: int) -> list[np.ndarray]:
    """构建分组列表：[截距组] + [每个源变量的 max_order 个基函数列]。"""
    groups: list[np.ndarray] = [np.array([0])]
    for k in range(n_features):
        start = 1 + k * max_order
        groups.append(np.arange(start, start + max_order))
    return groups


def _lasso_cd_col(
    Xv: np.ndarray,
    y: np.ndarray,
    alpha: float,
    protected: set[int],
    coef_alphas: np.ndarray | None = None,
    max_iter: int = 10_000,
    tol: float = 1e-5,
) -> np.ndarray:
    """坐标下降 Lasso，protected 中的列索引不施加软阈值。"""
    XtX = Xv.T @ Xv
    Xty = Xv.T @ y
    d = np.diag(XtX)
    w = np.zeros(Xv.shape[1])
    for _ in range(max_iter):
        w_prev = w.copy()
        for k in range(len(w)):
            if d[k] == 0:
                continue
            r_k = float(Xty[k] - XtX[k] @ w + d[k] * w[k])
            if k in protected:
                w[k] = r_k / d[k]
            else:
                eff_alpha = float(coef_alphas[k]) if coef_alphas is not None else alpha
                w[k] = np.sign(r_k) * max(abs(r_k) - eff_alpha, 0.0) / d[k]
        if np.linalg.norm(w - w_prev) < tol:
            break
    return w


class IDOPRegressor:
    """多输出线性回归：支持 OLS、Lasso、ASGL。"""

    def __init__(
        self,
        max_order: int,
        solver: str = "ols",
        alpha: float = 1.0,
        mix: float = 0.5,
        fix_mix: bool = False,
        nonneg_self: bool = True,
        basis_decay: float = 0.0,
        max_interactions: int = 0,
        multi_task: bool = False,
        basis_type: str = "integral",
        basis_kind: str = "legendre",
        ebic_gamma: float = 0.0,
    ):
        if basis_kind not in SUPPORTED_BASIS_KINDS:
            raise ValueError(
                f"basis_kind 必须为 {SUPPORTED_BASIS_KINDS} 之一，实际收到 {basis_kind!r}"
            )
        if solver not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"solver 必须为 {SUPPORTED_SOLVERS} 之一，实际收到 {solver!r}"
            )
        self.max_order = max_order
        self.solver = solver
        self.alpha = alpha
        self.mix = mix
        self.fix_mix = fix_mix
        self.protect_self = nonneg_self
        self.nonneg_self = nonneg_self
        self.basis_decay = basis_decay
        self.max_interactions = max_interactions
        self.multi_task = multi_task
        self.basis_type = basis_type
        self.basis_kind = basis_kind
        self.ebic_gamma = ebic_gamma
        self.coef_: pd.DataFrame | None = None
        self.mse_: float | None = None
        self.bic_order_path_: pd.DataFrame | None = None
        self.bic_alpha_path_: pd.DataFrame | None = None

    def _design(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        basis_raw = polynomial_basis_expansion(
            power_function_sample_df,
            self.max_order,
            kind=self.basis_kind,
        )
        if self.basis_type == "derivative":
            basis = basis_raw
        else:
            basis = polynomial_basis_expansion_integral(basis_raw)
        if self.basis_decay > 0.0:
            n_feat = power_function_sample_df.shape[1]
            for r in range(self.max_order):
                scale = float(np.exp(-self.basis_decay * r))
                cols = [k * self.max_order + r for k in range(n_feat)]
                basis.iloc[:, cols] *= max(scale, 1e-10)
        intercept = pd.DataFrame(1.0, index=basis.index, columns=["intercept"])
        return pd.concat([intercept, basis], axis=1)

    def _fit_asgl_bic(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "IDOPRegressor":
        n_vars = power_function_sample_df.shape[1]
        user_max_order = self.max_order

        upper = min(user_max_order, 20)
        best_order, best_order_bic = 1, np.inf
        order_rows: list[dict[str, float | int | bool]] = []
        for order_c in range(1, upper + 1):
            self.max_order = order_c
            X_c = self._design(power_function_sample_df)
            Y_c = quasi_dynamic_df.reindex(X_c.index).values.astype(float)
            Xv_c = X_c.values.astype(float)
            n_obs, p_c = Xv_c.shape
            n_t = Y_c.shape[1]
            y0_c = (
                np.asarray(intercept_values, dtype=float)
                if intercept_values is not None
                else Y_c[0, :].copy()
            )

            Xb = Xv_c[:, 1:]
            W_c = np.zeros((p_c, n_t))
            for j in range(n_t):
                W_c[0, j] = y0_c[j]
                ya = Y_c[:, j] - y0_c[j]
                if p_c - 1 >= n_obs:
                    W_c[1:, j] = np.linalg.solve(
                        Xb.T @ Xb + 0.01 * np.eye(p_c - 1), Xb.T @ ya
                    )
                else:
                    W_c[1:, j] = np.linalg.lstsq(Xb, ya, rcond=None)[0]

            rss = float(np.sum((Y_c - Xv_c @ W_c) ** 2))
            if rss <= 0:
                order_rows.append(
                    {
                        "max_order": order_c,
                        "bic": np.nan,
                        "rss": rss,
                        "df": (p_c - 1) * n_t,
                        "n_obs": n_obs,
                        "n_targets": n_t,
                    }
                )
                continue
            df_total = (p_c - 1) * n_t
            bic = n_obs * n_t * np.log(rss / (n_obs * n_t)) + df_total * np.log(n_obs)
            if self.ebic_gamma > 0:
                from math import lgamma as _lg

                _p_total = order_c * n_vars * n_t
                _d = min(df_total, _p_total)
                if _d > 0 and _p_total > 0:
                    bic += self.ebic_gamma * 2 * (
                        _lg(_p_total + 1) - _lg(_d + 1) - _lg(_p_total - _d + 1)
                    )
            order_rows.append(
                {
                    "max_order": order_c,
                    "bic": float(bic),
                    "rss": rss,
                    "df": df_total,
                    "n_obs": n_obs,
                    "n_targets": n_t,
                }
            )
            if bic < best_order_bic:
                best_order_bic = bic
                best_order = order_c

        self.max_order = best_order
        X = self._design(power_function_sample_df)
        Y = quasi_dynamic_df.reindex(X.index).values.astype(float)
        Xv = X.values.astype(float)
        n, p = Xv.shape
        n_targets = Y.shape[1]
        groups = _build_groups((p - 1) // self.max_order, self.max_order)
        XtX = Xv.T @ Xv
        y0 = (
            np.asarray(intercept_values, dtype=float)
            if intercept_values is not None
            else Y[0, :].copy()
        )

        W_ols, *_ = np.linalg.lstsq(Xv, Y, rcond=None)
        eps = 1e-6
        use_mt = self.multi_task

        if use_mt:
            caw_mat = 1.0 / (np.abs(W_ols) + eps)
            gaw_vec = np.empty(len(groups))
            for gi, g in enumerate(groups):
                self_col = gi - 1
                active_cols = [
                    j for j in range(n_targets) if not (self.nonneg_self and j == self_col)
                ]
                if active_cols:
                    gaw_vec[gi] = 1.0 / (
                        float(np.linalg.norm(W_ols[np.ix_(g, active_cols)])) + eps
                    )
                else:
                    gaw_vec[gi] = 1.0
        else:
            caws = [1.0 / (np.abs(W_ols[:, j]) + eps) for j in range(n_targets)]
            gaws: list[np.ndarray] = []
            for j in range(n_targets):
                g_aw = np.empty(len(groups))
                for gi, g in enumerate(groups):
                    g_aw[gi] = 1.0 / (float(np.linalg.norm(W_ols[g, j])) + eps)
                gaws.append(g_aw)

        prots = [{j + 1} if self.protect_self else set() for j in range(n_targets)]
        pinvs = _precompute_pinvs(Xv, groups) if self.nonneg_self else None
        L_val = float(np.linalg.norm(XtX, ord=2))
        XtY = Xv.T @ Y
        Xtys = [XtY[:, j] for j in range(n_targets)]

        mix_cands = [self.mix] if self.fix_mix else [0.0, 0.25, 0.5, 0.75, 1.0]
        best_bic = np.inf
        best_lam, best_mix = 1.0, 0.5
        best_W: np.ndarray | None = None
        alpha_rows: list[dict[str, float | int | bool]] = []

        for mc in mix_cands:
            lam_max = 0.0
            for j in range(n_targets):
                for gi, g in enumerate(groups):
                    if gi == 0 or gi in prots[j]:
                        continue
                    gw = gaw_vec[gi] if use_mt else gaws[j][gi]
                    if mc < 1:
                        s = float(np.linalg.norm(Xtys[j][g])) / max(
                            n * float(gw) * np.sqrt(len(g)) * (1.0 - mc), 1e-12
                        )
                        lam_max = max(lam_max, s)
                    if mc > 0:
                        cw = caw_mat[:, j] if use_mt else caws[j]
                        for i in g:
                            s = abs(float(Xtys[j][i])) / max(n * float(cw[i]) * mc, 1e-12)
                            lam_max = max(lam_max, s)
            if lam_max <= 0:
                continue
            lam_grid = np.geomspace(lam_max * 1e-4, lam_max, 20)

            prev: np.ndarray | None = None
            rising = 0
            for lc in lam_grid:
                if use_mt:
                    Wc = _asgl_multi_task(
                        Xv,
                        Y,
                        groups,
                        lam=lc,
                        mix=mc,
                        coef_weights=caw_mat,
                        group_weights=gaw_vec,
                        nonneg_self=self.nonneg_self,
                        pinvs=pinvs,
                        XtX=XtX,
                        XtY=XtY,
                        L=L_val,
                        fixed_intercepts=y0,
                        W_init=prev,
                    )
                else:
                    Wc = np.column_stack(
                        [
                            _asgl_col(
                                Xv,
                                Y[:, j],
                                groups,
                                lam=lc,
                                mix=mc,
                                coef_weights=caws[j],
                                group_weights=gaws[j],
                                protected_ids=prots[j],
                                target_idx=j + 1,
                                nonneg_self=self.nonneg_self,
                                pinvs=pinvs,
                                XtX=XtX,
                                Xty=Xtys[j],
                                L=L_val,
                                fixed_intercept=float(y0[j]),
                                w_init=prev[:, j] if prev is not None else None,
                            )
                            for j in range(n_targets)
                        ]
                    )
                prev = Wc

                tb = 0.0
                valid = True
                rss_floor = 1e-12 * n
                df_model = 0
                _total_active_groups = 0
                _total_pen_groups = 0
                for j in range(n_targets):
                    r = Y[:, j] - Xv @ Wc[:, j]
                    rss_j = max(float(r @ r), rss_floor)
                    df_j = int(np.sum(np.abs(Wc[:, j]) > 1e-10))
                    df_model += df_j
                    if n <= df_j:
                        valid = False
                        break
                    tb += n * np.log(rss_j / n) + df_j * np.log(n)
                    if self.ebic_gamma > 0:
                        for gi, g in enumerate(groups):
                            if gi == 0 or gi in prots[j]:
                                continue
                            _total_pen_groups += 1
                            if np.any(np.abs(Wc[g, j]) > 1e-10):
                                _total_active_groups += 1
                if not valid:
                    continue
                if self.ebic_gamma > 0 and _total_pen_groups > 0:
                    from math import lgamma as _lg

                    _d = min(_total_active_groups, _total_pen_groups)
                    if _d > 0:
                        tb += self.ebic_gamma * 2 * (
                            _lg(_total_pen_groups + 1)
                            - _lg(_d + 1)
                            - _lg(_total_pen_groups - _d + 1)
                        )

                alpha_rows.append(
                    {
                        "alpha": float(lc),
                        "mix": float(mc),
                        "bic": float(tb) if valid else np.nan,
                        "df": df_model,
                        "active_groups": _total_active_groups,
                    }
                )
                if tb < best_bic:
                    best_bic = tb
                    best_lam = lc
                    best_mix = mc
                    best_W = Wc.copy()
                    rising = 0
                else:
                    rising += 1
                    if rising >= 5:
                        break

        self.alpha = best_lam
        self.mix = best_mix
        self.bic_order_path_ = pd.DataFrame(order_rows)
        if self.bic_order_path_ is not None and not self.bic_order_path_.empty:
            self.bic_order_path_["selected"] = (
                self.bic_order_path_["max_order"] == self.max_order
            )
        self.bic_alpha_path_ = pd.DataFrame(alpha_rows)
        if self.bic_alpha_path_ is not None and not self.bic_alpha_path_.empty:
            self.bic_alpha_path_["selected"] = (
                self.bic_alpha_path_["alpha"] == self.alpha
            ) & (self.bic_alpha_path_["mix"] == self.mix)

        if best_W is None:
            best_W = np.zeros((p, n_targets))
            for j in range(n_targets):
                best_W[0, j] = y0[j]

        self.coef_ = pd.DataFrame(best_W, index=X.columns, columns=quasi_dynamic_df.columns)

        K = self.max_interactions
        if K > 0:
            w_arr = self.coef_.values.copy()
            for j in range(n_targets):
                self_gi = j + 1
                amplitudes: list[tuple[int, float]] = []
                for gi, g in enumerate(groups):
                    if gi == 0 or gi == self_gi:
                        continue
                    amp = float(np.max(np.abs(Xv[:, g] @ w_arr[g, j])))
                    amplitudes.append((gi, amp))
                amplitudes.sort(key=lambda x: x[1], reverse=True)
                keep = {gi for gi, _ in amplitudes[:K]}
                for gi, _ in amplitudes[K:]:
                    w_arr[groups[gi], j] = 0.0
                active_cols = list(groups[self_gi])
                for gi in keep:
                    active_cols.extend(groups[gi].tolist())
                if active_cols:
                    X_sub = Xv[:, active_cols]
                    y_adj = Y[:, j] - y0[j]
                    theta_sub = np.linalg.lstsq(X_sub, y_adj, rcond=None)[0]
                    w_arr[active_cols, j] = theta_sub
                    w_arr[0, j] = y0[j]
            self.coef_ = pd.DataFrame(w_arr, index=self.coef_.index, columns=self.coef_.columns)

        self.mse_ = float(np.mean((Y - Xv @ self.coef_.values) ** 2))
        return self

    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "IDOPRegressor":
        self.bic_order_path_ = None
        self.bic_alpha_path_ = None
        X = self._design(power_function_sample_df)
        Y = quasi_dynamic_df.reindex(X.index).values.astype(float)
        Xv = X.values.astype(float)
        n_targets = Y.shape[1]
        p = Xv.shape[1]

        n_feat = (p - 1) // self.max_order
        groups = _build_groups(n_feat, self.max_order)

        if self.solver == "asgl":
            return self._fit_asgl_bic(
                power_function_sample_df, quasi_dynamic_df, intercept_values
            )

        y0 = (
            np.asarray(intercept_values, dtype=float)
            if intercept_values is not None
            else Y[0, :].copy()
        )

        if self.solver == "lasso":
            if not self.protect_self and not self.nonneg_self:
                from sklearn.linear_model import Lasso

                lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10_000, tol=1e-4)
                Xv_body = Xv[:, 1:]
                W = np.zeros((p, n_targets))
                for j in range(n_targets):
                    W[0, j] = y0[j]
                    Y_adj = Y[:, j] - y0[j]
                    W[1:, j] = lasso.fit(Xv_body, Y_adj).coef_
            else:
                W = np.column_stack(
                    [
                        _lasso_cd_col(
                            Xv,
                            Y[:, j],
                            self.alpha,
                            protected=set(groups[j + 1].tolist()) | {0}
                            if self.protect_self
                            else {0},
                        )
                        for j in range(n_targets)
                    ]
                )
                for j in range(n_targets):
                    W[0, j] = y0[j]

        else:
            Xv_body = Xv[:, 1:]
            W = np.zeros((p, n_targets))
            for j in range(n_targets):
                W[0, j] = y0[j]
                Y_adj = Y[:, j] - y0[j]
                W[1:, j] = np.linalg.lstsq(Xv_body, Y_adj, rcond=None)[0]

        self.coef_ = pd.DataFrame(W, index=X.columns, columns=quasi_dynamic_df.columns)

        if self.nonneg_self and self.solver not in ("asgl",):
            w_arr = self.coef_.values.copy()
            for j in range(n_targets):
                _project_nonneg_self_keep_intercept(w_arr[:, j], Xv, 0, groups[j + 1])
            self.coef_ = pd.DataFrame(w_arr, index=self.coef_.index, columns=self.coef_.columns)

        K = self.max_interactions
        if K > 0:
            w_arr = self.coef_.values.copy()
            for j in range(n_targets):
                self_gi = j + 1
                amplitudes: list[tuple[int, float]] = []
                for gi, g in enumerate(groups):
                    if gi == 0 or gi == self_gi:
                        continue
                    amp = float(np.max(np.abs(Xv[:, g] @ w_arr[g, j])))
                    amplitudes.append((gi, amp))
                amplitudes.sort(key=lambda x: x[1], reverse=True)
                keep = {gi for gi, _ in amplitudes[:K]}
                for gi, _ in amplitudes[K:]:
                    w_arr[groups[gi], j] = 0.0

                active_cols = list(groups[self_gi])
                for gi in keep:
                    active_cols.extend(groups[gi].tolist())
                if active_cols:
                    X_sub = Xv[:, active_cols]
                    y_adj = Y[:, j] - y0[j]
                    theta_sub = np.linalg.lstsq(X_sub, y_adj, rcond=None)[0]
                    w_arr[active_cols, j] = theta_sub
                    w_arr[0, j] = y0[j]
            self.coef_ = pd.DataFrame(w_arr, index=self.coef_.index, columns=self.coef_.columns)

        self.mse_ = float(np.mean((Y - Xv @ self.coef_.values) ** 2))
        return self

    def predict(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        X = self._design(power_function_sample_df)
        return pd.DataFrame(
            X.values.astype(float) @ self.coef_.values,
            index=X.index,
            columns=self.coef_.columns,
        )

    def effect(self, power_function_sample_df: pd.DataFrame) -> list[pd.DataFrame]:
        """按每个源特征聚合积分基×系数，得到各目标上的源特征效应（不含截距）。"""
        if self.coef_ is None:
            raise RuntimeError("call fit before effect")
        X = self._design(power_function_sample_df)
        basis_int_df = X.drop(columns=["intercept"])
        n_feature = power_function_sample_df.shape[1]
        max_order = self.max_order
        feature_names = list(power_function_sample_df.columns)
        effect_df_list: list[pd.DataFrame] = []
        for target in self.coef_.columns:
            coef_row = self.coef_.loc[basis_int_df.columns, target]
            weighted = basis_int_df.multiply(coef_row, axis=1)
            cols: list[pd.Series] = []
            for j in range(n_feature):
                cols.append(weighted.iloc[:, j * max_order : (j + 1) * max_order].sum(axis=1))
            collapsed = pd.concat(cols, axis=1)
            collapsed.columns = feature_names
            effect_df_list.append(collapsed)
        return effect_df_list

    def adjacency_matrix(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        """G = sum_r Θ^(r) D^(r), ψ_k^(r)=sum_τ φ_k^(r)(τ)。"""
        if self.coef_ is None:
            raise RuntimeError("call fit before adjacency_matrix")
        X = self._design(power_function_sample_df)
        basis_int_df = X.drop(columns=["intercept"])
        m = power_function_sample_df.shape[1]
        max_order = self.max_order
        targets = list(self.coef_.columns)
        names = list(power_function_sample_df.columns)
        G = np.zeros((m, m), dtype=float)
        for r in range(max_order):
            theta_r = np.zeros((m, m), dtype=float)
            psi = np.zeros(m, dtype=float)
            for k in range(m):
                col_idx = k * max_order + r
                col_name = basis_int_df.columns[col_idx]
                theta_r[:, k] = self.coef_.loc[col_name, targets].values.astype(float)
                psi[k] = float(basis_int_df.iloc[:, col_idx].sum())
            G += theta_r * psi[np.newaxis, :]
        return pd.DataFrame(G, index=targets, columns=names)
