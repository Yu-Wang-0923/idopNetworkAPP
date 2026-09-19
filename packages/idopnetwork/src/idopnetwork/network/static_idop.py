"""可选建网算法：把 :class:`IDOPRegressor` 的**交叉边选边器**换成新版多窗口 LASSO。

设计原则（重要）
----------------
新版 ``idop.py`` 与仓库的建网求解**本来就是同一套东西**：同样是 TIGER-style 逐目标
约束凸优化（cvxpy、CLARABEL→SCS、``ridge=1e-6``、``gap_min=1e-6``、跨源 L1
``5e-4``、同样两个方向各解一次取残差最小者），基函数 ``tiger_lop_basis_expansion``
与新版脚本里的积分 Legendre 基也逐行等价。

因此本模块**不再重写求解器**，只提供：

1. :func:`select_edges_lasso` —— 新版选边器（在**原始拟动态数据**上跑多窗口
   LASSO，按**出现频率**过阈值入选；仓库版是在**基函数列**上沿 alpha 路径取第一个
   非零解 + 分组 Top-K）；
2. :class:`StaticIDOPRegressor` —— 继承 :class:`IDOPRegressor`，仅覆写
   ``_select_lasso_cross_support``，其余（基函数、cvxpy 约束、效应约束校验、
   ``max_order`` 的 BIC 网格、``effect`` / ``adjacency_matrix``）**全部复用父类**；
3. :func:`fit_static_idop_network` —— 按页面约定产出 network 字典。

这样做的好处是：**效应分解、预测曲线、邻接矩阵的形态与原来完全一致**，唯一的差别
只来自支撑集不同 —— 而这正是这个可选项存在的意义。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from idopnetwork.network.construction import IDOPRegressor

__all__ = [
    "StaticIDOPRegressor",
    "fit_static_idop_network",
    "select_edges_lasso",
]


def select_edges_lasso(
    data_ecg: pd.DataFrame,
    alpha: float = 0.1,
    k: int = 10,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """多窗口 LASSO：按「跨窗口出现频率」恢复稀疏有向支撑集。

    把样本切成 ``k`` 个窗口；每个窗口内对每个 target 做一次逐列 z-score 后的
    LASSO；系数还原回原始尺度后，**在超过 ``threshold`` 比例的窗口里都非零**的
    source 才进入该 target 的支撑集。

    Parameters
    ----------
    data_ecg:
        拟动态数据（样本 × 特征）；若含 ``time`` 列会被排除在特征之外。
    alpha:
        LASSO 正则强度。
    k:
        窗口数；``k=1`` 即全样本一次。
    threshold:
        出现频率阈值，取值 ``[0, 1]``。

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
) -> dict[str, set[str]]:
    """``{"a,b"}`` → ``{target: {source, ...}}``（只保留已知特征）。"""
    support_sets: dict[str, set[str]] = {}
    for _, row in edge_supports.iterrows():
        target = str(row["target"])
        inner = str(row["source"]).strip().strip("{}")
        support_sets[target] = {
            s.strip() for s in inner.split(",") if s.strip() and s.strip() in features
        }
    return support_sets


class StaticIDOPRegressor(IDOPRegressor):
    """与 :class:`IDOPRegressor` **完全相同的建网流程**，只替换交叉边选边器。

    覆写 :meth:`_select_lasso_cross_support` 使用 :func:`select_edges_lasso`
    （原始数据上的多窗口 LASSO + 频率阈值），继承的基函数、cvxpy 约束求解、
    效应约束校验、``max_order`` 的 BIC 网格因此全部保持不变。
    """

    def __init__(
        self,
        *,
        lasso_alpha: float = 0.1,
        lasso_windows: int = 1,
        lasso_threshold: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.lasso_alpha = float(lasso_alpha)
        self.lasso_windows = int(lasso_windows)
        self.lasso_threshold = float(lasso_threshold)
        self.edge_supports_: pd.DataFrame | None = None
        self._support_names_: dict[str, set[str]] = {}
        self._feature_names_: list[str] = []
        self._target_names_: list[str] = []

    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        *,
        power_function_params: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "StaticIDOPRegressor":
        """先跑新版选边器，再走父类完全相同的拟合流程。"""
        features = [str(c) for c in power_function_sample_df.columns]
        targets = [str(c) for c in quasi_dynamic_df.columns]

        self._feature_names_ = features
        self._target_names_ = targets
        self.edge_supports_ = select_edges_lasso(
            quasi_dynamic_df,
            alpha=self.lasso_alpha,
            k=self.lasso_windows,
            threshold=self.lasso_threshold,
        )
        self._support_names_ = _parse_support_sets(self.edge_supports_, features)

        return super().fit(
            power_function_sample_df,
            quasi_dynamic_df,
            power_function_params=power_function_params,
            intercept_values=intercept_values,
        )

    def _select_lasso_cross_support(
        self,
        basis: np.ndarray,
        y_adj: np.ndarray,
        groups: list[np.ndarray],
        target_idx: int,
    ) -> list[int]:
        """用新版选边器的结果替换父类的 alpha 路径 LASSO。

        ``groups[k + 1]`` 对应第 ``k`` 个特征（``groups[0]`` 是截距组），
        因此特征名 → 组号 = 特征下标 + 1。
        """
        if target_idx >= len(self._target_names_):
            return []
        target = self._target_names_[target_idx]
        wanted = self._support_names_.get(target, set())
        if not wanted:
            return []

        selected = [
            index + 1
            for index, name in enumerate(self._feature_names_)
            if name != target and name in wanted
        ]
        if not selected:
            return []

        # Top-K 与父类同口径：按组与残差的相关强度排序
        if self.max_interactions > 0 and len(selected) > self.max_interactions:
            residual = y_adj - float(np.mean(y_adj))
            scored = [
                (gi, float(np.linalg.norm(basis[:, groups[gi] - 1].T @ residual)))
                for gi in selected
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            selected = [gi for gi, _ in scored[: int(self.max_interactions)]]

        return selected


def fit_static_idop_network(
    curve_sample_df: pd.DataFrame,
    response_df: pd.DataFrame,
    *,
    max_order: int,
    nonneg_self: bool,
    max_interactions: int,
    adjacency_aggregation: str,
    power_function_params: pd.DataFrame,
    lasso_alpha: float = 0.1,
    lasso_windows: int = 1,
    lasso_threshold: float = 0.5,
) -> dict:
    """按页面 ``_fit_idop_network_from_curve_sample`` 的约定产出 network 字典。

    与仓库路线的唯一差别是模型换成了 :class:`StaticIDOPRegressor`，其余步骤
    （``fit`` / ``predict`` / ``effect`` / ``adjacency_matrix`` / ``_design``）
    完全一致，因此效应分解的形态保持一致。
    """
    if curve_sample_df.shape[1] < 2:
        raise ValueError("至少需要 2 条曲线才能构建交互网络")
    features = [str(c) for c in curve_sample_df.columns]
    response_df = response_df.loc[:, features]

    from idopnetwork.network.construction import align_response_to_design

    model = StaticIDOPRegressor(
        max_order=int(max_order),
        mix=0.5,
        fix_mix=False,
        nonneg_self=bool(nonneg_self),
        max_interactions=int(max_interactions),
        adaptive_weights=False,
        lasso_alpha=float(lasso_alpha),
        lasso_windows=int(lasso_windows),
        lasso_threshold=float(lasso_threshold),
    )
    model.fit(
        curve_sample_df,
        response_df,
        power_function_params=power_function_params,
    )

    design_X = model._design(curve_sample_df)
    return {
        "model": model,
        "quasi_dynamic_df": response_df,
        "curve_sample_df": curve_sample_df,
        "design_X": design_X,
        "response_Y": align_response_to_design(response_df, design_X.index),
        "predicted_df": model.predict(curve_sample_df),
        "effect_df_list": model.effect(curve_sample_df),
        "adj_df": model.adjacency_matrix(
            curve_sample_df, aggregation=str(adjacency_aggregation)
        ),
        "adjacency_aggregation": str(adjacency_aggregation),
    }
