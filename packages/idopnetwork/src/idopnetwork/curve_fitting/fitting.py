import io
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


# 数据加载
def load_csv(file):
    return pd.read_csv(file, index_col=0)


def _numeric_frame_for_transform(data: pd.DataFrame) -> pd.DataFrame:
    """将列转为 float；非数值列无法解析时抛出明确错误，避免 sklearn 在 object 上失败。"""
    if data.empty:
        return data.copy()
    try:
        return data.apply(pd.to_numeric, errors="raise").astype(np.float64)
    except ValueError as e:
        raise ValueError(
            "数据变换要求所有数据列为可解析数值（object/文本列无法缩放或取 log1p）。"
            "请检查 CSV：仅将应作为行索引的一列放在首列，其余列应为数值。"
        ) from e


# 数据变换（funclu_v4）
def preprocess(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Apply the supplied v4 column-wise transform; nonnumeric cells become NaN."""
    df = df.apply(pd.to_numeric, errors="coerce")
    transforms = {
        "None": lambda x: x,
        "Log10_1p": lambda x: np.log10(1 + x),
        "Minmax_0_1": lambda x: (x - x.min()) / (x.max() - x.min()).replace(0, 1),
        "Z_min_add1": lambda x: x - x.min() + 1,
    }
    if method not in transforms:
        raise ValueError(f"不支持的数据变换类型: {method}")
    return transforms[method](df)


def data_transformation(data: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
    """Application entry point for the v4 preprocessing kernel."""
    return preprocess(data, scaler_type)


# 从 Static DataFrame 变换到 quasi-dynamic DataFrame.
# y_j(s_i) -> y_j(tau_i).
# T_i = \sum_{j=1}^p y_j(s_i), i = 1, 2, ..., n.
# tau_i = sigma(T_i), s.t. tau_1 ≦ ... ≦ tau_n.
def get_quasi_dynamic_df(
    data: pd.DataFrame,
    log_index: bool = False,
) -> pd.DataFrame:
    """Build quasi-dynamic data, optionally using natural log of the row-sum index."""
    row_sum = data.sum(axis=1)
    # 用位置索引排序，避免原始行标签有重复时 .loc 展开多行导致长度不匹配
    sort_pos = np.argsort(row_sum.values, kind="stable")
    quasi_dynamic_df = data.iloc[sort_pos].copy()
    quasi_index = row_sum.values[sort_pos].astype(float)
    if log_index:
        valid = np.isfinite(quasi_index) & (quasi_index > 0)
        quasi_dynamic_df = quasi_dynamic_df.iloc[valid].copy()
        quasi_index = np.log(quasi_index[valid])
    quasi_dynamic_df.index = pd.Index(quasi_index)
    return quasi_dynamic_df


# y = a * x^{b}.
def power_equation(
    x: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    return a * np.power(x, b)


def fit_power_loglinear(
    x: np.ndarray,
    y: np.ndarray,
    *,
    clip_a: Tuple[float, float] = (-np.inf, np.inf),
    clip_b: Tuple[float, float] = (-np.inf, np.inf),
) -> Tuple[float, float]:
    """以双对数线性回归（log-log OLS）拟合幂律 ``y = a * x^b``。

    异速生长律（Huxley, 1932）的标准做法：在 log-log 空间做最小二乘等价于
    假设乘性（对数正态）噪声，对跨量级的生物学数据更合理；同时具有闭式解，
    数值稳定、不依赖优化器初值。

    自动过滤 ``x_i <= 0 / y_i <= 0 / 非有限`` 的样本对（``log`` 不能取非正）。
    若过滤后有效样本数不足 2，或 ``log x`` 方差为 0，回退到 ``(a, b) = (1.0, 0.5)``。

    Args:
        x: 一维数组。
        y: 与 ``x`` 等长的一维数组。
        clip_a: 对 ``a`` 的 ``(lo, hi)`` 裁剪范围，默认 ``(-inf, +inf)``，
            防止数值上为零或负。
        clip_b: 对 ``b`` 的 ``(lo, hi)`` 裁剪范围，默认不裁剪。

    Returns:
        ``(a, b)`` 两个 Python ``float``。

    Raises:
        ValueError: ``x`` 与 ``y`` 长度不一致。
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"x, y 长度不一致: x.shape={x_arr.shape}, y.shape={y_arr.shape}"
        )

    mask = (x_arr > 0) & (y_arr > 0) & np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return 1.0, 0.5

    log_x = np.log(x_arr[mask])
    log_y = np.log(y_arr[mask])
    try:
        result = stats.linregress(log_x, log_y)
        slope = float(result.slope)
        intercept = float(result.intercept)
    except Exception:
        return 1.0, 0.5
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return 1.0, 0.5

    # 防止 np.exp(intercept) 溢出为 Inf，限制 a 值在 float64 安全范围内。
    # exp(690) ≈ 1.4e299, exp(-690) ≈ 1e-300，足以覆盖所有生物学实际量级。
    safe_intercept = float(np.clip(intercept, -690.0, 690.0))
    a = float(np.exp(safe_intercept))
    b = slope
    a = float(np.clip(a, clip_a[0], clip_a[1]))
    b = float(np.clip(b, clip_b[0], clip_b[1]))
    return a, b


def power_fitting(df_qd: pd.DataFrame, n_samples: int = 30):
    """Fit v4 power curves in standardized log-time and sample a uniform grid.

    Nonpositive/nonfinite observations are ignored per feature. Features with
    fewer than two usable observations retain NaN parameters for exclusion by
    clustering, rather than being replaced with an arbitrary default curve.
    """
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples 必须是至少为 2 的整数")
    t, y = df_qd.index.to_numpy(float), df_qd.to_numpy(float)
    keep = np.isfinite(t) & (t > 0)
    t, y = t[keep], y[keep]

    if len(t) < 2 or np.unique(t).size < 2:
        raise ValueError("幂律拟合至少需要两个不同的正数时间点")
    u = np.log(t)
    m, d = u.mean(), u.std(ddof=0)
    z = (u - m) / d

    positive = (y > 0) & np.isfinite(y)
    n_positive = positive.sum(axis=0)
    n_safe = np.maximum(n_positive, 1)

    r = np.zeros_like(y)
    np.log(y, out=r, where=positive)

    sum_z = positive.T @ z
    sum_z2 = positive.T @ (z**2)
    sum_r = r.sum(axis=0)
    sum_zr = z @ r

    z_bar, r_bar = sum_z / n_safe, sum_r / n_safe
    S_zz = sum_z2 - sum_z**2 / n_safe
    S_zr = sum_zr - sum_z * sum_r / n_safe

    valid = (n_positive >= 2) & (S_zz > 1e-12)
    beta = np.divide(S_zr, S_zz, out=np.full(y.shape[1], np.nan), where=valid)
    alpha = r_bar - beta * z_bar

    c = np.exp(alpha)
    b = beta / d
    a = np.exp(alpha - b * m)

    params = pd.DataFrame(
        {"c": c, "beta": beta, "a": a, "b": b, "n_positive": n_positive},
        index=df_qd.columns,
    )
    params.index.name = "feature"

    sample_t = np.linspace(t.min(), t.max(), n_samples)
    sample_z = (np.log(sample_t) - m) / d
    fitted = np.exp(alpha + sample_z[:, None] * beta)

    samples = pd.DataFrame(
        fitted, index=pd.Index(sample_t, name="quasi time"), columns=df_qd.columns,
    )

    return params, samples


def get_power_function_params(quasi_dynamic_df: pd.DataFrame) -> pd.DataFrame:
    return power_fitting(quasi_dynamic_df)[0]


# 生成切比雪夫节点
def chebyshev_nodes(
    n: int,
    a: float,
    b: float,
) -> np.ndarray:
    nodes = [
        0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) * np.pi / (2 * n))
        for i in range(n)
    ]
    return np.sort(np.array(nodes))


# 计算幂律拟合曲线采样值
def get_power_function_sample(
    quasi_dynamic_df: pd.DataFrame, n_samples: int = 30,
) -> pd.DataFrame:
    return power_fitting(quasi_dynamic_df, n_samples=n_samples)[1]


def show_data_expander(title, df):
    import warnings
    warnings.warn(
        "show_data_expander is deprecated and no longer supports Streamlit display. "
        "Use your application layer's display framework instead.",
        DeprecationWarning,
        stacklevel=2,
    )













