import io
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


# 数据加载
@st.cache_data
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


# 数据变换
@st.cache_data
def data_transformation(
    data: pd.DataFrame,
    scaler_type: str,
) -> pd.DataFrame:
    if scaler_type == "none":
        return data.copy()
    if scaler_type == "rescale_to_-1_1":
        num = _numeric_frame_for_transform(data)
        scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(num)
    elif scaler_type == "rescale_to_0_1":
        num = _numeric_frame_for_transform(data)
        scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(num)
    elif scaler_type == "log1p":
        num = _numeric_frame_for_transform(data)
        if (num < -1).any().any():
            raise ValueError("log1p 变换要求所有数值列数据均大于等于 -1。")
        scaled = num.apply(np.log1p, axis=0)
        return pd.DataFrame(scaled, columns=data.columns, index=data.index)
    else:
        raise ValueError(f"不支持的数据变换类型: {scaler_type}")
    return pd.DataFrame(scaled, columns=data.columns, index=data.index)



# 从 Static DataFrame 变换到 quasi-dynamic DataFrame.
# y_j(s_i) -> y_j(tau_i).
# T_i = \sum_{j=1}^p y_j(s_i), i = 1, 2, ..., n.
# tau_i = sigma(T_i), s.t. tau_1 ≦ ... ≦ tau_n.
@st.cache_data
def get_quasi_dynamic_df(
    data: pd.DataFrame,
) -> pd.DataFrame:
    row_sum = data.sum(axis=1)
    # 用位置索引排序，避免原始行标签有重复时 .loc 展开多行导致长度不匹配
    sort_pos = np.argsort(row_sum.values, kind="stable")
    quasi_dynamic_df = data.iloc[sort_pos].copy()
    quasi_dynamic_df.index = pd.Index(row_sum.values[sort_pos])
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
        clip_a: 对 ``a`` 的 ``(lo, hi)`` 裁剪范围，默认 ``(1e-8, +inf)``，
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

    a = float(np.exp(intercept))
    b = slope
    a = float(np.clip(a, clip_a[0], clip_a[1]))
    b = float(np.clip(b, clip_b[0], clip_b[1]))
    return a, b


# 拟合幂函数参数 a_j, b_j（双对数线性回归；与 FunClu 初始化共用底层）
@st.cache_data
def get_power_function_params(
    quasi_dynamic_df: pd.DataFrame,
) -> pd.DataFrame:
    results = {}
    for col in quasi_dynamic_df.columns:
        x = quasi_dynamic_df.index.values.astype(float)
        y = quasi_dynamic_df[col].values.astype(float)
        a_hat, b_hat = fit_power_loglinear(x, y)
        results[col] = [a_hat, b_hat]
    return pd.DataFrame(results, index=["a", "b"]).T


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
@st.cache_data
def get_power_function_sample(
    quasi_dynamic_df: pd.DataFrame,
) -> pd.DataFrame:
    power_function_params = get_power_function_params(quasi_dynamic_df)
    params = power_function_params.reindex(quasi_dynamic_df.columns)
    tau_lo = float(quasi_dynamic_df.index.min())
    tau_hi = float(quasi_dynamic_df.index.max())
    allometric_index = chebyshev_nodes(len(quasi_dynamic_df), tau_lo, tau_hi)
    a = params["a"].to_numpy(dtype=float)
    b = params["b"].to_numpy(dtype=float)
    y = power_equation(allometric_index[:, np.newaxis], a, b)
    return pd.DataFrame(y, index=allometric_index, columns=quasi_dynamic_df.columns)


def show_data_expander(title, df):
    with st.expander(title, expanded=False):
        with st.expander("Data Overview", expanded=False):
            st.dataframe(df, use_container_width=True)
        with st.expander("Descriptive Statistics", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)
        with st.expander("Scatter Plot", expanded=False):
            st.write("To Be Updated")













