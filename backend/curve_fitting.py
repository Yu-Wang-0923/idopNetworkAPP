import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit


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


# 拟合幂函数参数 a_j, b_j
@st.cache_data
def get_power_function_params(
    quasi_dynamic_df: pd.DataFrame,
) -> pd.DataFrame:
    results = {}
    for col in quasi_dynamic_df.columns:
        x = quasi_dynamic_df.index.values.astype(float)
        y = quasi_dynamic_df[col].values.astype(float)
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        a_hat, b_hat = curve_fit(power_equation, x, y, maxfev=50_000)[0]
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













