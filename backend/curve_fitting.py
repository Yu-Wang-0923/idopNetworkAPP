import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit





# 数据加载
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


# 数据变换
@st.cache_data
def data_transformation(
    data: pd.DataFrame,
    scaler_type: str,
) -> pd.DataFrame:
    if scaler_type == "none":
        return data.copy()
    if scaler_type == "rescale_to_-1_1":
        scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)
    elif scaler_type == "rescale_to_0_1":
        scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    elif scaler_type == "log1p":
        if (data < -1).any().any():
            raise ValueError("log1p 变换要求所有数值列数据均大于等于 -1。")
        scaled = data.apply(np.log1p, axis=0)
    else:
        raise ValueError(f"不支持的数据变换类型: {scaler_type}")
    return pd.DataFrame(scaled, columns=data.columns, index=data.index)



# 从 Static DataFrame 变换到 quasi-dynamic DataFrame.
# y_j(s_i) -> y_j(tau_i).
# T_i = \sum_{j=1}^p y_j(s_i), i = 1, 2, ..., n.
# tau_i = sigma(T_i), s.t. tau_1 ≦ ... ≦ tau_n.
def get_quasi_dynamic_df(
    data: pd.DataFrame,
) -> pd.DataFrame:
    row_sum = data.sum(axis=1).sort_values()
    quasi_dynamic_df = data.loc[row_sum.index].copy()
    quasi_dynamic_df.index = pd.Index(row_sum.values)
    return quasi_dynamic_df


# y = a * x^{b}.
def power_equation(
    x: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    return a * np.power(x, b)


# 拟合幂函数参数 a_j, b_j
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
















