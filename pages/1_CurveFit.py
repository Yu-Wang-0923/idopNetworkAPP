import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

st.set_page_config(page_title="曲线拟合", page_icon="📈")
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

st.markdown("<h1 style='text-align: center;'>数据拟合</h1>", unsafe_allow_html=True)


def data_transformation(
    data: pd.DataFrame,
    scaler_type: str,
) -> pd.DataFrame:
    """
    数据变换.
    """
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


def apply_data_transformation(
    data: pd.DataFrame,
    scaler_type: str,
) -> pd.DataFrame:
    """对数值列应用变换。"""
    transformed_data = data.copy()
    numeric_cols = transformed_data.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        return transformed_data

    transformed_numeric = data_transformation(
        transformed_data[numeric_cols].astype(float), scaler_type
    )
    non_transformed_data = transformed_data.drop(columns=numeric_cols)
    transformed_data = pd.concat([non_transformed_data, transformed_numeric], axis=1)
    return transformed_data[data.columns]


def get_quasi_dynamic_df(
    data: pd.DataFrame,
) -> pd.DataFrame:
    r"""
    从 Static DataFrame 变换到 quasi-dynamic DataFrame.
    y_j(s_i) -> y_j(tau_i).
    T_i = \sum_{j=1}^p y_j(s_i), i = 1, 2, ..., n.
    tau_i = sigma(T_i), s.t. tau_1 ≦ ... ≦ tau_n.
    """
    row_sum = data.sum(axis=1).sort_values()
    quasi_dynamic_df = data.loc[row_sum.index].copy()
    quasi_dynamic_df.index = pd.Index(row_sum.values)
    return quasi_dynamic_df


def power_equation(
    x: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """y = a * x^{b}."""
    return a * np.power(x, b)


def get_power_function_params(
    quasi_dynamic_df: pd.DataFrame,
) -> pd.DataFrame:
    """拟合幂函数参数 a_j, b_j。"""
    x = quasi_dynamic_df.index.values.astype(float)
    results = {}
    for col in quasi_dynamic_df.columns:
        y = quasi_dynamic_df[col].values.astype(float)
        a_hat, b_hat = curve_fit(power_equation, x, y, maxfev=50_000)[0]
        results[col] = [a_hat, b_hat]
    return pd.DataFrame(results, index=["a", "b"]).T


def chebyshev_nodes(
    n: int,
    a: float,
    b: float,
) -> np.ndarray:
    """生成切比雪夫节点。"""
    nodes = [
        0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) * np.pi / (2 * n))
        for i in range(n)
    ]
    return np.sort(np.array(nodes))


def get_power_function_sample(
    quasi_dynamic_df: pd.DataFrame,
) -> pd.DataFrame:
    """计算幂律拟合曲线采样值。"""
    power_function_params = get_power_function_params(quasi_dynamic_df)
    params = power_function_params.reindex(quasi_dynamic_df.columns)
    tau_lo = float(quasi_dynamic_df.index.min())
    tau_hi = float(quasi_dynamic_df.index.max())
    allometric_index = chebyshev_nodes(len(quasi_dynamic_df), tau_lo, tau_hi)
    a = params["a"].to_numpy(dtype=float)
    b = params["b"].to_numpy(dtype=float)
    y = power_equation(allometric_index[:, np.newaxis], a, b)
    return pd.DataFrame(y, index=allometric_index, columns=quasi_dynamic_df.columns)


def build_allometric_fit_with_filter(
    quasi_dynamic_df: pd.DataFrame,
    min_points: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """按列过滤非正值后执行幂律拟合，并返回拟合统计。"""
    x_all = quasi_dynamic_df.index.to_numpy(dtype=float)
    params = {}
    filtered_points = {}
    stats = []
    skipped = {}

    for col in quasi_dynamic_df.columns:
        y_all = quasi_dynamic_df[col].to_numpy(dtype=float)
        valid_mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
        x_fit = x_all[valid_mask]
        y_fit = y_all[valid_mask]
        removed_count = int(len(y_all) - len(y_fit))

        if len(y_fit) < min_points:
            skipped[col] = f"过滤后有效点不足 {min_points}（仅 {len(y_fit)} 个）"
            continue

        try:
            a_hat, b_hat = curve_fit(power_equation, x_fit, y_fit, maxfev=50_000)[0]
        except Exception as exc:
            skipped[col] = f"拟合失败: {exc}"
            continue

        params[col] = [a_hat, b_hat]
        filtered_points[col] = (x_fit, y_fit)
        stats.append(
            {
                "column": col,
                "total_points": len(y_all),
                "removed_points": removed_count,
                "used_points": len(y_fit),
            }
        )

    if not params:
        empty_df = pd.DataFrame()
        return empty_df, empty_df, pd.DataFrame(stats), skipped

    power_params = pd.DataFrame(params, index=["a", "b"]).T
    positive_x = x_all[x_all > 0]
    tau_lo = float(positive_x.min())
    tau_hi = float(positive_x.max())
    allometric_index = chebyshev_nodes(len(quasi_dynamic_df), tau_lo, tau_hi)
    y = power_equation(
        allometric_index[:, np.newaxis],
        power_params["a"].to_numpy(dtype=float),
        power_params["b"].to_numpy(dtype=float),
    )
    power_sample = pd.DataFrame(y, index=allometric_index, columns=power_params.index)
    fit_stats_df = pd.DataFrame(stats).set_index("column")
    return power_params, power_sample, fit_stats_df, skipped

if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "show_data" not in st.session_state:
    st.session_state.show_data = False
if "show_plot" not in st.session_state:
    st.session_state.show_plot = False
if "show_quasi_data" not in st.session_state:
    st.session_state.show_quasi_data = False
if "show_quasi_plot" not in st.session_state:
    st.session_state.show_quasi_plot = False
if "show_allometric_fit" not in st.session_state:
    st.session_state.show_allometric_fit = False

# 侧边栏
st.sidebar.header("📁 上传数据")
use_first_col_as_index = st.sidebar.checkbox("使用第 0 列作为 index", value=False)
transpose_uploaded_data = st.sidebar.checkbox("上传数据转置", value=False)
enable_data_slice = st.sidebar.checkbox("仅加载前 n 行 p 列", value=False)
slice_n_rows = st.sidebar.number_input("n（行数）", min_value=1, value=100, step=1)
slice_p_cols = st.sidebar.number_input("p（列数）", min_value=1, value=10, step=1)
scaler_type = st.sidebar.selectbox(
    "数据变换",
    options=["none", "rescale_to_-1_1", "rescale_to_0_1", "log1p"],
    index=0,
)
uploaded_files = st.sidebar.file_uploader(
    "选择 CSV 文件", type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    st.session_state.datasets = {}
    for file in uploaded_files:
        df = pd.read_csv(
            file,
            index_col=0 if use_first_col_as_index else None,
            nrows=int(slice_n_rows)
            if enable_data_slice and not transpose_uploaded_data
            else None,
        )
        if transpose_uploaded_data:
            df = df.T
        if enable_data_slice:
            df = df.iloc[: int(slice_n_rows), : int(slice_p_cols)]
        st.session_state.datasets[file.name] = df
    st.session_state.show_data = False
    st.session_state.show_plot = False
    st.session_state.show_quasi_data = False
    st.session_state.show_quasi_plot = False
    st.session_state.show_allometric_fit = False

st.sidebar.markdown("### 数据展示")
if st.sidebar.button("展示数据"):
    st.session_state.show_data = True
if st.sidebar.button("生成可视化"):
    st.session_state.show_plot = True

st.sidebar.markdown("### 拟动态分析")
use_loglog_quasi_plot = st.sidebar.checkbox("拟动态图使用 log-log 空间", value=True)
if st.sidebar.button("生成拟动态数据"):
    st.session_state.show_quasi_data = True
if st.sidebar.button("生成拟动态可视化"):
    st.session_state.show_quasi_plot = True
if st.sidebar.button("执行异速生长拟合"):
    st.session_state.show_allometric_fit = True

if not st.session_state.datasets:
    st.info("请上传数据文件")
    st.stop()

subplot_rows = st.sidebar.number_input("子图行数", min_value=1, value=3, step=1)
subplot_cols = st.sidebar.number_input("子图列数", min_value=1, value=2, step=1)


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["数据", "拟合", "参数"])


# ========== Tab 1 ==========
with tab1:
    if not st.session_state.show_data:
        st.info("已上传文件后，点击侧边栏“展示数据”按钮查看表格预览。")
    else:
        for name, data in st.session_state.datasets.items():
            try:
                transformed_data = apply_data_transformation(data, scaler_type)
            except ValueError as exc:
                st.error(f"{name} 数据变换失败：{exc}")
                continue

            with st.expander(
                f"{name} ({transformed_data.shape[0]}×{transformed_data.shape[1]})"
            ):
                st.dataframe(transformed_data.head(10), use_container_width=True)

    if st.session_state.show_plot:
        st.subheader("散点图可视化")
        max_subplots = int(subplot_rows) * int(subplot_cols)
        for name, data in st.session_state.datasets.items():
            try:
                transformed_data = apply_data_transformation(data, scaler_type)
            except ValueError as exc:
                st.error(f"{name} 数据变换失败：{exc}")
                continue

            numeric_cols = transformed_data.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                st.warning(f"{name} 中没有可用于散点图的数值列。")
                continue

            if len(numeric_cols) > max_subplots:
                st.warning(
                    f"{name} 的数值列较多，仅展示前 {max_subplots} 列。"
                )
            plot_cols = numeric_cols[:max_subplots]

            fig, axes = plt.subplots(
                int(subplot_rows),
                int(subplot_cols),
                figsize=(5 * int(subplot_cols), 3.5 * int(subplot_rows)),
            )
            axes = np.array(axes).reshape(-1)
            x_values = np.arange(len(data))

            for idx, col in enumerate(plot_cols):
                axes[idx].scatter(
                    x_values,
                    transformed_data[col],
                    s=100,
                    alpha=0.85,
                    facecolors="none",
                    edgecolors="C0",
                    linewidths=1.0,
                )
                axes[idx].margins(x=0.2, y=0.3)
                axes[idx].set_title(col)
                axes[idx].set_xlabel("Index")
                axes[idx].set_ylabel("Value")

            for idx in range(len(plot_cols), len(axes)):
                fig.delaxes(axes[idx])

            fig.suptitle(f"{name} - 多子图散点图", fontsize=12)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("点击侧边栏“生成可视化”按钮后显示散点图。")


# ========== Tab 2 ==========
with tab2:
    subtab1, subtab2, subtab3 = st.tabs(["拟动态数据", "拟动态散点图", "异速生长拟合"])

    with subtab1:
        if not st.session_state.show_quasi_data:
            st.info("点击侧边栏“生成拟动态数据”按钮后显示。")
        else:
            for name, data in st.session_state.datasets.items():
                try:
                    transformed_data = apply_data_transformation(data, scaler_type)
                except ValueError as exc:
                    st.error(f"{name} 数据变换失败：{exc}")
                    continue

                numeric_data = transformed_data.select_dtypes(include=["number"])
                if numeric_data.empty:
                    st.warning(f"{name} 中没有可用于构建拟动态数据的数值列。")
                    continue

                quasi_dynamic_df = get_quasi_dynamic_df(numeric_data)
                with st.expander(
                    f"{name} 拟动态数据 ({quasi_dynamic_df.shape[0]}×{quasi_dynamic_df.shape[1]})"
                ):
                    st.dataframe(quasi_dynamic_df.head(10), use_container_width=True)

    with subtab2:
        if not st.session_state.show_quasi_plot:
            st.info("点击侧边栏“生成拟动态可视化”按钮后显示。")
        else:
            max_subplots = int(subplot_rows) * int(subplot_cols)
            for name, data in st.session_state.datasets.items():
                try:
                    transformed_data = apply_data_transformation(data, scaler_type)
                except ValueError as exc:
                    st.error(f"{name} 数据变换失败：{exc}")
                    continue

                numeric_data = transformed_data.select_dtypes(include=["number"])
                if numeric_data.empty:
                    st.warning(f"{name} 中没有可用于拟动态散点图的数值列。")
                    continue

                quasi_dynamic_df = get_quasi_dynamic_df(numeric_data)
                numeric_cols = quasi_dynamic_df.columns.tolist()

                if len(numeric_cols) > max_subplots:
                    st.warning(
                        f"{name} 的数值列较多，仅展示前 {max_subplots} 列拟动态散点图。"
                    )
                plot_cols = numeric_cols[:max_subplots]

                fig, axes = plt.subplots(
                    int(subplot_rows),
                    int(subplot_cols),
                    figsize=(5 * int(subplot_cols), 3.5 * int(subplot_rows)),
                )
                axes = np.array(axes).reshape(-1)
                x_values = quasi_dynamic_df.index.to_numpy()

                for idx, col in enumerate(plot_cols):
                    y_values = quasi_dynamic_df[col].to_numpy()
                    if use_loglog_quasi_plot:
                        valid_mask = (x_values > 0) & (y_values > 0)
                        if not valid_mask.any():
                            st.warning(
                                f"{name} 的列 {col} 含非正值，无法在 log-log 空间绘图。"
                                "请先进行数据变换。"
                            )
                            continue
                        axes[idx].loglog(
                            x_values[valid_mask],
                            y_values[valid_mask],
                            linestyle="None",
                            marker="o",
                            markersize=6,
                            markerfacecolor="none",
                            markeredgecolor="C0",
                            markeredgewidth=1.0,
                            alpha=0.85,
                        )
                        axes[idx].set_xlabel("log(Tau)")
                        axes[idx].set_ylabel("log(Value)")
                    else:
                        axes[idx].scatter(
                            x_values,
                            y_values,
                            s=100,
                            alpha=0.85,
                            facecolors="none",
                            edgecolors="C0",
                            linewidths=1.0,
                        )
                        axes[idx].set_xlabel("Tau")
                        axes[idx].set_ylabel("Value")
                    axes[idx].margins(x=0.2, y=0.3)
                    axes[idx].set_title(col)

                for idx in range(len(plot_cols), len(axes)):
                    fig.delaxes(axes[idx])

                fig.suptitle(f"{name} - 拟动态多子图散点图", fontsize=12)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with subtab3:
        if not st.session_state.show_allometric_fit:
            st.info("点击侧边栏“执行异速生长拟合”按钮后显示。")
        else:
            max_subplots = int(subplot_rows) * int(subplot_cols)
            for name, data in st.session_state.datasets.items():
                try:
                    transformed_data = apply_data_transformation(data, scaler_type)
                except ValueError as exc:
                    st.error(f"{name} 数据变换失败：{exc}")
                    continue

                numeric_data = transformed_data.select_dtypes(include=["number"])
                if numeric_data.empty:
                    st.warning(f"{name} 中没有可用于异速生长拟合的数值列。")
                    continue

                quasi_dynamic_df = get_quasi_dynamic_df(numeric_data)
                (
                    power_params,
                    power_sample,
                    _fit_stats,
                    _skipped_cols,
                ) = build_allometric_fit_with_filter(quasi_dynamic_df)
                if power_params.empty:
                    st.info(f"{name} 过滤后没有可用于幂律拟合的列，请到参数页查看详情。")
                    continue

                valid_cols = power_params.index.tolist()

                if len(valid_cols) > max_subplots:
                    st.warning(
                        f"{name} 的可拟合列较多，仅展示前 {max_subplots} 列拟合图。"
                    )
                plot_cols = valid_cols[:max_subplots]
                fig, axes = plt.subplots(
                    int(subplot_rows),
                    int(subplot_cols),
                    figsize=(5 * int(subplot_cols), 3.5 * int(subplot_rows)),
                )
                axes = np.array(axes).reshape(-1)

                for idx, col in enumerate(plot_cols):
                    valid_mask = (quasi_dynamic_df.index.to_numpy(dtype=float) > 0) & (
                        quasi_dynamic_df[col].to_numpy(dtype=float) > 0
                    )
                    x_scatter = quasi_dynamic_df.index.to_numpy(dtype=float)[valid_mask]
                    y_scatter = quasi_dynamic_df[col].to_numpy(dtype=float)[valid_mask]
                    x_curve = power_sample.index.to_numpy(dtype=float)
                    y_curve = power_sample[col].to_numpy(dtype=float)
                    if use_loglog_quasi_plot:
                        axes[idx].loglog(
                            x_scatter,
                            y_scatter,
                            linestyle="None",
                            marker="o",
                            markersize=6,
                            markerfacecolor="none",
                            markeredgecolor="C0",
                            markeredgewidth=1.0,
                            alpha=0.85,
                        )
                        axes[idx].loglog(
                            x_curve,
                            y_curve,
                            color="red",
                            linewidth=2.5,
                        )
                        axes[idx].set_xlabel("log(Tau)")
                        axes[idx].set_ylabel("log(Value)")
                    else:
                        axes[idx].scatter(
                            x_scatter,
                            y_scatter,
                            s=100,
                            alpha=0.85,
                            facecolors="none",
                            edgecolors="C0",
                            linewidths=1.0,
                        )
                        axes[idx].plot(
                            x_curve,
                            y_curve,
                            color="red",
                            linewidth=2.5,
                        )
                        axes[idx].set_xlabel("Tau")
                        axes[idx].set_ylabel("Value")
                    axes[idx].margins(x=0.2, y=0.3)
                    axes[idx].set_title(col)

                for idx in range(len(plot_cols), len(axes)):
                    fig.delaxes(axes[idx])

                fig.suptitle(f"{name} - 异速生长拟合", fontsize=12)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# ========== Tab 3 ==========
with tab3:
    if not st.session_state.show_allometric_fit:
        st.info("点击侧边栏“执行异速生长拟合”按钮后显示参数信息。")
    else:
        for name, data in st.session_state.datasets.items():
            try:
                transformed_data = apply_data_transformation(data, scaler_type)
            except ValueError as exc:
                st.error(f"{name} 数据变换失败：{exc}")
                continue

            numeric_data = transformed_data.select_dtypes(include=["number"])
            if numeric_data.empty:
                st.warning(f"{name} 中没有可用于异速生长拟合的数值列。")
                continue

            quasi_dynamic_df = get_quasi_dynamic_df(numeric_data)
            power_params, _power_sample, _fit_stats_df, _skipped_cols = (
                build_allometric_fit_with_filter(quasi_dynamic_df)
            )
            valid_cols = power_params.index.tolist()

            st.markdown(f"#### {name}")
            if not valid_cols:
                continue

            st.dataframe(power_params, use_container_width=True)
