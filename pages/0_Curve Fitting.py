# 调试 Curve Fitting
import io
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

# sys.path.append("..")

# # sys.path.append(str(Path(__file__).parent.parent))
# from backend.curve_fitting import *
# from backend.plot_curve_fitting import *

# # 字体文件路径（你已经放在 static 文件夹里了）
# font_path = Path(__file__).parent.parent / "static" / "SimHei.ttf"
# font_prop = fm.FontProperties(fname=font_path)


# 页面设置
st.set_page_config(
    page_title="Curve Fitting", 
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        # 右上角 ⋮ 三点菜单里，帮助选项保留, 点击后跳转到：https://www.streamlit.io
        "Get help": "https://www.streamlit.io",
        "Report a Bug": None,
        "About": "# 数据看板\n版本 1.0.0" 
    }
)

# 页面标题
st.title("Curve Fitting", text_alignment="center")




# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# @st.cache_data
# def load_csv(file):
#     return pd.read_csv(file)

# @st.cache_data
# def data_transformation(
#     data: pd.DataFrame,
#     scaler_type: str,
# ) -> pd.DataFrame:
#     """
#     数据变换.
#     """
#     if scaler_type == "none":
#         return data.copy()
#     if scaler_type == "rescale_to_-1_1":
#         scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)
#     elif scaler_type == "rescale_to_0_1":
#         scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
#     elif scaler_type == "log1p":
#         if (data < -1).any().any():
#             raise ValueError("log1p 变换要求所有数值列数据均大于等于 -1。")
#         scaled = data.apply(np.log1p, axis=0)
#     else:
#         raise ValueError(f"不支持的数据变换类型: {scaler_type}")
#     return pd.DataFrame(scaled, columns=data.columns, index=data.index)


# def plot_scatter_matrix(df, use_seq, n_cols, max_plots):
#     x = np.arange(1, len(df)+1) if use_seq else df.index
#     cols = st.columns(n_cols)
#     selected_cols = df.columns[:max_plots]
#     for i, col in enumerate(selected_cols):
#         with cols[i % n_cols]:
#             fig, ax = plt.subplots(figsize=(4,3)) # , dpi=300
#             ax.scatter(x, df[col], s=150, alpha=0.7, facecolors='none', edgecolors='#4285F4', linewidth=1)
#             ax.set_title(col, fontproperties=font_prop)
#             ax.set_xlabel("Sequence" if use_seq else "Index", fontproperties=font_prop)
#             ax.set_ylabel(col, fontproperties=font_prop)
#             for label in ax.get_xticklabels() + ax.get_yticklabels():
#                 label.set_fontproperties(font_prop)
#             ax.grid(alpha=0.3)
#             ax.margins(x=0.2, y=0.3)
#             ax.xaxis.set_major_locator(plt.MaxNLocator(5))
#             ax.yaxis.set_major_locator(plt.MaxNLocator(5))
#             st.pyplot(fig)
#             plt.close()





# ========== Sidebar ==========
# with st.sidebar:
#     st.write("To Be Updated")
#     st.info('stpe-1: Uploaded Data')
#     st.info('stpe-2: Transformation Data')




# # ========== Tabs ==========
# tab1, tab2, tab3 = st.tabs([
#     "Uploaded Data", 
#     "Curve Fitting", 
#     "To Be Updated",
#     ])

# with tab1:
#     uploaded_files = st.file_uploader(
#         label="Please upload your files",
#         type=["csv"],
#         accept_multiple_files=True,
#         help="Supports CSV, multiple files allowed",
#         label_visibility="visible",
#         max_upload_size=500,
#     )

#     subtab1_1, subtab1_2, subtab1_3 = st.tabs([
#         "Data Overview", 
#         "Data Transformation", 
#         "To Be Updated",
#     ])

#     with subtab1_1:
#         if uploaded_files:
#             with st.expander("⚙️ Data Settings", expanded=False):
#                 use_first_col_as_index = st.checkbox(
#                     "Use first column as index",
#                     value=True  # 默认勾选
#                 )
#                 st.session_state["use_first_col_as_index"] = use_first_col_as_index

#             st.divider()

#             original_data_session = {}
#             for file in uploaded_files:
#                 with st.expander(f"📄 Original Data: {file.name}", expanded=False):
#                     df = load_csv(file)

#                     if st.session_state["use_first_col_as_index"]:
#                         df = df.set_index(df.columns[0])
                    
#                     # 缓存, key: 文件名, value: df
#                     original_data_session[file.name] = df
#                     st.session_state["original_data_session"] = original_data_session

#                     with st.expander(f"📄 Original Data Overview: {file.name}", expanded=False):
#                         st.dataframe(df, use_container_width=True)
#                         st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

#                     with st.expander(f"📄 Descriptive Statistics: {file.name}", expanded=False):
#                         st.dataframe(df.describe(),use_container_width=True)

#                     # key_id = id(df)
#                     # with st.expander("📊 Scatter Plot", expanded=False):
#                     #     with st.expander("⚙️ Plot Settings", expanded=False):
#                     #         col1,col2,col3 = st.columns(3)
#                     #         use_seq = col1.checkbox("Use sequential X-axis", key=f"original_data_seq_{key_id}", value=True)
#                     #         n_cols = col2.selectbox("Subplots per row", [1,2,3,4,5,6], index=2, key=f"original_data_col_{key_id}")
#                     #         max_plots = col3.selectbox("Max plots", [3,6,9], index=1, key=f"original_data_plots_{key_id}")

#                     #     plot_scatter_matrix(df, use_seq, n_cols, max_plots)
#         else:
#             st.info("Please upload CSV file(s) to view data overview")
    
#     with subtab1_2:
#         if uploaded_files:
#             with st.expander("⚙️ Transformation Settings", expanded=False):
#                 scaler_type = st.selectbox(
#                     "Transformation Type", 
#                     [
#                         "none", 
#                         "rescale_to_0_1", 
#                         "rescale_to_-1_1", 
#                         "log1p"
#                     ]
#                 )
            
#             st.divider()

#             original_data_session = st.session_state.get("original_data_session", {})
            

#             transform_data_session = {}
#             for fname, df in original_data_session.items():
#                 with st.expander(f"📄 Transformation Data: {file.name}", expanded=False):
#                     df_transform = data_transformation(df, scaler_type)

#                     transform_data_session[file.name] = df_transform
#                     st.session_state["transform_data_session"] = transform_data_session

#                     with st.expander(f"📄 Transformation Data Overview: {file.name}", expanded=False):
#                         st.dataframe(df_transform, use_container_width=True)
#                         st.info(f"Rows: {df_transform.shape[0]} | Columns: {df_transform.shape[1]}")

#                     with st.expander(f"📄 Descriptive Statistics: {file.name}", expanded=False):
#                         st.dataframe(df_transform.describe().round(2), use_container_width=True)

#                     # with st.expander("📊 Scatter Plot", expanded=False):
#                     #     with st.expander("⚙️ Plot Settings", expanded=False):
#                     #         col1,col2,col3 = st.columns(3)
#                     #         use_seq = col1.checkbox("Use sequential X-axis", key=f"transform_data_seq_{fname}", value=True)
#                     #         n_cols = col2.selectbox("Subplots per row", [1,2,3,4,5,6], index=2, key=f"transform_data_col_{fname}")
#                     #         max_plots = col3.selectbox("Max plots", [3,6,9], index=1, key=f"transform_data_plots_{fname}")
#                     #     plot_scatter_matrix(df_transform, use_seq, n_cols, max_plots)
#         else:
#             st.info("Please upload CSV file(s) to view data overview")

#     with subtab1_3:
#         st.write("To Be Updated")













# # def get_quasi_dynamic_df(
# #     data: pd.DataFrame,
# # ) -> pd.DataFrame:
# #     r"""
# #     从 Static DataFrame 变换到 quasi-dynamic DataFrame.
# #     y_j(s_i) -> y_j(tau_i).
# #     T_i = \sum_{j=1}^p y_j(s_i), i = 1, 2, ..., n.
# #     tau_i = sigma(T_i), s.t. tau_1 ≦ ... ≦ tau_n.
# #     """
# #     row_sum = data.sum(axis=1).sort_values()
# #     quasi_dynamic_df = data.loc[row_sum.index].copy()
# #     quasi_dynamic_df.index = pd.Index(row_sum.values)
# #     return quasi_dynamic_df



# # def power_equation(
# #     x: np.ndarray,
# #     a: float,
# #     b: float,
# # ) -> np.ndarray:
# #     """y = a * x^{b}."""
# #     return a * np.power(x, b)


# # def get_power_function_params(
# #     quasi_dynamic_df: pd.DataFrame,
# # ) -> pd.DataFrame:
# #     """拟合幂函数参数 a_j, b_j。"""
    
# #     results = {}
# #     for col in quasi_dynamic_df.columns:
# #         x = quasi_dynamic_df.index.values.astype(float)
# #         y = quasi_dynamic_df[col].values.astype(float)
# #         mask = (x > 0) & (y > 0)
# #         x = x[mask]
# #         y = y[mask]
# #         a_hat, b_hat = curve_fit(power_equation, x, y, maxfev=50_000)[0]
# #         results[col] = [a_hat, b_hat]
# #     return pd.DataFrame(results, index=["a", "b"]).T


# # def chebyshev_nodes(
# #     n: int,
# #     a: float,
# #     b: float,
# # ) -> np.ndarray:
# #     """生成切比雪夫节点。"""
# #     nodes = [
# #         0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) * np.pi / (2 * n))
# #         for i in range(n)
# #     ]
# #     return np.sort(np.array(nodes))


# # def get_power_function_sample(
# #     quasi_dynamic_df: pd.DataFrame,
# # ) -> pd.DataFrame:
# #     """计算幂律拟合曲线采样值。"""
# #     power_function_params = get_power_function_params(quasi_dynamic_df)
# #     params = power_function_params.reindex(quasi_dynamic_df.columns)
# #     tau_lo = float(quasi_dynamic_df.index.min())
# #     tau_hi = float(quasi_dynamic_df.index.max())
# #     allometric_index = chebyshev_nodes(len(quasi_dynamic_df), tau_lo, tau_hi)
# #     a = params["a"].to_numpy(dtype=float)
# #     b = params["b"].to_numpy(dtype=float)
# #     y = power_equation(allometric_index[:, np.newaxis], a, b)
# #     return pd.DataFrame(y, index=allometric_index, columns=quasi_dynamic_df.columns)



# # def plot_curve_fitting(
# #     df_quasi_dynamic, 
# #     df_curve_sample,
# #     use_seq, 
# #     n_cols, 
# #     max_plots
# # ):
# #     x = np.arange(1, len(df_quasi_dynamic)+1) if use_seq else df_quasi_dynamic.index
# #     a_index = df_curve_sample.index
# #     cols = st.columns(n_cols)
# #     selected_cols = df_quasi_dynamic.columns[:max_plots]
# #     for i, col in enumerate(selected_cols):
# #         with cols[i % n_cols]:
# #             fig, ax = plt.subplots(figsize=(4,3)) # , dpi=300
# #             ax.scatter(a_index, df_quasi_dynamic[col], s=150, alpha=0.7, facecolors='none', edgecolors='#4285F4', linewidth=1)
# #             ax.plot(a_index, df_curve_sample[col], alpha=0.7, color='#ff0000', linewidth=4)
# #             ax.set_title(col, fontproperties=font_prop)
# #             ax.set_xlabel("Sequence" if use_seq else "Index", fontproperties=font_prop)
# #             ax.set_ylabel(col, fontproperties=font_prop)
# #             for label in ax.get_xticklabels() + ax.get_yticklabels():
# #                 label.set_fontproperties(font_prop)
# #             ax.grid(alpha=0.3)
# #             ax.margins(x=0.2, y=0.3)
# #             ax.xaxis.set_major_locator(plt.MaxNLocator(5))
# #             ax.yaxis.set_major_locator(plt.MaxNLocator(5))
# #             st.pyplot(fig)
# #             plt.close()




# # with tab2:
# #     subtab2_1, subtab2_2, subtab2_3 = st.tabs([
# #         "Quasi Dynamic", 
# #         "异速生长拟合", 
# #         "To Be Updated",
# #     ])

# #     with subtab2_1:
# #         if uploaded_files:
# #             with st.expander("⚙️ Quasi Dynamic", expanded=False):
# #                 st.write("To Be Updated")

# #             transform_data_session = st.session_state.get("transform_data_session", {})

# #             for fname, df_transform in transform_data_session.items():
# #                     with st.expander(f"📄 Transformation Data: {file.name}", expanded=False):
# #                         df_quasi_dynamic = get_quasi_dynamic_df(df_transform)

# #                         with st.expander(f"📄 Transformation Data Overview: {file.name}", expanded=False):
# #                             st.dataframe(df_quasi_dynamic, use_container_width=True)
# #                             st.info(f"Rows: {df_quasi_dynamic.shape[0]} | Columns: {df_quasi_dynamic.shape[1]}")
                        
# #                         with st.expander("📊 Scatter Plot", expanded=False):
# #                             with st.expander("⚙️ Plot Settings", expanded=False):
# #                                 col1,col2,col3 = st.columns(3)
# #                                 use_seq = col1.checkbox("Use sequential X-axis", key=f"quasi_dynamic_data_seq_{fname}", value=False)
# #                                 n_cols = col2.selectbox("Subplots per row", [1,2,3,4,5,6], index=2, key=f"quasi_dynamic_data_col_{fname}")
# #                                 max_plots = col3.selectbox("Max plots", [3,6,9], index=1, key=f"quasi_dynamic_data_plots_{fname}")
# #                             plot_scatter_matrix(df_quasi_dynamic, use_seq, n_cols, max_plots)

# #     with subtab2_2:
# #         if uploaded_files:
# #             st.write("To Be Updated")

# #             curve_sample = get_power_function_sample(df_quasi_dynamic)
# #             st.dataframe(df_quasi_dynamic, use_container_width=True)
# #             st.dataframe(curve_sample, use_container_width=True)
# #             plot_curve_fitting(df_quasi_dynamic=df_quasi_dynamic, df_curve_sample=curve_sample, use_seq = None, n_cols=3, max_plots=6)
# # "Sequence"





