# 调试 Curve Fitting
import io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit


# 字体文件路径（你已经放在 static 文件夹里了）
font_path = Path(__file__).parent.parent / "static" / "SimHei.ttf"
font_prop = fm.FontProperties(fname=font_path)


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
plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
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


def plot_scatter_matrix(df, use_seq, n_cols):
    x = np.arange(1, len(df)+1) if use_seq else df.index
    cols = st.columns(n_cols)
    for i, col in enumerate(df.columns):
        with cols[i % n_cols]:
            fig, ax = plt.subplots(figsize=(6,3), dpi=300)
            ax.scatter(x, df[col], s=70, alpha=0.7, facecolors='none', edgecolors='#4285F4', linewidth=1)
            ax.set_title(col, fontproperties=font_prop)
            ax.set_xlabel("Sequence" if use_seq else "Index", fontproperties=font_prop)
            ax.set_ylabel(col, fontproperties=font_prop)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_prop)
            ax.grid(alpha=0.3)
            ax.margins(x=0.2, y=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            st.pyplot(fig, dpi=300)
            plt.close()












# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs([
    "Uploaded Data", 
    "To Be Updated", 
    "To Be Updated",
    ])

with tab1:
    uploaded_files = st.file_uploader(
        label="Please upload your files",
        type=["csv"],
        accept_multiple_files=True,
        help="Supports CSV, multiple files allowed",
        label_visibility="visible",
        max_upload_size=500,
    )

    subtab1_1, subtab1_2, subtab1_3 = st.tabs([
        "Data Overview", 
        "Data Transformation", 
        "To Be Updated",
    ])

    with subtab1_1:
        if uploaded_files:
            with st.expander("⚙️ Data Settings", expanded=False):
                use_first_col_as_index = st.checkbox(
                    "Use first column as index",
                    value=True  # 默认勾选
                )
                st.session_state["use_first_col_as_index"] = use_first_col_as_index

            st.divider()

            cache = {}
            for file in uploaded_files:
                with st.expander(f"📄 Original Data: {file.name}", expanded=False):
                    df = load_csv(file)

                    if st.session_state["use_first_col_as_index"]:
                        df = df.set_index(df.columns[0])
                    
                    # 缓存, key: 文件名, value: df
                    cache[file.name] = df
                    st.session_state["file_df_cache"] = cache

                    with st.expander(f"📄 Original Data Overview: {file.name}", expanded=False):
                        st.dataframe(df, use_container_width=True)
                        st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

                    with st.expander(f"📄 Descriptive Statistics: {file.name}", expanded=False):
                        st.dataframe(df.describe(),use_container_width=True)

                    # key_id = id(df)
                    # with st.expander("📊 Scatter Plot", expanded=False):
                    #     with st.expander("⚙️ Plot Settings", expanded=False):
                    #         col1,col2 = st.columns(2)
                    #         use_seq = col1.checkbox("Use sequential X-axis", key=f"seq_{key_id}", value=True)
                    #         n_cols = col2.selectbox("Subplots per row", [1,2,3,4,5,6], index=2, key=f"col_{key_id}")

                    #     plot_scatter_matrix(df, use_seq, n_cols)
        else:
            st.info("Please upload CSV file(s) to view data overview")
    
    with subtab1_2:
        if uploaded_files:
            with st.expander("⚙️ Transformation Settings", expanded=False):
                scaler_type = st.selectbox(
                    "Transformation Type", 
                    [
                        "none", 
                        "rescale_to_0_1", 
                        "rescale_to_-1_1", 
                        "log1p"
                    ]
                )
            
            st.divider()

            cache = st.session_state.get("file_df_cache", {})
            
            for fname, df in cache.items():
                with st.expander(f"📄 Transformation Data: {file.name}", expanded=False):

                    # file = uploaded_files[0]
                    # df = load_csv(file)

                    # if st.session_state.get("use_index", True):
                    #     df = df.set_index(df.columns[0])

                    df_transform = data_transformation(df, scaler_type)

                    with st.expander(f"📄 Transformation Data Overview: {file.name}", expanded=False):
                        st.dataframe(df_transform, use_container_width=True)
                        st.info(f"Rows: {df_transform.shape[0]} | Columns: {df_transform.shape[1]}")

                    with st.expander(f"📄 Descriptive Statistics: {file.name}", expanded=False):
                        st.dataframe(df_transform.describe().round(2), use_container_width=True)

                    with st.expander("📊 Scatter Plot", expanded=False):
                        with st.expander("⚙️ Plot Settings", expanded=False):
                            col1,col2 = st.columns(2)
                            use_seq = col1.checkbox("Use sequential X-axis", key=f"seq_{fname}", value=True)
                            n_cols = col2.selectbox("Subplots per row", [1,2,3,4,5,6], index=2, key=f"col_{fname}")

                        plot_scatter_matrix(df_transform, use_seq, n_cols)
        else:
            st.info("Please upload CSV file(s) to view data overview")

