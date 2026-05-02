# 调试 Curve Fitting
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

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




@st.cache_data
def load_csv(file):
    return pd.read_csv(file)
















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
        "To Be Updated", 
        "To Be Updated",
    ])

    with subtab1_1:
        if uploaded_files:
            with st.expander("⚙️ Data Settings", expanded=False):
                use_first_col_as_index = st.checkbox(
                    "Use first column as index",
                    value=True  # 默认勾选
                )
            st.divider()

            for file in uploaded_files:
                with st.expander(f"📄 Original Data: {file.name}", expanded=False):
                    df = load_csv(file)

                    if use_first_col_as_index:
                        df = df.set_index(df.columns[0])

                    st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                    st.dataframe(df, use_container_width=True)
                    st.subheader("Descriptive Statistics")
                    st.dataframe(df.describe(),use_container_width=True)
        else:
            st.info("Please upload CSV file(s) to view data overview")

