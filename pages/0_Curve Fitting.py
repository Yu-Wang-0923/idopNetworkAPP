import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

# 调试 Curve Fitting



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

# # 标题
# st.markdown(
#     """
#     <h1 style='text-align: center; color: #2E86AB; 
#                 font-size: 48px; 
#                 font-weight: 700; 
#                 margin-bottom: 30px;
#                 text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
#         Curve Fitting
#     </h1>
#     """,
#     unsafe_allow_html=True
# )

# st.title("Curve Fitting")

# 全局美化 Tabs CSS
st.markdown(
    """
    <style>
    /* 标签整体圆角、间距 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    /* 单个标签样式 */
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 10px;
        padding: 0 25px;
        font-size: 18px !important;
        font-weight: 600;
        color: #666;
        background-color: #f5f7fa;
    }

    /* 选中激活的标签 */
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    </style>
    """, 
    unsafe_allow_html=True,
)

# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs([
    "Uploaded Data", 
    "To Be Updated", 
    "To Be Updated",
    ])

with tab1:
    st.markdown(
    """
    <div class="affiliation">
        To Be Updated
    </div>
    """,
    unsafe_allow_html=True,
    )

    st.header("📁 上传数据")


    subtab1_1, subtab1_2, subtab1_3 = st.tabs([
        "To Be Updated", 
        "To Be Updated", 
        "To Be Updated",
    ])

