# 调试 Curve Fitting
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit


# 添加自定义CSS
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# local_css("static/css/custom_style.css")


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


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs([
    "Uploaded Data", 
    "To Be Updated", 
    "To Be Updated",
    ])

with tab1:
    # st.markdown(
    # """
    # <div class="affiliation">
    #     To Be Updated
    # </div>
    # """,
    # unsafe_allow_html=True,
    # )

    uploaded_files = st.file_uploader(
        type=["csv"],
        accept_multiple_files=True,
        help="支持 csv, 可多选文件",
    )



    subtab1_1, subtab1_2, subtab1_3 = st.tabs([
        "To Be Updated", 
        "To Be Updated", 
        "To Be Updated",
    ])

