import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

# 调试 Curve Fitting

st.set_page_config(
    page_title="Curve Fitting", 
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": "# 数据看板\n版本 1.0.0"
    }
    )


st.markdown("<h1 style='text-align: center;'>Curve Fitting</h1>", unsafe_allow_html=True)

# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["数据", "拟合", "参数"])

with tab1:
    st.markdown(
    """
    <div class="affiliation">
        复杂系统拓扑统计理论及应用北京市重点实验室<br/>
        北京雁栖湖应用数学研究院
    </div>
    """,
    unsafe_allow_html=True,
    )

