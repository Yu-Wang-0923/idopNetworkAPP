import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Network Analysis", page_icon="📈")
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

st.markdown("<h1 style='text-align: center;'>Network Analysis</h1>", unsafe_allow_html=True)


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["GLMY 同调", "中心网络", "待更新..."])


# ========== Tab 1 ==========
with tab1:
    st.write("待更新...")

# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")

# ========== Tab 3 ==========
with tab3:
    st.write("待更新...")
