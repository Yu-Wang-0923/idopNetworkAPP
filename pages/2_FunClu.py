import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

st.set_page_config(page_title="功能聚类", page_icon="📈")
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

st.markdown("<h1 style='text-align: center;'>功能聚类</h1>", unsafe_allow_html=True)

# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["一元功能聚类", "多元功能聚类", "待更新..."])


# ========== Tab 1 ==========
with tab1:
    st.write("待更新...")

# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")
