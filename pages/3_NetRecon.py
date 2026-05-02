import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Network Construction", page_icon="📈")
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

st.markdown("<h1 style='text-align: center;'>Network Construction</h1>", unsafe_allow_html=True)


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["单层Idop网络", "多层Idop网络", "待更新..."])


# ========== Tab 1 ==========
with tab1:
    st.write("待更新...")

# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")
