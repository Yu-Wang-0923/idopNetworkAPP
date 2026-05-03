import sys
import numpy as np
import pandas as pd
import streamlit as st
sys.path.append("..")
from backend.curve_fitting import *
from backend.plot_curve_fitting import *


st.set_page_config(page_title="Functional Clustering", page_icon=None, layout="wide", initial_sidebar_state="auto")
st.title("Functional Clustering", text_alignment="center")


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["固定簇的功能聚类", "BIC功能聚类", "待更新..."])


# ========== Tab 1 ==========
with tab1:
    st.write("待更新...")

# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")
