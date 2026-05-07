import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

# ========== 页面设置 ==========
st.set_page_config(page_title="Network Construction", page_icon="TSA.png", layout="wide", initial_sidebar_state="expanded")

# ========== 加载 CSS ==========
from backend.utils import load_css, setup_sidebar
load_css()
setup_sidebar()

# ========== 页面标题 ==========
st.title("Network Construction", text_alignment="center")


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["IdopNetwork", "Multi-Layer IdopNetwork", "To Be Updated..."])


# ========== Tab 1 GLMY ==========
with tab1:
    st.write("待更新...")