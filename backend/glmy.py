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

# ========== 侧边栏 ==========
# with st.sidebar:
#     st.write("To Be Updated...")
#     st.divider()


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["IdopNetwork", "Multi-Layer IdopNetwork", "To Be Updated..."])


# ========== Tab 1 IdopNetwork ==========
with tab1:
    st.write("uploaded_zip 待更新...")

    tab1_1, tab1_2, tab1_3 = st.tabs(["Data Overview", "IdopNetwork Construction", "Export"])

    # ========== Tab 1_1 Data Overview ==========