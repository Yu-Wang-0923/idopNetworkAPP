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
tab1, tab2 = st.tabs(["IdopNetwork", "Multi-Layer IdopNetwork"])


# ========== Tab 1 IdopNetwork ==========
with tab1:
    st.write("uploaded_zip 待更新...")

    tab1_1, tab1_2, tab1_3 = st.tabs(["Data Overview", "IdopNetwork Construction", "Export"])
    
    # ========== Tab 1_1 Data Overview ==========
    with tab1_1:
        st.write("待更新...")

    # ========== Tab 1_2 IdopNetwork Construction ==========
    with tab1_2:
        st.write("待更新...")

    tab1_2_1, tab1_2_2, tab1_2_3, tab1_2_4 = st.tabs(["Network", "Effect Decomposition", "Adjacency Matrix", "Debug"])
        
        # ========== Tab 1_2_1 Network ==========
        with tab1_2_1:
            st.write("待更新...")

        # ========== Tab 1_2_2 Effect Decomposition ==========
        with tab1_2_2:
            st.write("待更新...")

        # ========== Tab 1_2_3 Adjacency Matrix ==========
        with tab1_2_3:
            st.write("待更新...")

        # ========== Tab 1_2_4 Debug ==========
        with tab1_2_4:
            st.write("待更新...")

    # ========== Tab 1_3 Export ==========
    with tab1_3:
        st.write("待更新...")

# ========== Tab 2 Multi-Layer IdopNetwork ==========
with tab2:
    st.write("uploaded_zip 待更新...")

    tab2_1, tab2_2, tab2_3 = st.tabs(["Data Overview", "Multi-Layer IdopNetwork Construction", "Export"])

    # ========== Tab 2_1 Data Overview ==========
    with tab2_1:
        st.write("待更新...")

    # ========== Tab 2_2 IdopNetwork Construction ==========
    with tab2_2:
        st.write("待更新...")

    tab2_2_1, tab2_2_2, tab2_2_3, tab2_2_4 = st.tabs(["Network", "Effect Decomposition", "Adjacency Matrix", "Debug"])

    # ========== Tab 2_3 Export ==========
    with tab2_3:
        st.write("待更新...")

