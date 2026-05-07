import streamlit as st

# 🌟 修改 1：统一小图标和侧边栏状态
st.set_page_config(page_title="Network Analysis", page_icon="TSA.png", layout="wide", initial_sidebar_state="expanded")

# 🌟 修改 2：导入 setup_sidebar
from backend.utils import load_css, setup_sidebar

# 🌟 修改 3：一键加载
load_css()
setup_sidebar()

st.title("Network Analysis", text_alignment="center")


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["GLMY", "Machine Learning", "Center Network"])


# ========== Tab 1 GLMY ==========
with tab1:
    st.write("待更新...")

    tab1_1, tab1_2 = st.tabs(["Uploaded Data", "GLMY Analysis"])

    # ========== Tab 1_1 Uploaded Data ==========
    with tab1_1:
        st.write("待更新...")

    # ========== Tab 1_1 GLMY Analysis ==========
    with tab1_2:
        st.write("待更新...")

# ========== Tab 2 Machine Learning ==========
with tab2:
    st.write("待更新...")

    tab2_1, tab2_2, tab2_3 = st.tabs(["Uploaded Data", "Classification", "Regression"])
    
    # ========== Tab 2_1 Uploaded Data ==========
    with tab2_1:
        st.write("待更新...")

    # ========== Tab 2_2 Classification ==========
    with tab2_2:
        st.write("待更新...")

    # ========== Tab 2_3 Regression ==========
    with tab2_3:
        st.write("待更新...")


# ========== Tab 3 Center Network ==========
with tab3:
    st.write("待更新...")
