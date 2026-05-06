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
