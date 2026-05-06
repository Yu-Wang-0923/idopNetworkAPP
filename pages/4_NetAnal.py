import streamlit as st

st.set_page_config(page_title="Network Analysis", page_icon=None, layout="wide", initial_sidebar_state="auto")

from backend.utils import load_css
load_css()

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
