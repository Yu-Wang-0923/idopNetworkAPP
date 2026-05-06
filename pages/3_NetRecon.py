import streamlit as st

st.set_page_config(page_title="Network Construction", page_icon=None, layout="wide", initial_sidebar_state="auto")


from backend.utils import setup_matplotlib_chinese

from backend.utils import load_css

load_css()

st.title("Network Construction", text_alignment="center")


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["单层Idop网络", "多层Idop网络", "待更新..."])


# ========== Tab 1 ==========
with tab1:
    st.write("待更新...")

# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")
