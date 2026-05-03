import streamlit as st

from backend.utils import setup_matplotlib_chinese

st.set_page_config(page_title="Network Construction", page_icon="📈")
setup_matplotlib_chinese()

st.markdown("<h1 style='text-align: center;'>Network Construction</h1>", unsafe_allow_html=True)


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["单层Idop网络", "多层Idop网络", "待更新..."])


# ========== Tab 1 ==========
with tab1:
    st.write("待更新...")

# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")
