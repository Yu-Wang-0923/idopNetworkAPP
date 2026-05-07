import streamlit as st

# ========== 页面设置 ==========
st.set_page_config(page_title="Network Construction111", page_icon="TSA.png", layout="wide", initial_sidebar_state="expanded")

# ========== 加载 CSS ==========
# from backend.utils import load_css, setup_sidebar
# load_css()
# setup_sidebar()

# ========== 页面标题 ==========
st.title("Network Construction111", text_alignment="center")



# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["GLMY", "Machine Learning", "Center Network"])


# ========== Tab 1 GLMY ==========
with tab1:
    st.write("待更新...")