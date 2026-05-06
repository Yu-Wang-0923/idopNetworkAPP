import streamlit as st

st.set_page_config(page_title="Network Construction", page_icon=None, layout="wide", initial_sidebar_state="auto")


from backend.utils import load_css

load_css()

# NetworkConstruction
st.title("Network Construction", text_alignment="center")


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["IdopNetwork", "Multi-Layer IdopNetwork", "To Be Updated..."])


# ========== Tab 1 ==========
with tab1:
    st.write("To Be Updated...")

# ========== Tab 2 ==========
with tab2:
    st.write("To Be Updated...")
