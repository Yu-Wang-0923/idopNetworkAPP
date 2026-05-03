import sys
import numpy as np
import pandas as pd
import streamlit as st
sys.path.append("..")
from backend.curve_fitting import *
from backend.functional_clustering import *


# FunClu
st.set_page_config(page_title="Functional Clustering", page_icon=None, layout="wide", initial_sidebar_state="auto")
st.title("Functional Clustering", text_alignment="center")


if "df_original" not in st.session_state:
    st.session_state.df_original = {}
if "df_transform" not in st.session_state:
    st.session_state.df_transform = {}
if "df_quasi_dynamic" not in st.session_state:
    st.session_state.df_quasi_dynamic = {}
if "df_curve_sample" not in st.session_state:
    st.session_state.df_curve_sample = {}


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["Uploaded Data", "BIC功能聚类", "待更新..."])



# ========== Tab 1 ==========
with tab1:
    uploaded_files = st.file_uploader(
        label="Please upload your files",
        type=["csv"],
        accept_multiple_files=True,
        help="Supports CSV, multiple files allowed",
        label_visibility="visible",
        max_upload_size=500,
    )

    subtab1_1, subtab1_2, subtab1_3 = st.tabs(["Data Overview", "Data Transformation", "To Be Updated"])

    with subtab1_1:
        if uploaded_files:     
            for file in uploaded_files:
                df_original = load_csv(file)
                st.session_state.df_original[file.name] = df_original
                with st.expander(f"Original Data: {file.name}", expanded=False):
                    with st.expander("Data Overview", expanded=False):
                        st.dataframe(df_original, use_container_width=True)
                    with st.expander("Descriptive Statistics", expanded=False):
                        st.dataframe(df_original.describe(), use_container_width=True)   
        else:
            st.info("Please upload CSV file(s)")


# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")
