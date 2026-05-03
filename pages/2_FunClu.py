import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Functional Clustering", page_icon=None, layout="wide", initial_sidebar_state="auto")

_APP_ROOT = Path(__file__).resolve().parent.parent
_root = str(_APP_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from backend.curve_fitting import *
from backend.functional_clustering import *

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
    
    with subtab1_2:
        if uploaded_files:
            with st.form(key="transform_form"):
                with st.expander("⚙️ Transform Settings", expanded=False):
                    scaler_type = st.selectbox("Transform Type", ["none", "rescale_to_0_1", "rescale_to_-1_1", "log1p"],key="transform_data")
                    submit_transform = st.form_submit_button("Run Transform")
            if submit_transform:
                for file in uploaded_files:
                    df_original = st.session_state.df_original[file.name]
                    df_transform = data_transformation(df_original, scaler_type)
                    st.session_state.df_transform[file.name] = df_transform
                st.success("Success")
            if st.session_state.df_transform:
                for file in uploaded_files:
                    if file.name in st.session_state.df_transform:
                        df_transform = st.session_state.df_transform[file.name]
                        with st.expander(f"Transform Data: {file.name}", expanded=False):
                            with st.expander("Data Overview", expanded=False):
                                st.dataframe(df_transform, use_container_width=True)
                            with st.expander("Descriptive Statistics", expanded=False):
                                st.dataframe(df_transform.describe(), use_container_width=True)
        else:
            st.info("Please upload CSV file(s)")


# ========== Tab 2 ==========
with tab2:
    st.write("待更新...")
