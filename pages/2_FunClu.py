import streamlit as st

st.set_page_config(page_title="Functional Clustering", page_icon=None, layout="wide", initial_sidebar_state="auto")

import numpy as np
import pandas as pd

from backend.curve_fitting import load_csv, data_transformation
from backend import functional_clustering  # noqa: F401  — 待补充具名导入

st.title("Functional Clustering", text_alignment="center")


if "df_original" not in st.session_state:
    st.session_state.df_original = {}
if "df_transform" not in st.session_state:
    st.session_state.df_transform = {}
if "df_quasi_dynamic" not in st.session_state:
    st.session_state.df_quasi_dynamic = {}
if "df_curve_sample" not in st.session_state:
    st.session_state.df_curve_sample = {}

uploaded_file = st.file_uploader(
        label="Please upload your file",
        type=["zip"],
        accept_multiple_files=False,
        help=(
            "Upload the ZIP from Curve Fitting: Export Result → "
            "Export ZIP → Download curve_fitting_export.zip (one file)."
        ),
        label_visibility="visible",
        max_upload_size=500,
    )

# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["FunClu-K", "FunClu-BIC", "To Be Updated..."])

# ========== Tab 1 ==========
with tab1:
    st.write("To Be Updated...")



# ========== Tab 2 ==========
with tab2:
    st.write("To Be Updated...")
