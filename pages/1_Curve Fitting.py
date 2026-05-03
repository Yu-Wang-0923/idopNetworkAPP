import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_ROOT = Path(__file__).resolve().parent.parent
_root = str(_APP_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.curve_fitting import *
from backend.plot_curve_fitting import *

# 添加自定义CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("static/css/custom_style.css")


st.set_page_config(page_title="Curve Fitting", page_icon=None, layout="wide", initial_sidebar_state="auto")
st.title("Curve Fitting", text_alignment="center")


if "df_original" not in st.session_state:
    st.session_state.df_original = {}
if "df_transform" not in st.session_state:
    st.session_state.df_transform = {}
if "df_quasi_dynamic" not in st.session_state:
    st.session_state.df_quasi_dynamic = {}
if "df_curve_sample" not in st.session_state:
    st.session_state.df_curve_sample = {}


with st.sidebar:
    st.write("To Be Updated...")
    # st.divider()

tab1, tab2, tab3 = st.tabs(["Uploaded Data", "Static Data", "Dynamic Data"])

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
                    with st.expander("Scatter Plot", expanded=False):
                        plot_curve_fitting(
                                    df_scatter=df_original,
                                    df_curve=None,
                                    show_curve=False,
                                    nrow=2,
                                    ncol=3,
                                    nsubfig=6,  # 最多画4张图
                                    )
                
                
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
                            with st.expander("Scatter Plot", expanded=False):
                                plot_curve_fitting(
                                            df_scatter=df_transform,
                                            df_curve=None,
                                            show_curve=False,
                                            nrow=2,
                                            ncol=3,
                                            nsubfig=6,
                                            )
        else:
            st.info("Please upload CSV file(s)")


    with subtab1_3:
        st.write("To Be Updated...")



with tab2:
    subtab2_1, subtab2_2, subtab2_3 = st.tabs([
        "Quasi Dynamic", 
        "Allometric Scaling Law", 
        "Export",
    ])

    with subtab2_1:
        if uploaded_files:
            with st.form(key="quasi_dynamic_form"):
                submit_quasi = st.form_submit_button("Run Quasi Dynamic")
            if submit_quasi:
                for file in uploaded_files:
                    if file.name in st.session_state.df_transform:
                        df_transform = st.session_state.df_transform[file.name]
                        df_quasi_dynamic = get_quasi_dynamic_df(df_transform)
                        st.session_state.df_quasi_dynamic[file.name] = df_quasi_dynamic
                st.success("Success")
            if st.session_state.df_quasi_dynamic:
                for file in uploaded_files:
                    if file.name in st.session_state.df_quasi_dynamic:
                        df_quasi = st.session_state.df_quasi_dynamic[file.name]
                        with st.expander(f"Quasi Dynamic: {file.name}", expanded=False):
                            with st.expander("Data Overview", expanded=False):
                                st.dataframe(df_quasi, use_container_width=True)
                            with st.expander("Descriptive Statistics", expanded=False):
                                st.dataframe(df_quasi.describe(), use_container_width=True)
                            with st.expander("Scatter Plot", expanded=False):
                                plot_curve_fitting(
                                            df_scatter=df_quasi,
                                            df_curve=None,
                                            show_curve=False,
                                            nrow=2,
                                            ncol=3,
                                            nsubfig=6,
                                            )
        else:
            st.info("Please upload CSV file(s)")


    with subtab2_2:
        if uploaded_files:
            with st.form(key="Allometric Scaling Law"):
                submit_fit = st.form_submit_button("Run Allometric Scaling Law")
            if submit_fit:
                scatter_list = []
                curve_list = []
                name_list = []
                
                for file in uploaded_files:
                    if file.name in st.session_state.df_quasi_dynamic:
                        df_quasi_dynamic = st.session_state.df_quasi_dynamic[file.name]        
                        df_curve_sample = get_power_function_sample(df_quasi_dynamic)
                        st.session_state.df_curve_sample[file.name] = df_curve_sample

                        scatter_list.append(df_quasi_dynamic)
                        curve_list.append(df_curve_sample)
                        name_list.append(file.name)
                st.success("Success")
            if st.session_state.df_curve_sample:
                for file in uploaded_files:
                    if file.name in st.session_state.df_curve_sample:
                        df_quasi = st.session_state.df_quasi_dynamic[file.name]
                        df_curve = st.session_state.df_curve_sample[file.name]
                        with st.expander(f"Allometric Scaling Law: {file.name}", expanded=False):
                            with st.expander("Data Overview", expanded=False):
                                st.dataframe(df_quasi, use_container_width=True)
                            with st.expander("Descriptive Statistics", expanded=False):
                                st.dataframe(df_quasi.describe(), use_container_width=True)
                            with st.expander("Curve Fitting Plot", expanded=False):
                                plot_curve_fitting(
                                            df_scatter=df_quasi,
                                            df_curve=df_curve,
                                            show_curve=True,
                                            nrow=2,
                                            ncol=3,
                                            nsubfig=6,
                                            )
                with st.expander("Allometric Scaling Law Compare", expanded=False):
                    scatter_list_compare = []
                    curve_list_compare = []
                    name_list_compare = []

                    for file in uploaded_files:
                        if file.name in st.session_state.df_quasi_dynamic:
                            scatter_list_compare.append(st.session_state.df_quasi_dynamic[file.name])
                            curve_list_compare.append(st.session_state.df_curve_sample[file.name])
                            name_list_compare.append(file.name)
                    if scatter_list_compare:
                        plot_curve_fitting_compare(
                            df_scatter_list=scatter_list_compare,
                            df_curve_list=curve_list_compare,
                            label_list=name_list_compare,
                            show_curve=True,
                            nrow=2,
                            ncol=3,
                            nsubfig=6
                            )
        else:
            st.info("Please upload CSV file(s)")

    with subtab2_3:
        st.write("Export Result")
        if not st.session_state.df_original:
            st.info("Please upload CSV file(s) first.")
        else:
            for fname, df in st.session_state.df_original.items():
                st.download_button(
                    label=f"Download original: {fname}",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                    key=f"export_original_{fname}",
                )


with tab3:
    st.write("To Be Updated")
    subtab3_1, subtab3_2, subtab3_3, subtab3_4, subtab3_5 = st.tabs([
        "Polynomial Fitting", 
        "Logistic Growth Fitting", 
        "Fourier Series Fitting",
        "Wavelet Fitting",
        "To Be Updated...",
    ])

    with subtab3_1:
        st.write("To Be Updated...")
    
    with subtab3_2:
        st.write("To Be Updated...")

    with subtab3_3:
        st.write("To Be Updated...")

    with subtab3_4:
        st.write("To Be Updated...")

    with subtab3_5:
        st.write("To Be Updated...")






