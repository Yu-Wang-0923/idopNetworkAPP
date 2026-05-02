import sys
import numpy as np
import pandas as pd
import streamlit as st
sys.path.append("..")
from backend.curve_fitting import *
from backend.plot_curve_fitting import *


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
            with st.expander("⚙️ Transform Settings", expanded=False):
                scaler_type = st.selectbox("Transform Type", ["none", "rescale_to_0_1", "rescale_to_-1_1", "log1p"],key="transform_data")
            for file in uploaded_files:
                df_original = st.session_state.df_original[file.name]
                df_transform = data_transformation(df_original, scaler_type)
                st.session_state.df_transform[file.name] = df_transform
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
        "To Be Updated...",
    ])

    with subtab2_1:
        if uploaded_files:
            for file in uploaded_files:
                df_transform = st.session_state.df_transform[file.name]
                df_quasi_dynamic = get_quasi_dynamic_df(df_transform)
                st.session_state.df_quasi_dynamic[file.name] = df_quasi_dynamic
                with st.expander(f"Quasi Dynamic: {file.name}", expanded=False):
                    with st.expander("Data Overview", expanded=False):
                        st.dataframe(df_quasi_dynamic, use_container_width=True)
                    with st.expander("Descriptive Statistics", expanded=False):
                        st.dataframe(df_quasi_dynamic.describe(), use_container_width=True)
                    with st.expander("Scatter Plot", expanded=False):
                        plot_curve_fitting(
                                    df_scatter=df_quasi_dynamic,
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
            scatter_list = []
            curve_list = []
            name_list = []
            for file in uploaded_files:
                df_quasi_dynamic = st.session_state.df_quasi_dynamic[file.name]        
                df_curve_sample = get_power_function_sample(df_quasi_dynamic)
                st.session_state.df_curve_sample[file.name] = df_curve_sample

                scatter_list.append(df_quasi_dynamic)
                curve_list.append(df_curve_sample)
                name_list.append(file.name)  # 图例用

                with st.expander(f"Allometric Scaling Law: {file.name}", expanded=False):
                    with st.expander("Data Overview", expanded=False):
                        st.dataframe(df_quasi_dynamic, use_container_width=True)
                    with st.expander("Descriptive Statistics", expanded=False):
                        st.dataframe(df_quasi_dynamic.describe(), use_container_width=True)
                    with st.expander("Curve Fitting Plot", expanded=False):
                        plot_curve_fitting(
                                    df_scatter=df_quasi_dynamic,
                                    df_curve=df_curve_sample,
                                    show_curve=True,
                                    nrow=2,
                                    ncol=3,
                                    nsubfig=6,
                                    )
            with st.expander("Allometric Scaling Law Compare", expanded=False):                        
                plot_curve_fitting_compare(
                                            df_scatter_list=scatter_list,
                                            df_curve_list=curve_list,
                                            label_list=name_list,
                                            show_curve=True,
                                            nrow=2,
                                            ncol=3,
                                            nsubfig=6
                                            )
        else:
            st.info("Please upload CSV file(s)")

    with subtab2_3:
        st.write("To Be Updated...")

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






