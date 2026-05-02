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


with st.sidebar:
    st.write("To Be Updated")
    # st.divider()

tab1, tab2, tab3 = st.tabs(["Uploaded Data", "Curve Fitting", "To Be Updated"])

with tab1:
    uploaded_files = st.file_uploader(
        label="Please upload your files",
        type=["csv"],
        accept_multiple_files=True,
        help="Supports CSV, multiple files allowed",
        label_visibility="visible",
        max_upload_size=500,
    )

    subtab1_1, subtab1_2, subtab1_3 = st.tabs(["Data Overview", "Data Transformation", "To Be Updated",])

    with subtab1_1:
        if uploaded_files:     
            for file in uploaded_files:

                df_original = load_csv(file)
                st.session_state.df_original[file.name] = df_original

                with st.expander(f"Original Data: {file.name}, Rows: {df_original.shape[0]}, Columns: {df_original.shape[1]}", expanded=False):
                    with st.expander("Original Data Overview", expanded=False):
                        st.dataframe(df_original, use_container_width=True)
                    with st.expander("Descriptive Statistics", expanded=False):
                        st.dataframe(df_original.describe(), use_container_width=True)
                    with st.expander("Scatter Plot", expanded=False):
                        st.write("To Be Updated")
        else:
            st.info("Please upload CSV file(s) to view data overview")

    
    with subtab1_2:
        if uploaded_files:
            with st.expander("⚙️ Transform Settings", expanded=False):
                scaler_type = st.selectbox("Transform Type", ["none", "rescale_to_0_1", "rescale_to_-1_1", "log1p"],key="transform_data")
            for file in uploaded_files:

                # df = st.session_state.original_data[file.name]
                df_transform = data_transformation(original_data, scaler_type)
                st.session_state.df_transform[file.name] = df_transform
            
                with st.expander(f"Transform Data: {file.name}, Rows: {df_transform.shape[0]}, Columns: {df_transform.shape[1]}", expanded=False):
                        with st.expander("Transform Data Overview", expanded=False):
                            st.dataframe(df_transform, use_container_width=True)
                        with st.expander("Descriptive Statistics", expanded=False):
                            st.dataframe(df_transform.describe(), use_container_width=True)
                        with st.expander("Scatter Plot", expanded=False):
                            st.write("To Be Updated")
        else:
            st.info("Please upload CSV file(s) to view data overview")


    with subtab1_3:
        st.write("To Be Updated")



with tab2:
    subtab2_1, subtab2_2, subtab2_3 = st.tabs([
        "Quasi Dynamic", 
        "异速生长拟合", 
        "To Be Updated",
    ])

    with subtab2_1:
        if uploaded_files:
            for 
                df_quasi_dynamic = get_quasi_dynamic_df(df_transform)

                # with st.expander("⚙️ Quasi Dynamic", expanded=False):
                # st.write("To Be Updated")

# #             transform_data_session = st.session_state.get("transform_data_session", {})

# #             for fname, df_transform in transform_data_session.items():
# #                     with st.expander(f"📄 Transformation Data: {file.name}", expanded=False):
# #                         df_quasi_dynamic = get_quasi_dynamic_df(df_transform)

# #                         with st.expander(f"📄 Transformation Data Overview: {file.name}", expanded=False):
# #                             st.dataframe(df_quasi_dynamic, use_container_width=True)
# #                             st.info(f"Rows: {df_quasi_dynamic.shape[0]} | Columns: {df_quasi_dynamic.shape[1]}")
                        
# #                         with st.expander("📊 Scatter Plot", expanded=False):
# #                             with st.expander("⚙️ Plot Settings", expanded=False):
# #                                 col1,col2,col3 = st.columns(3)
# #                                 use_seq = col1.checkbox("Use sequential X-axis", key=f"quasi_dynamic_data_seq_{fname}", value=False)
# #                                 n_cols = col2.selectbox("Subplots per row", [1,2,3,4,5,6], index=2, key=f"quasi_dynamic_data_col_{fname}")
# #                                 max_plots = col3.selectbox("Max plots", [3,6,9], index=1, key=f"quasi_dynamic_data_plots_{fname}")
# #                             plot_scatter_matrix(df_quasi_dynamic, use_seq, n_cols, max_plots)

# #     with subtab2_2:
# #         if uploaded_files:
# #             st.write("To Be Updated")

# #             curve_sample = get_power_function_sample(df_quasi_dynamic)
# #             st.dataframe(df_quasi_dynamic, use_container_width=True)
# #             st.dataframe(curve_sample, use_container_width=True)
# #             plot_curve_fitting(df_quasi_dynamic=df_quasi_dynamic, df_curve_sample=curve_sample, use_seq = None, n_cols=3, max_plots=6)
# # "Sequence"





