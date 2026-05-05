import streamlit as st

st.set_page_config(page_title="Curve Fitting", page_icon=None, layout="wide", initial_sidebar_state="auto")

import numpy as np
import pandas as pd

from backend.curve_fitting import (
    load_csv,
    data_transformation,
    get_quasi_dynamic_df,
    get_power_function_sample,
    get_power_function_params,
)
from backend.plot_curve_fitting import plot_curve_fitting, plot_curve_fitting_compare
from backend.utils import load_css

load_css()

st.title("Curve Fitting", text_alignment="center")


if "df_original" not in st.session_state:
    st.session_state.df_original = {}
if "df_transform" not in st.session_state:
    st.session_state.df_transform = {}
if "df_quasi_dynamic" not in st.session_state:
    st.session_state.df_quasi_dynamic = {}
if "df_curve_sample" not in st.session_state:
    st.session_state.df_curve_sample = {}
if "df_curve_params" not in st.session_state:
    st.session_state.df_curve_params = {}
if "original_plot_params" not in st.session_state:
    st.session_state.original_plot_params = {}
if "transform_plot_params" not in st.session_state:
    st.session_state.transform_plot_params = {}
if "show_original_data" not in st.session_state:
    st.session_state.show_original_data = {}
if "show_transform_data" not in st.session_state:
    st.session_state.show_transform_data = {}
if "quasi_show_data" not in st.session_state:
    st.session_state.quasi_show_data = {}
if "quasi_plot_params" not in st.session_state:
    st.session_state.quasi_plot_params = {}
if "allometric_show_data" not in st.session_state:
    st.session_state.allometric_show_data = {}
if "allometric_plot_params" not in st.session_state:
    st.session_state.allometric_plot_params = {}
if "allometric_show_curve_params" not in st.session_state:
    st.session_state.allometric_show_curve_params = {}
if "allometric_compare_params" not in st.session_state:
    st.session_state.allometric_compare_params = None


with st.sidebar:
    st.write("To Be Updated...")
    # st.divider()

tab1, tab2, tab3 = st.tabs(["Uploaded Data", "Static Data", "Dynamic Data"])

# 上传数据
with tab1:
    uploaded_files = st.file_uploader(label="Please upload your files",type=["csv"],accept_multiple_files=True)
    
    tab1_1, tab1_2, tab1_3 = st.tabs(["Data Overview", "Data Transformation", "To Be Updated"])
    
    # 数据概览
    with tab1_1:
        if uploaded_files:
            for file in uploaded_files:
                
                # 加载并缓存原始数据 df_original
                df_original = load_csv(file=file)
                st.session_state.df_original[file.name] = df_original
                
                with st.expander(f"Original Data: {file.name}", expanded=False):
                    tab1_1_1, tab1_1_2, tab1_1_3 = st.tabs(["View Data", "Scatter Plot", "To Be Updated..."])

                    # 查看数据
                    with tab1_1_1:
                        with st.form(key=f"original_data_view_{file.name}"):
                            submit_original_data_view = st.form_submit_button("View Data", help="May lag for very large datasets")
                            if submit_original_data_view:
                                st.session_state.show_original_data[file.name] = True
                        if st.session_state.show_original_data.get(file.name):
                            st.dataframe(df_original, use_container_width=True)
                    
                    # 绘制原始数据散点图
                    with tab1_1_2:
                        # 原始数据散点图绘图参数
                        with st.form(key=f"original_data_plot_{file.name}"):
                            with st.expander("⚙️ Original Data Plot Settings", expanded=False):
                                tab_layout, tab_color = st.tabs(["Layout", "Color"])
                                # 布局
                                with tab_layout:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        nrow = st.number_input("Rows", value=4, min_value=1, max_value=10, step=1)
                                        ncol = st.number_input("Cols", value=3, min_value=1, max_value=10, step=1)
                                    with col2:
                                        plot_scatter_type = st.selectbox("Plot Type", ["line","scatter"], key=f"plot_scatter_type_{file.name}")
                                    with col3:
                                        scatter_size = st.number_input("Scatter Size", value=10, min_value=1, max_value=1000, step=1)
                                        scatter_linewidth = st.number_input("Line Size", value=1, min_value=1, max_value=10, step=1)
                                   
                                    nsubfig = nrow * ncol
                                # 颜色
                                with tab_color:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        color_scatter = st.color_picker("Data Color", value="#FF7678")
                                        # color_curve = st.color_picker("Curve Color", value="#A06EA5")
                                    with col2:
                                        subfig_background_color = st.color_picker("Subfig Background Color", value="#FFFFFF")
  
                            submit_original_data_plot = st.form_submit_button("Run Original Data Plot")
                            # 原始数据绘图区
                            if submit_original_data_plot:
                                st.session_state.original_plot_params[file.name] = dict(
                                    plot_scatter_type=plot_scatter_type,
                                    show_curve=False,
                                    nrow=nrow,
                                    ncol=ncol,
                                    nsubfig=nsubfig,
                                    scatter_size=scatter_size,
                                    scatter_x="sequence",
                                    scatter_linewidth=scatter_linewidth,
                                    color_scatter=color_scatter,
                                    subfig_background_color=subfig_background_color,
                                )

                        if file.name in st.session_state.original_plot_params:
                            plot_curve_fitting(
                                df_scatter=df_original,
                                df_curve=None,
                                **st.session_state.original_plot_params[file.name],
                            )
        else:
            st.info("Please upload CSV file(s)")
 
    # 数据变换
    with tab1_2:
        if uploaded_files:
            with st.form(key="transform_form"):
                scaler_type = st.selectbox("Transform Type", ["none", "rescale_to_0_1", "rescale_to_-1_1", "log1p"],key="transform_data")
                submit_transform = st.form_submit_button("Run Transform")
            # 执行数据变换
            if submit_transform:
                for file in uploaded_files:
                    df_original = st.session_state.df_original[file.name]
                    # 加载并缓存数据变换结果 df_transform
                    df_transform = data_transformation(df_original, scaler_type)
                    st.session_state.df_transform[file.name] = df_transform
                st.success("Success")

            if uploaded_files:
                for file in uploaded_files:
                    if file.name in st.session_state.df_transform:
                        df_transform = st.session_state.df_transform[file.name]
                        with st.expander(f"Transform Data: {file.name}", expanded=False):
                            tab1_2_1, tab1_2_2, tab1_2_3 = st.tabs(["Data Overview", "Scatter Plot", "To Be Updated..."])
                            # 查看数据
                            with tab1_2_1:
                                if st.button("View Data", key=f"view_transformed_data_{file.name}"):
                                    st.session_state.show_transform_data[file.name] = True
                                if st.session_state.show_transform_data.get(file.name):
                                    st.dataframe(df_transform, use_container_width=True)
                            # 绘制数据散点图
                            with tab1_2_2:
                                with st.form(key=f"transform_data_plot_{file.name}"):
                                    with st.expander("⚙️ Transform Data Plot Settings", expanded=False):
                                        tab_layout, tab_color = st.tabs(["Layout", "Color"])
                                        # 布局
                                        with tab_layout:
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                nrow = st.number_input("Rows", value=4, min_value=1, max_value=10, step=1)
                                                ncol = st.number_input("Cols", value=3, min_value=1, max_value=10, step=1)
                                            with col2:
                                                plot_scatter_type = st.selectbox("Plot Type", ["line","scatter"], key=f"plot_transform_scatter_type_{file.name}")
                                            with col3:
                                                scatter_size = st.number_input("Scatter Size", value=100, min_value=1, max_value=1000, step=1)
                                                scatter_linewidth = st.number_input("Line Size", value=1, min_value=1, max_value=10, step=1)
                                        with tab_color:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                color_scatter = st.color_picker("Data Color", value="#FF7678")
                                                # color_curve = st.color_picker("Curve Color", value="#A06EA5")
                                            with col2:
                                                subfig_background_color = st.color_picker("Subfig Background Color", value="#FFFFFF")
                                    submit_transform_data_plot = st.form_submit_button("Run Transform Data Plot")
                                    if submit_transform_data_plot:
                                        nsubfig = nrow * ncol
                                        st.session_state.transform_plot_params[file.name] = dict(
                                            plot_scatter_type=plot_scatter_type,
                                            show_curve=False,
                                            nrow=nrow,
                                            ncol=ncol,
                                            nsubfig=nsubfig,
                                            scatter_size=scatter_size,
                                            scatter_linewidth=scatter_linewidth,
                                            color_scatter=color_scatter,
                                            subfig_background_color=subfig_background_color,
                                        )

                                if file.name in st.session_state.transform_plot_params:
                                    plot_curve_fitting(
                                        df_scatter=df_transform,
                                        df_curve=None,
                                        **st.session_state.transform_plot_params[file.name],
                                    )
        else:
            st.info("Please upload CSV file(s)")

    
    with tab1_3:
        st.write("To Be Updated...")


# 静态数据
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
                        # 加载并缓存 quasi-dynamic DataFrame df_quasi_dynamic
                        df_transform = st.session_state.df_transform[file.name]
                        df_quasi_dynamic = get_quasi_dynamic_df(df_transform)
                        st.session_state.df_quasi_dynamic[file.name] = df_quasi_dynamic
                st.success("Success")
            if st.session_state.df_quasi_dynamic:
                for file in uploaded_files:
                    if file.name in st.session_state.df_quasi_dynamic:
                        # 加载并缓存 quasi-dynamic DataFrame df_quasi
                        df_quasi = st.session_state.df_quasi_dynamic[file.name]
                        with st.expander(f"Quasi Dynamic: {file.name}", expanded=False):
                            tab_qd_data, tab_qd_plot = st.tabs([
                                "Data Overview", "Scatter Plot"
                            ])
                            with tab_qd_data:
                                if st.button("View Data", key=f"quasi_view_{file.name}"):
                                    st.session_state.quasi_show_data[file.name] = True
                                if st.session_state.quasi_show_data.get(file.name):
                                    st.dataframe(df_quasi, use_container_width=True)
                            with tab_qd_plot:
                                with st.form(key=f"quasi_plot_{file.name}"):
                                    with st.expander("⚙️ Scatter Plot Settings", expanded=False):
                                        tab_layout, tab_color = st.tabs(["Layout", "Color"])
                                        with tab_layout:
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                nrow = st.number_input("Rows", value=2, min_value=1, max_value=10, step=1, key=f"quasi_nrow_{file.name}")
                                                ncol = st.number_input("Cols", value=3, min_value=1, max_value=10, step=1, key=f"quasi_ncol_{file.name}")
                                            with col2:
                                                plot_scatter_type = st.selectbox("Plot Type", ["scatter", "line"], key=f"quasi_scatter_type_{file.name}")
                                            with col3:
                                                scatter_size = st.number_input("Scatter Size", value=100, min_value=1, max_value=1000, step=1, key=f"quasi_scatter_size_{file.name}")
                                                scatter_linewidth = st.number_input("Line Size", value=1, min_value=1, max_value=10, step=1, key=f"quasi_scatter_lw_{file.name}")
                                        with tab_color:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                color_scatter = st.color_picker("Data Color", value="#F9B3AD", key=f"quasi_color_{file.name}")
                                            with col2:
                                                subfig_bg = st.color_picker("Subfig Background Color", value="#FFFFFF", key=f"quasi_bg_{file.name}")
                                    submit_quasi_plot = st.form_submit_button("Run Scatter Plot")
                                    if submit_quasi_plot:
                                        st.session_state.quasi_plot_params[file.name] = dict(
                                            plot_scatter_type=plot_scatter_type,
                                            scatter_x = "index",
                                            show_curve=False,
                                            nrow=nrow,
                                            ncol=ncol,
                                            nsubfig=nrow * ncol,
                                            scatter_size=scatter_size,
                                            scatter_linewidth=scatter_linewidth,
                                            color_scatter=color_scatter,
                                            subfig_background_color=subfig_bg,
                                        )
                                if st.session_state.quasi_plot_params.get(file.name):
                                    plot_curve_fitting(
                                        df_scatter=df_quasi,
                                        df_curve=None,
                                        **st.session_state.quasi_plot_params[file.name],
                                    )
        else:
            st.info("Please upload CSV file(s)")

    # Allometric Scaling Law 
    with subtab2_2:
        if uploaded_files:
            with st.form(key="Allometric Scaling Law"):
                submit_fit = st.form_submit_button("Run Allometric Scaling Law")
            if submit_fit:
                for file in uploaded_files:
                    if file.name in st.session_state.df_quasi_dynamic:
                        df_quasi_dynamic = st.session_state.df_quasi_dynamic[file.name]
                        # 加载并缓存幂函数拟合曲线采样值 df_curve_sample
                        df_curve_sample = get_power_function_sample(df_quasi_dynamic)
                        st.session_state.df_curve_sample[file.name] = df_curve_sample
                        # 加载并缓存幂函数拟合曲线参数 df_curve_params
                        df_curve_params = get_power_function_params(df_quasi_dynamic)
                        st.session_state.df_curve_params[file.name] = df_curve_params
                st.success("Success")
            if st.session_state.df_curve_sample:
                for file in uploaded_files:
                    if file.name in st.session_state.df_curve_sample:
                        df_quasi = st.session_state.df_quasi_dynamic[file.name]
                        df_curve = st.session_state.df_curve_sample[file.name]
                        with st.expander(f"Allometric Scaling Law: {file.name}", expanded=False):
                            tab_al_data, tab_al_plot, tab_al_params = st.tabs([
                                "Data Overview", "Curve Fitting Plot","Curve Parameters"
                            ])
                            with tab_al_data:
                                if st.button("View Data", key=f"allometric_view_{file.name}"):
                                    st.session_state.allometric_show_data[file.name] = True
                                if st.session_state.allometric_show_data.get(file.name):
                                    st.dataframe(df_quasi, use_container_width=True)
                            with tab_al_plot:
                                with st.form(key=f"allometric_plot_{file.name}"):
                                    with st.expander("⚙️ Curve Fitting Plot Settings", expanded=False):
                                        tab_layout, tab_color = st.tabs(["Layout", "Color"])
                                        with tab_layout:
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                nrow = st.number_input("Rows", value=2, min_value=1, max_value=10, step=1, key=f"allometric_nrow_{file.name}")
                                                ncol = st.number_input("Cols", value=3, min_value=1, max_value=10, step=1, key=f"allometric_ncol_{file.name}")
                                            with col2:
                                                plot_scatter_type = st.selectbox("Plot Type", ["scatter", "line"], key=f"allometric_scatter_type_{file.name}")
                                            with col3:
                                                scatter_size = st.number_input("Scatter Size", value=100, min_value=1, max_value=1000, step=1, key=f"allometric_scatter_size_{file.name}")
                                                scatter_linewidth = st.number_input("Line Size", value=1, min_value=1, max_value=10, step=1, key=f"allometric_scatter_lw_{file.name}")
                                        # 幂函数拟合图配色
                                        with tab_color:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                color_scatter = st.color_picker("Data Color", value="#F9B3AD", key=f"allometric_color_{file.name}")
                                                color_curve = st.color_picker("Curve Color", value="#F9B3AD", key=f"allometric_color_curve_{file.name}")
                                            with col2:
                                                subfig_bg = st.color_picker("Subfig Background Color", value="#FFFFFF", key=f"allometric_bg_{file.name}")
                                    submit_allometric_plot = st.form_submit_button("Run Curve Fitting Plot")
                                    if submit_allometric_plot:
                                        st.session_state.allometric_plot_params[file.name] = dict(
                                            plot_scatter_type=plot_scatter_type,
                                            show_curve=True,
                                            nrow=nrow,
                                            ncol=ncol,
                                            nsubfig=nrow * ncol,
                                            scatter_size=scatter_size,
                                            scatter_linewidth=scatter_linewidth,
                                            scatter_x = "index",
                                            color_scatter=color_scatter,
                                            color_curve=color_curve,
                                            subfig_background_color=subfig_bg,
                                        )
                                if st.session_state.allometric_plot_params.get(file.name):
                                    plot_curve_fitting(
                                        df_scatter=df_quasi,
                                        df_curve=df_curve,
                                        **st.session_state.allometric_plot_params[file.name],
                                    )
                            # 曲线参数
                            with tab_al_params:
                                if st.button("View Curve Parameters", key=f"allometric_params_view_{file.name}"):
                                    st.session_state.allometric_show_curve_params[file.name] = (
                                        not st.session_state.allometric_show_curve_params.get(file.name, False)
                                    )
                                if st.session_state.allometric_show_curve_params.get(file.name):
                                    st.dataframe(st.session_state.df_curve_params[file.name], use_container_width=True)
                with st.expander("Allometric Scaling Law Compare", expanded=False):
                    scatter_list_compare = [
                        st.session_state.df_quasi_dynamic[f.name]
                        for f in uploaded_files
                        if f.name in st.session_state.df_quasi_dynamic
                    ]
                    curve_list_compare = [
                        st.session_state.df_curve_sample[f.name]
                        for f in uploaded_files
                        if f.name in st.session_state.df_quasi_dynamic
                    ]
                    name_list_compare = [
                        f.name
                        for f in uploaded_files
                        if f.name in st.session_state.df_quasi_dynamic
                    ]
                    if scatter_list_compare:
                        with st.form(key="allometric_compare_plot"):
                            with st.expander("⚙️ Compare Plot Settings", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    nrow_cmp = st.number_input("Rows", value=2, min_value=1, max_value=10, step=1, key="compare_nrow")
                                    ncol_cmp = st.number_input("Cols", value=3, min_value=1, max_value=10, step=1, key="compare_ncol")
                            submit_compare_plot = st.form_submit_button("Run Compare Plot")
                            if submit_compare_plot:
                                st.session_state.allometric_compare_params = dict(
                                    show_curve=True,
                                    nrow=nrow_cmp,
                                    ncol=ncol_cmp,
                                    nsubfig=nrow_cmp * ncol_cmp,
                                )
                        if st.session_state.allometric_compare_params:
                            plot_curve_fitting_compare(
                                df_scatter_list=scatter_list_compare,
                                df_curve_list=curve_list_compare,
                                label_list=name_list_compare,
                                **st.session_state.allometric_compare_params,
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






