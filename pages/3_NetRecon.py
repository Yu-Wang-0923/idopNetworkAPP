import io
import zipfile

import pandas as pd
import streamlit as st

from backend.curve_fitting import data_transformation
from backend.network_construction import IDOPRegressor
from backend.plot_curve_fitting import plot_curve_fitting
from backend.plot_network_construction import plot_effect, plot_network
from backend.utils import load_css

st.set_page_config(
    page_title="Network Construction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

load_css()
st.title("Network Construction", text_alignment="center")


def _load_netrecon_inputs_from_zip(
    zip_bytes: bytes,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """从 curve_fitting_export.zip 读取 quasi_dynamic 与 curve_sample。"""
    quasi_map: dict[str, pd.DataFrame] = {}
    curve_map: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("quasi_dynamic.csv"):
                cond = name.rsplit("/", 1)[0] if "/" in name else name
                with zf.open(name) as f:
                    quasi_map[cond] = pd.read_csv(f, index_col=0)
            elif name.endswith("curve_sample.csv"):
                cond = name.rsplit("/", 1)[0] if "/" in name else name
                with zf.open(name) as f:
                    curve_map[cond] = pd.read_csv(f, index_col=0)
    return dict(sorted(quasi_map.items())), dict(sorted(curve_map.items()))


if "netrecon_quasi_dynamic" not in st.session_state:
    st.session_state.netrecon_quasi_dynamic = {}
if "netrecon_curve_sample" not in st.session_state:
    st.session_state.netrecon_curve_sample = {}
if "netrecon_uploaded_zip_name" not in st.session_state:
    st.session_state.netrecon_uploaded_zip_name = None
if "netrecon_result" not in st.session_state:
    st.session_state.netrecon_result = None

uploaded_zip = st.file_uploader(
    label="Please upload curve_fitting_export.zip",
    type=["zip"],
    accept_multiple_files=False,
    help="Input comes directly from Curve Fitting -> Export ZIP",
)

if uploaded_zip is not None and st.session_state.netrecon_uploaded_zip_name != uploaded_zip.name:
    try:
        quasi_map, curve_map = _load_netrecon_inputs_from_zip(uploaded_zip.getvalue())
        st.session_state.netrecon_quasi_dynamic = quasi_map
        st.session_state.netrecon_curve_sample = curve_map
        st.session_state.netrecon_uploaded_zip_name = uploaded_zip.name
        st.session_state.netrecon_result = None
    except Exception as e:
        st.error(f"读取 ZIP 失败：{e}")
        st.session_state.netrecon_quasi_dynamic = {}
        st.session_state.netrecon_curve_sample = {}
        st.session_state.netrecon_uploaded_zip_name = None
        st.session_state.netrecon_result = None


tab1, tab2, tab3 = st.tabs(["IdopNetwork", "Multi-Layer IdopNetwork", "To Be Updated..."])

with tab1:
    quasi_map = st.session_state.netrecon_quasi_dynamic
    curve_map = st.session_state.netrecon_curve_sample
    common_conditions = [k for k in quasi_map if k in curve_map]

    if not common_conditions:
        st.info("Please upload curve_fitting_export.zip first.")
    else:
        default_condition = common_conditions[0]
        selected_condition = st.selectbox(
            "Condition (single-layer default picks the first one)",
            options=common_conditions,
            index=0,
        )

        quasi_dynamic_df = quasi_map[selected_condition]
        curve_sample_df = curve_map[selected_condition]

        st.markdown(f"当前 Condition: `{selected_condition}`")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**quasi_dynamic preview**")
            st.dataframe(quasi_dynamic_df.head(), use_container_width=True)
        with col2:
            st.markdown("**curve_sample preview**")
            st.dataframe(curve_sample_df.head(), use_container_width=True)

        with st.expander("IdopNetwork parameter settings", expanded=True):
            with st.form(key="netrecon_form_single"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    solver = st.selectbox(
                        "Solver",
                        options=["ols", "ridge", "lasso", "asgl", "adsiht"],
                        index=0,
                    )
                    max_order = st.number_input(
                        "max_order",
                        min_value=1,
                        max_value=100,
                        value=6,
                        step=1,
                    )
                with c2:
                    alpha = st.number_input(
                        "alpha",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        help="ridge/lasso/asgl/adsiht regularization strength",
                    )
                    mix = st.slider(
                        "mix (ASGL only)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                    )
                with c3:
                    nonneg_self = st.checkbox("nonneg_self", value=True)
                    top_k = st.number_input(
                        "max_interactions (Top-K)",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                    )
                submit_run = st.form_submit_button("Run IdopNetwork")

        if submit_run:
            try:
                # Legendre 基要求自变量 ∈ [-1, 1]，curve_sample 通常超出该域，
                # 必须先做 rescale_to_-1_1 才能保证数值稳定
                curve_sample_scaled = data_transformation(
                    curve_sample_df, "rescale_to_-1_1"
                )
                model = IDOPRegressor(
                    max_order=int(max_order),
                    solver=str(solver),
                    alpha=float(alpha),
                    mix=float(mix),
                    nonneg_self=bool(nonneg_self),
                    max_interactions=int(top_k),
                )
                model.fit(curve_sample_scaled, quasi_dynamic_df)
                predicted_df = model.predict(curve_sample_scaled)
                effect_df_list = model.effect(curve_sample_scaled)
                adj_df = model.adjacency_matrix(curve_sample_scaled)
            except Exception as e:
                st.error(f"IdopNetwork 运行失败：{e}")
                st.session_state.netrecon_result = None
            else:
                st.session_state.netrecon_result = {
                    "condition": selected_condition,
                    "model": model,
                    "quasi_dynamic_df": quasi_dynamic_df,
                    "curve_sample_df": curve_sample_df,
                    "predicted_df": predicted_df,
                    "effect_df_list": effect_df_list,
                    "adj_df": adj_df,
                }
                st.success("Done")

        result = st.session_state.netrecon_result
        if result is not None:
            st.markdown("### Fitting vs Prediction")
            plot_curve_fitting(
                df_scatter=result["quasi_dynamic_df"],
                df_curve=result["predicted_df"],
                plot_scatter_type="line",
                show_curve=True,
                nrow=2,
                ncol=3,
                nsubfig=6,
                scatter_x="index",
                scatter_size=30,
                scatter_linewidth=1,
            )

            st.markdown("### Effect Decomposition")
            plot_effect(
                quasi_dynamic_df=result["quasi_dynamic_df"],
                curve_df=result["predicted_df"],
                effect_df_list=result["effect_df_list"],
                intercept=result["model"].coef_.loc["intercept"],
                plot_ncols=4,
            )

            st.markdown("### Adjacency Matrix")
            st.dataframe(result["adj_df"], use_container_width=True)

            st.markdown("### Interaction Network")
            target_node = st.selectbox(
                "Target node filter",
                options=[""] + list(result["adj_df"].index),
                format_func=lambda x: "ALL" if x == "" else x,
                key="netrecon_target_node",
            )
            plot_network(result["adj_df"], target_node=target_node)

with tab2:
    st.write("To Be Updated...")
