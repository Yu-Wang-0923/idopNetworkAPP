import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

# 🌟 修改 1：统一小图标和侧边栏状态（建议紧跟在 import st 后面）
st.set_page_config(
    page_title="Network Construction",
    page_icon="TSA.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backend.curve_fitting import data_transformation
from backend.network_construction import IDOPRegressor
from backend.plot_curve_fitting import plot_curve_fitting
from backend.plot_network_construction import plot_effect, plot_network

# 🌟 修改 2：导入 setup_sidebar
from backend.utils import load_css, setup_sidebar

# 🌟 修改 3：一键加载
load_css()
setup_sidebar()

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
                    basis_kind = st.selectbox(
                        "basis_kind",
                        options=["legendre", "laguerre", "bspline"],
                        index=0,
                        help=(
                            "Basis family for per-feature expansion. "
                            "legendre/laguerre: orthogonal polynomials (laguerre is "
                            "internally remapped to [0, max_order]). "
                            "bspline: clamped uniform B-spline with `bspline_degree` "
                            "knots, column count still equals max_order."
                        ),
                    )
                    bspline_degree = st.number_input(
                        "bspline_degree",
                        min_value=1,
                        max_value=10,
                        value=3,
                        step=1,
                        help=(
                            "B-spline degree (only used when basis_kind == 'bspline'). "
                            "Auto-clamped to max_order - 1 if too large."
                        ),
                        disabled=(basis_kind != "bspline"),
                    )
                with c2:
                    alpha = st.number_input(
                        "alpha",
                        min_value=0.0,
                        value=0.001,
                        step=0.001,
                        format="%.4f",
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
                    basis_kind=str(basis_kind),
                    bspline_degree=int(bspline_degree),
                )
                # 响应 Y 必须与设计矩阵 X 共享同一 index（curve_sample 的 Chebyshev 节点）
                # quasi_dynamic 的 τ 索引与 curve_sample 不同域，仅用于后续可视化散点
                model.fit(curve_sample_scaled, curve_sample_df)
                predicted_df = model.predict(curve_sample_scaled)
                effect_df_list = model.effect(curve_sample_scaled)
                adj_df = model.adjacency_matrix(curve_sample_scaled)
                design_X = model._design(curve_sample_scaled)
                response_Y = curve_sample_df.reindex(design_X.index)
            except Exception as e:
                st.error(f"IdopNetwork 运行失败：{e}")
                st.session_state.netrecon_result = None
            else:
                st.session_state.netrecon_result = {
                    "condition": selected_condition,
                    "model": model,
                    "quasi_dynamic_df": quasi_dynamic_df,
                    "curve_sample_df": curve_sample_df,
                    "curve_sample_scaled": curve_sample_scaled,
                    "design_X": design_X,
                    "response_Y": response_Y,
                    "predicted_df": predicted_df,
                    "effect_df_list": effect_df_list,
                    "adj_df": adj_df,
                }
                st.success("Done")

        result = st.session_state.netrecon_result
        if result is not None:
            with st.expander("Debug: design matrix, response, coefficients", expanded=True):
                model_dbg: IDOPRegressor = result["model"]
                X_dbg: pd.DataFrame = result["design_X"]
                Y_dbg: pd.DataFrame = result["response_Y"]
                cs_raw = result["curve_sample_df"]
                cs_scl = result["curve_sample_scaled"]
                pred = result["predicted_df"]
                coef = model_dbg.coef_

                def _summary(name: str, df: pd.DataFrame) -> dict:
                    arr = df.to_numpy(dtype=float, copy=False)
                    finite = np.isfinite(arr)
                    finite_arr = arr[finite] if finite.any() else np.array([np.nan])
                    return {
                        "name": name,
                        "shape": str(arr.shape),
                        "n_nan": int(np.isnan(arr).sum()),
                        "n_inf": int(np.isinf(arr).sum()),
                        "min": float(np.min(finite_arr)) if finite.any() else float("nan"),
                        "max": float(np.max(finite_arr)) if finite.any() else float("nan"),
                        "abs_max": float(np.max(np.abs(finite_arr))) if finite.any() else float("nan"),
                        "mean": float(np.mean(finite_arr)) if finite.any() else float("nan"),
                        "std": float(np.std(finite_arr)) if finite.any() else float("nan"),
                        "n_zero": int((arr == 0).sum()),
                    }

                _basis_label = model_dbg.basis_kind.capitalize()
                if model_dbg.basis_kind == "bspline":
                    _basis_label = f"B-spline(k={model_dbg.bspline_degree})"
                summary_rows = [
                    _summary("curve_sample (raw)", cs_raw),
                    _summary("curve_sample (scaled to [-1,1])", cs_scl),
                    _summary(f"design X = [intercept | {_basis_label} integral]", X_dbg),
                    _summary("response Y = curve_sample (raw)", Y_dbg),
                    _summary("coef_", coef),
                    _summary("predicted", pred),
                ]
                st.markdown("**Numeric summary**")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                st.markdown(
                    f"`max_order` = **{model_dbg.max_order}** &nbsp; | &nbsp; "
                    f"`solver` = **{model_dbg.solver}** &nbsp; | &nbsp; "
                    f"`alpha` = **{model_dbg.alpha}** &nbsp; | &nbsp; "
                    f"`basis_kind` = **{model_dbg.basis_kind}** &nbsp; | &nbsp; "
                    f"`mse_` = **{model_dbg.mse_}**"
                )

                st.markdown("**curve_sample (raw) — head**")
                st.dataframe(cs_raw.head(), use_container_width=True)
                st.markdown("**curve_sample (scaled) — head**")
                st.dataframe(cs_scl.head(), use_container_width=True)
                st.markdown("**design matrix X — head (first 8 cols)**")
                st.dataframe(X_dbg.iloc[:5, :8], use_container_width=True)
                st.markdown("**response Y — head**")
                st.dataframe(Y_dbg.head(), use_container_width=True)
                st.markdown("**coef_**")
                st.dataframe(coef, use_container_width=True)
                st.markdown("**predicted — head**")
                st.dataframe(pred.head(), use_container_width=True)

            st.markdown("### Fitting vs Prediction")
            plot_curve_fitting(
                df_scatter=result["quasi_dynamic_df"],
                df_curve=result["predicted_df"],
                plot_scatter_type="scatter",
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
