import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ========== 页面配置 ==========
st.set_page_config(
    page_title="Network Construction",
    page_icon="TSA.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backend.curve_fitting import data_transformation
from backend.network_construction import IDOPRegressor, polynomial_basis_expansion
from backend.plot_curve_fitting import plot_curve_fitting
from backend.plot_network_construction import plot_effect, plot_network
from backend.utils import font_prop, load_css, setup_sidebar

# ========== 加载 CSS ==========
load_css()
setup_sidebar()

# ========== 页面标题 ==========
st.title("Network Construction", text_alignment="center")

# ========== Backend Functions ==========
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


def _load_funclu_export_from_zip(
    zip_bytes: bytes,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, dict[str, pd.DataFrame]],
]:
    """从 funclu_k_export.zip 读取标签、簇中心曲线与簇内成员曲线。"""
    labels_df: pd.DataFrame | None = None
    cluster_sizes_df: pd.DataFrame | None = None
    centers: dict[str, pd.DataFrame] = {}
    members: dict[str, dict[str, pd.DataFrame]] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            parts = name.split("/")
            with zf.open(name) as f:
                if name == "labels.csv":
                    labels_df = pd.read_csv(f)
                elif name == "cluster_sizes.csv":
                    cluster_sizes_df = pd.read_csv(f)
                elif (
                    len(parts) == 3
                    and parts[0] == "cluster_centers"
                    and parts[2] == "cluster_center_curve_sample.csv"
                ):
                    centers[parts[1]] = pd.read_csv(f, index_col=0)
                elif (
                    len(parts) == 3
                    and parts[0] == "cluster_members"
                    and parts[2].endswith("_curve_sample.csv")
                ):
                    cluster_name = parts[2].removesuffix("_curve_sample.csv")
                    members.setdefault(parts[1], {})[cluster_name] = pd.read_csv(
                        f, index_col=0
                    )

    if labels_df is None:
        raise ValueError("FunClu-K export ZIP 缺少 labels.csv")
    if cluster_sizes_df is None:
        cluster_sizes_df = pd.DataFrame()
    if not centers:
        raise ValueError("FunClu-K export ZIP 缺少 cluster center curve_sample")
    if not members:
        raise ValueError("FunClu-K export ZIP 缺少 cluster member curve_sample")

    sorted_centers = dict(sorted(centers.items()))
    sorted_members = {
        cond: dict(sorted(cluster_map.items()))
        for cond, cluster_map in sorted(members.items())
    }
    return labels_df, cluster_sizes_df, sorted_centers, sorted_members


def _fit_idop_network_from_curve_sample(
    curve_sample_df: pd.DataFrame,
    *,
    max_order: int,
    solver: str,
    alpha: float,
    nonneg_self: bool,
    max_interactions: int,
    basis_kind: str,
) -> dict:
    """复用单层 IdopNetwork 流程，从 curve_sample 构建一个网络。"""
    if curve_sample_df.shape[1] < 2:
        raise ValueError("至少需要 2 条曲线才能构建交互网络")

    curve_sample_scaled = data_transformation(curve_sample_df, "rescale_to_-1_1")
    model = IDOPRegressor(
        max_order=int(max_order),
        solver=str(solver),
        alpha=float(alpha),
        mix=0.5,
        fix_mix=(solver == "asgl"),
        nonneg_self=bool(nonneg_self),
        max_interactions=int(max_interactions),
        basis_kind=str(basis_kind),
    )
    model.fit(curve_sample_scaled, curve_sample_df)
    predicted_df = model.predict(curve_sample_scaled)
    effect_df_list = model.effect(curve_sample_scaled)
    adj_df = model.adjacency_matrix(curve_sample_scaled)
    return {
        "model": model,
        "curve_sample_df": curve_sample_df,
        "curve_sample_scaled": curve_sample_scaled,
        "predicted_df": predicted_df,
        "effect_df_list": effect_df_list,
        "adj_df": adj_df,
    }


def _adjacency_to_from_to(adj_df: pd.DataFrame, *, eps: float = 1e-12) -> pd.DataFrame:
    """将邻接矩阵转换为边表（保留自环，忽略绝对值很小的权重）。"""
    rows: list[dict[str, str | float]] = []
    for to_node in adj_df.index:
        for from_node in adj_df.columns:
            weight = float(adj_df.loc[to_node, from_node])
            if abs(weight) < eps:
                continue
            rows.append(
                {
                    "from": str(from_node),
                    "to": str(to_node),
                    "weight": weight,
                    "type": "+" if weight > 0 else "-",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["from", "to", "weight", "type"])
    return pd.DataFrame(rows, columns=["from", "to", "weight", "type"])


def _params_to_df(params: dict[str, object]) -> pd.DataFrame:
    """参数字典转 param-value 两列表格。"""
    return pd.DataFrame(
        [{"param": key, "value": value} for key, value in params.items()],
        columns=["param", "value"],
    )


def _collect_network_export_artifacts(
    network: dict,
    *,
    extra_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """统一提取一个网络导出的 adjacency/effect/from_to/params。"""
    adj_df: pd.DataFrame = network["adj_df"]
    effect_df_list: list[pd.DataFrame] = network["effect_df_list"]
    model: IDOPRegressor = network["model"]

    effect_map = {
        str(target): effect_df
        for target, effect_df in zip(adj_df.index, effect_df_list)
    }
    from_to_df = _adjacency_to_from_to(adj_df)

    params: dict[str, object] = {
        "solver": model.solver,
        "max_order": int(model.max_order),
        "alpha": float(model.alpha),
        "mix": float(model.mix),
        "nonneg_self": bool(model.nonneg_self),
        "max_interactions": int(model.max_interactions),
        "basis_kind": model.basis_kind,
        "basis_type": model.basis_type,
        "ebic_gamma": float(model.ebic_gamma),
        "mse": float(model.mse_) if model.mse_ is not None else np.nan,
        "n_nodes": int(adj_df.shape[0]),
        "n_edges_nonzero": int(from_to_df.shape[0]),
    }
    if extra_params:
        params.update(extra_params)

    return {
        "adj_df": adj_df,
        "effect_map": effect_map,
        "from_to_df": from_to_df,
        "params_df": _params_to_df(params),
    }


def _build_singlelayer_export_zip(result: dict) -> bytes:
    """打包单层 IdopNetwork 导出 ZIP。"""
    artifacts = _collect_network_export_artifacts(
        result,
        extra_params={"condition": result.get("condition", "")},
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("adjacency_matrix.csv", artifacts["adj_df"].to_csv(index=True))
        zf.writestr("from_to.csv", artifacts["from_to_df"].to_csv(index=False))
        zf.writestr("params.csv", artifacts["params_df"].to_csv(index=False))
        for target_name, effect_df in artifacts["effect_map"].items():
            zf.writestr(f"effect/{target_name}.csv", effect_df.to_csv(index=True))
    return buffer.getvalue()


def _build_multilayer_export_zip(
    *,
    result: dict,
    labels_df: pd.DataFrame,
    cluster_sizes_df: pd.DataFrame,
) -> bytes:
    """把 Multi-Layer IdopNetwork 邻接矩阵与元数据打包为 ZIP。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata/labels.csv", labels_df.to_csv(index=False))
        if not cluster_sizes_df.empty:
            zf.writestr(
                "metadata/cluster_sizes.csv", cluster_sizes_df.to_csv(index=False)
            )

        for cond_name, network in result["inter_cluster"].items():
            artifacts = _collect_network_export_artifacts(
                network,
                extra_params={
                    "layer": "inter_cluster",
                    "condition": cond_name,
                    "cluster": "",
                },
            )
            base_dir = f"inter_cluster/{cond_name}"
            zf.writestr(
                f"{base_dir}/adjacency_matrix.csv",
                artifacts["adj_df"].to_csv(index=True),
            )
            zf.writestr(f"{base_dir}/from_to.csv", artifacts["from_to_df"].to_csv(index=False))
            zf.writestr(f"{base_dir}/params.csv", artifacts["params_df"].to_csv(index=False))
            for target_name, effect_df in artifacts["effect_map"].items():
                zf.writestr(
                    f"{base_dir}/effect/{target_name}.csv",
                    effect_df.to_csv(index=True),
                )

        for cond_name, cluster_map in result["intra_cluster"].items():
            for cluster_name, network in cluster_map.items():
                artifacts = _collect_network_export_artifacts(
                    network,
                    extra_params={
                        "layer": "intra_cluster",
                        "condition": cond_name,
                        "cluster": cluster_name,
                    },
                )
                base_dir = f"intra_cluster/{cond_name}/{cluster_name}"
                zf.writestr(
                    f"{base_dir}/adjacency_matrix.csv",
                    artifacts["adj_df"].to_csv(index=True),
                )
                zf.writestr(
                    f"{base_dir}/from_to.csv",
                    artifacts["from_to_df"].to_csv(index=False),
                )
                zf.writestr(
                    f"{base_dir}/params.csv",
                    artifacts["params_df"].to_csv(index=False),
                )
                for target_name, effect_df in artifacts["effect_map"].items():
                    zf.writestr(
                        f"{base_dir}/effect/{target_name}.csv",
                        effect_df.to_csv(index=True),
                    )

        skipped_df = pd.DataFrame(result.get("skipped", []))
        if not skipped_df.empty:
            zf.writestr("metadata/skipped_networks.csv", skipped_df.to_csv(index=False))

    return buffer.getvalue()


# ========== Session State Initialization ==========
if "netrecon_quasi_dynamic" not in st.session_state:
    st.session_state.netrecon_quasi_dynamic = {}
if "netrecon_curve_sample" not in st.session_state:
    st.session_state.netrecon_curve_sample = {}
if "netrecon_uploaded_zip_name" not in st.session_state:
    st.session_state.netrecon_uploaded_zip_name = None
if "netrecon_result" not in st.session_state:
    st.session_state.netrecon_result = None
if "netrecon_funclu_labels" not in st.session_state:
    st.session_state.netrecon_funclu_labels = pd.DataFrame()
if "netrecon_funclu_cluster_sizes" not in st.session_state:
    st.session_state.netrecon_funclu_cluster_sizes = pd.DataFrame()
if "netrecon_funclu_centers" not in st.session_state:
    st.session_state.netrecon_funclu_centers = {}
if "netrecon_funclu_members" not in st.session_state:
    st.session_state.netrecon_funclu_members = {}
if "netrecon_funclu_uploaded_zip_name" not in st.session_state:
    st.session_state.netrecon_funclu_uploaded_zip_name = None
if "netrecon_multilayer_result" not in st.session_state:
    st.session_state.netrecon_multilayer_result = None

# ========== Tabs ==========
tab1, tab2 = st.tabs(["IdopNetwork", "Multi-Layer IdopNetwork"])

# ==========================================
# ========== Tab 1 IdopNetwork ==========
# ==========================================
with tab1:
    uploaded_zip = st.file_uploader(
        label="Please upload curve_fitting_export.zip",
        type=["zip"],
        accept_multiple_files=False,
        help="Input comes directly from Curve Fitting -> Export ZIP",
        key="netrecon_single_upload"
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

    quasi_map = st.session_state.netrecon_quasi_dynamic
    curve_map = st.session_state.netrecon_curve_sample
    common_conditions = [k for k in quasi_map if k in curve_map]

    tab1_1, tab1_2, tab1_3 = st.tabs(["Data Overview", "IdopNetwork Construction", "Export"])
    
    # ========== Tab 1_1 Data Overview ==========
    with tab1_1:
        if not common_conditions:
            st.info("Please upload curve_fitting_export.zip first.")
        else:
            default_condition = common_conditions[0]
            selected_condition = st.selectbox(
                "Condition (single-layer default picks the first one)",
                options=common_conditions,
                index=0,
                key="tab1_condition_select"
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

    # ========== Tab 1_2 IdopNetwork Construction ==========
    with tab1_2:
        if not common_conditions:
            st.info("Waiting for data upload...")
        else:
            # 使用在 Tab 1_1 中选择的条件
            selected_condition = st.session_state.get("tab1_condition_select", common_conditions[0])
            quasi_dynamic_df = quasi_map[selected_condition]
            curve_sample_df = curve_map[selected_condition]

            with st.expander("IdopNetwork parameter settings", expanded=True):
                with st.form(key="netrecon_form_single"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        solver = st.selectbox(
                            "Solver",
                            options=["ols", "lasso", "asgl"],
                            index=0,
                        )
                        max_order = st.number_input(
                            "max_order_upper (BIC)" if solver == "asgl" else "max_order",
                            min_value=1,
                            max_value=100,
                            value=6,
                            step=1,
                            help=(
                                "ASGL uses BIC to select max_order from 1..this value."
                                if solver == "asgl"
                                else "Fixed basis order for this solver."
                            ),
                        )
                        basis_kind = st.selectbox(
                            "basis_kind",
                            options=["legendre", "laguerre", "polynomial"],
                            index=0,
                            help=(
                                "Basis family for per-feature expansion. "
                                "legendre/laguerre: orthogonal polynomials on [-1, 1]. "
                                "polynomial: powers x, x^2, ..., with column count "
                                "still equals max_order."
                            ),
                        )
                    with c2:
                        if solver == "lasso":
                            alpha = st.number_input(
                                "alpha",
                                min_value=0.0,
                                value=0.001,
                                step=0.001,
                                format="%.4f",
                                help="lasso regularization strength",
                            )
                        else:
                            alpha = 1.0
                        if solver == "asgl":
                            st.caption("ASGL: max_order and alpha are selected by BIC.")
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
                    curve_sample_scaled = data_transformation(curve_sample_df, "rescale_to_-1_1")
                    model = IDOPRegressor(
                        max_order=int(max_order),
                        solver=str(solver),
                        alpha=float(alpha),
                        mix=0.5,
                        fix_mix=(solver == "asgl"),
                        nonneg_self=bool(nonneg_self),
                        max_interactions=int(top_k),
                        basis_kind=str(basis_kind),
                    )
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
                # 嵌套 Tab 展示结果
                tab1_2_1, tab1_2_2, tab1_2_3, tab1_2_4 = st.tabs(["Network", "Effect Decomposition", "Adjacency Matrix", "Debug"])
                
                # ========== Tab 1_2_1 Network ==========
                with tab1_2_1:
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

                    st.markdown("### Interaction Network")
                    target_node = st.selectbox(
                        "Target node filter",
                        options=[""] + list(result["adj_df"].index),
                        format_func=lambda x: "ALL" if x == "" else x,
                        key="netrecon_target_node",
                    )
                    plot_network(result["adj_df"], target_node=target_node)

                # ========== Tab 1_2_2 Effect Decomposition ==========
                with tab1_2_2:
                    st.markdown("### Effect Decomposition")
                    plot_effect(
                        quasi_dynamic_df=result["quasi_dynamic_df"],
                        curve_df=result["predicted_df"],
                        effect_df_list=result["effect_df_list"],
                        intercept=result["model"].coef_.loc["intercept"],
                        plot_ncols=4,
                    )

                # ========== Tab 1_2_3 Adjacency Matrix ==========
                with tab1_2_3:
                    st.markdown("### Adjacency Matrix")
                    st.dataframe(result["adj_df"], use_container_width=True)

                # ========== Tab 1_2_4 Debug ==========
                with tab1_2_4:
                    st.markdown("### Debug: design matrix, response, coefficients")
                    model_dbg: IDOPRegressor = result["model"]
                    X_dbg: pd.DataFrame = result["design_X"]
                    Y_dbg: pd.DataFrame = result["response_Y"]
                    cs_raw = result["curve_sample_df"]
                    cs_scl = result["curve_sample_scaled"]
                    pred = result["predicted_df"]
                    coef = model_dbg.coef_
                    basis_raw_dbg = polynomial_basis_expansion(
                        cs_scl,
                        model_dbg.max_order,
                        kind=model_dbg.basis_kind,
                    )

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

                    def _plot_matrix_lines(df: pd.DataFrame, title: str, x_label: str, exclude_intercept: bool = False) -> None:
                        if exclude_intercept:
                            cols = [c for c in df.columns if c != "intercept"]
                        else:
                            cols = list(df.columns)
                        if cols:
                            fig, ax = plt.subplots(figsize=(12, 4.5))
                            try:
                                x_idx = df.index.astype(float).to_numpy(dtype=float, copy=False)
                                if not np.all(np.isfinite(x_idx)):
                                    raise ValueError("non-finite float index")
                            except (TypeError, ValueError):
                                x_idx = np.arange(len(df), dtype=float)
                            for col in cols:
                                ax.plot(
                                    x_idx,
                                    df[col].to_numpy(dtype=float, copy=False),
                                    linewidth=1.0,
                                    alpha=0.75,
                                )
                            ax.set_xlabel(x_label, fontproperties=font_prop)
                            ax.set_ylabel("列取值", fontproperties=font_prop)
                            ax.set_title(title, fontproperties=font_prop)
                            ax.grid(True, alpha=0.35)
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        else:
                            st.caption("没有可绘制的矩阵列。")

                    def _plot_bic_curve(df: pd.DataFrame | None, title: str, x_col: str, x_label: str, log_x: bool = False) -> None:
                        if df is None or df.empty:
                            st.caption("没有可用的 BIC 搜索轨迹。")
                            return
                        plot_df = (
                            df.replace([np.inf, -np.inf], np.nan)
                            .dropna(subset=[x_col, "bic"])
                            .sort_values(x_col)
                        )
                        if plot_df.empty:
                            st.caption("没有可绘制的 BIC 点。")
                            return
                        fig, ax = plt.subplots(figsize=(6, 3.6))
                        ax.plot(
                            plot_df[x_col].to_numpy(dtype=float, copy=False),
                            plot_df["bic"].to_numpy(dtype=float, copy=False),
                            marker="o",
                            linewidth=1.4,
                        )
                        if "selected" in plot_df.columns:
                            selected = plot_df[plot_df["selected"].astype(bool)]
                            if not selected.empty:
                                ax.scatter(
                                    selected[x_col].to_numpy(dtype=float, copy=False),
                                    selected["bic"].to_numpy(dtype=float, copy=False),
                                    color="red",
                                    s=50,
                                    zorder=3,
                                )
                        if log_x:
                            ax.set_xscale("log")
                        ax.set_xlabel(x_label, fontproperties=font_prop)
                        ax.set_ylabel("BIC", fontproperties=font_prop)
                        ax.set_title(title, fontproperties=font_prop)
                        ax.grid(True, alpha=0.35)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    _basis_label = model_dbg.basis_kind.capitalize()
                    summary_rows = [
                        _summary("curve_sample (raw)", cs_raw),
                        _summary("curve_sample (scaled to [-1,1])", cs_scl),
                        _summary(f"basis raw = {_basis_label} before integral", basis_raw_dbg),
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
                    if model_dbg.solver == "asgl":
                        bic_left, bic_right = st.columns(2)
                        with bic_left:
                            st.markdown("**BIC vs max_order**")
                            _plot_bic_curve(
                                model_dbg.bic_order_path_,
                                "BIC 选择 max_order",
                                "max_order",
                                "max_order",
                            )
                        with bic_right:
                            st.markdown("**BIC vs alpha**")
                            _plot_bic_curve(
                                model_dbg.bic_alpha_path_,
                                "BIC 选择 alpha",
                                "alpha",
                                "alpha",
                                log_x=True,
                            )

                    st.markdown("**curve_sample (raw) — head**")
                    st.dataframe(cs_raw.head(), use_container_width=True)
                    st.markdown("**curve_sample (scaled) — head**")
                    st.dataframe(cs_scl.head(), use_container_width=True)
                    left_col, right_col = st.columns(2)
                    with left_col:
                        st.markdown("**basis matrix before integral — head (first 8 cols)**")
                        st.dataframe(basis_raw_dbg.iloc[:5, :8], use_container_width=True)
                        st.markdown("**basis matrix before integral — 沿 index 折线图**")
                        _plot_matrix_lines(
                            basis_raw_dbg,
                            "积分前基函数矩阵（全部列）",
                            "basis_raw.index",
                        )
                    with right_col:
                        st.markdown("**design matrix X — head (first 8 cols)**")
                        st.dataframe(X_dbg.iloc[:5, :8], use_container_width=True)
                        st.markdown("**design matrix X — 沿 index 折线图（不含 intercept）**")
                        _plot_matrix_lines(
                            X_dbg,
                            "设计矩阵 X（全部列，不含 intercept）",
                            "design_X.index",
                            exclude_intercept=True,
                        )
                    st.markdown("**response Y — head**")
                    st.dataframe(Y_dbg.head(), use_container_width=True)
                    st.markdown("**coef_**")
                    st.dataframe(coef, use_container_width=True)
                    st.markdown("**predicted — head**")
                    st.dataframe(pred.head(), use_container_width=True)

    # ========== Tab 1_3 Export ==========
    with tab1_3:
        result = st.session_state.netrecon_result
        if result is not None:
            try:
                export_zip = _build_singlelayer_export_zip(result)
            except Exception as e:
                st.error(f"Single-layer export failed: {e}")
            else:
                st.markdown("**Export contents**")
                st.write(
                    "- `adjacency_matrix.csv`: 邻接矩阵。\n"
                    "- `from_to.csv`: 边表（`from,to,weight,type`，包含自环，`type` 为 `+/-`）。\n"
                    "- `params.csv`: 本次网络拟合参数与统计量。\n"
                    "- `effect/*.csv`: 每个目标节点一个 effect 分解表。"
                )
                st.download_button(
                    label="Download Single-Layer IdopNetwork ZIP",
                    data=export_zip,
                    file_name="single_layer_idopnetwork_export.zip",
                    mime="application/zip",
                    key="netrecon_single_export_download",
                )
        else:
            st.info("Please run IdopNetwork Construction first.")


# =======================================================
# ========== Tab 2 Multi-Layer IdopNetwork ==========
# =======================================================
with tab2:
    uploaded_funclu_zip = st.file_uploader(
        label="Please upload funclu_k_export.zip",
        type=["zip"],
        accept_multiple_files=False,
        help="Input comes from Functional Clustering -> FunClu-K -> Export",
        key="netrecon_funclu_export_upload",
    )

    if (
        uploaded_funclu_zip is not None
        and st.session_state.netrecon_funclu_uploaded_zip_name
        != uploaded_funclu_zip.name
    ):
        try:
            labels_df, cluster_sizes_df, centers, members = _load_funclu_export_from_zip(
                uploaded_funclu_zip.getvalue()
            )
        except Exception as e:
            st.error(f"读取 FunClu-K ZIP 失败：{e}")
            st.session_state.netrecon_funclu_labels = pd.DataFrame()
            st.session_state.netrecon_funclu_cluster_sizes = pd.DataFrame()
            st.session_state.netrecon_funclu_centers = {}
            st.session_state.netrecon_funclu_members = {}
            st.session_state.netrecon_funclu_uploaded_zip_name = None
            st.session_state.netrecon_multilayer_result = None
        else:
            st.session_state.netrecon_funclu_labels = labels_df
            st.session_state.netrecon_funclu_cluster_sizes = cluster_sizes_df
            st.session_state.netrecon_funclu_centers = centers
            st.session_state.netrecon_funclu_members = members
            st.session_state.netrecon_funclu_uploaded_zip_name = uploaded_funclu_zip.name
            st.session_state.netrecon_multilayer_result = None

    labels_df = st.session_state.netrecon_funclu_labels
    cluster_sizes_df = st.session_state.netrecon_funclu_cluster_sizes
    centers = st.session_state.netrecon_funclu_centers
    members = st.session_state.netrecon_funclu_members

    tab2_1, tab2_2, tab2_3 = st.tabs(["Data Overview", "Multi-Layer IdopNetwork Construction", "Export"])

    # ========== Tab 2_1 Data Overview ==========
    with tab2_1:
        if not centers:
            st.info("Please upload funclu_k_export.zip first.")
        else:
            condition_names = list(centers.keys())
            cluster_names = sorted(
                {
                    cluster_name
                    for cluster_map in members.values()
                    for cluster_name in cluster_map
                }
            )

            st.markdown("### FunClu-K Export Overview")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("n_conditions", len(condition_names))
            with c2:
                st.metric("n_clusters", len(cluster_names))
            with c3:
                st.metric("n_features", len(labels_df))

            overview_rows = []
            for cond_name in condition_names:
                center_df = centers[cond_name]
                overview_rows.append(
                    {
                        "condition": cond_name,
                        "center_shape": str(center_df.shape),
                        "member_clusters": len(members.get(cond_name, {})),
                    }
                )
            st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

            with st.expander("FunClu labels and cluster sizes", expanded=False):
                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown("**labels.csv**")
                    st.dataframe(labels_df, use_container_width=True)
                with right_col:
                    st.markdown("**cluster_sizes.csv**")
                    if cluster_sizes_df.empty:
                        st.caption("No cluster_sizes.csv found in ZIP.")
                    else:
                        st.dataframe(cluster_sizes_df, use_container_width=True)

    # ========== Tab 2_2 IdopNetwork Construction ==========
    with tab2_2:
        if not centers:
            st.info("Waiting for data upload...")
        else:
            condition_names = list(centers.keys())
            
            with st.expander("Multi-Layer IdopNetwork parameter settings", expanded=True):
                with st.form(key="netrecon_form_multilayer"):
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        ml_solver = st.selectbox(
                            "Solver",
                            options=["ols", "lasso", "asgl"],
                            index=0,
                            key="netrecon_ml_solver",
                        )
                        ml_max_order = st.number_input(
                            "max_order_upper (BIC)" if ml_solver == "asgl" else "max_order",
                            min_value=1,
                            max_value=100,
                            value=6,
                            step=1,
                            key="netrecon_ml_max_order",
                        )
                        ml_basis_kind = st.selectbox(
                            "basis_kind",
                            options=["legendre", "laguerre", "polynomial"],
                            index=0,
                            key="netrecon_ml_basis_kind",
                        )
                    with mc2:
                        if ml_solver == "lasso":
                            ml_alpha = st.number_input(
                                "alpha",
                                min_value=0.0,
                                value=0.001,
                                step=0.001,
                                format="%.4f",
                                key="netrecon_ml_alpha",
                            )
                        else:
                            ml_alpha = 1.0
                        if ml_solver == "asgl":
                            st.caption("ASGL: max_order and alpha are selected by BIC.")
                    with mc3:
                        ml_nonneg_self = st.checkbox(
                            "nonneg_self",
                            value=True,
                            key="netrecon_ml_nonneg_self",
                        )
                        ml_top_k = st.number_input(
                            "max_interactions (Top-K)",
                            min_value=0,
                            max_value=100,
                            value=0,
                            step=1,
                            key="netrecon_ml_top_k",
                        )
                    submit_multilayer = st.form_submit_button("Run Multi-Layer IdopNetwork")

            if submit_multilayer:
                params = {
                    "max_order": int(ml_max_order),
                    "solver": str(ml_solver),
                    "alpha": float(ml_alpha),
                    "nonneg_self": bool(ml_nonneg_self),
                    "max_interactions": int(ml_top_k),
                    "basis_kind": str(ml_basis_kind),
                }
                inter_cluster: dict[str, dict] = {}
                intra_cluster: dict[str, dict[str, dict]] = {}
                skipped: list[dict[str, str | int]] = []

                try:
                    for cond_name in condition_names:
                        center_df = centers[cond_name]
                        try:
                            inter_cluster[cond_name] = _fit_idop_network_from_curve_sample(
                                center_df,
                                **params,
                            )
                        except Exception as e:
                            skipped.append(
                                {
                                    "layer": "inter_cluster",
                                    "condition": cond_name,
                                    "cluster": "",
                                    "n_nodes": int(center_df.shape[1]),
                                    "reason": str(e),
                                }
                            )

                        intra_cluster[cond_name] = {}
                        for cluster_name, member_df in members.get(cond_name, {}).items():
                            if member_df.shape[1] < 2:
                                skipped.append(
                                    {
                                        "layer": "intra_cluster",
                                        "condition": cond_name,
                                        "cluster": cluster_name,
                                        "n_nodes": int(member_df.shape[1]),
                                        "reason": "less than 2 member curves",
                                    }
                                )
                                continue
                            try:
                                intra_cluster[cond_name][cluster_name] = (
                                    _fit_idop_network_from_curve_sample(
                                        member_df,
                                        **params,
                                    )
                                )
                            except Exception as e:
                                skipped.append(
                                    {
                                        "layer": "intra_cluster",
                                        "condition": cond_name,
                                        "cluster": cluster_name,
                                        "n_nodes": int(member_df.shape[1]),
                                        "reason": str(e),
                                    }
                                )
                except Exception as e:
                    st.error(f"Multi-Layer IdopNetwork 运行失败：{e}")
                    st.session_state.netrecon_multilayer_result = None
                else:
                    st.session_state.netrecon_multilayer_result = {
                        "params": params,
                        "inter_cluster": inter_cluster,
                        "intra_cluster": intra_cluster,
                        "skipped": skipped,
                    }
                    st.success("Done")

            multilayer_result = st.session_state.netrecon_multilayer_result
            if multilayer_result is not None:
                # 嵌套 Tab 展示结果
                tab2_2_1, tab2_2_2, tab2_2_3, tab2_2_4 = st.tabs(["Network", "Effect Decomposition", "Adjacency Matrix", "Debug"])

                # ========== Tab 2_2_1 Network ==========
                with tab2_2_1:
                    st.markdown("### Multi-Layer Network Summary")
                    summary_rows = []
                    for cond_name, network in multilayer_result["inter_cluster"].items():
                        adj_df = network["adj_df"]
                        summary_rows.append(
                            {
                                "layer": "inter_cluster",
                                "condition": cond_name,
                                "cluster": "",
                                "n_nodes": int(adj_df.shape[0]),
                                "mse": network["model"].mse_,
                                "status": "built",
                            }
                        )
                    for cond_name, cluster_map in multilayer_result["intra_cluster"].items():
                        for cluster_name, network in cluster_map.items():
                            adj_df = network["adj_df"]
                            summary_rows.append(
                                {
                                    "layer": "intra_cluster",
                                    "condition": cond_name,
                                    "cluster": cluster_name,
                                    "n_nodes": int(adj_df.shape[0]),
                                    "mse": network["model"].mse_,
                                    "status": "built",
                                }
                            )
                    for row in multilayer_result.get("skipped", []):
                        summary_rows.append({**row, "mse": np.nan, "status": "skipped"})
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    if multilayer_result["inter_cluster"]:
                        st.markdown("### Inter-Cluster Network")
                        inter_condition = st.selectbox(
                            "Inter-cluster condition",
                            options=list(multilayer_result["inter_cluster"].keys()),
                            key="netrecon_ml_inter_condition",
                        )
                        inter_adj_df = multilayer_result["inter_cluster"][inter_condition]["adj_df"]
                        inter_target_node = st.selectbox(
                            "Inter-cluster target node filter",
                            options=[""] + list(inter_adj_df.index),
                            format_func=lambda x: "ALL" if x == "" else x,
                            key="netrecon_ml_inter_target_node",
                        )
                        plot_network(inter_adj_df, target_node=inter_target_node)
                    else:
                        st.warning("No inter-cluster network was built.")

                    available_intra = {
                        cond_name: cluster_map
                        for cond_name, cluster_map in multilayer_result["intra_cluster"].items()
                        if cluster_map
                    }
                    if available_intra:
                        st.markdown("### Intra-Cluster Network")
                        intra_condition = st.selectbox(
                            "Intra-cluster condition",
                            options=list(available_intra.keys()),
                            key="netrecon_ml_intra_condition",
                        )
                        intra_cluster_name = st.selectbox(
                            "Cluster",
                            options=list(available_intra[intra_condition].keys()),
                            key="netrecon_ml_intra_cluster",
                        )
                        intra_adj_df = available_intra[intra_condition][intra_cluster_name]["adj_df"]
                        intra_target_node = st.selectbox(
                            "Intra-cluster target node filter",
                            options=[""] + list(intra_adj_df.index),
                            format_func=lambda x: "ALL" if x == "" else x,
                            key="netrecon_ml_intra_target_node",
                        )
                        plot_network(intra_adj_df, target_node=intra_target_node)
                    else:
                        st.warning("No intra-cluster network was built.")

                # ========== Tab 2_2_2 Effect Decomposition ==========
                with tab2_2_2:
                    st.info("Effect Decomposition for Multi-Layer IdopNetwork is to be updated...")

                # ========== Tab 2_2_3 Adjacency Matrix ==========
                with tab2_2_3:
                    if multilayer_result["inter_cluster"]:
                        st.markdown(f"### Inter-Cluster Adjacency Matrix ({st.session_state.get('netrecon_ml_inter_condition', '')})")
                        if "netrecon_ml_inter_condition" in st.session_state:
                            st.dataframe(multilayer_result["inter_cluster"][st.session_state.netrecon_ml_inter_condition]["adj_df"], use_container_width=True)

                    if available_intra:
                        st.markdown(f"### Intra-Cluster Adjacency Matrix ({st.session_state.get('netrecon_ml_intra_condition', '')} - {st.session_state.get('netrecon_ml_intra_cluster', '')})")
                        if "netrecon_ml_intra_condition" in st.session_state and "netrecon_ml_intra_cluster" in st.session_state:
                            st.dataframe(available_intra[st.session_state.netrecon_ml_intra_condition][st.session_state.netrecon_ml_intra_cluster]["adj_df"], use_container_width=True)

                # ========== Tab 2_2_4 Debug ==========
                with tab2_2_4:
                    st.info("Debug panel for Multi-Layer IdopNetwork is to be updated...")

    # ========== Tab 2_3 Export ==========
    with tab2_3:
        multilayer_result = st.session_state.netrecon_multilayer_result
        if multilayer_result is not None:
            try:
                export_zip = _build_multilayer_export_zip(
                    result=multilayer_result,
                    labels_df=labels_df,
                    cluster_sizes_df=cluster_sizes_df,
                )
            except Exception as e:
                st.error(f"Multi-layer export failed: {e}")
            else:
                st.markdown("**Export contents**")
                st.write(
                    "- `metadata/labels.csv`, `metadata/cluster_sizes.csv`, `metadata/skipped_networks.csv`.\n"
                    "- `inter_cluster/<condition>/adjacency_matrix.csv`.\n"
                    "- `inter_cluster/<condition>/from_to.csv`.\n"
                    "- `inter_cluster/<condition>/params.csv`.\n"
                    "- `inter_cluster/<condition>/effect/*.csv`.\n"
                    "- `intra_cluster/<condition>/<cluster>/adjacency_matrix.csv`.\n"
                    "- `intra_cluster/<condition>/<cluster>/from_to.csv`.\n"
                    "- `intra_cluster/<condition>/<cluster>/params.csv`.\n"
                    "- `intra_cluster/<condition>/<cluster>/effect/*.csv`."
                )
                st.download_button(
                    label="Download Multi-Layer IdopNetwork ZIP",
                    data=export_zip,
                    file_name="multi_layer_idopnetwork_export.zip",
                    mime="application/zip",
                    key="netrecon_ml_export_download",
                )
        else:
            st.info("Please run Multi-Layer IdopNetwork Construction first.")