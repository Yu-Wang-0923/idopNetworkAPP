import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from importlib.resources import files

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")

# ========== 页面配置 ==========
st.set_page_config(
    page_title="Network Construction",
    page_icon=_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

from idopnetwork.curve_fitting.fitting import get_power_function_params
from idopnetwork.network.construction import (
    IDOPRegressor,
    align_response_to_design,
    polynomial_basis_expansion,
)
from idopnetwork.curve_fitting.plot import plot_curve_fitting
from idopnetwork.network.plot import (
    plot_effect,
    plot_network,
    plot_adjusted_matrix_heatmap,
)
from idopnetwork_app.utils import font_prop, load_css, setup_sidebar

# ========== 加载 CSS ==========
load_css()
setup_sidebar()

# ========== 登录门禁 ==========
if not st.session_state.get("logged_in", False):
    st.warning("请先返回首页登录后使用。")
    st.stop()

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
    dict[str, pd.DataFrame],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, pd.DataFrame]],
]:
    """从 funclu_k_export.zip 读取多层曲线与响应变量。"""
    labels_df: pd.DataFrame | None = None
    cluster_sizes_df: pd.DataFrame | None = None
    centers: dict[str, pd.DataFrame] = {}
    center_responses: dict[str, pd.DataFrame] = {}
    members: dict[str, dict[str, pd.DataFrame]] = {}
    member_responses: dict[str, dict[str, pd.DataFrame]] = {}

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
                    and parts[0] == "cluster_centers"
                    and parts[2] == "cluster_center_quasi_dynamic.csv"
                ):
                    center_responses[parts[1]] = pd.read_csv(f, index_col=0)
                elif (
                    len(parts) == 3
                    and parts[0] == "cluster_members"
                    and parts[2].endswith("_curve_sample.csv")
                ):
                    cluster_name = parts[2].removesuffix("_curve_sample.csv")
                    members.setdefault(parts[1], {})[cluster_name] = pd.read_csv(
                        f, index_col=0
                    )
                elif (
                    len(parts) == 3
                    and parts[0] == "cluster_members"
                    and parts[2].endswith("_quasi_dynamic.csv")
                ):
                    cluster_name = parts[2].removesuffix("_quasi_dynamic.csv")
                    member_responses.setdefault(parts[1], {})[
                        cluster_name
                    ] = pd.read_csv(f, index_col=0)

    if labels_df is None:
        raise ValueError("FunClu-K export ZIP 缺少 labels.csv")
    if cluster_sizes_df is None:
        cluster_sizes_df = pd.DataFrame()
    if not centers:
        raise ValueError("FunClu-K export ZIP 缺少 cluster center curve_sample")
    if not center_responses:
        raise ValueError("FunClu-K export ZIP 缺少 cluster center quasi_dynamic")
    if not members:
        raise ValueError("FunClu-K export ZIP 缺少 cluster member curve_sample")
    if not member_responses:
        raise ValueError("FunClu-K export ZIP 缺少 cluster member quasi_dynamic")

    sorted_centers = dict(sorted(centers.items()))
    sorted_center_responses = dict(sorted(center_responses.items()))
    sorted_members = {
        cond: dict(sorted(cluster_map.items()))
        for cond, cluster_map in sorted(members.items())
    }
    sorted_member_responses = {
        cond: dict(sorted(cluster_map.items()))
        for cond, cluster_map in sorted(member_responses.items())
    }
    return (
        labels_df,
        cluster_sizes_df,
        sorted_centers,
        sorted_center_responses,
        sorted_members,
        sorted_member_responses,
    )


def _fit_idop_network_from_curve_sample(
    curve_sample_df: pd.DataFrame,
    response_df: pd.DataFrame,
    *,
    max_order: int,
    nonneg_self: bool,
    max_interactions: int,
    adjacency_aggregation: str,
    power_function_params: pd.DataFrame,
) -> dict:
    """复用单层 IdopNetwork 流程：curve_sample 做设计矩阵，response_df 做响应。"""
    if curve_sample_df.shape[1] < 2:
        raise ValueError("至少需要 2 条曲线才能构建交互网络")
    response_df = response_df.loc[:, list(curve_sample_df.columns)]

    model = IDOPRegressor(
        max_order=int(max_order),
        mix=0.5,
        fix_mix=False,
        nonneg_self=bool(nonneg_self),
        max_interactions=int(max_interactions),
        adaptive_weights=False,
    )
    model.fit(
        curve_sample_df,
        response_df,
        power_function_params=power_function_params,
    )
    predicted_df = model.predict(curve_sample_df)
    effect_df_list = model.effect(curve_sample_df)
    adj_df = model.adjacency_matrix(
        curve_sample_df,
        aggregation=str(adjacency_aggregation),
    )
    design_X = model._design(curve_sample_df)
    response_Y = align_response_to_design(response_df, design_X.index)
    return {
        "model": model,
        "quasi_dynamic_df": response_df,
        "curve_sample_df": curve_sample_df,
        "design_X": design_X,
        "response_Y": response_Y,
        "predicted_df": predicted_df,
        "effect_df_list": effect_df_list,
        "adj_df": adj_df,
        "adjacency_aggregation": str(adjacency_aggregation),
    }


def _cluster_response_from_quasi(
    labels_df: pd.DataFrame,
    quasi_dynamic_df: pd.DataFrame,
    center_columns: list[str],
) -> pd.DataFrame:
    """按 FunClu-K 标签把成员 quasi_dynamic 聚合为 cluster-level 响应。"""
    rows: dict[str, pd.Series] = {}
    for cluster_name in center_columns:
        member_cols = labels_df.loc[
            labels_df["cluster"].astype(str) == str(cluster_name),
            "feature",
        ].astype(str).tolist()
        available_cols = [col for col in member_cols if col in quasi_dynamic_df.columns]
        if not available_cols:
            raise ValueError(f"cluster {cluster_name} 在 quasi_dynamic 中没有成员列")
        rows[str(cluster_name)] = quasi_dynamic_df.loc[:, available_cols].mean(axis=1)
    return pd.DataFrame(rows, index=quasi_dynamic_df.index)


def _adjacency_to_from_to(adj_df: pd.DataFrame, *, eps: float = 1e-12) -> pd.DataFrame:
    """将邻接矩阵转换为边表（保留自环，忽略绝对值很小的权重）。

    Direction rule:
        adj_df.loc[source, target] = source -> target
    """
    rows: list[dict[str, str | float]] = []

    for from_node in adj_df.index:
        for to_node in adj_df.columns:
            weight = float(adj_df.loc[from_node, to_node])
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


def _to_csv_utf8_sig_bytes(df: pd.DataFrame, *, index: bool) -> bytes:
    """DataFrame 导出为 UTF-8-SIG 编码的 CSV 字节串。"""
    return df.to_csv(index=index).encode("utf-8-sig")


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
        "max_order": int(model.max_order),
        "alpha": float(model.alpha),
        "mix": float(model.mix),
        "nonneg_self": bool(model.nonneg_self),
        "max_interactions": int(model.max_interactions),
        "basis_type": model.basis_type,
        "ebic_gamma": float(model.ebic_gamma),
        "enforce_effect_constraints": bool(model.enforce_effect_constraints),
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
        extra_params={
            "condition": result.get("condition", ""),
            "adjacency_aggregation": result.get("adjacency_aggregation", "mean"),
        },
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "adjacency_matrix.csv",
            _to_csv_utf8_sig_bytes(artifacts["adj_df"], index=True),
        )
        zf.writestr(
            "from_to.csv",
            _to_csv_utf8_sig_bytes(artifacts["from_to_df"], index=False),
        )
        zf.writestr(
            "params.csv",
            _to_csv_utf8_sig_bytes(artifacts["params_df"], index=False),
        )
        for target_name, effect_df in artifacts["effect_map"].items():
            zf.writestr(
                f"effect/{target_name}.csv",
                _to_csv_utf8_sig_bytes(effect_df, index=True),
            )
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
        zf.writestr(
            "metadata/labels.csv",
            _to_csv_utf8_sig_bytes(labels_df, index=False),
        )
        if not cluster_sizes_df.empty:
            zf.writestr(
                "metadata/cluster_sizes.csv",
                _to_csv_utf8_sig_bytes(cluster_sizes_df, index=False),
            )

        for cond_name, network in result["inter_cluster"].items():
            artifacts = _collect_network_export_artifacts(
                network,
                extra_params={
                    "layer": "inter_cluster",
                    "condition": cond_name,
                    "cluster": "",
                    "adjacency_aggregation": network.get("adjacency_aggregation", "mean"),
                },
            )
            base_dir = f"inter_cluster/{cond_name}"
            zf.writestr(
                f"{base_dir}/adjacency_matrix.csv",
                _to_csv_utf8_sig_bytes(artifacts["adj_df"], index=True),
            )
            zf.writestr(
                f"{base_dir}/from_to.csv",
                _to_csv_utf8_sig_bytes(artifacts["from_to_df"], index=False),
            )
            zf.writestr(
                f"{base_dir}/params.csv",
                _to_csv_utf8_sig_bytes(artifacts["params_df"], index=False),
            )
            for target_name, effect_df in artifacts["effect_map"].items():
                zf.writestr(
                    f"{base_dir}/effect/{target_name}.csv",
                    _to_csv_utf8_sig_bytes(effect_df, index=True),
                )

        for cond_name, cluster_map in result["intra_cluster"].items():
            for cluster_name, network in cluster_map.items():
                artifacts = _collect_network_export_artifacts(
                    network,
                    extra_params={
                        "layer": "intra_cluster",
                        "condition": cond_name,
                        "cluster": cluster_name,
                        "adjacency_aggregation": network.get(
                            "adjacency_aggregation", "mean"
                        ),
                    },
                )
                base_dir = f"intra_cluster/{cond_name}/{cluster_name}"
                zf.writestr(
                    f"{base_dir}/adjacency_matrix.csv",
                    _to_csv_utf8_sig_bytes(artifacts["adj_df"], index=True),
                )
                zf.writestr(
                    f"{base_dir}/from_to.csv",
                    _to_csv_utf8_sig_bytes(artifacts["from_to_df"], index=False),
                )
                zf.writestr(
                    f"{base_dir}/params.csv",
                    _to_csv_utf8_sig_bytes(artifacts["params_df"], index=False),
                )
                for target_name, effect_df in artifacts["effect_map"].items():
                    zf.writestr(
                        f"{base_dir}/effect/{target_name}.csv",
                        _to_csv_utf8_sig_bytes(effect_df, index=True),
                    )

        skipped_df = pd.DataFrame(result.get("skipped", []))
        if not skipped_df.empty:
            zf.writestr(
                "metadata/skipped_networks.csv",
                _to_csv_utf8_sig_bytes(skipped_df, index=False),
            )

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
if "netrecon_funclu_center_responses" not in st.session_state:
    st.session_state.netrecon_funclu_center_responses = {}
if "netrecon_funclu_members" not in st.session_state:
    st.session_state.netrecon_funclu_members = {}
if "netrecon_funclu_member_responses" not in st.session_state:
    st.session_state.netrecon_funclu_member_responses = {}
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
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        max_order = st.number_input(
                            "max_order_upper (BIC)",
                            min_value=1,
                            max_value=100,
                            value=1,
                            step=1,
                            help="ASGL uses BIC to select max_order from 1..this value.",
                        )
                    with c2:
                        nonneg_self = st.checkbox("nonneg_self", value=True)
                    with c3:
                        top_k = st.number_input(
                            "max_interactions (Top-K)",
                            min_value=0,
                            max_value=100,
                            value=0,
                            step=1,
                        )
                    with c4:
                        adjacency_aggregation = st.selectbox(
                            "adjacency_aggregation",
                            options=["mean", "integral"],
                            index=0,
                            help="邻接矩阵列聚合方式：mean=离散点均值，integral=按 index 梯形积分。",
                        )
                    with c5:
                        submit_run = st.form_submit_button("Run IdopNetwork")

            if submit_run:
                try:
                    with st.status(
                        f"Building IdopNetwork for `{selected_condition}`...",
                        expanded=True,
                    ) as sl_status:
                        progress_bar = st.progress(0, text="Starting...")
                        log = st.empty()

                        progress_bar.progress(10, text="Estimating power-function parameters...")
                        log.write("Estimating power-function parameters...")
                        power_function_params = get_power_function_params(quasi_dynamic_df)

                        progress_bar.progress(20, text="Initializing constrained ASGL model...")
                        log.write("Initializing constrained ASGL model...")
                        model = IDOPRegressor(
                            max_order=int(max_order),
                            mix=0.5,
                            fix_mix=False,
                            nonneg_self=bool(nonneg_self),
                            max_interactions=int(top_k),
                            adaptive_weights=False,
                        )

                        progress_bar.progress(35, text="Fitting model and enforcing effect constraints...")
                        log.write("Fitting model (this may take a while)...")
                        model.fit(
                            curve_sample_df,
                            quasi_dynamic_df,
                            power_function_params=power_function_params,
                        )

                        progress_bar.progress(70, text="Generating prediction curves...")
                        log.write("Generating prediction curves...")
                        predicted_df = model.predict(curve_sample_df)

                        progress_bar.progress(80, text="Computing effect decomposition...")
                        log.write("Computing effect decomposition...")
                        effect_df_list = model.effect(curve_sample_df)

                        progress_bar.progress(90, text="Building adjacency matrix...")
                        log.write("Building adjacency matrix...")
                        adj_df = model.adjacency_matrix(
                            curve_sample_df,
                            aggregation=str(adjacency_aggregation),
                        )

                        progress_bar.progress(95, text="Finalizing...")
                        log.write("Preparing model diagnostics...")
                        design_X = model._design(curve_sample_df)
                        response_Y = align_response_to_design(quasi_dynamic_df, design_X.index)
                except Exception as e:
                    st.error(f"IdopNetwork 运行失败：{e}")
                    st.session_state.netrecon_result = None
                else:
                    st.session_state.netrecon_result = {
                        "condition": selected_condition,
                        "model": model,
                        "quasi_dynamic_df": quasi_dynamic_df,
                        "curve_sample_df": curve_sample_df,
                        "design_X": design_X,
                        "response_Y": response_Y,
                        "predicted_df": predicted_df,
                        "effect_df_list": effect_df_list,
                        "adj_df": adj_df,
                        "adjacency_aggregation": str(adjacency_aggregation),
                    }
                    # 重置所有子 Tab 的渲染标记，避免新结果自动触发旧绘图
                    st.session_state["single_effect_plot_render"] = False
                    st.session_state["single_adj_heatmap_render"] = False
                    st.session_state["single_network_plot_mode"] = None
                    sl_status.update(
                        label=f"IdopNetwork for `{selected_condition}` completed.",
                        state="complete",
                    )

            result = st.session_state.netrecon_result
            if result is not None:
                # 嵌套 Tab 展示结果
                tab1_2_1, tab1_2_2, tab1_2_3 = st.tabs(["Network", "Effect Decomposition", "Adjacency Matrix"])
                
                # ========== Tab 1_2_1 Network ==========
                with tab1_2_1:
                    st.markdown("### Fitting vs Prediction")
                    fig = plot_curve_fitting(
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
                    st.pyplot(fig)

                    st.markdown("### Interaction Network")

                    with st.form(key="single_network_form"):
                        target_node = st.selectbox(
                            "Target node filter",
                            options=[""] + list(result["adj_df"].columns),
                            format_func=lambda x: "ALL" if x == "" else x,
                            key="netrecon_target_node",
                        )
                        top_edges_for_plot = st.number_input(
                            "Top edges for plot",
                            min_value=1,
                            max_value=500,
                            value=60,
                            step=1,
                            key="single_top_edges_for_plot",
                        )
                        cplot1, cplot2 = st.columns(2)
                        with cplot1:
                            run_network_plot = st.form_submit_button(
                                "Run Network Plot",
                            )
                        with cplot2:
                            run_network_degree_plot = st.form_submit_button(
                                "Run Network + Degree Plot",
                            )

                    if st.button("Clear Plot", key="clear_single_network_plot_btn"):
                        st.session_state["single_network_plot_mode"] = None

                    if run_network_plot:
                        st.session_state["single_network_plot_mode"] = "network_only"
                        st.session_state["single_network_target"] = target_node
                        st.session_state["single_network_top_edges"] = int(top_edges_for_plot)

                    if run_network_degree_plot:
                        st.session_state["single_network_plot_mode"] = "network_degree"
                        st.session_state["single_network_target"] = target_node
                        st.session_state["single_network_top_edges"] = int(top_edges_for_plot)

                    single_network_plot_mode = st.session_state.get("single_network_plot_mode", None)

                    if single_network_plot_mode == "network_only":
                        fig = plot_network(
                            result["adj_df"],
                            target_node=st.session_state.get("single_network_target", ""),
                            top_edges=st.session_state.get("single_network_top_edges", 60),
                            show_degree_panel=False,
                        )
                        st.pyplot(fig)
                    elif single_network_plot_mode == "network_degree":
                        fig = plot_network(
                            result["adj_df"],
                            target_node=st.session_state.get("single_network_target", ""),
                            top_edges=st.session_state.get("single_network_top_edges", 60),
                            show_degree_panel=True,
                        )
                        st.pyplot(fig)
                    else:
                        st.info("点击 `Run Network Plot` 或 `Run Network + Degree Plot` 渲染网络图。")
                # ========== Tab 1_2_2 Effect Decomposition ==========
                with tab1_2_2:
                    st.markdown("### Effect Decomposition")

                    with st.form(key="single_effect_form"):
                        ceff1, ceff2, ceff3, ceff4 = st.columns(4)

                        with ceff1:
                            run_effect_plot = st.form_submit_button(
                                "Run Effect Decomposition Plot",
                            )

                        with ceff2:
                            st.write("")

                        with ceff3:
                            single_effect_ncols = st.number_input(
                                "Subplot cols",
                                min_value=1,
                                max_value=12,
                                value=4,
                                step=1,
                                key="single_effect_ncols",
                            )

                        with ceff4:
                            single_effect_max_vars = st.number_input(
                                "Max variables (0=all)",
                                min_value=0,
                                max_value=500,
                                value=0,
                                step=1,
                                key="single_effect_max_vars",
                                help="限制显示的目标变量数量；0 表示绘制全部变量。",
                            )

                    clear_effect_plot = st.button(
                        "Clear Effect Plot",
                        key="clear_single_effect_plot_btn",
                    )

                    if run_effect_plot:
                        st.session_state["single_effect_plot_render"] = True

                    if clear_effect_plot:
                        st.session_state["single_effect_plot_render"] = False

                    if st.session_state.get("single_effect_plot_render", False):
                        fig = plot_effect(
                            quasi_dynamic_df=result["quasi_dynamic_df"],
                            curve_df=result["predicted_df"],
                            effect_df_list=result["effect_df_list"],
                            intercept=result["model"].coef_.loc["intercept"],
                            plot_ncols=st.session_state.get("single_effect_ncols", 4),
                            plot_max_vars=st.session_state.get("single_effect_max_vars", 0),
                        )
                        st.pyplot(fig)
                    else:
                        st.info("Click `Run Effect Decomposition Plot` to render effect curves.")

                # ========== Tab 1_2_3 Adjacency Matrix ==========
                # ========== Tab 1_2_3 Adjacency Matrix ==========
                with tab1_2_3:
                    st.markdown("### Adjacency Matrix")
                    st.dataframe(result["adj_df"], use_container_width=True)

                    st.markdown("### Adjusted Matrix Heatmap")

                    with st.form(key="single_heatmap_form"):
                        hcol1, hcol2, hcol3 = st.columns(3)
                        with hcol1:
                            show_single_heatmap_values = st.checkbox(
                                "Show values",
                                value=False,
                                key="show_single_adj_heatmap_values",
                                help="节点数较多时不建议显示数值。",
                            )
                        with hcol2:
                            normalize_single_heatmap = st.checkbox(
                                "Normalize",
                                value=True,
                                key="normalize_single_adj_heatmap",
                                help="勾选后分别归一化非对角线边权和对角线自身效应；取消后显示原始量级。",
                            )
                        with hcol3:
                            run_single_heatmap = st.form_submit_button("Run Heatmap")

                    if st.button("Clear Heatmap", key="clear_single_adj_heatmap_btn"):
                        st.session_state["single_adj_heatmap_render"] = False

                    if run_single_heatmap:
                        st.session_state["single_adj_heatmap_render"] = True

                    if st.session_state.get("single_adj_heatmap_render", False):
                        fig = plot_adjusted_matrix_heatmap(
                            result["adj_df"],
                            title="Adjusted Matrix Heatmap",
                            show_values=bool(show_single_heatmap_values),
                            normalize=bool(normalize_single_heatmap),
                        )
                        st.pyplot(fig)
                    else:
                        st.info("点击 `Run Heatmap` 渲染热力图。")
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
            (
                labels_df,
                cluster_sizes_df,
                centers,
                center_responses,
                members,
                member_responses,
            ) = _load_funclu_export_from_zip(uploaded_funclu_zip.getvalue())
        except Exception as e:
            st.error(f"读取 FunClu-K ZIP 失败：{e}")
            st.session_state.netrecon_funclu_labels = pd.DataFrame()
            st.session_state.netrecon_funclu_cluster_sizes = pd.DataFrame()
            st.session_state.netrecon_funclu_centers = {}
            st.session_state.netrecon_funclu_center_responses = {}
            st.session_state.netrecon_funclu_members = {}
            st.session_state.netrecon_funclu_member_responses = {}
            st.session_state.netrecon_funclu_uploaded_zip_name = None
            st.session_state.netrecon_multilayer_result = None
        else:
            st.session_state.netrecon_funclu_labels = labels_df
            st.session_state.netrecon_funclu_cluster_sizes = cluster_sizes_df
            st.session_state.netrecon_funclu_centers = centers
            st.session_state.netrecon_funclu_center_responses = center_responses
            st.session_state.netrecon_funclu_members = members
            st.session_state.netrecon_funclu_member_responses = member_responses
            st.session_state.netrecon_funclu_uploaded_zip_name = uploaded_funclu_zip.name
            st.session_state.netrecon_multilayer_result = None

    labels_df = st.session_state.netrecon_funclu_labels
    cluster_sizes_df = st.session_state.netrecon_funclu_cluster_sizes
    centers = st.session_state.netrecon_funclu_centers
    center_responses = st.session_state.netrecon_funclu_center_responses
    members = st.session_state.netrecon_funclu_members
    member_responses = st.session_state.netrecon_funclu_member_responses

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
                center_response_df = center_responses.get(cond_name)
                member_response_count = len(member_responses.get(cond_name, {}))
                overview_rows.append(
                    {
                        "condition": cond_name,
                        "center_shape": str(center_df.shape),
                        "center_response_shape": (
                            str(center_response_df.shape)
                            if center_response_df is not None
                            else "missing"
                        ),
                        "member_clusters": len(members.get(cond_name, {})),
                        "member_response_clusters": member_response_count,
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
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    with mc1:
                        ml_max_order = st.number_input(
                            "max_order_upper (BIC)",
                            min_value=1,
                            max_value=100,
                            value=1,
                            step=1,
                            key="netrecon_ml_max_order",
                            help="ASGL uses BIC to select max_order from 1..this value.",
                        )
                    with mc2:
                        ml_nonneg_self = st.checkbox(
                            "nonneg_self",
                            value=True,
                            key="netrecon_ml_nonneg_self",
                        )
                    with mc3:
                        ml_top_k = st.number_input(
                            "max_interactions (Top-K)",
                            min_value=0,
                            max_value=100,
                            value=0,
                            step=1,
                            key="netrecon_ml_top_k",
                        )
                    with mc4:
                        ml_adjacency_aggregation = st.selectbox(
                            "adjacency_aggregation",
                            options=["mean", "integral"],
                            index=0,
                            key="netrecon_ml_adjacency_aggregation",
                            help="邻接矩阵列聚合方式：mean=离散点均值，integral=按 index 梯形积分。",
                        )
                    with mc5:
                        submit_multilayer = st.form_submit_button("Run Multi-Layer IdopNetwork")

            if submit_multilayer:
                params = {
                    "max_order": int(ml_max_order),
                    "nonneg_self": bool(ml_nonneg_self),
                    "max_interactions": int(ml_top_k),
                    "adjacency_aggregation": str(ml_adjacency_aggregation),
                }
                inter_cluster: dict[str, dict] = {}
                intra_cluster: dict[str, dict[str, dict]] = {}
                skipped: list[dict[str, str | int]] = []
                total_jobs = len(condition_names) + sum(
                    len(cluster_map) for cluster_map in members.values()
                )
                total_jobs = max(total_jobs, 1)
                completed_jobs = 0

                def _update_multilayer_progress(message: str) -> None:
                    """更新多层网络构建进度条。"""
                    progress = min(completed_jobs / total_jobs, 1.0)
                    progress_bar.progress(progress, text=message)
                    status_log.write(f"`[{progress:.0%}]` {message}")

                try:
                    with st.status(
                        "Preparing Multi-Layer IdopNetwork...",
                        expanded=True,
                    ) as ml_status:
                        status_log = st.empty()
                        progress_bar = st.progress(0, text="Preparing...")
                        for cond_name in condition_names:
                            center_df = centers[cond_name]
                            if cond_name not in center_responses:
                                skipped.append(
                                    {
                                        "layer": "condition",
                                        "condition": cond_name,
                                        "cluster": "",
                                        "n_nodes": int(center_df.shape[1]),
                                        "reason": (
                                            "missing cluster_center_quasi_dynamic "
                                            "in FunClu-K export"
                                        ),
                                    }
                                )
                                completed_jobs += 1 + len(members.get(cond_name, {}))
                                _update_multilayer_progress(
                                    f"Skipped condition `{cond_name}`: missing center response."
                                )
                                continue
                            center_response_df = center_responses[cond_name]
                            _update_multilayer_progress(
                                f"Building inter-cluster network: `{cond_name}`..."
                            )
                            try:
                                inter_cluster[cond_name] = _fit_idop_network_from_curve_sample(
                                    center_df,
                                    center_response_df,
                                    **params,
                                    power_function_params=get_power_function_params(
                                        center_response_df
                                    ),
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
                            completed_jobs += 1
                            _update_multilayer_progress(
                                f"Finished inter-cluster network: `{cond_name}`."
                            )

                            intra_cluster[cond_name] = {}
                            for cluster_name, member_df in members.get(cond_name, {}).items():
                                _update_multilayer_progress(
                                    "Building intra-cluster network: "
                                    f"`{cond_name}` / `{cluster_name}`..."
                                )
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
                                    completed_jobs += 1
                                    _update_multilayer_progress(
                                        "Skipped intra-cluster network: "
                                        f"`{cond_name}` / `{cluster_name}`."
                                    )
                                    continue
                                try:
                                    member_response_df = member_responses.get(
                                        cond_name, {}
                                    )[cluster_name]
                                    intra_cluster[cond_name][cluster_name] = (
                                        _fit_idop_network_from_curve_sample(
                                            member_df,
                                            member_response_df,
                                            **params,
                                            power_function_params=get_power_function_params(
                                                member_response_df
                                            ),
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
                                completed_jobs += 1
                                _update_multilayer_progress(
                                    "Finished intra-cluster network: "
                                    f"`{cond_name}` / `{cluster_name}`."
                                )
                except Exception as e:
                    ml_status.update(
                        label="Multi-Layer IdopNetwork failed.",
                        state="error",
                    )
                    st.error(f"Multi-Layer IdopNetwork 运行失败：{e}")
                    st.session_state.netrecon_multilayer_result = None
                else:
                    st.session_state.netrecon_multilayer_result = {
                        "params": params,
                        "inter_cluster": inter_cluster,
                        "intra_cluster": intra_cluster,
                        "skipped": skipped,
                    }
                    # 重置所有子 Tab 的渲染标记，避免新结果自动触发旧绘图
                    st.session_state["ml_inter_network_plot_request"] = None
                    st.session_state["ml_intra_network_plot_request"] = None
                    st.session_state["ml_effect_plot_request"] = None
                    st.session_state["ml_inter_heatmap_request"] = None
                    st.session_state["ml_intra_heatmap_request"] = None
                    ml_status.update(
                        label=f"Multi-Layer IdopNetwork completed ({completed_jobs}/{total_jobs} jobs).",
                        state="complete",
                    )

            multilayer_result = st.session_state.netrecon_multilayer_result
            if multilayer_result is not None:
                available_intra = {
                    cond_name: cluster_map
                    for cond_name, cluster_map in multilayer_result["intra_cluster"].items()
                    if cluster_map
                }
                # 嵌套 Tab 展示结果
                tab2_2_1, tab2_2_2, tab2_2_3 = st.tabs(["Network", "Effect Decomposition", "Adjacency Matrix"])

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

                        with st.form(key="ml_inter_network_form"):
                            inter_condition = st.selectbox(
                                "Inter-cluster condition",
                                options=list(multilayer_result["inter_cluster"].keys()),
                                key="netrecon_ml_inter_condition",
                            )
                            inter_adj_df = multilayer_result["inter_cluster"][inter_condition]["adj_df"]
                            inter_target_node = st.selectbox(
                                "Inter-cluster target node filter",
                                options=[""] + list(inter_adj_df.columns),
                                format_func=lambda x: "ALL" if x == "" else x,
                                key="netrecon_ml_inter_target_node",
                            )
                            inter_top_edges_for_plot = st.number_input(
                                "Top edges",
                                min_value=1,
                                max_value=500,
                                value=60,
                                step=1,
                                key="ml_inter_top_edges_for_plot",
                            )
                            c1, c2 = st.columns(2)
                            with c1:
                                run_inter_network_plot = st.form_submit_button("Run Inter Network")
                            with c2:
                                run_inter_network_degree_plot = st.form_submit_button("Run Inter Network + Degree")

                        clear_inter_network_plot = st.button(
                            "Clear Inter Plot",
                            key="clear_ml_inter_network_plot_btn",
                        )

                        if run_inter_network_plot:
                            st.session_state["ml_inter_network_plot_request"] = {
                                "condition": inter_condition,
                                "target_node": inter_target_node,
                                "top_edges": int(inter_top_edges_for_plot),
                                "show_degree_panel": False,
                            }

                        if run_inter_network_degree_plot:
                            st.session_state["ml_inter_network_plot_request"] = {
                                "condition": inter_condition,
                                "target_node": inter_target_node,
                                "top_edges": int(inter_top_edges_for_plot),
                                "show_degree_panel": True,
                            }

                        if clear_inter_network_plot:
                            st.session_state["ml_inter_network_plot_request"] = None

                        inter_plot_request = st.session_state.get("ml_inter_network_plot_request", None)

                        if (
                            inter_plot_request is not None
                            and inter_plot_request.get("condition") == inter_condition
                            and inter_plot_request.get("target_node") == inter_target_node
                            and int(inter_plot_request.get("top_edges", 60)) == int(inter_top_edges_for_plot)
                        ):
                            fig = plot_network(
                                inter_adj_df,
                                target_node=inter_target_node,
                                top_edges=int(inter_plot_request["top_edges"]),
                                show_degree_panel=bool(inter_plot_request["show_degree_panel"]),
                            )
                            st.pyplot(fig)
                        else:
                            st.info("Click `Run Inter Network` or `Run Inter Network + Degree` to render.")
                    else:
                        st.warning("No inter-cluster network was built.")

                    if available_intra:
                        st.markdown("### Intra-Cluster Network")

                        with st.form(key="ml_intra_network_form"):
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
                                options=[""] + list(intra_adj_df.columns),
                                format_func=lambda x: "ALL" if x == "" else x,
                                key="netrecon_ml_intra_target_node",
                            )

                            with st.expander("Intra-cluster plot controls", expanded=True):
                                c1, c2, c3 = st.columns(3)

                                with c1:
                                    intra_top_edges_for_plot = st.number_input(
                                        "Top edges",
                                        min_value=1,
                                        max_value=500,
                                        value=60,
                                        step=1,
                                        key="ml_intra_top_edges_for_plot",
                                    )

                                with c2:
                                    run_intra_network_plot = st.form_submit_button(
                                        "Run Intra Network",
                                    )

                                with c3:
                                    run_intra_network_degree_plot = st.form_submit_button(
                                        "Run Intra Network + Degree",
                                    )

                        clear_intra_network_plot = st.button(
                            "Clear Intra Plot",
                            key="clear_ml_intra_network_plot_btn",
                        )

                        if run_intra_network_plot:
                            st.session_state["ml_intra_network_plot_request"] = {
                                "condition": intra_condition,
                                "cluster": intra_cluster_name,
                                "target_node": intra_target_node,
                                "top_edges": int(intra_top_edges_for_plot),
                                "show_degree_panel": False,
                            }

                        if run_intra_network_degree_plot:
                            st.session_state["ml_intra_network_plot_request"] = {
                                "condition": intra_condition,
                                "cluster": intra_cluster_name,
                                "target_node": intra_target_node,
                                "top_edges": int(intra_top_edges_for_plot),
                                "show_degree_panel": True,
                            }

                        if clear_intra_network_plot:
                            st.session_state["ml_intra_network_plot_request"] = None

                        intra_plot_request = st.session_state.get("ml_intra_network_plot_request", None)

                        if (
                            intra_plot_request is not None
                            and intra_plot_request.get("condition") == intra_condition
                            and intra_plot_request.get("cluster") == intra_cluster_name
                            and intra_plot_request.get("target_node") == intra_target_node
                            and int(intra_plot_request.get("top_edges", 60)) == int(intra_top_edges_for_plot)
                        ):
                            fig = plot_network(
                                intra_adj_df,
                                target_node=intra_target_node,
                                top_edges=int(intra_plot_request["top_edges"]),
                                show_degree_panel=bool(intra_plot_request["show_degree_panel"]),
                            )
                            st.pyplot(fig)
                        else:
                            st.info("Click `Run Intra Network` or `Run Intra Network + Degree` to render.")
                    else:
                        st.warning("No intra-cluster network was built.")

                # ========== Tab 2_2_2 Effect Decomposition ==========
                with tab2_2_2:
                    has_inter = bool(multilayer_result["inter_cluster"])
                    has_intra = bool(available_intra)
                    layer_options = []
                    if has_inter:
                        layer_options.append("inter_cluster")
                    if has_intra:
                        layer_options.append("intra_cluster")

                    if not layer_options:
                        st.warning("No built network available for effect decomposition.")
                    else:
                        with st.form(key="ml_effect_form"):
                            effect_layer = st.selectbox(
                                "Layer",
                                options=layer_options,
                                key="netrecon_ml_effect_layer",
                            )
                            if effect_layer == "inter_cluster":
                                effect_condition = st.selectbox(
                                    "Condition",
                                    options=list(multilayer_result["inter_cluster"].keys()),
                                    key="netrecon_ml_effect_inter_condition",
                                )
                                effect_network = multilayer_result["inter_cluster"][effect_condition]
                                st.markdown(f"当前网络: `inter_cluster / {effect_condition}`")
                            else:
                                effect_condition = st.selectbox(
                                    "Condition",
                                    options=list(available_intra.keys()),
                                    key="netrecon_ml_effect_intra_condition",
                                )
                                effect_cluster = st.selectbox(
                                    "Cluster",
                                    options=list(available_intra[effect_condition].keys()),
                                    key="netrecon_ml_effect_intra_cluster",
                                )
                                effect_network = available_intra[effect_condition][effect_cluster]
                                st.markdown(
                                    f"当前网络: `intra_cluster / {effect_condition} / {effect_cluster}`"
                                )

                            ce1, ce2, ce3, ce4 = st.columns(4)

                            with ce1:
                                run_ml_effect_plot = st.form_submit_button(
                                    "Run Multi-Layer Effect Plot",
                                )
                            with ce2:
                                st.write("")

                            with ce3:
                                effect_ncols = st.number_input(
                                    "Subplot cols",
                                    min_value=1,
                                    max_value=12,
                                    value=4,
                                    step=1,
                                    key="ml_effect_ncols",
                                )
                            with ce4:
                                effect_max_vars = st.number_input(
                                    "Max variables (0=all)",
                                    min_value=0,
                                    max_value=500,
                                    value=0,
                                    step=1,
                                    key="ml_effect_max_vars",
                                    help="限制显示的目标变量数量；0 表示绘制全部变量。",
                                )

                        clear_ml_effect_plot = st.button(
                            "Clear Multi-Layer Effect Plot",
                            key="clear_ml_effect_plot_btn",
                        )

                        if run_ml_effect_plot:
                            if effect_layer == "inter_cluster":
                                st.session_state["ml_effect_plot_request"] = {
                                    "layer": effect_layer,
                                    "condition": effect_condition,
                                    "cluster": "",
                                    "ncols": int(effect_ncols),
                                    "max_vars": int(effect_max_vars),
                                }
                            else:
                                st.session_state["ml_effect_plot_request"] = {
                                    "layer": effect_layer,
                                    "condition": effect_condition,
                                    "cluster": effect_cluster,
                                    "ncols": int(effect_ncols),
                                    "max_vars": int(effect_max_vars),
                                }

                        if clear_ml_effect_plot:
                            st.session_state["ml_effect_plot_request"] = None

                        effect_plot_request = st.session_state.get("ml_effect_plot_request", None)

                        if effect_layer == "inter_cluster":
                            current_effect_key = {
                                "layer": effect_layer,
                                "condition": effect_condition,
                                "cluster": "",
                                "ncols": int(effect_ncols),
                                "max_vars": int(effect_max_vars),
                            }
                        else:
                            current_effect_key = {
                                "layer": effect_layer,
                                "condition": effect_condition,
                                "cluster": effect_cluster,
                                "ncols": int(effect_ncols),
                                "max_vars": int(effect_max_vars),
                            }

                        if effect_plot_request == current_effect_key:
                            fig = plot_effect(
                                quasi_dynamic_df=effect_network["quasi_dynamic_df"],
                                curve_df=effect_network["predicted_df"],
                                effect_df_list=effect_network["effect_df_list"],
                                intercept=effect_network["model"].coef_.loc["intercept"],
                                plot_ncols=int(effect_plot_request.get("ncols", 4)),
                                plot_max_vars=int(effect_plot_request.get("max_vars", 0)),
                            )
                            st.pyplot(fig)
                        else:
                            st.info("Click `Run Multi-Layer Effect Plot` to render effect curves.")

                # ========== Tab 2_2_3 Adjacency Matrix ==========
                # ========== Tab 2_2_3 Adjacency Matrix ==========
                with tab2_2_3:
                    if multilayer_result["inter_cluster"]:
                        st.markdown(
                            f"### Inter-Cluster Adjacency Matrix "
                            f"({st.session_state.get('netrecon_ml_inter_condition', '')})"
                        )

                        inter_conditions = list(multilayer_result["inter_cluster"].keys())

                        inter_adj_condition = st.selectbox(
                            "Inter-cluster adjacency condition",
                            options=inter_conditions,
                            key="netrecon_ml_adj_inter_condition",
                        )

                        inter_adj_for_table = multilayer_result["inter_cluster"][inter_adj_condition]["adj_df"]

                        st.caption("Direction rule: rows are source nodes, columns are target nodes.")
                        st.dataframe(inter_adj_for_table, use_container_width=True)

                        st.markdown("#### Inter-Cluster Heatmap")

                        with st.form(key="ml_inter_heatmap_form"):
                            ih1, ih2, ih3 = st.columns(3)

                            with ih1:
                                run_inter_heatmap = st.form_submit_button(
                                    "Run Inter Heatmap",
                                )

                            with ih2:
                                show_inter_heatmap_values = st.checkbox(
                                    "Show values",
                                    value=False,
                                    key="show_ml_inter_heatmap_values",
                                )
                            with ih3:
                                normalize_inter_heatmap = st.checkbox(
                                    "Normalize",
                                    value=True,
                                    key="normalize_ml_inter_heatmap",
                                    help="勾选后分别归一化非对角线边权和对角线自身效应；取消后显示原始量级。",
                                )

                        clear_inter_heatmap = st.button(
                            "Clear Inter Heatmap",
                            key="clear_ml_inter_heatmap_btn",
                        )
                            
                        if run_inter_heatmap:
                            st.session_state["ml_inter_heatmap_request"] = {
                                "condition": inter_adj_condition,
                            }

                        if clear_inter_heatmap:
                            st.session_state["ml_inter_heatmap_request"] = None

                        inter_heatmap_request = st.session_state.get("ml_inter_heatmap_request", None)

                        if (
                            inter_heatmap_request is not None
                            and inter_heatmap_request.get("condition") == inter_adj_condition
                        ):
                            fig = plot_adjusted_matrix_heatmap(
                                inter_adj_for_table,
                                title=f"Inter-Cluster Heatmap: {inter_adj_condition}",
                                show_values=bool(show_inter_heatmap_values),
                                normalize=bool(normalize_inter_heatmap),
                            )
                            st.pyplot(fig)
                        else:
                            st.info("Click `Run Inter Heatmap` to render inter-cluster heatmap.")

                    if available_intra:
                        st.markdown(
                            f"### Intra-Cluster Adjacency Matrix "
                            f"({st.session_state.get('netrecon_ml_intra_condition', '')} - "
                            f"{st.session_state.get('netrecon_ml_intra_cluster', '')})"
                        )

                        intra_adj_condition = st.selectbox(
                            "Intra-cluster adjacency condition",
                            options=list(available_intra.keys()),
                            key="netrecon_ml_adj_intra_condition",
                        )

                        intra_adj_cluster = st.selectbox(
                            "Intra-cluster adjacency cluster",
                            options=list(available_intra[intra_adj_condition].keys()),
                            key="netrecon_ml_adj_intra_cluster",
                        )

                        intra_adj_for_table = available_intra[intra_adj_condition][intra_adj_cluster]["adj_df"]

                        st.caption("Direction rule: rows are source nodes, columns are target nodes.")
                        st.dataframe(intra_adj_for_table, use_container_width=True)

                        st.markdown("#### Intra-Cluster Heatmap")

                        with st.form(key="ml_intra_heatmap_form"):
                            ah1, ah2, ah3 = st.columns(3)

                            with ah1:
                                run_intra_heatmap = st.form_submit_button(
                                    "Run Intra Heatmap",
                                )

                            with ah2:
                                show_intra_heatmap_values = st.checkbox(
                                    "Show values",
                                    value=False,
                                    key="show_ml_intra_heatmap_values",
                                )
                            with ah3:
                                normalize_intra_heatmap = st.checkbox(
                                    "Normalize",
                                    value=True,
                                    key="normalize_ml_intra_heatmap",
                                    help="勾选后分别归一化非对角线边权和对角线自身效应；取消后显示原始量级。",
                                )

                        clear_intra_heatmap = st.button(
                            "Clear Intra Heatmap",
                            key="clear_ml_intra_heatmap_btn",
                        )
                            
                        if run_intra_heatmap:
                            st.session_state["ml_intra_heatmap_request"] = {
                                "condition": intra_adj_condition,
                                "cluster": intra_adj_cluster,
                            }

                        if clear_intra_heatmap:
                            st.session_state["ml_intra_heatmap_request"] = None

                        intra_heatmap_request = st.session_state.get("ml_intra_heatmap_request", None)

                        if (
                            intra_heatmap_request is not None
                            and intra_heatmap_request.get("condition") == intra_adj_condition
                            and intra_heatmap_request.get("cluster") == intra_adj_cluster
                        ):
                            fig = plot_adjusted_matrix_heatmap(
                                intra_adj_for_table,
                                title=f"Intra-Cluster Heatmap: {intra_adj_condition} / {intra_adj_cluster}",
                                show_values=bool(show_intra_heatmap_values),
                                normalize=bool(normalize_intra_heatmap),
                            )
                            st.pyplot(fig)
                        else:
                            st.info("Click `Run Intra Heatmap` to render intra-cluster heatmap.")

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