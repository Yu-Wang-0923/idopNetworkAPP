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

from backend.curve_fitting import get_power_function_params
from backend.network_construction import (
    IDOPRegressor,
    align_response_to_design,
    polynomial_basis_expansion,
)
from backend.plot_curve_fitting import plot_curve_fitting
from backend.plot_network_construction import (
    plot_effect,
    plot_network,
    plot_adjusted_matrix_heatmap,
)
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


def _summarize_df_for_debug(name: str, df: pd.DataFrame) -> dict[str, object]:
    """生成调试面板中的数值摘要。"""
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


def _plot_matrix_lines_for_debug(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    *,
    exclude_intercept: bool = False,
) -> None:
    """绘制矩阵列随 index 变化的折线图（Debug 用）。"""
    cols = [c for c in df.columns if c != "intercept"] if exclude_intercept else list(df.columns)
    if not cols:
        st.caption("没有可绘制的矩阵列。")
        return

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


def _plot_bic_curve_for_debug(
    df: pd.DataFrame | None,
    title: str,
    x_col: str,
    x_label: str,
    *,
    log_x: bool = False,
) -> None:
    """绘制 BIC 搜索轨迹（Debug 用）。"""
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


def _render_multilayer_debug_panel(network: dict) -> None:
    """多层网络 Debug 面板，展示与单层一致的核心矩阵与统计。"""
    model_dbg: IDOPRegressor = network["model"]
    cs_raw: pd.DataFrame = network["curve_sample_df"]
    X_dbg: pd.DataFrame = network["design_X"]
    Y_dbg: pd.DataFrame = network["response_Y"]
    pred: pd.DataFrame = network["predicted_df"]
    coef: pd.DataFrame = model_dbg.coef_
    basis_raw_dbg = polynomial_basis_expansion(cs_raw, model_dbg.max_order)

    summary_rows = [
        _summarize_df_for_debug("curve_sample (raw)", cs_raw),
        _summarize_df_for_debug(
            "basis raw = y_k(τ) · Legendre_r(τ̂) (derivative-mode point values)",
            basis_raw_dbg,
        ),
        _summarize_df_for_debug(
            "design X = [intercept | ∫_{τ_1}^{τ} a_k s^{b_k} · Legendre_r(τ̂(s)) ds (analytic)]",
            X_dbg,
        ),
        _summarize_df_for_debug("response Y = curve_sample (raw)", Y_dbg),
        _summarize_df_for_debug("coef_", coef),
        _summarize_df_for_debug("predicted", pred),
    ]
    st.markdown("**Numeric summary**")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.markdown(
        f"`max_order` = **{model_dbg.max_order}** &nbsp; | &nbsp; "
        f"`alpha` = **{model_dbg.alpha}** &nbsp; | &nbsp; "
        f"`mse_` = **{model_dbg.mse_}**"
    )
    if model_dbg.effect_constraint_diagnostics_ is not None:
        st.markdown("**Effect constraint diagnostics**")
        st.dataframe(model_dbg.effect_constraint_diagnostics_, use_container_width=True)
    bic_left, bic_right = st.columns(2)
    with bic_left:
        st.markdown("**BIC vs max_order**")
        _plot_bic_curve_for_debug(
            model_dbg.bic_order_path_,
            "BIC 选择 max_order",
            "max_order",
            "max_order",
        )
    with bic_right:
        st.markdown("**BIC vs alpha**")
        _plot_bic_curve_for_debug(
            model_dbg.bic_alpha_path_,
            "BIC 选择 alpha",
            "alpha",
            "alpha",
            log_x=True,
        )

    st.markdown("**curve_sample (raw) — head**")
    st.dataframe(cs_raw.head(), use_container_width=True)
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**basis matrix before integral — head (first 8 cols)**")
        st.dataframe(basis_raw_dbg.iloc[:5, :8], use_container_width=True)
        st.markdown("**basis matrix before integral — 沿 index 折线图**")
        _plot_matrix_lines_for_debug(
            basis_raw_dbg,
            "积分前基函数矩阵（全部列）",
            "basis_raw.index",
        )
    with right_col:
        st.markdown("**design matrix X — head (first 8 cols)**")
        st.dataframe(X_dbg.iloc[:5, :8], use_container_width=True)
        st.markdown("**design matrix X — 沿 index 折线图（不含 intercept）**")
        _plot_matrix_lines_for_debug(
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
                    c1, c2 = st.columns(2)
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
                        top_k = st.number_input(
                            "max_interactions (Top-K)",
                            min_value=0,
                            max_value=100,
                            value=0,
                            step=1,
                        )
                        adjacency_aggregation = st.selectbox(
                            "adjacency_aggregation",
                            options=["mean", "integral"],
                            index=0,
                            help="邻接矩阵列聚合方式：mean=离散点均值，integral=按 index 梯形积分。",
                        )
                    submit_run = st.form_submit_button("Run IdopNetwork")

            if submit_run:
                progress_bar = st.progress(
                    0,
                    text=f"Preparing IdopNetwork for `{selected_condition}`...",
                )
                try:
                    progress_bar.progress(10, text="Estimating power-function parameters...")
                    power_function_params = get_power_function_params(quasi_dynamic_df)
                    progress_bar.progress(20, text="Initializing constrained ASGL model...")
                    model = IDOPRegressor(
                        max_order=int(max_order),
                        mix=0.5,
                        fix_mix=False,
                        nonneg_self=bool(nonneg_self),
                        max_interactions=int(top_k),
                        adaptive_weights=False,
                    )
                    progress_bar.progress(
                        35,
                        text="Fitting model and enforcing effect constraints...",
                    )
                    model.fit(
                        curve_sample_df,
                        quasi_dynamic_df,
                        power_function_params=power_function_params,
                    )
                    progress_bar.progress(70, text="Generating prediction curves...")
                    predicted_df = model.predict(curve_sample_df)
                    progress_bar.progress(80, text="Computing effect decomposition...")
                    effect_df_list = model.effect(curve_sample_df)
                    progress_bar.progress(90, text="Building adjacency matrix...")
                    adj_df = model.adjacency_matrix(
                        curve_sample_df,
                        aggregation=str(adjacency_aggregation),
                    )
                    progress_bar.progress(95, text="Preparing debug matrices...")
                    design_X = model._design(curve_sample_df)
                    response_Y = align_response_to_design(quasi_dynamic_df, design_X.index)
                except Exception as e:
                    progress_bar.progress(100, text="IdopNetwork failed.")
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
                    progress_bar.progress(100, text="IdopNetwork completed.")
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
                        options=[""] + list(result["adj_df"].columns),
                        format_func=lambda x: "ALL" if x == "" else x,
                        key="netrecon_target_node",
                    )

                    with st.expander("Network plot controls", expanded=True):
                        cplot1, cplot2, cplot3, cplot4 = st.columns(4)

                        with cplot1:
                            top_edges_for_plot = st.number_input(
                                "Top edges for plot",
                                min_value=1,
                                max_value=500,
                                value=60,
                                step=1,
                                key="single_top_edges_for_plot",
                            )

                        with cplot2:
                            run_network_plot = st.button(
                                "Run Network Plot",
                                key="run_single_network_plot_btn",
                            )

                        with cplot3:
                            run_network_degree_plot = st.button(
                                "Run Network + Degree Plot",
                                key="run_single_network_degree_plot_btn",
                            )

                        with cplot4:
                            clear_network_plot = st.button(
                                "Clear Plot",
                                key="clear_single_network_plot_btn",
                            )

                    if run_network_plot:
                        st.session_state["single_network_plot_mode"] = "network_only"

                    if run_network_degree_plot:
                        st.session_state["single_network_plot_mode"] = "network_degree"

                    if clear_network_plot:
                        st.session_state["single_network_plot_mode"] = None

                    single_network_plot_mode = st.session_state.get("single_network_plot_mode", None)

                    if single_network_plot_mode == "network_only":
                        plot_network(
                            result["adj_df"],
                            target_node=target_node,
                            top_edges=int(top_edges_for_plot),
                            show_degree_panel=False,
                        )
                    elif single_network_plot_mode == "network_degree":
                        plot_network(
                            result["adj_df"],
                            target_node=target_node,
                            top_edges=int(top_edges_for_plot),
                            show_degree_panel=True,
                        )
                    else:
                        st.info("Click `Run Network Plot` or `Run Network + Degree Plot` to render.")
                # ========== Tab 1_2_2 Effect Decomposition ==========
                with tab1_2_2:
                    st.markdown("### Effect Decomposition")

                    ceff1, ceff2 = st.columns(2)

                    with ceff1:
                        run_effect_plot = st.button(
                            "Run Effect Decomposition Plot",
                            key="run_single_effect_plot_btn",
                        )

                    with ceff2:
                        clear_effect_plot = st.button(
                            "Clear Effect Plot",
                            key="clear_single_effect_plot_btn",
                        )

                    if run_effect_plot:
                        st.session_state["single_effect_plot_render"] = True

                    if clear_effect_plot:
                        st.session_state["single_effect_plot_render"] = False

                    if st.session_state.get("single_effect_plot_render", False):
                        plot_effect(
                            quasi_dynamic_df=result["quasi_dynamic_df"],
                            curve_df=result["predicted_df"],
                            effect_df_list=result["effect_df_list"],
                            intercept=result["model"].coef_.loc["intercept"],
                            plot_ncols=4,
                        )
                    else:
                        st.info("Click `Run Effect Decomposition Plot` to render effect curves.")

                # ========== Tab 1_2_3 Adjacency Matrix ==========
                # ========== Tab 1_2_3 Adjacency Matrix ==========
                with tab1_2_3:
                    st.markdown("### Adjacency Matrix")
                    st.dataframe(result["adj_df"], use_container_width=True)

                    st.markdown("### Adjusted Matrix Heatmap")

                    hcol1, hcol2, hcol3 = st.columns(3)

                    with hcol1:
                        run_single_heatmap = st.button(
                            "Run Heatmap",
                            key="run_single_adj_heatmap_btn",
                        )

                    with hcol2:
                        clear_single_heatmap = st.button(
                            "Clear Heatmap",
                            key="clear_single_adj_heatmap_btn",
                        )

                    with hcol3:
                        show_single_heatmap_values = st.checkbox(
                            "Show values",
                            value=False,
                            key="show_single_adj_heatmap_values",
                            help="节点数较多时不建议显示数值。",
                        )

                    if run_single_heatmap:
                        st.session_state["single_adj_heatmap_render"] = True

                    if clear_single_heatmap:
                        st.session_state["single_adj_heatmap_render"] = False

                    if st.session_state.get("single_adj_heatmap_render", False):
                        plot_adjusted_matrix_heatmap(
                            result["adj_df"],
                            title="Adjusted Matrix Heatmap",
                            normalize_pm1=True,
                            diag_cmap="RdBu_r",
                            offdiag_cmap="PRGn",
                            show_values=bool(show_single_heatmap_values),
                        )
                    else:
                        st.info("Click `Run Heatmap` to render the adjusted matrix heatmap.")
                # ========== Tab 1_2_4 Debug ==========
                with tab1_2_4:
                    st.markdown("### Debug: design matrix, response, coefficients")
                    model_dbg: IDOPRegressor = result["model"]
                    X_dbg: pd.DataFrame = result["design_X"]
                    Y_dbg: pd.DataFrame = result["response_Y"]
                    cs_raw = result["curve_sample_df"]
                    pred = result["predicted_df"]
                    coef = model_dbg.coef_
                    basis_raw_dbg = polynomial_basis_expansion(
                        cs_raw,
                        model_dbg.max_order,
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

                    summary_rows = [
                        _summary("curve_sample (raw)", cs_raw),
                        _summary(
                            "basis raw = y_k(τ) · Legendre_r(τ̂) (derivative-mode point values)",
                            basis_raw_dbg,
                        ),
                        _summary(
                            "design X = [intercept | ∫_{τ_1}^{τ} a_k s^{b_k} · Legendre_r(τ̂(s)) ds (analytic)]",
                            X_dbg,
                        ),
                        _summary(
                            "response Y = quasi_dynamic (interp → Chebyshev nodes)",
                            Y_dbg,
                        ),
                        _summary("coef_", coef),
                        _summary("predicted", pred),
                    ]
                    st.markdown("**Numeric summary**")
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    st.markdown(
                        f"`max_order` = **{model_dbg.max_order}** &nbsp; | &nbsp; "
                        f"`alpha` = **{model_dbg.alpha}** &nbsp; | &nbsp; "
                        f"`mse_` = **{model_dbg.mse_}**"
                    )
                    if model_dbg.effect_constraint_diagnostics_ is not None:
                        st.markdown("**Effect constraint diagnostics**")
                        st.dataframe(
                            model_dbg.effect_constraint_diagnostics_,
                            use_container_width=True,
                        )
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
                    mc1, mc2 = st.columns(2)
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
                        ml_top_k = st.number_input(
                            "max_interactions (Top-K)",
                            min_value=0,
                            max_value=100,
                            value=0,
                            step=1,
                            key="netrecon_ml_top_k",
                        )
                        ml_adjacency_aggregation = st.selectbox(
                            "adjacency_aggregation",
                            options=["mean", "integral"],
                            index=0,
                            key="netrecon_ml_adjacency_aggregation",
                            help="邻接矩阵列聚合方式：mean=离散点均值，integral=按 index 梯形积分。",
                        )
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
                progress_bar = st.progress(
                    0,
                    text="Preparing Multi-Layer IdopNetwork...",
                )

                def _update_multilayer_progress(message: str) -> None:
                    """更新多层网络构建进度条。"""
                    progress = min(completed_jobs / total_jobs, 1.0)
                    progress_bar.progress(progress, text=message)

                try:
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
                    progress_bar.progress(1.0, text="Multi-Layer IdopNetwork failed.")
                    st.error(f"Multi-Layer IdopNetwork 运行失败：{e}")
                    st.session_state.netrecon_multilayer_result = None
                else:
                    st.session_state.netrecon_multilayer_result = {
                        "params": params,
                        "inter_cluster": inter_cluster,
                        "intra_cluster": intra_cluster,
                        "skipped": skipped,
                    }
                    progress_bar.progress(1.0, text="Multi-Layer IdopNetwork completed.")
                    st.success("Done")

            multilayer_result = st.session_state.netrecon_multilayer_result
            if multilayer_result is not None:
                available_intra = {
                    cond_name: cluster_map
                    for cond_name, cluster_map in multilayer_result["intra_cluster"].items()
                    if cluster_map
                }
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
                            options=[""] + list(inter_adj_df.columns),
                            format_func=lambda x: "ALL" if x == "" else x,
                            key="netrecon_ml_inter_target_node",
                        )

                        with st.expander("Inter-cluster plot controls", expanded=True):
                            c1, c2, c3, c4 = st.columns(4)

                            with c1:
                                inter_top_edges_for_plot = st.number_input(
                                    "Top edges",
                                    min_value=1,
                                    max_value=500,
                                    value=60,
                                    step=1,
                                    key="ml_inter_top_edges_for_plot",
                                )

                            with c2:
                                run_inter_network_plot = st.button(
                                    "Run Inter Network",
                                    key="run_ml_inter_network_plot_btn",
                                )

                            with c3:
                                run_inter_network_degree_plot = st.button(
                                    "Run Inter Network + Degree",
                                    key="run_ml_inter_network_degree_plot_btn",
                                )

                            with c4:
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
                            plot_network(
                                inter_adj_df,
                                target_node=inter_target_node,
                                top_edges=int(inter_plot_request["top_edges"]),
                                show_degree_panel=bool(inter_plot_request["show_degree_panel"]),
                            )
                        else:
                            st.info("Click `Run Inter Network` or `Run Inter Network + Degree` to render.")
                    else:
                        st.warning("No inter-cluster network was built.")

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
                            options=[""] + list(intra_adj_df.columns),
                            format_func=lambda x: "ALL" if x == "" else x,
                            key="netrecon_ml_intra_target_node",
                        )

                        with st.expander("Intra-cluster plot controls", expanded=True):
                            c1, c2, c3, c4 = st.columns(4)

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
                                run_intra_network_plot = st.button(
                                    "Run Intra Network",
                                    key="run_ml_intra_network_plot_btn",
                                )

                            with c3:
                                run_intra_network_degree_plot = st.button(
                                    "Run Intra Network + Degree",
                                    key="run_ml_intra_network_degree_plot_btn",
                                )

                            with c4:
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
                            plot_network(
                                intra_adj_df,
                                target_node=intra_target_node,
                                top_edges=int(intra_plot_request["top_edges"]),
                                show_degree_panel=bool(intra_plot_request["show_degree_panel"]),
                            )
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

                        plot_effect(
                            quasi_dynamic_df=effect_network["quasi_dynamic_df"],
                            curve_df=effect_network["predicted_df"],
                            effect_df_list=effect_network["effect_df_list"],
                            intercept=effect_network["model"].coef_.loc["intercept"],
                            plot_ncols=4,
                        )

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
                    has_inter = bool(multilayer_result["inter_cluster"])
                    has_intra = bool(available_intra)
                    layer_options = []
                    if has_inter:
                        layer_options.append("inter_cluster")
                    if has_intra:
                        layer_options.append("intra_cluster")

                    if not layer_options:
                        st.warning("No built network available for debug panel.")
                    else:
                        debug_layer = st.selectbox(
                            "Layer",
                            options=layer_options,
                            key="netrecon_ml_debug_layer",
                        )
                        if debug_layer == "inter_cluster":
                            debug_condition = st.selectbox(
                                "Condition",
                                options=list(multilayer_result["inter_cluster"].keys()),
                                key="netrecon_ml_debug_inter_condition",
                            )
                            debug_network = multilayer_result["inter_cluster"][debug_condition]
                            st.markdown(f"当前网络: `inter_cluster / {debug_condition}`")
                        else:
                            debug_condition = st.selectbox(
                                "Condition",
                                options=list(available_intra.keys()),
                                key="netrecon_ml_debug_intra_condition",
                            )
                            debug_cluster = st.selectbox(
                                "Cluster",
                                options=list(available_intra[debug_condition].keys()),
                                key="netrecon_ml_debug_intra_cluster",
                            )
                            debug_network = available_intra[debug_condition][debug_cluster]
                            st.markdown(
                                f"当前网络: `intra_cluster / {debug_condition} / {debug_cluster}`"
                            )

                        _render_multilayer_debug_panel(debug_network)

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