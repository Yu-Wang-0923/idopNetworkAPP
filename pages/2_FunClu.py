import io
import zipfile

import streamlit as st

# 🌟 统一小图标和侧边栏状态
st.set_page_config(
    page_title="Functional Clustering", 
    page_icon="TSA.png", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd

from backend.functional_clustering import FunClu
from backend.plot_functional_clustering import plot_cluster_profiles
from backend.utils import load_css, setup_sidebar

# 🌟 一键加载 CSS 和侧边栏
load_css()
setup_sidebar()

st.title("Functional Clustering", text_alignment="center")

# ========== Session State ==========
if "funclu_curve_sample" not in st.session_state:
    st.session_state.funclu_curve_sample = {}  # {condition_name: pd.DataFrame}
if "funclu_uploaded_zip_name" not in st.session_state:
    st.session_state.funclu_uploaded_zip_name = None
if "funclu_em_result" not in st.session_state:
    st.session_state.funclu_em_result = None

# ========== Helper Functions ==========
def _load_curve_sample_from_zip(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    """从 curve_fitting 导出的 ZIP 中按子目录读取 ``curve_sample.csv``。"""
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if not name.endswith("curve_sample.csv"):
                continue
            cond = name.rsplit("/", 1)[0] if "/" in name else name
            with zf.open(name) as f:
                df = pd.read_csv(f, index_col=0)
            out[cond] = df
    return dict(sorted(out.items(), key=lambda kv: kv[0]))

def _build_funclu_k_export_zip(
    *,
    model: FunClu,
    cond_names: list[str],
    curve_sample_dict: dict[str, pd.DataFrame],
) -> bytes:
    """Build a ZIP package for downstream Multi-Layer idopNetwork construction."""
    if model.labels is None or model.common_cols is None:
        raise ValueError("FunClu-K result is incomplete: missing labels/common columns.")

    labels_np = model.labels.detach().cpu().numpy().astype(int)
    common_cols = list(model.common_cols)
    if len(labels_np) != len(common_cols):
        raise ValueError("FunClu-K labels do not match common curve_sample columns.")

    labels_df = pd.DataFrame(
        {
            "feature": common_cols,
            "cluster_id": labels_np + 1,
            "cluster": [f"M{k + 1}" for k in labels_np],
        }
    )
    cluster_sizes_df = (
        labels_df.groupby(["cluster_id", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_features"})
        .sort_values("cluster_id")
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labels.csv", labels_df.to_csv(index=False))
        zf.writestr("cluster_sizes.csv", cluster_sizes_df.to_csv(index=False))

        for cond_idx, cond_name in enumerate(cond_names):
            curve_sample = curve_sample_dict[cond_name].loc[:, common_cols]

            for cluster_idx in range(model.n_components):
                cluster_name = f"M{cluster_idx + 1}"
                member_cols = labels_df.loc[
                    labels_df["cluster_id"] == cluster_idx + 1, "feature"
                ].tolist()
                member_df = curve_sample.loc[:, member_cols]
                zf.writestr(
                    f"cluster_members/{cond_name}/{cluster_name}_curve_sample.csv",
                    member_df.to_csv(index=True),
                )

            center_times, center_curves = model.get_cluster_curves(cond_idx)
            center_df = pd.DataFrame(
                center_curves.T,
                index=center_times,
                columns=[f"M{k + 1}" for k in range(model.n_components)],
            )
            center_df.index.name = curve_sample.index.name or "time"
            zf.writestr(
                f"cluster_centers/{cond_name}/cluster_center_curve_sample.csv",
                center_df.to_csv(index=True),
            )

    return buffer.getvalue()


# ========== Top-Level Tabs ==========
tab1, tab2, tab3 = st.tabs(["Uploaded Data", "FunClu-K", "FunClu-BIC"])

# =====================================================================
# TAB 1: Uploaded Data (File uploader & Data Overview)
# =====================================================================
with tab1:
    # ---------- 1. 文件上传（直接位于 Tab 1 顶级） ----------
    st.markdown("### Data Initialization")
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

    if uploaded_file is not None:
        if st.session_state.funclu_uploaded_zip_name != uploaded_file.name:
            try:
                st.session_state.funclu_curve_sample = _load_curve_sample_from_zip(
                    uploaded_file.getvalue()
                )
                st.session_state.funclu_uploaded_zip_name = uploaded_file.name
                # 当上传新数据时，清空之前的 EM 拟合结果
                st.session_state.funclu_em_result = None 
                st.success("✅ File loaded successfully!")
            except Exception as e:
                st.error(f"读取 ZIP 失败：{e}")
                st.session_state.funclu_curve_sample = {}
                st.session_state.funclu_uploaded_zip_name = None

    # ---------- 2. 数据概览（拆分为多个子 Tab） ----------
    curve_sample_dict = st.session_state.get("funclu_curve_sample", {})
    
    if not curve_sample_dict:
        st.info("💡 Please upload `curve_fitting_export.zip` above to proceed.")
    else:
        st.divider()
        st.markdown("### Data Overview")
        cond_names = list(curve_sample_dict.keys())
        data_list = [curve_sample_dict[n] for n in cond_names]

        st.markdown(
            f"**Detected conditions ({len(cond_names)}):** "
            + ", ".join(f"`{n}`" for n in cond_names)
        )

        try:
            model = FunClu()
            X_list, times_list = model._prepare_data(data_list)
        except Exception as e:
            st.error(f"⚠️ 数据准备失败 (Data preparation failed): {e}")
        else:
            # 🌟 创建数据概览的子标签页
            overview_tab1, overview_tab2, overview_tab3 = st.tabs([
                "📊 Summary", 
                "🏷️ Common Features", 
                "👀 Data Preview"
            ])

            # --- 子标签页 1: Summary (核心指标与维度摘要) ---
            with overview_tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Conditions", model.n_conditions)
                with col2:
                    st.metric("Common Features", model.n_features)
                with col3:
                    per_cond_str = ", ".join(str(d) for d in model.n_times_conditions)
                    st.metric(
                        "Time Points (Total)",
                        f"{sum(model.n_times_conditions)}",
                        delta=f"Per condition: {per_cond_str}",
                        delta_color="off",
                    )
                
                st.write("")
                st.markdown("#### Per-Condition Summary")
                summary_df = pd.DataFrame([
                    {
                        "Condition": n,
                        "Time Points": int(model.n_times_conditions[i]),
                        "Shape (Features × Times)": f"{X_list[i].shape[0]} × {X_list[i].shape[1]}",
                        "Time Min": float(times_list[i].min()),
                        "Time Max": float(times_list[i].max()),
                    }
                    for i, n in enumerate(cond_names)
                ])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # --- 子标签页 2: Common Features (共有特征列表) ---
            with overview_tab2:
                st.markdown(f"#### Shared Features Across All Conditions ({model.n_features})")
                features_df = pd.DataFrame({"Feature Name": model.common_cols})
                # 给 dataframe 限制一下高度，防止特征太多拉得太长
                st.dataframe(features_df, use_container_width=True, hide_index=True, height=300)

            # --- 子标签页 3: Data Preview (数据预览) ---
            with overview_tab3:
                # 避免嵌套 tabs 引发 Streamlit 报错，使用 selectbox 切换查看不同的 condition 数据
                col_sel, col_empty = st.columns([1, 2])
                with col_sel:
                    selected_cond = st.selectbox(
                        "Select Condition to Preview:", 
                        options=cond_names,
                        key="overview_preview_select"
                    )
                
                st.markdown(f"**Top 5 rows × 5 cols for `{selected_cond}`:**")
                st.dataframe(
                    curve_sample_dict[selected_cond].iloc[:5, :5],
                    use_container_width=True,
                )

# =====================================================================
# TAB 2: FunClu-K (EM Fitting & Export)
# =====================================================================
with tab2:
    tab2_1, tab2_2 = st.tabs(["EM Fitting", "Export"])

    # ---------- EM Fitting ----------
    with tab2_1:
        curve_sample_dict = st.session_state.get("funclu_curve_sample", {})
        if not curve_sample_dict:
            st.info("Please upload data in the 'Uploaded Data' tab first.")
        else:
            cond_names = list(curve_sample_dict.keys())
            data_list = [curve_sample_dict[n] for n in cond_names]
            n_features_max = min((d.shape[1] for d in data_list), default=2)
            k_upper = max(2, min(20, n_features_max))

            with st.expander("FunClu parameter settings", expanded=True):
                with st.form(key="funclu_em_form"):
                    K_em = st.slider(
                        "n_components (K)",
                        min_value=2,
                        max_value=k_upper,
                        value=min(3, k_upper),
                        step=1,
                        key="funclu_em_K",
                    )
                    submit_em = st.form_submit_button("Run EM Fitting")

            if submit_em:
                try:
                    em_model = FunClu(
                        n_components=int(K_em),
                        use_minibatch_kmeans=None,
                        random_state=42,
                    )
                    em_model.fit(data_list)
                except Exception as e:
                    st.error(f"EM 拟合失败：{e}")
                    st.session_state.funclu_em_result = None
                else:
                    st.session_state.funclu_em_result = {
                        "model": em_model,
                        "cond_names": cond_names,
                    }
                    st.success(
                        f"Done. converged={em_model.converged}, "
                        f"iters={em_model.n_iter_run}/{em_model.max_iter}, "
                        f"log-lik={em_model.log_likelihood:.4g}, "
                        f"BIC={em_model.bic:.4g}"
                    )

            em = st.session_state.funclu_em_result
            if em is not None:
                em_model: FunClu = em["model"]
                em_cond_names = em["cond_names"]

                # 簇大小 & 权重
                em_labels_np = em_model.labels.detach().cpu().numpy()
                em_sizes = np.bincount(
                    em_labels_np, minlength=em_model.n_components
                )
                em_weights_np = em_model.weights.detach().cpu().numpy()
                em_sw_df = pd.DataFrame(
                    {
                        "cluster": [
                            f"M{k + 1}" for k in range(em_model.n_components)
                        ],
                        "size": em_sizes.astype(int),
                        "weight": em_weights_np,
                    }
                )
                st.markdown("**Cluster sizes & weights (after EM)**")
                st.dataframe(em_sw_df, use_container_width=True)

                # 参数表
                em_mu_np = em_model.params_mu.detach().cpu().numpy()
                em_cov_np = em_model.params_cov.detach().cpu().numpy()
                em_mu_rows = [
                    {
                        "cluster": f"M{k + 1}",
                        "condition": em_cond_names[i],
                        "a": float(em_mu_np[k, i, 0]),
                        "b": float(em_mu_np[k, i, 1]),
                    }
                    for k in range(em_model.n_components)
                    for i in range(em_model.n_conditions)
                ]
                em_cov_rows = [
                    {
                        "cluster": f"M{k + 1}",
                        "condition": em_cond_names[i],
                        "phi": float(em_cov_np[k, i, 0]),
                        "gamma": float(em_cov_np[k, i, 1]),
                    }
                    for k in range(em_model.n_components)
                    for i in range(em_model.n_conditions)
                ]
                
                col_mu, col_cov = st.columns(2)
                with col_mu:
                    st.markdown(r"**mu_params (a, b)** — $\mu = a \cdot t^{b}$")
                    st.dataframe(pd.DataFrame(em_mu_rows), use_container_width=True)
                with col_cov:
                    st.markdown(r"**cov_params (phi, gamma)** — SAD1")
                    st.dataframe(pd.DataFrame(em_cov_rows), use_container_width=True)

                st.markdown("**Cluster profiles**")
                with st.expander("Cluster profiles plot settings", expanded=False):
                    with st.form(key="funclu_cluster_profiles_form"):
                        layout_col, member_col, scale_col = st.columns(3)
                        with layout_col:
                            em_profile_layout = st.selectbox(
                                "Profile layout",
                                options=["combined", "k_by_l", "l_by_k"],
                                format_func={
                                    "combined": "Combined: conditions in one panel",
                                    "k_by_l": "K x L: cluster rows",
                                    "l_by_k": "L x K: condition rows",
                                }.get,
                                key="funclu_em_profile_layout",
                            )
                            em_n_cols = st.number_input(
                                "Subplot cols (combined only)",
                                min_value=1,
                                max_value=max(em_model.n_components, 1),
                                value=min(3, em_model.n_components),
                                step=1,
                                key="funclu_em_ncols",
                            )
                        with member_col:
                            em_member_source = st.selectbox(
                                "Member source",
                                ["curve", "qd_df"],
                                index=0,
                                key="funclu_em_member_src",
                                help=(
                                    "curve: 每个成员一条细线（基于 curve_sample）；"
                                    "qd_df: 每个时间点 × 每成员的散点"
                                    "（适合配 quasi_dynamic）。"
                                ),
                            )
                            em_show_members = st.checkbox(
                                "Show members",
                                value=True,
                                key="funclu_em_show_members",
                            )
                            em_show_mean = st.checkbox(
                                "Show mean curve",
                                value=True,
                                key="funclu_em_show_mean",
                            )
                            em_show_ci = st.checkbox(
                                "Show CI band (±1.96 SE)",
                                value=False,
                                key="funclu_em_show_ci",
                            )
                        with scale_col:
                            em_use_log_y = st.checkbox(
                                "log-scale Y",
                                value=False,
                                key="funclu_em_log_y",
                            )
                            em_use_log_x = st.checkbox(
                                "log-scale X",
                                value=False,
                                key="funclu_em_log_x",
                            )
                            em_show_legend = st.checkbox(
                                "Show legend",
                                value=True,
                                key="funclu_em_show_legend",
                            )
                        st.form_submit_button("Apply plot settings")

                plot_cluster_profiles(
                    data_scatter=[curve_sample_dict[n] for n in em_cond_names],
                    labels=em_labels_np,
                    common_cols=em_model.common_cols,
                    n_components=em_model.n_components,
                    condition_labels=em_cond_names,
                    member_source=em_member_source,
                    show_members=bool(em_show_members),
                    show_mean=bool(em_show_mean),
                    show_mean_ci=bool(em_show_ci),
                    use_semilogy=bool(em_use_log_y),
                    use_log_x=bool(em_use_log_x),
                    layout=str(em_profile_layout),
                    n_cols=int(em_n_cols),
                    show_legend=bool(em_show_legend),
                )

    # ---------- Export ----------
    with tab2_2:
        curve_sample_dict = st.session_state.get("funclu_curve_sample", {})
        em = st.session_state.funclu_em_result
        
        if not curve_sample_dict:
            st.info("Please upload data in the 'Uploaded Data' tab first.")
        elif em is None:
            st.info("Please run EM Fitting in the adjacent tab first.")
        else:
            em_model: FunClu = em["model"]
            em_cond_names = em["cond_names"]

            try:
                export_zip_bytes = _build_funclu_k_export_zip(
                    model=em_model,
                    cond_names=em_cond_names,
                    curve_sample_dict=curve_sample_dict,
                )
            except Exception as e:
                st.error(f"FunClu-K export failed: {e}")
            else:
                st.markdown("**Export contents**")
                st.write(
                    "- `labels.csv`: feature-to-cluster labels.\n"
                    "- `cluster_sizes.csv`: number of features in each cluster.\n"
                    "- `cluster_members/<condition>/M*_curve_sample.csv`: "
                    "all member curves for each cluster and condition.\n"
                    "- `cluster_centers/<condition>/cluster_center_curve_sample.csv`: "
                    "cluster-center curves generated by `FunClu.get_cluster_curves()`."
                )

                st.download_button(
                    label="Download FunClu-K export ZIP",
                    data=export_zip_bytes,
                    file_name="funclu_k_export.zip",
                    mime="application/zip",
                    key="funclu_k_export_zip_download",
                )


# =====================================================================
# TAB 3: FunClu-BIC
# =====================================================================
with tab3:
    st.write("To Be Updated...")