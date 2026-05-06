import io
import zipfile

import streamlit as st

st.set_page_config(page_title="Functional Clustering", page_icon=None, layout="wide", initial_sidebar_state="auto")

import numpy as np
import pandas as pd

from backend.functional_clustering import FunClu
from backend.plot_functional_clustering import (
    plot_cluster_profiles,
    plot_initialization_grid,
    plot_loglik_history,
)
from backend.utils import load_css

load_css()

st.title("Functional Clustering", text_alignment="center")


# ========== Session State ==========
if "funclu_curve_sample" not in st.session_state:
    st.session_state.funclu_curve_sample = {}  # {condition_name: pd.DataFrame}
if "funclu_uploaded_zip_name" not in st.session_state:
    st.session_state.funclu_uploaded_zip_name = None
if "funclu_init_result" not in st.session_state:
    st.session_state.funclu_init_result = None
if "funclu_em_result" not in st.session_state:
    st.session_state.funclu_em_result = None


def _load_curve_sample_from_zip(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    """从 curve_fitting 导出的 ZIP 中按子目录读取 ``curve_sample.csv``。

    期望的 ZIP 结构（与 ``1_Curve Fitting.py`` 的导出一致）：
        ``<condition_name>/curve_sample.csv``
        ``<condition_name>/quasi_dynamic.csv``
        ``<condition_name>/curve_params.csv``

    Args:
        zip_bytes: ZIP 文件的字节内容。

    Returns:
        按 ``condition_name`` 升序排序的有序映射：``{condition_name: DataFrame}``，
        DataFrame 的第一列被作为行索引（即时间/τ）。
    """
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


# ========== File Upload ==========
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
        except Exception as e:
            st.error(f"读取 ZIP 失败：{e}")
            st.session_state.funclu_curve_sample = {}
            st.session_state.funclu_uploaded_zip_name = None


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["FunClu-K", "FunClu-BIC", "To Be Updated..."])

# ========== Tab 1 ==========
with tab1:
    tab1_1, tab1_2, tab1_3 = st.tabs(["Data Overview", "Initialization", "EM Fitting"])

    # ---------- Data Overview ----------
    with tab1_1:
        curve_sample_dict = st.session_state.funclu_curve_sample
        if not curve_sample_dict:
            st.info("Please upload curve_fitting_export.zip first.")
        else:
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
                st.error(f"数据准备失败：{e}")
            else:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("n_conditions", model.n_conditions)
                with col_b:
                    st.metric("n_features (common cols)", model.n_features)
                with col_c:
                    st.metric(
                        "n_times (sum / per-cond)",
                        f"{sum(model.n_times_conditions)}",
                        delta=", ".join(str(d) for d in model.n_times_conditions),
                        delta_color="off",
                    )

                shape_rows = [
                    {
                        "condition": n,
                        "n_times": int(model.n_times_conditions[i]),
                        "X_shape (n_features, n_times)": str(tuple(X_list[i].shape)),
                        "time_min": float(times_list[i].min()),
                        "time_max": float(times_list[i].max()),
                    }
                    for i, n in enumerate(cond_names)
                ]
                st.markdown("**Per-condition summary**")
                st.dataframe(pd.DataFrame(shape_rows), use_container_width=True)

                with st.expander(
                    f"Common columns ({model.n_features})", expanded=False
                ):
                    st.write(model.common_cols)

                st.markdown("**curve_sample preview** (first 5 rows × first 5 cols)")
                for n in cond_names:
                    with st.expander(f"`{n}`", expanded=False):
                        st.dataframe(
                            curve_sample_dict[n].iloc[:5, :5],
                            use_container_width=True,
                        )

    # ---------- Initialization ----------
    with tab1_2:
        curve_sample_dict = st.session_state.funclu_curve_sample
        if not curve_sample_dict:
            st.info("Please upload curve_fitting_export.zip first.")
        else:
            cond_names = list(curve_sample_dict.keys())
            data_list = [curve_sample_dict[n] for n in cond_names]
            n_features_max = min((d.shape[1] for d in data_list), default=2)
            k_upper = max(2, min(20, n_features_max))

            with st.form(key="funclu_init_form"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    K = st.slider(
                        "n_components (K)",
                        min_value=2,
                        max_value=k_upper,
                        value=min(3, k_upper),
                        step=1,
                    )
                with col2:
                    mb_choice = st.selectbox(
                        "MiniBatchKMeans",
                        ["auto", "on", "off"],
                        help=(
                            "auto: 当 n_features ≥ kmeans_minibatch_threshold (默认 8000) "
                            "时自动启用 MiniBatchKMeans；否则用全量 KMeans。"
                        ),
                    )
                with col3:
                    random_state = st.number_input(
                        "random_state", value=42, min_value=0, max_value=2**31 - 1, step=1
                    )
                with col4:
                    mb_threshold = st.number_input(
                        "minibatch_threshold",
                        value=8000,
                        min_value=100,
                        max_value=10**7,
                        step=100,
                    )
                submit_init = st.form_submit_button("Run Initialization")

            if submit_init:
                mb_map = {"auto": None, "on": True, "off": False}
                try:
                    model = FunClu(
                        n_components=int(K),
                        use_minibatch_kmeans=mb_map[mb_choice],
                        random_state=int(random_state),
                        kmeans_minibatch_threshold=int(mb_threshold),
                    )
                    X_list, _ = model._prepare_data(data_list)
                    init = model._initialize(X_list)
                except Exception as e:
                    st.error(f"初始化失败：{e}")
                    st.session_state.funclu_init_result = None
                else:
                    st.session_state.funclu_init_result = {
                        "model": model,
                        "init": init,
                        "X_list": X_list,
                        "cond_names": cond_names,
                    }
                    st.success(
                        f"Done. Backend = {init['backend']}, K = {model.n_components}, "
                        f"N = {model.n_features}."
                    )

            res = st.session_state.funclu_init_result
            if res is not None:
                model = res["model"]
                init = res["init"]
                X_list = res["X_list"]
                cond_names = res["cond_names"]

                # 概览 metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("N (features)", model.n_features)
                m2.metric("K (clusters)", model.n_components)
                m3.metric("L (conditions)", model.n_conditions)
                m4.metric("KMeans backend", init["backend"])

                # 簇大小 & 权重
                labels_np = init["labels"].cpu().numpy()
                sizes = np.bincount(labels_np, minlength=model.n_components)
                weights_np = init["weights"].cpu().numpy()
                sw_df = pd.DataFrame(
                    {
                        "cluster": [f"M{k + 1}" for k in range(model.n_components)],
                        "size": sizes.astype(int),
                        "weight": weights_np,
                    }
                )
                st.markdown("**Cluster sizes & weights**")
                st.dataframe(sw_df, use_container_width=True)

                # mu_params (a, b)
                mu_np = init["mu_params"].cpu().numpy()
                mu_rows = [
                    {
                        "cluster": f"M{k + 1}",
                        "condition": cond_names[i],
                        "a": float(mu_np[k, i, 0]),
                        "b": float(mu_np[k, i, 1]),
                    }
                    for k in range(model.n_components)
                    for i in range(model.n_conditions)
                ]
                st.markdown(r"**mu_params (a, b)** — power law $\mu = a \cdot t^{b}$")
                st.dataframe(pd.DataFrame(mu_rows), use_container_width=True)

                # cov_params (phi, gamma)
                cov_np = init["cov_params"].cpu().numpy()
                cov_rows = [
                    {
                        "cluster": f"M{k + 1}",
                        "condition": cond_names[i],
                        "phi": float(cov_np[k, i, 0]),
                        "gamma": float(cov_np[k, i, 1]),
                    }
                    for k in range(model.n_components)
                    for i in range(model.n_conditions)
                ]
                st.markdown(r"**cov_params (phi, gamma)** — SAD1 covariance")
                st.dataframe(pd.DataFrame(cov_rows), use_container_width=True)

                # 网格图 + 布局切换（不重跑 init）
                st.markdown("**Initialization grid**")
                grid_col1, grid_col2 = st.columns([1, 3])
                with grid_col1:
                    layout_choice = st.selectbox(
                        "Grid layout",
                        ["k_by_l", "l_by_k"],
                        index=0,
                        key="funclu_grid_layout",
                        help=(
                            "k_by_l: K 行 × L 列（每行一个 cluster）。"
                            "l_by_k: L 行 × K 列（每行一个 condition）。"
                        ),
                    )
                    use_log_y = st.checkbox(
                        "log-scale Y", value=False, key="funclu_grid_log_y"
                    )
                    use_log_x = st.checkbox(
                        "log-scale X", value=False, key="funclu_grid_log_x"
                    )
                    share_x = st.checkbox(
                        "Share X", value=True, key="funclu_grid_share_x"
                    )
                    share_y = st.checkbox(
                        "Share Y", value=True, key="funclu_grid_share_y"
                    )
                plot_initialization_grid(
                    X_list=X_list,
                    times_list=model.times_list,
                    labels=labels_np,
                    centers_kl=init["centers_kl"],
                    params_mu=mu_np,
                    condition_labels=cond_names,
                    layout=layout_choice,
                    use_semilogy=bool(use_log_y),
                    use_semilogx=bool(use_log_x),
                    share_x=share_x,
                    share_y=share_y,
                )

    # ---------- EM Fitting ----------
    with tab1_3:
        curve_sample_dict = st.session_state.funclu_curve_sample
        if not curve_sample_dict:
            st.info("Please upload curve_fitting_export.zip first.")
        else:
            cond_names = list(curve_sample_dict.keys())
            data_list = [curve_sample_dict[n] for n in cond_names]
            n_features_max = min((d.shape[1] for d in data_list), default=2)
            k_upper = max(2, min(20, n_features_max))

            with st.form(key="funclu_em_form"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    K_em = st.slider(
                        "n_components (K)",
                        min_value=2,
                        max_value=k_upper,
                        value=min(3, k_upper),
                        step=1,
                        key="funclu_em_K",
                    )
                with col2:
                    mb_choice_em = st.selectbox(
                        "MiniBatchKMeans",
                        ["auto", "on", "off"],
                        key="funclu_em_mb",
                        help=(
                            "auto: 当 n_features ≥ kmeans_minibatch_threshold 时自动启用。"
                        ),
                    )
                with col3:
                    random_state_em = st.number_input(
                        "random_state",
                        value=42,
                        min_value=0,
                        max_value=2**31 - 1,
                        step=1,
                        key="funclu_em_seed",
                    )
                with col4:
                    mb_threshold_em = st.number_input(
                        "minibatch_threshold",
                        value=8000,
                        min_value=100,
                        max_value=10**7,
                        step=100,
                        key="funclu_em_mb_thr",
                    )
                col5, col6, col7 = st.columns(3)
                with col5:
                    max_iter = st.number_input(
                        "max_iter",
                        value=50,
                        min_value=1,
                        max_value=1000,
                        step=1,
                        key="funclu_em_max_iter",
                        help="EM 主循环最大迭代次数。",
                    )
                with col6:
                    tol = st.number_input(
                        "tol",
                        value=1e-4,
                        min_value=1e-12,
                        max_value=1.0,
                        step=1e-5,
                        format="%.1e",
                        key="funclu_em_tol",
                        help="|Δlog_likelihood| < tol 视为收敛。",
                    )
                with col7:
                    verbose_em = st.checkbox(
                        "verbose (terminal)",
                        value=False,
                        key="funclu_em_verbose",
                        help="勾选后会在运行 Streamlit 的终端打印每轮 log-lik。",
                    )
                submit_em = st.form_submit_button("Run EM Fitting")

            if submit_em:
                mb_map = {"auto": None, "on": True, "off": False}
                try:
                    em_model = FunClu(
                        n_components=int(K_em),
                        max_iter=int(max_iter),
                        tol=float(tol),
                        use_minibatch_kmeans=mb_map[mb_choice_em],
                        random_state=int(random_state_em),
                        kmeans_minibatch_threshold=int(mb_threshold_em),
                    )
                    em_model.fit(data_list, verbose=bool(verbose_em))
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

                # 概览 metrics
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("K", em_model.n_components)
                m2.metric("L", em_model.n_conditions)
                m3.metric("converged", "yes" if em_model.converged else "no")
                m4.metric(
                    "iters",
                    f"{em_model.n_iter_run}/{em_model.max_iter}",
                )
                m5.metric(
                    "log-lik",
                    f"{em_model.log_likelihood:.4g}"
                    if em_model.log_likelihood is not None
                    else "—",
                )
                m6.metric(
                    "BIC",
                    f"{em_model.bic:.4g}" if em_model.bic is not None else "—",
                )

                # 收敛曲线
                st.markdown("**Convergence curve**")
                plot_loglik_history(em_model.loglik_history)

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

                # 模型规模 / NLL / n_params
                m_a, m_b, m_c = st.columns(3)
                m_a.metric("n_features (N)", em_model.n_features)
                m_b.metric(
                    "n_params (p)",
                    em_model.n_params if em_model.n_params is not None else "—",
                )
                m_c.metric(
                    "neg log-lik",
                    f"{em_model.neg_log_likelihood:.4g}"
                    if em_model.neg_log_likelihood is not None
                    else "—",
                )

                # Cluster profiles
                st.markdown("**Cluster profiles**")
                opts_col, _ = st.columns([1, 3])
                with opts_col:
                    em_use_log_y = st.checkbox(
                        "log-scale Y", value=True, key="funclu_em_log_y"
                    )
                    em_use_log_x = st.checkbox(
                        "log-scale X", value=False, key="funclu_em_log_x"
                    )
                    em_member_source = st.selectbox(
                        "Member source",
                        ["curve", "qd_df"],
                        index=0,
                        key="funclu_em_member_src",
                        help=(
                            "curve: 每个成员一条细线（基于 curve_sample）；"
                            "qd_df: 每个时间点 × 每成员的散点（适合配 quasi_dynamic）。"
                        ),
                    )
                    em_show_ci = st.checkbox(
                        "Show CI band (±1.96 SE)",
                        value=False,
                        key="funclu_em_show_ci",
                    )
                    em_n_cols = st.number_input(
                        "Subplot cols",
                        min_value=1,
                        max_value=max(em_model.n_components, 1),
                        value=min(3, em_model.n_components),
                        step=1,
                        key="funclu_em_ncols",
                    )

                plot_cluster_profiles(
                    data_scatter=[curve_sample_dict[n] for n in em_cond_names],
                    labels=em_labels_np,
                    common_cols=em_model.common_cols,
                    n_components=em_model.n_components,
                    condition_labels=em_cond_names,
                    member_source=em_member_source,
                    show_mean_ci=bool(em_show_ci),
                    use_semilogy=bool(em_use_log_y),
                    use_log_x=bool(em_use_log_x),
                    n_cols=int(em_n_cols),
                )


# ========== Tab 2 ==========
with tab2:
    st.write("To Be Updated...")
