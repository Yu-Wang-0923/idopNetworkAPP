import io
import zipfile

import streamlit as st

st.set_page_config(page_title="Functional Clustering", page_icon=None, layout="wide", initial_sidebar_state="auto")

import numpy as np
import pandas as pd

from backend.functional_clustering import FunClu
from backend.plot_functional_clustering import plot_initialization_grid
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
    tab1_1, tab1_2, tab1_3 = st.tabs(["Data Overview", "Initialization", "To Be Updated..."])

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
                        "log-scale Y", value=True, key="funclu_grid_log_y"
                    )
                    use_log_x = st.checkbox(
                        "log-scale X", value=False, key="funclu_grid_log_x"
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
                )

    with tab1_3:
        st.write("To Be Updated...")


# ========== Tab 2 ==========
with tab2:
    st.write("To Be Updated...")
