

import io
import zipfile

import streamlit as st
from importlib.resources import files

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")

# 🌟 统一小图标和侧边栏状态
st.set_page_config(
    page_title="Functional Clustering",
    page_icon=_ICON,
    layout="wide",
    initial_sidebar_state="auto"
)

import numpy as np
import pandas as pd

from idopnetwork.clustering.funclu import FunClu, compute_bic_scores
from idopnetwork.curve_fitting import get_power_function_params
from idopnetwork.clustering.plot import (
    plot_bic_elbow,
    plot_cluster_profiles,
    plot_cluster_profiles_per_cluster,
    stitch_cluster_figures,
)
from idopnetwork_app.utils import load_css, setup_sidebar, upload_guidance
from idopnetwork_app.i18n import t

# 🌟 一键加载 CSS 和侧边栏
load_css()
setup_sidebar()

# ========== 登录门禁 ==========
if not st.session_state.get("logged_in", False):
    st.warning(t("auth.login_required"))
    st.stop()

st.title("Functional Clustering")

# ========== Session State ==========
if "funclu_curve_sample" not in st.session_state:
    st.session_state.funclu_curve_sample = {}  # {condition_name: pd.DataFrame}
if "funclu_quasi_dynamic" not in st.session_state:
    st.session_state.funclu_quasi_dynamic = {}  # {condition_name: pd.DataFrame}
if "funclu_uploaded_zip_name" not in st.session_state:
    st.session_state.funclu_uploaded_zip_name = None
if "funclu_em_result" not in st.session_state:
    st.session_state.funclu_em_result = None
if "funclu_bic_result" not in st.session_state:
    st.session_state.funclu_bic_result = None
if "funclu_bic_params" not in st.session_state:
    st.session_state.funclu_bic_params = {}
if "funclu_bic_checkmark_ks" not in st.session_state:
    st.session_state.funclu_bic_checkmark_ks = []
if "funclu_bic_default_best_k" not in st.session_state:
    st.session_state.funclu_bic_default_best_k = None
if "funclu_bic_default_mode" not in st.session_state:
    st.session_state.funclu_bic_default_mode = False

# ========== Helper Functions ==========
def _detect_bic_checkmark_candidates(
    bic_df: pd.DataFrame,
    *,
    max_short_arm: int = 3,
) -> list[int]:
    """Find checkmark-like local BIC minima on the plotted K path."""
    if bic_df is None or "K" not in bic_df.columns or "BIC" not in bic_df.columns:
        return []

    df = bic_df.dropna(subset=["K", "BIC"]).sort_values("K")
    if len(df) < 3:
        return []

    ks = df["K"].to_numpy(dtype=int)
    bics = df["BIC"].to_numpy(dtype=float)
    scale = float(np.nanmax(np.abs(bics))) if bics.size else 1.0
    eps = max(1e-9, scale * 1e-12)

    candidates: list[int] = []
    for idx in range(1, len(bics) - 1):
        if not (bics[idx] + eps < bics[idx - 1] and bics[idx] + eps < bics[idx + 1]):
            continue

        left_len = 0
        last = bics[idx]
        for j in range(idx - 1, -1, -1):
            if bics[j] > last + eps:
                left_len += 1
                last = bics[j]
            else:
                break

        right_len = 0
        last = bics[idx]
        for j in range(idx + 1, len(bics)):
            if bics[j] > last + eps:
                right_len += 1
                last = bics[j]
            else:
                break

        short_arm = min(left_len, right_len)
        long_arm = max(left_len, right_len)
        if left_len >= 1 and right_len >= 1 and short_arm <= max_short_arm and long_arm > short_arm:
            candidates.append(int(ks[idx]))

    return candidates


def _load_netrecon_inputs_from_zip(
    zip_bytes: bytes,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """从 curve_fitting 导出的 ZIP 读取 curve_sample 与 quasi_dynamic。"""
    curve_map: dict[str, pd.DataFrame] = {}
    quasi_map: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if not (
                name.endswith("curve_sample.csv")
                or name.endswith("quasi_dynamic.csv")
            ):
                continue
            cond = name.rsplit("/", 1)[0] if "/" in name else name
            with zf.open(name) as f:
                df = pd.read_csv(f, index_col=0)
            if name.endswith("curve_sample.csv"):
                curve_map[cond] = df
            else:
                quasi_map[cond] = df
    return (
        dict(sorted(curve_map.items(), key=lambda kv: kv[0])),
        dict(sorted(quasi_map.items(), key=lambda kv: kv[0])),
    )


def _build_funclu_k_export_zip(
    *,
    model: FunClu,
    cond_names: list[str],
    curve_sample_dict: dict[str, pd.DataFrame],
    quasi_dynamic_dict: dict[str, pd.DataFrame],
) -> bytes:
    """Build a ZIP package for downstream Multi-Layer idopNetwork construction."""
    if model.labels is None or model.common_cols is None:
        raise ValueError("FunClu-K result is incomplete: missing labels/common columns.")

    labels_np = model.labels.detach().cpu().numpy().astype(int)
    common_cols = list(model.common_cols)
    if len(labels_np) != len(common_cols):
        raise ValueError("FunClu-K labels do not match common curve_sample columns.")
    missing_quasi = [
        cond_name for cond_name in cond_names if cond_name not in quasi_dynamic_dict
    ]
    if missing_quasi:
        raise ValueError(
            "FunClu-K export requires quasi_dynamic for every condition; "
            f"missing: {', '.join(missing_quasi)}"
        )

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
            quasi_dynamic = quasi_dynamic_dict[cond_name].loc[:, common_cols]

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
                member_response_df = quasi_dynamic.loc[:, member_cols]
                zf.writestr(
                    f"cluster_members/{cond_name}/{cluster_name}_quasi_dynamic.csv",
                    member_response_df.to_csv(index=True),
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
            center_response_df = pd.DataFrame(index=quasi_dynamic.index)
            for cluster_idx in range(model.n_components):
                cluster_name = f"M{cluster_idx + 1}"
                member_cols = labels_df.loc[
                    labels_df["cluster_id"] == cluster_idx + 1, "feature"
                ].tolist()
                center_response_df[cluster_name] = quasi_dynamic.loc[
                    :, member_cols
                ].mean(axis=1)
            center_response_df.index.name = quasi_dynamic.index.name or "time"
            zf.writestr(
                f"cluster_centers/{cond_name}/cluster_center_quasi_dynamic.csv",
                center_response_df.to_csv(index=True),
            )

    return buffer.getvalue()


# ========== Top-Level Tabs ==========
tab1, tab2, tab3 = st.tabs([t("funclu.tab.data"), "FunClu-K", "FunClu-BIC"])

# =====================================================================
# TAB 1: Uploaded Data (compact header + overview)
# =====================================================================
with tab1:
    def _ingest_upload(uploaded_file) -> None:
        """Load the curve_fitting ZIP into session state and refresh the page."""
        try:
            curve_sample_map, quasi_dynamic_map = _load_netrecon_inputs_from_zip(
                uploaded_file.getvalue()
            )
            if not quasi_dynamic_map:
                raise ValueError("ZIP 中缺少 quasi_dynamic.csv，无法导出多层响应变量")
            st.session_state.funclu_curve_sample = curve_sample_map
            st.session_state.funclu_quasi_dynamic = quasi_dynamic_map
            st.session_state.funclu_uploaded_zip_name = uploaded_file.name
            st.session_state.funclu_em_result = None
        except Exception as exc:
            st.error(t("funclu.upload.failed", error=exc))
            st.session_state.funclu_curve_sample = {}
            st.session_state.funclu_quasi_dynamic = {}
            st.session_state.funclu_uploaded_zip_name = None
        else:
            st.rerun()

    curve_sample_dict = st.session_state.get("funclu_curve_sample", {})

    if not curve_sample_dict:
        st.markdown(f"## {t('funclu.init.title')}")
        uploaded_file = st.file_uploader(
            label=t("funclu.upload.label"),
            type=["zip"],
            accept_multiple_files=False,
            help=t("funclu.upload.help"),
            label_visibility="visible",
            max_upload_size=500,
        )
        if uploaded_file is not None and (
            st.session_state.funclu_uploaded_zip_name != uploaded_file.name
        ):
            _ingest_upload(uploaded_file)
        upload_guidance(
            "pages/1_Curve Fitting.py", "Curve Fitting", "curve_fitting_export.zip",
            t("funclu.need.step1"),
        )
    else:
        cond_names = list(curve_sample_dict.keys())
        n_features = int(curve_sample_dict[cond_names[0]].shape[1])
        n_timepoints = int(sum(len(df) for df in curve_sample_dict.values()))

        head_col, action_col = st.columns([3, 1], vertical_alignment="center")
        head_col.markdown(
            f"<p class='funclu-strip'>{t('funclu.strip', n=len(cond_names), f=n_features, tp=n_timepoints)}</p>",
            unsafe_allow_html=True,
        )
        head_col.caption(" · ".join(cond_names))
        with action_col.popover(t("funclu.upload.change"), use_container_width=True):
            replacement = st.file_uploader(
                label=t("funclu.upload.label"),
                type=["zip"],
                accept_multiple_files=False,
                help=t("funclu.upload.help"),
                key="funclu_reupload",
                label_visibility="collapsed",
            )
            if replacement is not None and (
                st.session_state.funclu_uploaded_zip_name != replacement.name
            ):
                _ingest_upload(replacement)

        st.markdown("### " + t("funclu.overview"))

        try:
            model = FunClu()
            X_list, times_list = model._prepare_data(
                [curve_sample_dict[n] for n in cond_names]
            )
        except Exception as exc:
            st.error(t("funclu.prepare_failed", error=exc))
        else:
            overview_tab1, overview_tab2 = st.tabs(
                [t("funclu.tab.summary"), t("funclu.tab.detail")]
            )

            with overview_tab1:
                col1, col2, col3 = st.columns(3)
                col1.metric(t("funclu.metric.conditions"), model.n_conditions)
                col2.metric(t("funclu.metric.features"), model.n_features)
                per_cond_str = ", ".join(str(d) for d in model.n_times_conditions)
                col3.metric(
                    t("funclu.metric.timepoints"),
                    f"{sum(model.n_times_conditions)}",
                    delta=f"Per condition: {per_cond_str}",
                    delta_color="off",
                )
                st.write("")
                st.markdown("#### " + t("funclu.per_condition"))
                summary_df = pd.DataFrame([
                    {
                        t("funclu.col.condition"): n,
                        t("funclu.col.timepoints"): int(model.n_times_conditions[i]),
                        t("funclu.col.shape"): f"{X_list[i].shape[0]} × {X_list[i].shape[1]}",
                        t("funclu.col.time_min"): float(times_list[i].min()),
                        t("funclu.col.time_max"): float(times_list[i].max()),
                    }
                    for i, n in enumerate(cond_names)
                ])
                st.dataframe(summary_df, width="stretch", hide_index=True)

            with overview_tab2:
                st.markdown("#### " + t("funclu.shared_features", n=model.n_features))
                features_df = pd.DataFrame({t("funclu.col.feature"): model.common_cols})
                st.dataframe(features_df, width="stretch", hide_index=True, height=220)

                st.markdown("#### " + t("funclu.preview.title"))
                selected_cond = st.selectbox(
                    t("funclu.select_condition"),
                    options=cond_names,
                    key="overview_preview_select",
                )
                st.caption(t("funclu.preview_caption", cond=selected_cond))
                st.dataframe(
                    curve_sample_dict[selected_cond].iloc[:5, :5],
                    width="stretch",
                )

# =====================================================================
# TAB 2: FunClu-K (EM Fitting & Export)
# =====================================================================
with tab2:
    tab2_1, tab2_2 = st.tabs([t("funclu.tab.em"), t("funclu.tab.export")])

    # ---------- EM Fitting ----------
    with tab2_1:
        curve_sample_dict = st.session_state.get("funclu_curve_sample", {})
        if not curve_sample_dict:
            st.info(t("funclu.need_upload"))
        else:
            cond_names = list(curve_sample_dict.keys())
            data_list = [curve_sample_dict[n] for n in cond_names]
            n_features_max = min((d.shape[1] for d in data_list), default=2)
            k_upper = max(2, n_features_max)

            with st.expander(t("funclu.settings"), expanded=True):
                with st.form(key="funclu_em_form"):
                    K_em = st.number_input(
                        "n_components (K)",
                        min_value=2,
                        max_value=k_upper,
                        value=min(3, k_upper),
                        step=1,
                        key="funclu_em_K",
                    )
                    submit_em = st.form_submit_button(t("funclu.run_em"))

            if submit_em:
                try:
                    em_model = FunClu(
                        n_components=int(K_em),
                        use_minibatch_kmeans=None,
                        random_state=42,
                    )
                    em_model.fit(data_list, parameter_data=[
                        get_power_function_params(st.session_state.funclu_quasi_dynamic[n])
                        for n in cond_names
                    ])
                except Exception as e:
                    st.error(t("funclu.em_failed", error=e))
                    st.session_state.funclu_em_result = None
                else:
                    st.session_state.funclu_em_result = {
                        "model": em_model,
                        "cond_names": cond_names,
                    }
                    st.success(
                        t("funclu.em_done", c=em_model.converged, i=em_model.n_iter_run, m=em_model.max_iter, l=round(em_model.log_likelihood, 4), b=round(em_model.bic, 4))
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
                st.markdown(t("funclu.cluster_sizes_weights"))
                st.dataframe(em_sw_df, width="stretch")

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
                    st.dataframe(pd.DataFrame(em_mu_rows), width="stretch")
                with col_cov:
                    st.markdown(r"**cov_params (phi, gamma)** — SAD1")
                    st.dataframe(pd.DataFrame(em_cov_rows), width="stretch")

                st.markdown(t("funclu.cluster_profiles"))
                with st.expander(t("funclu.plot_settings"), expanded=False):
                    with st.form(key="funclu_cluster_profiles_form"):
                        layout_col, member_col, scale_col = st.columns(3)
                        with layout_col:
                            em_profile_layout = st.selectbox(
                                t("funclu.profile_layout"),
                                options=["per_cluster", "combined", "k_by_l", "l_by_k"],
                                format_func={
                                    "per_cluster": "Per cluster: one 4x3 figure each (funclu_v4 style)",
                                    "combined": "Combined: conditions in one panel",
                                    "k_by_l": "K x L: cluster rows",
                                    "l_by_k": "L x K: condition rows",
                                }.get,
                                key="funclu_em_profile_layout",
                            )
                            em_n_cols = st.number_input(
                                t("funclu.subplot_cols"),
                                min_value=1,
                                max_value=max(em_model.n_components, 1),
                                value=min(3, em_model.n_components),
                                step=1,
                                key="funclu_em_ncols",
                                help=(
                                    t("funclu.subplot_cols_help")
                                ),
                            )
                        with member_col:
                            em_member_source = st.selectbox(
                                t("funclu.member_source"),
                                ["curve", "qd_df"],
                                index=0,
                                key="funclu_em_member_src",
                                help=(
                                    t("funclu.member_source_help")
                                ),
                            )
                            em_show_members = st.checkbox(
                                t("funclu.show_members"),
                                value=True,
                                key="funclu_em_show_members",
                            )
                            em_show_mean = st.checkbox(
                                t("funclu.show_mean"),
                                value=True,
                                key="funclu_em_show_mean",
                            )
                            em_show_ci = st.checkbox(
                                t("funclu.show_ci"),
                                value=False,
                                key="funclu_em_show_ci",
                            )
                        with scale_col:
                            em_use_log_y = st.checkbox(
                                t("funclu.log_y"),
                                value=False,
                                key="funclu_em_log_y",
                            )
                            em_use_log_x = st.checkbox(
                                t("funclu.log_x"),
                                value=False,
                                key="funclu_em_log_x",
                            )
                            em_show_legend = st.checkbox(
                                t("funclu.show_legend"),
                                value=True,
                                key="funclu_em_show_legend",
                            )
                        st.form_submit_button(t("funclu.apply_plot"))

                _profile_kwargs = dict(
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
                    show_legend=bool(em_show_legend),
                )
                if str(em_profile_layout) == "per_cluster":
                    # funclu_v4 风格：每簇一张 4:3 图，再按 n_cols 拼成完整网格图。
                    # 先离屏渲染每簇、再光栅化拼接，这样每个格子都保持 4:3，
                    # 不会被 subplots 的共享轴/图例留白重新挤压。
                    # pyplot 局部导入而非模块级：避免在页面入口新增 matplotlib
                    # 的模块级导入边（并发加载时易与 idopnetwork.* 形成导入竞争）。
                    import matplotlib.pyplot as plt

                    _cluster_figs = plot_cluster_profiles_per_cluster(
                        **_profile_kwargs,
                    )
                    _stitched = stitch_cluster_figures(
                        _cluster_figs, n_cols=int(em_n_cols),
                    )
                    for _fig in _cluster_figs:
                        plt.close(_fig)
                    st.pyplot(_stitched, width="stretch")
                    plt.close(_stitched)
                else:
                    plot_cluster_profiles(
                        **_profile_kwargs,
                        layout=str(em_profile_layout),
                        n_cols=int(em_n_cols),
                        show_in_streamlit=True,
                    )

    # ---------- Export ----------
    with tab2_2:
        curve_sample_dict = st.session_state.get("funclu_curve_sample", {})
        em = st.session_state.funclu_em_result
        
        if not curve_sample_dict:
            st.info(t("funclu.need_upload"))
        elif em is None:
            st.info(t("funclu.need_em"))
        else:
            em_model: FunClu = em["model"]
            em_cond_names = em["cond_names"]
            quasi_dynamic_dict = st.session_state.get("funclu_quasi_dynamic", {})

            try:
                export_zip_bytes = _build_funclu_k_export_zip(
                    model=em_model,
                    cond_names=em_cond_names,
                    curve_sample_dict=curve_sample_dict,
                    quasi_dynamic_dict=quasi_dynamic_dict,
                )
            except Exception as e:
                st.error(t("funclu.export_failed", error=e))
            else:
                st.markdown(t("funclu.export_contents"))
                st.write(
                    t("funclu.export_list")
                )

                st.download_button(
                    label=t("funclu.download_zip"),
                    data=export_zip_bytes,
                    file_name="funclu_k_export.zip",
                    mime="application/zip",
                    key="funclu_k_export_zip_download",
                )


# =====================================================================
# TAB 3: FunClu-BIC
# =====================================================================
with tab3:
    bic_tab1, bic_tab2 = st.tabs([t("funclu.tab.bic"), t("funclu.tab.export")])

    # ---------- BIC Analysis ----------
    with bic_tab1:
        curve_sample_dict = st.session_state.get("funclu_curve_sample", {})
        if not curve_sample_dict:
            st.info(t("funclu.need_upload"))
        else:
            cond_names = list(curve_sample_dict.keys())
            data_list = [curve_sample_dict[n] for n in cond_names]
            n_features = min((d.shape[1] for d in data_list), default=2)
            k_upper = max(2, n_features)
            min_cluster_lower = 2 if n_features >= 2 else 1
            min_cluster_default = min(
                max(1, n_features),
                max(min_cluster_lower, int(np.ceil(0.03 * max(n_features, 1)))),
            )

            with st.expander(t("funclu.bic_settings"), expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    bic_k_min = st.slider(
                        "k_min", min_value=2, max_value=k_upper - 1,
                        value=2, step=1, key="bic_k_min",
                    )
                with col2:
                    k_max_default = min(
                        k_upper,
                        max(bic_k_min + 1, int(np.sqrt(max(n_features, 1)))),
                    )
                    bic_k_max = st.slider(
                        "k_max", min_value=bic_k_min + 1, max_value=k_upper,
                        value=k_max_default, step=1, key="bic_k_max",
                    )
                with col3:
                    bic_step = st.slider(
                        "step", min_value=1, max_value=5,
                        value=1, step=1, key="bic_step",
                    )

                with st.expander(t("funclu.advanced"), expanded=True):
                    col_a1, col_a2, col_a3 = st.columns(3)
                    with col_a1:
                        bic_max_iter = st.slider(
                            "max_iter", 10, 2000, 1000, 10, key="bic_max_iter",
                        )
                    with col_a2:
                        bic_tol = st.number_input(
                            "tol", min_value=1e-10, max_value=1.0,
                            value=1e-4, format="%.1e", key="bic_tol",
                        )
                    with col_a3:
                        bic_rs = st.number_input(
                            "random_state(seeds)", 0, 9999, 42, key="bic_rs",
                        )
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        bic_random_starts = st.slider(
                            "random_starts",
                            min_value=1,
                            max_value=10,
                            value=1,
                            step=1,
                            key="bic_random_starts",
                            help=(
                                t("funclu.help.seeds")
                            ),
                        )
                    with col_b2:
                        bic_aggregation = st.selectbox(
                            "seed_aggregation",
                            options=["median", "mean"],
                            index=0,
                            key="bic_seed_aggregation",
                            help=(
                                t("funclu.help.aggregation")
                            ),
                        )
                    with col_b3:
                        bic_min_cluster_size = st.number_input(
                            "min_cluster_size",
                            min_value=min_cluster_lower,
                            max_value=max(1, n_features),
                            value=int(min_cluster_default),
                            step=1,
                            key="bic_min_cluster_size",
                            help=(
                                t("funclu.help.min_size")
                            ),
                        )

                button_col1, button_col2 = st.columns([1, 1])
                with button_col1:
                    try_default_best_k = st.button(
                        t("funclu.try_default"),
                        type="primary",
                        key="try_default_best_k_btn",
                        help=t("funclu.help.scan"),
                    )
                with button_col2:
                    run_bic = st.button(t("funclu.run_bic"), type="primary", key="run_bic_btn")

            if run_bic or try_default_best_k:
                if try_default_best_k:
                    scan_k_min = 2
                    scan_k_max = min(
                        n_features,
                        max(2, int(np.sqrt(max(n_features, 1)))),
                    )
                    scan_step = 1
                else:
                    scan_k_min = bic_k_min
                    scan_k_max = bic_k_max
                    scan_step = bic_step

                ks = list(range(scan_k_min, scan_k_max + 1, scan_step))
                with st.status(
                    f"Scanning K from {scan_k_min} to {scan_k_max} ({len(ks)} fits)..."
                ) as status:
                    try:
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                        def _update_progress(idx: int, total: int) -> None:
                            progress_bar.progress(idx / total)
                            status_text.text(
                                f"Fitting K={ks[idx - 1]} ({idx}/{total}); "
                                f"{bic_random_starts} seed(s) per K..."
                            )

                        bic_df = compute_bic_scores(
                            data=data_list,
                            parameter_data=[get_power_function_params(
                                st.session_state.funclu_quasi_dynamic[n]) for n in cond_names],
                            k_min=scan_k_min,
                            k_max=scan_k_max,
                            step=scan_step,
                            max_iter=bic_max_iter,
                            tol=bic_tol,
                            random_state=bic_rs,
                            n_random_starts=bic_random_starts,
                            aggregation=bic_aggregation,
                            min_cluster_size=int(bic_min_cluster_size),
                            progress_callback=_update_progress,
                        )
                        checkmark_ks = (
                            _detect_bic_checkmark_candidates(bic_df)
                            if try_default_best_k
                            else []
                        )
                        default_best_k = min(checkmark_ks) if checkmark_ks else None

                        st.session_state["funclu_bic_result"] = bic_df
                        st.session_state["funclu_bic_params"] = dict(
                            k_min=scan_k_min, k_max=scan_k_max, step=scan_step,
                            max_iter=bic_max_iter, tol=bic_tol, random_state=bic_rs,
                            n_random_starts=bic_random_starts,
                            aggregation=bic_aggregation,
                            min_cluster_size=int(bic_min_cluster_size),
                            mode="try_default_best_k" if try_default_best_k else "manual",
                        )
                        st.session_state["funclu_bic_checkmark_ks"] = checkmark_ks
                        st.session_state["funclu_bic_default_best_k"] = default_best_k
                        st.session_state["funclu_bic_default_mode"] = bool(try_default_best_k)
                        progress_bar.progress(1.0)
                        status_text.text("Done!")
                        status.update(label="BIC scan complete!", state="complete")
                        if try_default_best_k:
                            if checkmark_ks:
                                st.success(
                                    "Checkmark-like BIC candidates found: "
                                    + ", ".join(f"K={k}" for k in checkmark_ks)
                                    + f". Leftmost candidate K={default_best_k} is selected."
                                )
                            else:
                                st.warning(
                                    t("funclu.no_elbow")
                                )
                    except Exception as e:
                        st.error(t("funclu.bic_failed", error=e))
                        st.session_state["funclu_bic_result"] = None
                        st.session_state["funclu_bic_checkmark_ks"] = []
                        st.session_state["funclu_bic_default_best_k"] = None
                        st.session_state["funclu_bic_default_mode"] = False

            # Display results
            bic_result_df = st.session_state.get("funclu_bic_result")
            if bic_result_df is not None:
                valid = bic_result_df.dropna(subset=["BIC"])
                if not valid.empty:
                    raw_best_row = valid.loc[valid["BIC"].idxmin()]
                    checkmark_ks = [
                        int(k) for k in st.session_state.get("funclu_bic_checkmark_ks", [])
                    ]
                    default_best_k = st.session_state.get("funclu_bic_default_best_k")
                    default_mode = st.session_state.get("funclu_bic_default_mode", False)
                    use_default_best = (
                        default_mode
                        and default_best_k is not None
                        and int(default_best_k) in set(valid["K"].astype(int))
                    )
                    if "passes_min_cluster_size" in valid.columns:
                        eligible = valid[
                            valid["passes_min_cluster_size"].astype(bool)
                        ].copy()
                    else:
                        eligible = valid.copy()
                    if use_default_best:
                        best_row = valid[valid["K"].astype(int) == int(default_best_k)].iloc[0]
                    elif default_mode:
                        best_row = None
                    elif eligible.empty:
                        best_row = raw_best_row
                        st.warning(
                            t("funclu.min_size_violated")
                        )
                    else:
                        best_row = eligible.loc[eligible["BIC"].idxmin()]
                    best_k_val = int(best_row["K"]) if best_row is not None else None
                    raw_best_k_val = int(raw_best_row["K"])
                    si_valid = (
                        eligible.dropna(subset=["SI"])
                        if "SI" in eligible.columns and not eligible.empty
                        else valid.dropna(subset=["SI"])
                        if "SI" in valid.columns
                        else pd.DataFrame()
                    )
                    best_si_row = (
                        si_valid.loc[si_valid["SI"].idxmax()]
                        if not si_valid.empty
                        else None
                    )
                    best_si_label = (
                        f"K = {int(best_si_row['K'])}"
                        if best_si_row is not None
                        else "N/A"
                    )
                    best_si_value = (
                        f"SI={best_si_row['SI']:.4g}"
                        if best_si_row is not None
                        else "N/A"
                    )
                    si_at_bic = best_row.get("SI", np.nan) if best_row is not None else np.nan
                    si_at_bic_label = (
                        f"{si_at_bic:.4g}"
                        if np.isfinite(si_at_bic)
                        else "N/A"
                    )
                    min_size_label = (
                        f"{best_row['min_cluster_size']:.0f}"
                        if best_row is not None
                        and "min_cluster_size" in best_row
                        and np.isfinite(best_row["min_cluster_size"])
                        else "N/A"
                    )
                    size_ok_label = (
                        f"{best_row['size_ok_rate']:.0%}"
                        if best_row is not None
                        and "size_ok_rate" in best_row
                        and np.isfinite(best_row["size_ok_rate"])
                        else "N/A"
                    )
                    converged_label = (
                        f"{best_row['converged_rate']:.0%}"
                        if best_row is not None
                        and "converged_rate" in best_row
                        and np.isfinite(best_row["converged_rate"])
                        else str(best_row["converged"]) if best_row is not None else "N/A"
                    )

                    st.markdown("### " + t("funclu.best_k"))
                    col_c1, col_c2, col_c3, col_c4, col_c5, col_c6 = st.columns(6)
                    with col_c1:
                        st.metric(
                            t("funclu.metric.recommended"),
                            f"K = {best_k_val}" if best_k_val is not None else "N/A",
                        )
                    with col_c2:
                        st.metric(
                            "BIC",
                            f"{best_row['BIC']:.4g}" if best_row is not None else "N/A",
                        )
                    with col_c3:
                        st.metric(t("funclu.metric.raw_bic_k"), f"K = {raw_best_k_val}")
                    with col_c4:
                        st.metric(t("funclu.metric.si"), si_at_bic_label)
                    with col_c5:
                        st.metric(t("funclu.metric.min_size"), min_size_label, size_ok_label)
                    with col_c6:
                        st.metric(t("funclu.metric.converged"), converged_label)

                    if use_default_best and checkmark_ks:
                        st.caption(
                            "Try Default Best K candidates: "
                            + ", ".join(f"K = {k}" for k in checkmark_ks)
                            + f"; selected leftmost candidate K = {best_k_val}."
                        )
                    elif default_mode:
                        st.caption(
                            t("funclu.no_elbow")
                        )
                    st.caption(t("funclu.best_si_caption", label=best_si_label, value=best_si_value))

                    st.markdown("### " + t("funclu.elbow_plot"))
                    plot_bic_elbow(
                        bic_results=bic_result_df,
                        best_K=best_k_val,
                        candidate_Ks=checkmark_ks if default_mode else None,
                        auto_select_best=not default_mode,
                        show_in_streamlit=True,
                    )
                    st.caption(t("funclu.seed_note"))

                    st.markdown("### " + t("funclu.results_table"))
                    display_df = bic_result_df.rename(columns={
                        "log_likelihood": "Log-Lik",
                        "n_iter_run": "Iters",
                        "n_features": "Features",
                        "n_conditions": "Conds",
                        "BIC_mean": "BIC Mean",
                        "BIC_median": "BIC Median",
                        "BIC_sd": "BIC SD",
                        "SI_mean": "SI Mean",
                        "SI_median": "SI Median",
                        "converged_rate": "Converged Rate",
                        "min_cluster_size": "Min Cluster Size",
                        "min_cluster_size_min": "Min Size Across Seeds",
                        "max_cluster_size": "Max Cluster Size",
                        "small_cluster_count": "Small Cluster Count",
                        "small_cluster_count_max": "Max Small Clusters",
                        "empty_cluster_count": "Empty Cluster Count",
                        "size_ok_rate": "Size OK Rate",
                        "passes_min_cluster_size": "Passes Min Size",
                        "fit_successes": "Fit Successes",
                        "random_starts": "Random Starts",
                        "seed_start": "Seed Start",
                        "seed_end": "Seed End",
                        "min_cluster_size_threshold": "Min Size Threshold",
                    })
                    st.dataframe(
                        display_df, width="stretch", hide_index=True,
                    )

    # ---------- Export ----------
    with bic_tab2:
        bic_result_df = st.session_state.get("funclu_bic_result")
        if bic_result_df is None:
            st.info(t("funclu.need_bic"))
        else:
            st.markdown("### " + t("funclu.export_csv"))
            st.dataframe(
                bic_result_df, width="stretch", hide_index=True,
            )
            csv_data = bic_result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=t("funclu.download_bic"),
                data=csv_data,
                file_name="funclu_bic_results.csv",
                mime="text/csv",
                key="bic_csv_download",
            )
