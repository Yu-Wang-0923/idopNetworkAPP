"""Configure and run FunClu v4 fitting for uploaded static CSV data."""
from importlib.resources import files
from hashlib import sha256

import base64
import numpy as np
import streamlit as st

from idopnetwork_app.charts import curve_fitting_html
from idopnetwork_app.curve_fitting_workflow import (
    DATA_MODE_DYNAMIC,
    DATA_MODE_QUASI,
    build_fitting_export,
    fit_uploaded_csv,
)
from idopnetwork_app.utils import load_css, setup_sidebar
from idopnetwork_app.i18n import t

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")
st.set_page_config(page_title="Curve Fitting", page_icon=_ICON, layout="wide",
                   initial_sidebar_state="auto")
load_css()
setup_sidebar()
if not st.session_state.get("logged_in", False):
    st.warning(t("auth.login_required"))
    st.stop()


@st.cache_data(show_spinner=False, max_entries=32)
def _fit(content: bytes, **options):
    return fit_uploaded_csv(content, **options)


def _compact_layout() -> bool:
    """Phones and tablets get one plot per row and a single-line metric row."""
    try:
        ua = st.context.headers.get("User-Agent", "")
    except Exception:
        return False
    return any(token in ua for token in ("Android", "iPhone", "iPad", "Mobile"))


st.title("Curve Fitting")
with st.container(key="fit-intake"):
    st.markdown(f"## {t('fitting.intake.title')}")
    st.caption(t("fitting.intake.caption"))
    uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True,
                                key="curve_fitting_upload_v4", label_visibility="collapsed")
    with st.expander(t("fitting.example.expander")):
        education, ecg = st.columns(2)
        with education:
            st.download_button(
                "下载 7 科成绩 ZIP", data=(files("idopnetwork_app") / "data" / "education_example.zip").read_bytes(),
                file_name="教育示例数据.zip", mime="application/zip", key="download_education_example", on_click="ignore",
            )
            st.caption(t("fitting.example.education"))
        with ecg:
            st.download_button(
                "下载 100 Hz ECG CSV", data=(files("idopnetwork_app") / "data" / "ecg_100hz.csv").read_bytes(),
                file_name="ECG_100Hz_12通道.csv", mime="text/csv", key="download_ecg_example", on_click="ignore",
            )
            st.caption(t("fitting.example.ecg"))

if not uploaded:
    st.session_state.pop("curve_fitting_run_v4", None)
    st.stop()

names = [file.name for file in uploaded]
if len(names) != len(set(names)):
    st.session_state.pop("curve_fitting_run_v4", None)
    st.error("上传文件名不能重复，请重命名后重试。")
    st.stop()

st.markdown(f"## {t('fitting.settings.title')}")
MODE_OPTIONS = [DATA_MODE_QUASI, DATA_MODE_DYNAMIC]
MODE_LABELS = {DATA_MODE_QUASI: "Quasi-dynamic", DATA_MODE_DYNAMIC: "Dynamic"}
mode_help = {
    DATA_MODE_QUASI: t("fitting.mode.quasi"),
    DATA_MODE_DYNAMIC: t("fitting.mode.dynamic"),
}

with st.container(key="fit-settings"):
    data_mode = st.segmented_control(
        "Data type",
        options=MODE_OPTIONS,
        format_func=lambda mode: MODE_LABELS[mode],
        default=DATA_MODE_QUASI,
        required=True,
        key="fit_data_mode",
        label_visibility="collapsed",
    ) or DATA_MODE_QUASI
    st.caption(mode_help[data_mode])

    signature = (data_mode,) + tuple(
        (file.name, sha256(file.getvalue()).hexdigest()) for file in uploaded
    )
    previous = st.session_state.get("curve_fitting_run_v4")
    if previous is not None and previous["signature"] != signature:
        st.session_state.pop("curve_fitting_run_v4", None)

    with st.form("curve_fitting_parameters"):
        fields, action = st.columns([2.5, 1], gap="large", vertical_alignment="bottom")
        with fields:
            if data_mode == DATA_MODE_DYNAMIC:
                swt_cols = st.columns(3, gap="medium")
                swt_wavelet = swt_cols[0].selectbox(
                    "Wavelet", ["rbio3.9", "db4", "sym8", "coif3", "bior3.9"],
                    index=0, key="fit_swt_wavelet",
                    help="SWT 小波基；ECG 常用 rbio3.9。",
                )
                swt_level = swt_cols[1].number_input(
                    "SWT level", min_value=1, max_value=10, value=2, step=1,
                    key="fit_swt_level",
                    help="分解层数上限；实际取 min(level, swt_max_level(N))。",
                )
                swt_alpha = swt_cols[2].number_input(
                    "Threshold alpha", min_value=0.0, max_value=5.0, value=0.5,
                    step=0.05, format="%.2f", key="fit_swt_alpha",
                    help="阈值系数：threshold = alpha · sigma · sqrt(2 log N)，soft 阈值，"
                         "sigma 由最细层细节系数的 MAD/0.6745 估计。",
                )
                first_transform = "Z_min_add1"
                second_transform = "Log10_1p"
                n_samples = 30
                trim_percent = 1.0
            else:
                methods = ["None", "Z_min_add1", "Log10_1p", "Minmax_0_1"]
                transform_row = st.columns(2, gap="medium")
                first_transform = transform_row[0].selectbox(
                    "First transform", methods, index=1, key="fit_first_transform")
                second_transform = transform_row[1].selectbox(
                    "Second transform", methods, index=2, key="fit_second_transform")
                numeric_row = st.columns(2, gap="medium")
                n_samples = numeric_row[0].number_input(
                    "Sample points", min_value=2, max_value=10000, value=30, step=1,
                    key="fit_n_samples")
                trim_percent = numeric_row[1].number_input(
                    "Remove first rows (%)", min_value=0.0, max_value=99.0, value=1.0,
                    step=0.5, key="fit_trim_percent")
                swt_wavelet, swt_level, swt_alpha = "rbio3.9", 2, 0.5
        with action:
            submitted = st.form_submit_button("Run Fitting", type="primary", width="stretch")
        st.caption(t("fitting.dynamic.note") if data_mode == DATA_MODE_DYNAMIC
                   else t("fitting.transforms.hint"))

if submitted:
    options = dict(
        data_mode=data_mode,
        first_transform=first_transform, second_transform=second_transform,
        n_samples=int(n_samples), trim_percent=float(trim_percent),
        swt_wavelet=str(swt_wavelet), swt_level=int(swt_level),
        swt_alpha=float(swt_alpha),
    )
    results, errors = {}, {}
    for file in uploaded:
        try:
            with st.spinner(f"Fitting {file.name}..."):
                results[file.name] = _fit(file.getvalue(), **options)
        except Exception as exc:
            errors[file.name] = str(exc)
    st.session_state.curve_fitting_run_v4 = dict(
        signature=signature, options=options, results=results, errors=errors,
    )

run = st.session_state.get("curve_fitting_run_v4")
if run is None:
    st.divider()
    with st.container(key="fit-results-empty"):
        st.markdown(f"## {t('fitting.results.title')}")
        st.caption(t("fitting.results.empty"))
    st.stop()

st.divider()
st.markdown(f"## {t('fitting.results.title')}")
for name, error in run["errors"].items():
    st.error(f"{name} 拟合失败：{error}")
results = run["results"]
if not results:
    st.stop()
used = run["options"]
is_dynamic = used.get("data_mode") == DATA_MODE_DYNAMIC
if is_dynamic:
    st.caption(t("fitting.status.dynamic", files=len(results),
                 wavelet=used.get("swt_wavelet"), level=used.get("swt_level"),
                 alpha=used.get("swt_alpha")))
else:
    st.caption(t("fitting.status.quasi", files=len(results),
                 first=used["first_transform"], second=used["second_transform"],
                 samples=used["n_samples"], trim=f"{used['trim_percent']:g}"))
if len(results) != len(uploaded):
    st.warning(f"{len(results)} / {len(uploaded)} files fitted. Downloads contain successful files only.")

pick_col, export_col = st.columns([2.2, 1], gap="medium", vertical_alignment="bottom")
selected = pick_col.selectbox(t("fitting.results.for"), list(results))
try:
    export_col.download_button(t("fitting.download.zip"), data=build_fitting_export(results),
                               file_name="curve_fitting_export.zip", mime="application/zip",
                               width="stretch")
except ValueError as exc:
    st.error(str(exc))

result = results[selected]
quasi, params, samples = (result[k] for k in ("quasi_dynamic", "curve_params", "curve_sample"))
valid = np.isfinite(samples.to_numpy(float)).all(axis=0)
compact = _compact_layout()
if compact:
    st.caption(t("fitting.metrics.compact", fitted=f"{int(valid.sum())} / {len(valid)}",
                 rows=len(quasi), samples=len(samples)))
else:
    cols = st.columns(3)
    cols[0].metric(
        "Features denoised" if is_dynamic else "Features fitted",
        f"{int(valid.sum())} / {len(valid)}", border=True,
    )
    cols[1].metric(
        "Time points" if is_dynamic else "Quasi-dynamic rows", len(quasi), border=True
    )
    cols[2].metric(
        "Annotated samples" if is_dynamic else "Sample points", len(samples), border=True
    )
if not valid.all():
    st.warning("Some features have insufficient positive observations. Their parameters and samples contain NaN and will be excluded from clustering.")

plot_tab, params_tab, samples_tab, quasi_tab = st.tabs(
    ["Curve Fitting Plot", "Curve Parameters", "Curve Samples", "Quasi-dynamic Data"]
)
with plot_tab:
    if is_dynamic:
        per_page = 2 if compact else 6
        ecg_sampling_rate = st.number_input("ECG 采样率 (Hz)", min_value=1.0, value=100.0,
                                            help="有 time 时间索引时直接使用该时间；否则按此采样率生成秒单位横轴。")
    else:
        # The publication panel lives inside the chart component, so the figure
        # is configured there; the page only decides which features are drawn.
        nrow, ncol = 3, 4
        per_page = 12
    features = list(quasi.columns)
    if is_dynamic:
        from idopnetwork_app.ecg_visualization import render_ecg_fitting
        render_ecg_fitting(quasi, samples, features=features[:per_page], key=f"swt_{selected}",
                           sampling_rate=float(ecg_sampling_rate))
    else:
        panels = curve_fitting_html.build_panels(quasi, samples, features)
        chart_html = curve_fitting_html.render(panels)
        chart_url = "data:text/html;base64," + base64.b64encode(
            chart_html.encode("utf-8")).decode("ascii")
        st.iframe(chart_url, height=700)
    st.caption(
        f"共 {len(features)} 个 Feature；在图表左侧面板的「特征」里勾选要显示的面板，"
        "Rows × Cols 决定每页容量。"
    )

for tab, table, suffix in (
    (params_tab, params, "params"),
    (samples_tab, samples, "samples"),
    (quasi_tab, quasi, "quasidynamic"),
):
    with tab:
        st.dataframe(table, width="stretch")
        st.download_button(f"Download {suffix}.csv", table.to_csv(index=True),
                           file_name=f"{selected.rsplit('.', 1)[0]}_{suffix}.csv",
                           mime="text/csv", key=f"download_{suffix}")

if len(results) > 1:
    with st.expander("Compare fitted curves", expanded=False):
        if st.checkbox("Show comparison"):
            common = list(next(iter(results.values()))["quasi_dynamic"].columns)
            for tables in results.values():
                common = [c for c in common if c in tables["quasi_dynamic"].columns]
            if not common:
                st.info("The uploaded files have no shared features to compare.")
            else:
                compare_pages = max(1, (len(common) + per_page - 1) // per_page)
                compare_page = st.selectbox("Comparison page", range(1, compare_pages + 1))
                subset = common[(compare_page - 1) * per_page:compare_page * per_page]
                if is_dynamic:
                    from idopnetwork_app.ecg_visualization import render_ecg_fitting
                    for name, tables in results.items():
                        st.markdown(f"**{name}**")
                        render_ecg_fitting(tables["quasi_dynamic"], tables["curve_sample"],
                                           features=subset, key=f"compare_swt_{name}",
                                           sampling_rate=float(ecg_sampling_rate))
                else:
                    panels = curve_fitting_html.build_compare_panels(
                        [item["quasi_dynamic"].loc[:, subset] for item in results.values()],
                        [item["curve_sample"].loc[:, subset] for item in results.values()],
                        list(results),
                        subset,
                    )
                    compare_html = curve_fitting_html.render(panels)
                    compare_url = "data:text/html;base64," + base64.b64encode(
                        compare_html.encode("utf-8")).decode("ascii")
                    st.iframe(compare_url, height=700)
