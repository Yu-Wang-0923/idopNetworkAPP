"""Configure and run FunClu v4 fitting for uploaded static CSV data."""
from importlib.resources import files
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from idopnetwork.curve_fitting.plot import plot_curve_fitting, plot_curve_fitting_compare
from idopnetwork_app.curve_fitting_workflow import (
    DATA_MODE_DYNAMIC,
    DATA_MODE_LABELS,
    DATA_MODE_QUASI,
    build_fitting_export,
    fit_uploaded_csv,
)
from idopnetwork_app.utils import load_css, setup_sidebar

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")
st.set_page_config(page_title="Curve Fitting", page_icon=_ICON, layout="wide",
                   initial_sidebar_state="auto")
load_css()
setup_sidebar()
if not st.session_state.get("logged_in", False):
    st.warning("请先返回首页登录后使用。")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=32)
def _fit(content: bytes, **options):
    return fit_uploaded_csv(content, **options)


st.title("Curve Fitting")
with st.container(key="fit-intake"):
    st.subheader("导入观测数据")
    st.caption("支持多个 CSV 文件。第一列为样本或时间索引，其余列为分析特征。")
    uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True,
                                key="curve_fitting_upload_v4", label_visibility="collapsed")
    with st.expander("下载示例数据与使用说明"):
        education, ecg = st.columns(2)
        with education:
            st.download_button(
                "下载 7 科成绩 ZIP", data=(files("idopnetwork_app") / "data" / "education_example.zip").read_bytes(),
                file_name="教育示例数据.zip", mime="application/zip", key="download_education_example", on_click="ignore",
            )
            st.caption("教育数据已匿名化。解压后上传 CSV，选择 Quasi-dynamic 进行拟合。")
        with ecg:
            st.download_button(
                "下载 100 Hz ECG CSV", data=(files("idopnetwork_app") / "data" / "ecg_100hz.csv").read_bytes(),
                file_name="ECG_100Hz_12通道.csv", mime="text/csv", key="download_ecg_example", on_click="ignore",
            )
            st.caption("12 导联、1000 个采样点。选择 Dynamic 去噪，建网时选择 idopECG(4)，采样率保持 100 Hz。")

if not uploaded:
    st.session_state.pop("curve_fitting_run_v4", None)
    st.stop()

names = [file.name for file in uploaded]
if len(names) != len(set(names)):
    st.session_state.pop("curve_fitting_run_v4", None)
    st.error("上传文件名不能重复，请重命名后重试。")
    st.stop()

st.subheader("分析设置")
mode_tabs = st.tabs(
    [DATA_MODE_LABELS[DATA_MODE_QUASI], DATA_MODE_LABELS[DATA_MODE_DYNAMIC]],
    key="fit_data_mode_tabs",
    default=DATA_MODE_LABELS[DATA_MODE_QUASI],
    on_change="rerun",
)
data_mode = DATA_MODE_QUASI if mode_tabs[0].open or not mode_tabs[1].open else DATA_MODE_DYNAMIC
active_tab = mode_tabs[0] if data_mode == DATA_MODE_QUASI else mode_tabs[1]
mode_help = {
    DATA_MODE_QUASI: "静态数据 → 列变换 → 按行和排序成拟动态序列 → 幂律拟合。",
    DATA_MODE_DYNAMIC: "时间序列（如 12 导联 ECG）→ 逐通道 SWT 平稳小波软阈值去噪；"
                       "去噪后的波形可直接用于建网页的「Fourier + Legendre（动态 idop）」算法。",
}
with active_tab:
    st.caption(mode_help[data_mode])

signature = (data_mode,) + tuple(
    (file.name, sha256(file.getvalue()).hexdigest()) for file in uploaded
)
previous = st.session_state.get("curve_fitting_run_v4")
if previous is not None and previous["signature"] != signature:
    st.session_state.pop("curve_fitting_run_v4", None)

with active_tab, st.form("curve_fitting_parameters"):
    if data_mode == DATA_MODE_DYNAMIC:
        st.markdown("**SWT denoise parameters**")
        swt_left, swt_mid, swt_right = st.columns(3)
        swt_wavelet = swt_left.selectbox(
            "Wavelet", ["rbio3.9", "db4", "sym8", "coif3", "bior3.9"],
            index=0, key="fit_swt_wavelet",
            help="SWT 小波基；ECG 常用 rbio3.9。",
        )
        swt_level = swt_mid.number_input(
            "SWT level", min_value=1, max_value=10, value=2, step=1,
            key="fit_swt_level",
            help="分解层数上限；实际取 min(level, swt_max_level(N))。",
        )
        swt_alpha = swt_right.number_input(
            "Threshold alpha", min_value=0.0, max_value=5.0, value=0.5,
            step=0.05, format="%.2f", key="fit_swt_alpha",
            help="阈值系数：threshold = alpha · sigma · sqrt(2 log N)，soft 阈值，"
                 "sigma 由最细层细节系数的 MAD/0.6745 估计。",
        )
        st.caption(
            "Dynamic 模式下导出沿用同一套三表契约，但含义为："
            "quasi_dynamic = 原始波形，curve_sample = SWT 去噪后波形，"
            "curve_params = 每个通道的小波/层数/sigma/阈值诊断。"
        )
        first_transform = "Z_min_add1"
        second_transform = "Log10_1p"
        n_samples = 30
        trim_percent = 1.0
    else:
        st.markdown("**Fitting parameters**")
        left, right = st.columns(2)
        methods = ["None", "Z_min_add1", "Log10_1p", "Minmax_0_1"]
        first_transform = left.selectbox("First transform", methods, index=1,
                                         key="fit_first_transform")
        second_transform = right.selectbox("Second transform", methods, index=2,
                                            key="fit_second_transform")
        n_samples = left.number_input("Sample points", min_value=2, max_value=10000,
                                      value=30, step=1, key="fit_n_samples")
        trim_percent = right.number_input("Remove first rows (%)", min_value=0.0,
                                           max_value=99.0, value=1.0, step=0.5,
                                           key="fit_trim_percent")
        st.caption("Transforms run in order; None skips a step. Z_min_add1 shifts each column to a minimum of 1. "
                   "Rows are sorted by log1p of their sums, filtered to positive indices, then trimmed.")
        swt_wavelet, swt_level, swt_alpha = "rbio3.9", 2, 0.5
    submitted = st.form_submit_button("Run Fitting", type="primary")

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
    st.stop()
st.divider()
st.subheader("分析结果")
for name, error in run["errors"].items():
    st.error(f"{name} 拟合失败：{error}")
results = run["results"]
if not results:
    st.stop()
used = run["options"]
if used.get("data_mode") == DATA_MODE_DYNAMIC:
    st.caption(
        f"Displayed results: Dynamic (SWT) — wavelet `{used.get('swt_wavelet')}`, "
        f"level {used.get('swt_level')}, alpha {used.get('swt_alpha')}。"
        "quasi_dynamic = 原始波形，curve_sample = 去噪波形。"
        "After changing parameters, click Run Fitting to update results."
    )
else:
    st.caption(f"Displayed results: {used['first_transform']} → {used['second_transform']}; "
               f"{used['n_samples']} sample points; remove first {used['trim_percent']:g}%. "
               "After changing parameters, click Run Fitting to update results.")

if len(results) == len(uploaded):
    st.success(f"Fitting complete: {len(results)} file(s).")
else:
    st.warning(f"{len(results)} / {len(uploaded)} files fitted. Downloads contain successful files only.")
try:
    st.download_button("Download fitting results ZIP", data=build_fitting_export(results),
                       file_name="curve_fitting_export.zip", mime="application/zip")
except ValueError as exc:
    st.error(str(exc))

selected = st.selectbox("Results for", list(results))
result = results[selected]
quasi, params, samples = (result[k] for k in ("quasi_dynamic", "curve_params", "curve_sample"))
valid = np.isfinite(samples.to_numpy(float)).all(axis=0)
is_dynamic = used.get("data_mode") == DATA_MODE_DYNAMIC
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
    selected_features = st.multiselect(
        "Features to plot",
        options=list(quasi.columns),
        default=list(quasi.columns[:6]),
        key=f"fit_features_{selected}_{sha256(repr(list(quasi.columns)).encode()).hexdigest()[:12]}",
        help="默认只绘制前 6 个 Feature；可自行增减，超过每页容量时分页显示。",
    )
    if is_dynamic:
        per_page = 6
        ecg_sampling_rate = st.number_input("ECG 采样率 (Hz)", min_value=1.0, value=100.0,
                                            help="有 time 时间索引时直接使用该时间；否则按此采样率生成秒单位横轴。")
    else:
        with st.expander("Plot settings", expanded=False):
            c1, c2, c3 = st.columns(3)
            nrow = c1.number_input("Rows", min_value=1, max_value=10, value=2)
            ncol = c1.number_input("Cols", min_value=1, max_value=10, value=3)
            plot_type = c2.selectbox("Plot Type", ["scatter", "line"])
            scatter_size = c2.number_input("Scatter Size", min_value=1, max_value=1000, value=100)
            linewidth = c2.number_input("Line Size", min_value=1, max_value=10, value=1)
            scatter_color = c3.color_picker("Data Color", "#1F77B4")
            curve_color = c3.color_picker("Curve Color", "#D62728")
            background = c3.color_picker("Subfig Background Color", "#FFFFFF")
        per_page = int(nrow * ncol)
    pages = max(1, (len(selected_features) + per_page - 1) // per_page)
    page = st.selectbox("Feature page", range(1, pages + 1), key=f"fit_page_{selected}_{per_page}")
    features = selected_features[(page - 1) * per_page:page * per_page]
    st.caption(f"显示 {len(features)} / {len(quasi.columns)} 个 Feature；默认选择前 6 个。")
    if features:
        if is_dynamic:
            from idopnetwork_app.ecg_visualization import render_ecg_fitting
            render_ecg_fitting(quasi, samples, features=features, key=f"swt_{selected}",
                               sampling_rate=float(ecg_sampling_rate))
        else:
            fig = plot_curve_fitting(
                df_scatter=quasi.loc[:, features], df_curve=samples.loc[:, features],
                plot_scatter_type=plot_type, show_curve=True, scatter_x="index",
                nrow=int(nrow), ncol=int(ncol), nsubfig=per_page, scatter_size=scatter_size,
                scatter_linewidth=linewidth, color_scatter=scatter_color, color_curve=curve_color,
                subfig_background_color=background,
            )
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("请选择至少一个 Feature 以显示拟合图。")

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
                    fig = plot_curve_fitting_compare(
                        df_scatter_list=[t["quasi_dynamic"].loc[:, subset] for t in results.values()],
                        df_curve_list=[t["curve_sample"].loc[:, subset] for t in results.values()],
                        label_list=list(results), show_curve=True, nrow=int(nrow),
                        ncol=int(ncol), nsubfig=per_page,
                    )
                    st.pyplot(fig)
                    plt.close(fig)
