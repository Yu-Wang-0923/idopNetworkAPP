"""Configure and run FunClu v4 fitting for uploaded static CSV data."""
from importlib.resources import files
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from idopnetwork.curve_fitting.plot import plot_curve_fitting, plot_curve_fitting_compare
from idopnetwork_app.curve_fitting_workflow import fit_uploaded_csv, build_fitting_export
from idopnetwork_app.utils import load_css, setup_sidebar

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")
st.set_page_config(page_title="Curve Fitting", page_icon=_ICON, layout="wide",
                   initial_sidebar_state="expanded")
load_css()
setup_sidebar()
if not st.session_state.get("logged_in", False):
    st.warning("请先返回首页登录后使用。")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=32)
def _fit(content: bytes, **options):
    return fit_uploaded_csv(content, **options)


st.title("Curve Fitting", text_alignment="center")
st.caption("Upload static CSV files, choose parameters and click Run Fitting. The first column is the row index; remaining columns are features.")
uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True,
                            key="curve_fitting_upload_v4")
if not uploaded:
    st.session_state.pop("curve_fitting_run_v4", None)
    st.info("Upload CSV files, set parameters and run fitting to see results.")
    st.stop()

names = [file.name for file in uploaded]
if len(names) != len(set(names)):
    st.session_state.pop("curve_fitting_run_v4", None)
    st.error("上传文件名不能重复，请重命名后重试。")
    st.stop()

signature = tuple((file.name, sha256(file.getvalue()).hexdigest()) for file in uploaded)
previous = st.session_state.get("curve_fitting_run_v4")
if previous is not None and previous["signature"] != signature:
    st.session_state.pop("curve_fitting_run_v4", None)

with st.form("curve_fitting_parameters"):
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
    submitted = st.form_submit_button("Run Fitting", type="primary")

if submitted:
    options = dict(first_transform=first_transform, second_transform=second_transform,
                   n_samples=int(n_samples), trim_percent=float(trim_percent))
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
    st.info("Choose parameters above and click Run Fitting.")
    st.stop()
for name, error in run["errors"].items():
    st.error(f"{name} 拟合失败：{error}")
results = run["results"]
if not results:
    st.stop()
used = run["options"]
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
cols = st.columns(3)
cols[0].metric("Features fitted", f"{int(valid.sum())} / {len(valid)}")
cols[1].metric("Quasi-dynamic rows", len(quasi))
cols[2].metric("Sample points", len(samples))
if not valid.all():
    st.warning("Some features have insufficient positive observations. Their parameters and samples contain NaN and will be excluded from clustering.")

plot_tab, params_tab, samples_tab, quasi_tab = st.tabs(
    ["Curve Fitting Plot", "Curve Parameters", "Curve Samples", "Quasi-dynamic Data"]
)
with plot_tab:
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
    pages = max(1, (len(quasi.columns) + per_page - 1) // per_page)
    page = st.selectbox("Feature page", range(1, pages + 1), key=f"fit_page_{selected}_{per_page}")
    features = quasi.columns[(page - 1) * per_page:page * per_page]
    fig = plot_curve_fitting(
        df_scatter=quasi.loc[:, features], df_curve=samples.loc[:, features],
        plot_scatter_type=plot_type, show_curve=True, scatter_x="index",
        nrow=int(nrow), ncol=int(ncol), nsubfig=per_page, scatter_size=scatter_size,
        scatter_linewidth=linewidth, color_scatter=scatter_color, color_curve=curve_color,
        subfig_background_color=background,
    )
    st.pyplot(fig)
    plt.close(fig)

for tab, table, suffix in (
    (params_tab, params, "params"),
    (samples_tab, samples, "samples"),
    (quasi_tab, quasi, "quasidynamic"),
):
    with tab:
        st.dataframe(table, use_container_width=True)
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
                fig = plot_curve_fitting_compare(
                    df_scatter_list=[t["quasi_dynamic"].loc[:, subset] for t in results.values()],
                    df_curve_list=[t["curve_sample"].loc[:, subset] for t in results.values()],
                    label_list=list(results), show_curve=True, nrow=int(nrow),
                    ncol=int(ncol), nsubfig=per_page,
                )
                st.pyplot(fig)
                plt.close(fig)
