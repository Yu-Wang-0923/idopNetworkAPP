"""Upload static CSV data and immediately return FunClu v4 fitting results."""
from importlib.resources import files

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
def _fit(content: bytes):
    return fit_uploaded_csv(content)


st.title("Curve Fitting", text_alignment="center")
st.caption("Upload static CSV files to automatically fit curves. The first column is the row index; remaining columns are features.")
uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True,
                            key="curve_fitting_upload_v4")
with st.expander("Processing details", expanded=False):
    st.markdown(
        "Each column is shifted by its minimum + 1, then transformed with log10(1 + x). "
        "Rows are sorted by log1p of their sums; nonpositive indices and the first 1% "
        "of remaining rows are removed. Power curves are fitted at 30 equally spaced points."
    )

if not uploaded:
    st.info("Upload CSV files to see fitted curves, parameters and downloadable results.")
    st.stop()

names = [file.name for file in uploaded]
if len(names) != len(set(names)):
    st.error("上传文件名不能重复，请重命名后重试。")
    st.stop()

results = {}
for file in uploaded:
    try:
        with st.spinner(f"Fitting {file.name}..."):
            results[file.name] = _fit(file.getvalue())
    except Exception as exc:
        st.error(f"{file.name} 拟合失败：{exc}")

if not results:
    st.stop()

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
