import io
import zipfile

import streamlit as st

st.set_page_config(page_title="Functional Clustering", page_icon=None, layout="wide", initial_sidebar_state="auto")

import pandas as pd

from backend.functional_clustering import FunClu
from backend.utils import load_css

load_css()

st.title("Functional Clustering", text_alignment="center")


# ========== Session State ==========
if "funclu_curve_sample" not in st.session_state:
    st.session_state.funclu_curve_sample = {}  # {condition_name: pd.DataFrame}
if "funclu_uploaded_zip_name" not in st.session_state:
    st.session_state.funclu_uploaded_zip_name = None


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
    tab1_1, tab1_2 = st.tabs(["Data Overview", "To Be Updated..."])

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

    with tab1_2:
        st.write("To Be Updated...")


# ========== Tab 2 ==========
with tab2:
    st.write("To Be Updated...")
