import matplotlib

matplotlib.use("Agg", force=True)

import streamlit as st

from importlib.resources import files

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")

st.set_page_config(
    page_title="Network Analysis",
    page_icon=_ICON,
    layout="wide",
    initial_sidebar_state="auto",
)
# ========== 加载 CSS ==========
from idopnetwork_app.utils import load_css, setup_sidebar, upload_guidance
from idopnetwork_app.i18n import t
from idopnetwork.analysis.network_analysis import (
    list_from_to_members,
    member_display_label,
    load_from_to_from_zip,
)


load_css()
setup_sidebar()

# ========== 登录门禁 ==========
if not st.session_state.get("logged_in", False):
    st.warning(t("auth.login_required"))
    st.stop()

st.title("Network Analysis")

st.markdown("## GLMY (Path Homology)")


# ========== GLMY ==========
st.markdown(
    "上传 **NetRecon → Export** 导出的 ZIP（"
    "`single_layer_idopnetwork_export.zip` 或 "
    "`multi_layer_idopnetwork_export.zip`），"
    "对其中的 `from_to.csv` 使用内置 Python GLMY/path homology "
    "新版实现计算 F₂ 上的 regular path homology（β₀–β₂），展示 barcode、过滤过程和代表元图。"
)

uploaded_zip = st.file_uploader(
    label="Please upload IdopNetwork export ZIP",
    type=["zip"],
    key="netanal_glmy_zip_upload",
    help="Input comes from NetRecon -> Export ZIP",
)

zip_bytes: bytes | None = None
members: list[str] = []
st.session_state.pop("netanal_glmy_from_to_df", None)
st.session_state.pop("netanal_glmy_chosen_member", None)
if uploaded_zip is not None:
    zip_bytes = uploaded_zip.getvalue()
    try:
        members = list_from_to_members(zip_bytes)
    except Exception as e:
        st.error(f"无法读取 ZIP：{e}")
        zip_bytes = None
        members = []

    if zip_bytes is not None and not members:
        st.error("ZIP 中没有找到任何 `from_to.csv`。")
        zip_bytes = None

tab1_1, tab1_2 = st.tabs(["Uploaded Data", "GLMY Analysis"])

# ========== Uploaded Data ==========
with tab1_1:
    if zip_bytes is None:
        upload_guidance(
            "pages/3_NetRecon.py", "NetRecon", "single_layer_idopnetwork_export.zip",
            t("netanal.step1"),
        )
    else:
        labels = [member_display_label(m) for m in members]
        choice_label = st.selectbox(
            "Select a sub-network (`from_to.csv`)",
            options=labels,
            index=0,
            key="netanal_glmy_member_select",
        )
        chosen_member = members[labels.index(choice_label)]
        st.session_state["netanal_glmy_chosen_member"] = chosen_member

        try:
            from_to_df = load_from_to_from_zip(zip_bytes, chosen_member)
        except Exception as e:
            st.error(f"读取 `{chosen_member}` 失败：{e}")
        else:
            st.session_state["netanal_glmy_from_to_df"] = from_to_df

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Edges", f"{len(from_to_df):,}")
            n_nodes = len(set(from_to_df["from"]).union(set(from_to_df["to"])))
            col_b.metric("Nodes", f"{n_nodes:,}")
            if not from_to_df.empty:
                col_c.metric(
                    "|weight| max",
                    f"{from_to_df['weight'].abs().max():.4f}",
                )

            st.markdown(f"**Source:** `{chosen_member}`")
            st.dataframe(from_to_df, width="stretch", height=320)

# ========== GLMY Analysis ==========
with tab1_2:
    from idopnetwork_app.glmy_workflow import render_glmy_analysis

    render_glmy_analysis(from_to_df=st.session_state.get("netanal_glmy_from_to_df"),
                         member=st.session_state.get("netanal_glmy_chosen_member"))
