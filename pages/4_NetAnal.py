import io
import json
import sys

import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Network Analysis ss",
    page_icon="TSA.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ========== 加载 CSS ==========
from backend.utils import GLMY_EXE_PATH, load_css, setup_sidebar
from backend.network_analysis import (
    list_from_to_members,
    member_display_label,
    load_from_to_from_zip,
    run_glmy,
    suggest_max_x,
    sanitize_name,
)
from backend.plot_network_analysis import plot_glmy_barcode

load_css()
setup_sidebar()

st.title("Network Analysis", text_alignment="center")


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["GLMY", "Machine Learning", "Center Network"])


# ========== Tab 1 GLMY ==========
with tab1:
    st.markdown(
        "上传 **NetRecon → Export** 导出的 ZIP（"
        "`single_layer_idopnetwork_export.zip` 或 "
        "`multi_layer_idopnetwork_export.zip`），"
        "对其中的 `from_to.csv` 调用 GLMY 计算同调群（β₀–β₃）并展示 barcode 图。"
    )

    uploaded_zip = st.file_uploader(
        label="Please upload IdopNetwork export ZIP",
        type=["zip"],
        key="netanal_glmy_zip_upload",
        help="Input comes from NetRecon -> Export ZIP",
    )

    zip_bytes: bytes | None = None
    members: list[str] = []
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

    # ========== Tab 1_1 Uploaded Data ==========
    with tab1_1:
        if zip_bytes is None:
            st.info("Please upload an IdopNetwork export ZIP first.")
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
                st.dataframe(from_to_df, use_container_width=True, height=320)

    # ========== Tab 1_2 GLMY Analysis ==========
    with tab1_2:
        from_to_df = st.session_state.get("netanal_glmy_from_to_df")
        chosen_member = st.session_state.get("netanal_glmy_chosen_member")

        if from_to_df is None:
            st.info("Please upload a ZIP and pick a sub-network in **Uploaded Data** first.")
        elif from_to_df.empty:
            st.warning("当前选择的 `from_to.csv` 为空，无法运行 GLMY。")
        elif not GLMY_EXE_PATH.exists():
            st.error(f"未找到 GLMY 可执行文件：{GLMY_EXE_PATH}")
        else:
            if sys.platform != "win32":
                st.info(
                    "当前运行环境非 Windows（如 Streamlit Community Cloud 的 Linux 容器），"
                    "GLMY 通过 Wine 调用 `GLMY.exe`。**首次运行需要 30–90s** 预热 Wine prefix，"
                    "之后会快很多。"
                )

            default_max_x = float(suggest_max_x(from_to_df))
            col1, col2, col3 = st.columns([1.2, 1, 1])
            with col1:
                normalize_label = st.selectbox(
                    "Weight scaling",
                    options=[
                        "auto (rescale to [-1, 1] before +100 offset)",
                        "raw (only +100 offset)",
                    ],
                    index=0,
                    key="netanal_glmy_normalize",
                    help=(
                        "auto：先把 weight 等比缩放到 [-1, 1] 再加 +100 offset，"
                        "适合任意量级；barcode 输出会乘回原始 |weight| max，"
                        "横轴仍是原始 weight 尺度。"
                        "\nraw：保持原 GLMY1.py 行为（仅 +100 offset），"
                        "适合 weight 已经在 [-1, 1] 量级（如相关系数）。"
                    ),
                )
                normalize_mode = "auto" if normalize_label.startswith("auto") else "raw"
            with col2:
                max_x = st.number_input(
                    "Barcode max_x",
                    min_value=0.01,
                    value=round(default_max_x, 4),
                    step=0.1,
                    format="%.4f",
                    key="netanal_glmy_max_x",
                    help="Barcode 横轴右端位置；默认按 |weight| 最大值 + 10% buffer 自适应。",
                )
            with col3:
                st.caption("Auto suggestion (max_x)")
                st.code(f"{default_max_x:.4f}", language="text")

            run_clicked = st.button(
                "Run GLMY",
                type="primary",
                key="netanal_glmy_run_btn",
            )

            if run_clicked:
                spinner_msg = (
                    "Running GLMY.exe ..."
                    if sys.platform == "win32"
                    else "Running GLMY.exe via Wine (首次启动较慢，请耐心) ..."
                )
                with st.spinner(spinner_msg):
                    try:
                        glmy_result = run_glmy(from_to_df, normalize=normalize_mode)
                    except Exception as e:
                        st.error(f"GLMY 运行失败：{e}")
                        glmy_result = None
                if glmy_result is not None:
                    st.session_state["netanal_glmy_result"] = glmy_result
                    st.session_state["netanal_glmy_result_max_x"] = float(max_x)
                    st.session_state["netanal_glmy_result_member"] = chosen_member

            glmy_result = st.session_state.get("netanal_glmy_result")
            result_member = st.session_state.get("netanal_glmy_result_member")

            if glmy_result is None:
                st.info("Click **Run GLMY** to compute homology and draw the barcode.")
            else:
                if result_member != chosen_member:
                    st.warning(
                        f"当前展示的是 `{result_member}` 的旧结果；"
                        f"请重新点击 **Run GLMY** 以匹配当前选择 `{chosen_member}`。"
                    )

                homology = glmy_result["homology"]
                summary = {
                    f"β{dim}": len(homology.get(str(dim), []))
                    for dim in (0, 1, 2, 3)
                }
                st.markdown("**Betti number summary** (number of bars per dimension)")
                st.json(summary)
                st.caption(
                    f"normalize = `{glmy_result.get('normalize', 'auto')}`，"
                    f"scale_factor = `{glmy_result.get('scale_factor', 1.0):.6g}`；"
                    "barcode 横轴仍为原始 weight 尺度。"
                )

                fig = plot_glmy_barcode(
                    homology,
                    max_x=float(st.session_state.get("netanal_glmy_result_max_x", default_max_x)),
                )
                st.pyplot(fig, use_container_width=True)

                # PNG / PDF 下载
                png_buf = io.BytesIO()
                fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
                pdf_buf = io.BytesIO()
                fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
                plt.close(fig)

                base_name = sanitize_name(member_display_label(result_member or "glmy"))

                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.download_button(
                    label="Download barcode PNG",
                    data=png_buf.getvalue(),
                    file_name=f"{base_name}_glmy_barcode.png",
                    mime="image/png",
                    key="netanal_glmy_png_download",
                )
                col_d2.download_button(
                    label="Download barcode PDF",
                    data=pdf_buf.getvalue(),
                    file_name=f"{base_name}_glmy_barcode.pdf",
                    mime="application/pdf",
                    key="netanal_glmy_pdf_download",
                )
                col_d3.download_button(
                    label="Download homology.json",
                    data=json.dumps(homology, indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name=f"{base_name}_homology.json",
                    mime="application/json",
                    key="netanal_glmy_json_download",
                )

                with st.expander("GLMY raw stdout / stderr"):
                    st.text_area(
                        "stdout",
                        value=glmy_result.get("stdout", ""),
                        height=160,
                        key="netanal_glmy_stdout_text",
                    )
                    st.text_area(
                        "stderr",
                        value=glmy_result.get("stderr", ""),
                        height=80,
                        key="netanal_glmy_stderr_text",
                    )


# ========== Tab 2 Machine Learning ==========
with tab2:
    st.write("待更新...")

    tab2_1, tab2_2, tab2_3 = st.tabs(["Uploaded Data", "Classification", "Regression"])

    # ========== Tab 2_1 Uploaded Data ==========
    with tab2_1:
        st.write("待更新...")

    # ========== Tab 2_2 Classification ==========
    with tab2_2:
        st.write("待更新...")

    # ========== Tab 2_3 Regression ==========
    with tab2_3:
        st.write("待更新...")


# ========== Tab 3 Center Network ==========
with tab3:
    st.write("待更新...")

    tab3_1, tab3_2 = st.tabs(["Uploaded Data", "Center Network"])

    # ========== Tab 3_1 Uploaded Data ==========
    with tab3_1:
        st.write("待更新...")

    # ========== Tab 3_2 Classification ==========
    with tab3_2:
        st.write("待更新...")
