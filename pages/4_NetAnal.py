import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Network Analysis",
    page_icon="static/images/TSA.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ========== 加载 CSS ==========
from backend.utils import load_css, setup_sidebar 
from backend.analysis.network_analysis import (
    list_from_to_members,
    member_display_label,
    load_from_to_from_zip,
    run_glmy,
    suggest_max_x,
    sanitize_name,
)
from backend.analysis.plot_analysis import plot_glmy_barcode

# M3 / Paper §3.2 self-test：用 backend.analysis.digraph 复刻原 GLMY1.py 流程，仅用于诊断。
from backend.analysis.glmy_test import (
    DEFAULT_DIM as M3_DEFAULT_DIM,
    DEFAULT_M3_CSV,
    DEFAULT_MAX_X as M3_DEFAULT_MAX_X,
    DEFAULT_WEIGHT_OFFSET as M3_DEFAULT_WEIGHT_OFFSET,
    betti_summary as m3_betti_summary,
    load_m3_dataframe,
    paper_3_2_dataframe,
    run_digraph_on_m3,
)

PAPER_3_2_DEFAULT_MAX_X: float = 5.5

load_css()
setup_sidebar()

# ========== 登录门禁 ==========
if not st.session_state.get("logged_in", False):
    st.warning("请先返回首页登录后使用。")
    st.stop()

st.title("Network Analysis sss", text_alignment="center")


# ========== Tabs ==========
tab1, tab2, tab3 = st.tabs(["GLMY", "Machine Learning", "Center Network"])


# ========== Tab 1 GLMY ==========
with tab1:
    st.markdown(
        "上传 **NetRecon → Export** 导出的 ZIP（"
        "`single_layer_idopnetwork_export.zip` 或 "
        "`multi_layer_idopnetwork_export.zip`），"
        "对其中的 `from_to.csv` 使用内置 Python GLMY/path homology "
        "实现计算同调群（β₀–β₃）并展示 barcode 图。"
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

    tab1_1, tab1_2, tab1_3, tab1_4 = st.tabs(
        ["Uploaded Data", "GLMY Analysis", "M3 Self-test", "Paper §3.2 Self-test"]
    )

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
                st.dataframe(from_to_df, width="stretch", height=320)

    # ========== Tab 1_2 GLMY Analysis ==========
    with tab1_2:
        from_to_df = st.session_state.get("netanal_glmy_from_to_df")
        chosen_member = st.session_state.get("netanal_glmy_chosen_member")

        if from_to_df is None:
            st.info("Please upload a ZIP and pick a sub-network in **Uploaded Data** first.")
        elif from_to_df.empty:
            st.warning("当前选择的 `from_to.csv` 为空，无法运行 GLMY。")
        else:
            default_max_x = float(suggest_max_x(from_to_df))

            with st.form(key="netanal_glmy_form"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    max_x = st.number_input(
                        "Barcode max_x",
                        min_value=1e-9,
                        value=round(default_max_x, 4),
                        step=0.1,
                        format="%.4f",
                        key="netanal_glmy_max_x",
                        help=(
                            "Barcode 横轴右端位置；默认按 |weight| 最大值 + 10% buffer "
                            "自适应。仅作图横轴范围，**不**对 weight 做任何归一化。"
                        ),
                    )
                with col2:
                    st.caption("Auto suggestion (max_x)")
                    st.code(f"{default_max_x:.4f}", language="text")

                run_clicked = st.form_submit_button(
                    "Run GLMY",
                    type="primary",
                )

            if run_clicked:
                with st.spinner("Running Python GLMY/path homology ..."):
                    try:
                        glmy_result = run_glmy(from_to_df)
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
                st.markdown(
                    "**Betti number summary**：各维 **barcode 条数**（持久对个数），"
                    "与旧版 ``GLMY.exe`` 在顶点编号与 ``weight+100`` 流程下应对齐。"
                )
                st.json(summary)
                st.caption(
                    f"backend = `python`，weight_offset = `{glmy_result.get('weight_offset', 100.0):g}`；"
                    "barcode 横轴已减回 +offset 偏移，对应 `from_to.csv` 中的原始 "
                    "weight/Effect 尺度，**未做任何归一化**。"
                )

                fig = plot_glmy_barcode(
                    homology,
                    max_x=float(st.session_state.get("netanal_glmy_result_max_x", default_max_x)),
                )
                st.pyplot(fig, width="stretch")

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

    # ========== Tab 1_3 M3 Self-test ==========
    with tab1_3:
        st.markdown(
            "**目的**：用 `backend.Digraph` 复刻原 `GLMY1.py` 的内部流水线（"
            "`weight = Effect + 100` → `Digraph.get_persistence` → 端点减 100），"
            "对仓库根 `M3.csv` 跑一遍 path homology，**不**走 ZIP 上传、**不**调外部 "
            "`GLMY.exe`，仅用于诊断 barcode 是否与原 EXE 路径一致。"
        )

        m3_csv_path = DEFAULT_M3_CSV
        try:
            m3_df = load_m3_dataframe(m3_csv_path)
        except Exception as e:
            st.error(f"读取 `{m3_csv_path.name}` 失败：{e}")
            m3_df = None

        if m3_df is not None:
            m3_nodes = sorted(set(m3_df["From"]).union(set(m3_df["To"])))
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Edges", f"{len(m3_df):,}")
            col_m2.metric("Nodes", f"{len(m3_nodes):,}")
            col_m3.metric("|Effect| max", f"{m3_df['Effect'].abs().max():.4f}")

            st.markdown(f"**Source:** `{m3_csv_path.name}` (repo root)")
            st.dataframe(m3_df, width="stretch", height=240)

            with st.form(key="netanal_m3_form"):
                col_p1, col_p2 = st.columns([1, 1])
                with col_p1:
                    m3_max_x = st.number_input(
                        "Barcode max_x (M3)",
                        min_value=0.01,
                        value=float(M3_DEFAULT_MAX_X),
                        step=0.1,
                        format="%.4f",
                        key="netanal_m3_max_x",
                        help=(
                            "原 GLMY1.py 的默认值是 2.0；横轴范围为 [-max_x, max_x*1.1]。"
                        ),
                    )
                with col_p2:
                    st.caption("Default (matches original GLMY1.py)")
                    st.code(f"{M3_DEFAULT_MAX_X:.4f}", language="text")

                run_m3_clicked = st.form_submit_button(
                    "Run M3 self-test",
                    type="primary",
                )

            if run_m3_clicked:
                with st.spinner("Running backend.Digraph on M3.csv ..."):
                    try:
                        m3_homology, m3_vertex_map = run_digraph_on_m3(
                            m3_df,
                            dim=M3_DEFAULT_DIM,
                            weight_offset=M3_DEFAULT_WEIGHT_OFFSET,
                        )
                    except Exception as e:
                        st.error(f"M3 self-test 失败：{e}")
                        m3_homology = None
                        m3_vertex_map = None
                if m3_homology is not None:
                    st.session_state["netanal_m3_homology"] = m3_homology
                    st.session_state["netanal_m3_vertex_map"] = m3_vertex_map
                    st.session_state["netanal_m3_max_x_used"] = float(m3_max_x)

            m3_homology = st.session_state.get("netanal_m3_homology")
            m3_vertex_map = st.session_state.get("netanal_m3_vertex_map")

            if m3_homology is None:
                st.info("Click **Run M3 self-test** to compute homology and draw the barcode.")
            else:
                st.markdown("**Vertex map (name → integer id)**")
                st.json(m3_vertex_map or {})

                st.markdown(
                    "**Betti number summary**（barcode 条数；与原 `GLMY.exe` 在 "
                    "`Effect+100` 流程下应对齐）"
                )
                st.json(m3_betti_summary(m3_homology, dim=M3_DEFAULT_DIM))

                st.caption(
                    "barcode 横轴已减回 +100 偏移；β₀ 的 birth 来源于 "
                    "`backend.Digraph` 中顶点权重 0，因此还原后会出现在 -100 附近 ——"
                    "这是诊断 barcode 是否被横轴范围裁掉的关键信号。"
                )

                m3_fig = plot_glmy_barcode(
                    m3_homology,
                    max_x=float(
                        st.session_state.get("netanal_m3_max_x_used", M3_DEFAULT_MAX_X)
                    ),
                )
                st.pyplot(m3_fig, width="stretch")

                m3_png_buf = io.BytesIO()
                m3_fig.savefig(m3_png_buf, format="png", dpi=200, bbox_inches="tight")
                m3_pdf_buf = io.BytesIO()
                m3_fig.savefig(m3_pdf_buf, format="pdf", bbox_inches="tight")
                plt.close(m3_fig)

                col_md1, col_md2, col_md3 = st.columns(3)
                col_md1.download_button(
                    label="Download M3 barcode PNG",
                    data=m3_png_buf.getvalue(),
                    file_name="M3_barcode.png",
                    mime="image/png",
                    key="netanal_m3_png_download",
                )
                col_md2.download_button(
                    label="Download M3 barcode PDF",
                    data=m3_pdf_buf.getvalue(),
                    file_name="M3_barcode.pdf",
                    mime="application/pdf",
                    key="netanal_m3_pdf_download",
                )
                col_md3.download_button(
                    label="Download M3 homology.json",
                    data=json.dumps(m3_homology, indent=2, ensure_ascii=False).encode(
                        "utf-8"
                    ),
                    file_name="M3_homology.json",
                    mime="application/json",
                    key="netanal_m3_json_download",
                )

    # ========== Tab 1_4 Paper §3.2 Self-test ==========
    with tab1_4:
        st.markdown(
            "**目的**：用 `backend.Digraph` 跑刘祥 et al. *Computing (persistent) "
            "embedded homology of graded vector spaces in chain complexes* §3.2 / "
            "Figure 1 中的 filtered digraph，端点应与论文给出的 `(birth, death)` "
            "pairs 一致，作为 path-homology 实现的**地面真值**回归。"
        )

        paper_df = paper_3_2_dataframe()

        st.markdown("**Paper §3.2 filtered digraph (Effect = filtration index)**")
        st.dataframe(paper_df, width="stretch", height=240)

        st.markdown("**论文给出的标准答案（用作对照）**")
        expected_table = pd.DataFrame(
            [
                {"dim": "β₀", "(birth, death)": "(0, ∞)", "type": "infinite — vertex 1"},
                {"dim": "β₀", "(birth, death)": "(0, 1)", "type": "finite — vertex 2 ↔ arrow 21"},
                {"dim": "β₀", "(birth, death)": "(0, 3)", "type": "finite — vertex 3 ↔ arrow 23"},
                {"dim": "β₁", "(birth, death)": "(2, 5)", "type": "finite — arrow 12 ↔ path 213"},
                {"dim": "β₂", "(birth, death)": "(5, ∞)", "type": "infinite — path 131"},
                {"dim": "β₃", "(birth, death)": "—", "type": "empty"},
            ]
        )
        st.table(expected_table)
        st.caption(
            "Trivial pairs ``(31, 231)`` 和 ``(13, 123)`` 在论文里 persistence = 0，"
            "``backend.Digraph`` 内部已通过 ``death > birth`` 条件过滤掉，barcode 上不显示。"
            "β₀ 中 birth = 0 的 bar 会被还原成 -100（顶点权重 0 减 +100 偏移），"
            "横轴 ``[-max_x, max_x*1.1]`` 看不到 birth 端点 —— 这是与 M3 Self-test 一致的已知行为。"
        )

        with st.form(key="netanal_paper_form"):
            col_pp1, col_pp2 = st.columns([1, 1])
            with col_pp1:
                paper_max_x = st.number_input(
                    "Barcode max_x (Paper §3.2)",
                    min_value=1e-9,
                    value=float(PAPER_3_2_DEFAULT_MAX_X),
                    step=0.5,
                    format="%.4f",
                    key="netanal_paper_max_x",
                    help="filtration max = 5；默认 5.5（+ 10% buffer），仅作图横轴范围。",
                )
            with col_pp2:
                st.caption("Default (filtration max + 10% buffer)")
                st.code(f"{PAPER_3_2_DEFAULT_MAX_X:.4f}", language="text")

            run_paper_clicked = st.form_submit_button(
                "Run Paper §3.2 self-test",
                type="primary",
            )

        if run_paper_clicked:
            with st.spinner("Running backend.Digraph on Paper §3.2 data ..."):
                try:
                    paper_homology, paper_vertex_map = run_digraph_on_m3(
                        paper_df,
                        dim=M3_DEFAULT_DIM,
                        weight_offset=M3_DEFAULT_WEIGHT_OFFSET,
                    )
                except Exception as e:
                    st.error(f"Paper §3.2 self-test 失败：{e}")
                    paper_homology = None
                    paper_vertex_map = None
            if paper_homology is not None:
                st.session_state["netanal_paper_homology"] = paper_homology
                st.session_state["netanal_paper_vertex_map"] = paper_vertex_map
                st.session_state["netanal_paper_max_x_used"] = float(paper_max_x)

        paper_homology = st.session_state.get("netanal_paper_homology")
        paper_vertex_map = st.session_state.get("netanal_paper_vertex_map")

        if paper_homology is None:
            st.info("Click **Run Paper §3.2 self-test** to compute homology and draw the barcode.")
        else:
            st.markdown("**Vertex map (name → integer id)**")
            st.json(paper_vertex_map or {})

            st.markdown("**Betti number summary (computed)**")
            st.json(m3_betti_summary(paper_homology, dim=M3_DEFAULT_DIM))

            paper_fig = plot_glmy_barcode(
                paper_homology,
                max_x=float(
                    st.session_state.get(
                        "netanal_paper_max_x_used", PAPER_3_2_DEFAULT_MAX_X
                    )
                ),
            )
            st.pyplot(paper_fig, width="stretch")

            paper_png_buf = io.BytesIO()
            paper_fig.savefig(paper_png_buf, format="png", dpi=200, bbox_inches="tight")
            paper_pdf_buf = io.BytesIO()
            paper_fig.savefig(paper_pdf_buf, format="pdf", bbox_inches="tight")
            plt.close(paper_fig)

            col_pd1, col_pd2, col_pd3 = st.columns(3)
            col_pd1.download_button(
                label="Download Paper §3.2 barcode PNG",
                data=paper_png_buf.getvalue(),
                file_name="paper_3_2_barcode.png",
                mime="image/png",
                key="netanal_paper_png_download",
            )
            col_pd2.download_button(
                label="Download Paper §3.2 barcode PDF",
                data=paper_pdf_buf.getvalue(),
                file_name="paper_3_2_barcode.pdf",
                mime="application/pdf",
                key="netanal_paper_pdf_download",
            )
            col_pd3.download_button(
                label="Download Paper §3.2 homology.json",
                data=json.dumps(paper_homology, indent=2, ensure_ascii=False).encode(
                    "utf-8"
                ),
                file_name="paper_3_2_homology.json",
                mime="application/json",
                key="netanal_paper_json_download",
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
