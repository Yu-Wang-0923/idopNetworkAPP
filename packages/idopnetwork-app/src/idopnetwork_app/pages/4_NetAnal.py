import io
import json
import os

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from importlib.resources import files

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")

st.set_page_config(
    page_title="Network Analysis",
    page_icon=_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
# ========== 加载 CSS ==========
from idopnetwork_app.utils import load_css, setup_sidebar 
from idopnetwork.analysis.network_analysis import (
    list_from_to_members,
    member_display_label,
    load_from_to_from_zip,
    run_glmy,
    suggest_max_x,
    sanitize_name,
)
from idopnetwork.analysis.plot_analysis import plot_glmy_barcode
from idopnetwork.ml.core import (
    compare_hub_tables,
    funclu_k_export_summary,
    get_funclu_ml_matrix,
    hub_table_from_adjacency,
    infer_signed_hub_network,
    load_funclu_k_export,
    load_idopnetwork_adjacencies,
    matching_topology_key,
)
from idopnetwork.ml.plot import plot_hub_network

# M3 / Paper §3.2 self-test：用 backend.analysis.digraph 复刻原 GLMY1.py 流程，仅用于诊断。
from idopnetwork.analysis.glmy_test import (
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

            # ========== Run GLMY on All Sub-networks ==========
            if zip_bytes is not None and len(members) >= 2:
                st.divider()
                st.markdown("### Run GLMY on All Sub-networks")

                run_all = st.button(
                    f"Run GLMY on All ({len(members)} sub-networks)",
                    type="secondary",
                    key="netanal_glmy_run_all",
                    use_container_width=True,
                )

                if run_all:
                    all_results: dict[str, dict] = {}
                    with st.status(
                        f"Running GLMY on {len(members)} sub-networks..."
                    ) as status:
                        for member in members:
                            st.write(
                                f"Processing `{member_display_label(member)}`..."
                            )
                            try:
                                df = load_from_to_from_zip(zip_bytes, member)
                                result = run_glmy(df)
                                all_results[member] = {
                                    "result": result,
                                    "df": df,
                                }
                            except Exception as e:
                                all_results[member] = {"error": str(e)}
                        status.update(
                            label=f"Done ({len(all_results)} sub-networks)",
                            state="complete",
                        )
                    st.session_state["netanal_glmy_all_results"] = all_results

                all_results = st.session_state.get("netanal_glmy_all_results")
                if all_results:
                    for member, data in all_results.items():
                        label = member_display_label(member)
                        st.markdown(f"#### {label}")
                        if "error" in data:
                            st.error(f"GLMY 运行失败：{data['error']}")
                        else:
                            result = data["result"]
                            homology = result["homology"]
                            summary = {
                                f"β{dim}": len(homology.get(str(dim), []))
                                for dim in (0, 1, 2, 3)
                            }
                            st.json(summary)

                            max_x = suggest_max_x(data["df"])
                            fig = plot_glmy_barcode(
                                homology, max_x=max_x,
                            )
                            st.pyplot(fig, width="stretch")

                            base_name = sanitize_name(label)
                            png_buf = io.BytesIO()
                            fig.savefig(
                                png_buf, format="png", dpi=200,
                                bbox_inches="tight",
                            )
                            pdf_buf = io.BytesIO()
                            fig.savefig(
                                pdf_buf, format="pdf",
                                bbox_inches="tight",
                            )
                            plt.close(fig)

                            c1, c2, c3 = st.columns(3)
                            c1.download_button(
                                label="Download PNG",
                                data=png_buf.getvalue(),
                                file_name=f"{base_name}_barcode.png",
                                mime="image/png",
                                key=f"netanal_glmy_all_png_{member}",
                            )
                            c2.download_button(
                                label="Download PDF",
                                data=pdf_buf.getvalue(),
                                file_name=f"{base_name}_barcode.pdf",
                                mime="application/pdf",
                                key=f"netanal_glmy_all_pdf_{member}",
                            )
                            c3.download_button(
                                label="Download JSON",
                                data=json.dumps(
                                    homology, indent=2,
                                    ensure_ascii=False,
                                ).encode("utf-8"),
                                file_name=f"{base_name}_homology.json",
                                mime="application/json",
                                key=f"netanal_glmy_all_json_{member}",
                            )
                        st.divider()

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

            st.markdown(f"**Source:** `{os.path.basename(m3_csv_path)}` (repo root)")
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
    st.markdown("### Machine Learning Validation for Topology Hubs")
    st.markdown(
        "Upload the **FunClu-K export ZIP** as the ML input source, and optionally upload "
        "the **Multi-Layer IdopNetwork export ZIP** from NetRecon. The ML run uses the same "
        "inter-cluster or intra-cluster data layer selected by NetRecon, then compares ML Hub "
        "ranking against the topology Hub ranking."
    )
    st.caption(
        "Validation mode is IDOP-aligned by default: Hub ranking uses out-degree, matching "
        "NetRecon's default topology Hub selection."
    )

    tab2_1, tab2_2, tab2_3 = st.tabs(
        ["Aligned Inputs", "Run Validation", "Network View"]
    )

    # ========== Tab 2_1 Aligned Inputs ==========
    with tab2_1:
        funclu_zip = st.file_uploader(
            label="Upload FunClu-K export ZIP",
            type=["zip"],
            key="netanal_ml_funclu_zip_upload",
            help="Expected file: funclu_k_export.zip from Curve Fitting -> FunctionClu/K-Cluster.",
        )
        topology_zip = st.file_uploader(
            label="Upload NetRecon Multi-Layer IdopNetwork export ZIP (optional)",
            type=["zip"],
            key="netanal_ml_topology_zip_upload",
            help="Expected file: multi_layer_idopnetwork_export.zip from NetRecon -> Export.",
        )

        if funclu_zip is None:
            st.info("Please upload a FunClu-K export ZIP first.")
        else:
            funclu_signature = (funclu_zip.name, funclu_zip.size)
            if st.session_state.get("netanal_ml_funclu_signature") != funclu_signature:
                st.session_state.pop("netanal_ml_validation_result", None)
                st.session_state["netanal_ml_funclu_signature"] = funclu_signature

            try:
                funclu_export = load_funclu_k_export(funclu_zip.getvalue())
                funclu_summary = funclu_k_export_summary(funclu_export)
            except Exception as e:
                st.error(f"Unable to read the FunClu-K ZIP: {e}")
            else:
                st.session_state["netanal_ml_funclu_export"] = funclu_export
                st.session_state["netanal_ml_funclu_summary"] = funclu_summary

                labels_df = funclu_export["labels"]
                clusters = (
                    int(labels_df["cluster"].nunique())
                    if "cluster" in labels_df.columns
                    else int(labels_df["cluster_id"].nunique())
                    if "cluster_id" in labels_df.columns
                    else 0
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Features", f"{len(labels_df):,}")
                c2.metric("Clusters", f"{clusters:,}")
                c3.metric("Inter layers", f"{len(funclu_export['center_responses']):,}")
                c4.metric(
                    "Intra networks",
                    f"{sum(len(v) for v in funclu_export['member_responses'].values()):,}",
                )

                st.markdown("#### FunClu-K ZIP datasets")
                st.dataframe(funclu_summary, width="stretch", height=300)

                with st.expander("ZIP contents"):
                    st.code("\\n".join(funclu_export["zip_members"][:120]), language="text")
                    if len(funclu_export["zip_members"]) > 120:
                        st.caption(
                            f"Showing first 120 of {len(funclu_export['zip_members']):,} files."
                        )

        if topology_zip is not None:
            topology_signature = (topology_zip.name, topology_zip.size)
            if st.session_state.get("netanal_ml_topology_signature") != topology_signature:
                st.session_state.pop("netanal_ml_validation_result", None)
                st.session_state["netanal_ml_topology_signature"] = topology_signature

            try:
                topology_adjs = load_idopnetwork_adjacencies(topology_zip.getvalue())
            except Exception as e:
                st.error(f"Unable to read the NetRecon ZIP: {e}")
            else:
                st.session_state["netanal_ml_topology_adjs"] = topology_adjs
                topology_summary = pd.DataFrame(
                    [
                        {
                            "network": key,
                            "nodes": int(adj.shape[0]),
                            "edges_nonzero": int(
                                (
                                    adj.apply(pd.to_numeric, errors="coerce")
                                    .fillna(0.0)
                                    .to_numpy()
                                    != 0
                                ).sum()
                            ),
                        }
                        for key, adj in topology_adjs.items()
                    ]
                )
                st.markdown("#### NetRecon topology networks")
                st.dataframe(topology_summary, width="stretch", height=260)
        else:
            st.session_state.pop("netanal_ml_topology_adjs", None)

    # ========== Tab 2_2 Validation ==========
    with tab2_2:
        funclu_export = st.session_state.get("netanal_ml_funclu_export")
        topology_adjs = st.session_state.get("netanal_ml_topology_adjs", {})

        if funclu_export is None:
            st.info("Please upload a FunClu-K ZIP in **Aligned Inputs** first.")
        else:
            rank_options = {
                "out_degree": "IDOP-aligned: out-degree (default NetRecon Hub)",
                "out_strength": "IDOP-aligned: weighted out-degree",
                "total_degree": "Topology: total degree",
                "total_strength": "Topology: weighted total degree",
            }
            layer_options = []
            if funclu_export.get("center_responses"):
                layer_options.append("inter_cluster")
            if funclu_export.get("member_responses"):
                layer_options.append("intra_cluster")

            col_s1, col_s2, col_s3, col_s4 = st.columns([1, 1, 1, 1])
            with col_s1:
                selected_layer = st.selectbox(
                    "Layer",
                    options=layer_options,
                    format_func=lambda x: {
                        "inter_cluster": "Inter-cluster centers",
                        "intra_cluster": "Intra-cluster members",
                    }[x],
                    key="netanal_ml_selected_layer",
                )
            if selected_layer == "inter_cluster":
                condition_options = list(funclu_export["center_responses"].keys())
                with col_s2:
                    selected_condition = st.selectbox(
                        "Condition",
                        options=condition_options,
                        key="netanal_ml_inter_condition",
                    )
                selected_cluster = ""
            else:
                condition_options = list(funclu_export["member_responses"].keys())
                with col_s2:
                    selected_condition = st.selectbox(
                        "Condition",
                        options=condition_options,
                        key="netanal_ml_intra_condition",
                    )
                cluster_options = list(funclu_export["member_responses"][selected_condition].keys())
                with col_s3:
                    selected_cluster = st.selectbox(
                        "Cluster",
                        options=cluster_options,
                        key="netanal_ml_intra_cluster",
                    )
            with col_s4:
                data_source = st.selectbox(
                    "ML data source",
                    options=["quasi_dynamic", "curve_sample"],
                    format_func=lambda x: {
                        "quasi_dynamic": "Quasi-dynamic response",
                        "curve_sample": "Curve sample",
                    }[x],
                    key="netanal_ml_data_source",
                )

            topology_key = matching_topology_key(
                layer=str(selected_layer),
                condition=str(selected_condition),
                cluster=str(selected_cluster),
            )
            try:
                ml_matrix = get_funclu_ml_matrix(
                    funclu_export,
                    layer=str(selected_layer),
                    condition=str(selected_condition),
                    cluster=str(selected_cluster),
                    data_source=str(data_source),
                )
            except Exception as e:
                st.error(f"Unable to prepare this FunClu-K dataset for ML: {e}")
                ml_matrix = None

            if ml_matrix is not None:
                n_variables, n_samples = ml_matrix.shape
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ML variables", f"{n_variables:,}")
                c2.metric("ML samples", f"{n_samples:,}")
                c3.metric("Expected topology key", topology_key)
                if topology_adjs and topology_key in topology_adjs:
                    c4.metric("Topology match", "Found")
                    st.success(f"Matched NetRecon adjacency: `{topology_key}`")
                elif topology_adjs:
                    c4.metric("Topology match", "Missing")
                    st.warning(
                        f"No topology adjacency named `{topology_key}` was found in the NetRecon ZIP."
                    )
                else:
                    c4.metric("Topology match", "Not uploaded")
                    st.info("Upload the NetRecon export ZIP to compute ML-vs-topology overlap.")

                with st.expander("ML-ready matrix preview"):
                    st.dataframe(ml_matrix.head(80), width="stretch", height=300)

                if n_variables < 2 or n_samples < 3:
                    st.error("This selected dataset needs at least 2 variables and 3 samples.")
                else:
                    default_max_variables = min(250, int(n_variables))
                    default_cv = min(5, max(2, int(n_samples)))
                    max_possible_edges = max(1, int(n_variables) * max(1, int(n_variables) - 1))

                    if n_variables > default_max_variables:
                        st.warning(
                            f"This layer has {n_variables:,} variables. The default run keeps "
                            f"the top {default_max_variables:,} variables by variance."
                        )

                    with st.form(key="netanal_ml_validation_form"):
                        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                        with col_h1:
                            hub_rank_metric = st.selectbox(
                                "Hub ranking metric",
                                options=list(rank_options.keys()),
                                format_func=lambda x: rank_options[x],
                                index=0,
                                key="netanal_ml_hub_rank_metric",
                            )
                            max_variables = st.number_input(
                                "Max variables to analyze",
                                min_value=2,
                                max_value=max(2, int(n_variables)),
                                value=max(2, int(default_max_variables)),
                                step=10,
                                key="netanal_ml_max_variables",
                            )
                        with col_h2:
                            variable_selection = st.selectbox(
                                "If variables exceed the limit",
                                options=["variance", "mean_abs", "first"],
                                format_func=lambda x: {
                                    "variance": "Keep highest-variance variables",
                                    "mean_abs": "Keep highest mean absolute variables",
                                    "first": "Keep first variables",
                                }[x],
                                key="netanal_ml_variable_selection",
                            )
                            l1_ratio = st.slider(
                                "ElasticNet sparsity (L1 ratio)",
                                min_value=0.05,
                                max_value=1.00,
                                value=0.75,
                                step=0.05,
                                key="netanal_ml_l1_ratio",
                            )
                        with col_h3:
                            coefficient_threshold = st.number_input(
                                "Min |ML coefficient| edge",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.03,
                                step=0.01,
                                format="%.3f",
                                key="netanal_ml_coefficient_threshold",
                            )
                            topology_edge_threshold = st.number_input(
                                "Min |topology weight| edge",
                                min_value=0.0,
                                max_value=1000000.0,
                                value=0.0,
                                step=0.01,
                                format="%.4f",
                                key="netanal_ml_topology_edge_threshold",
                            )
                        with col_h4:
                            cv_folds = st.number_input(
                                "Cross-validation folds",
                                min_value=2,
                                max_value=max(2, min(10, int(n_samples))),
                                value=int(default_cv),
                                step=1,
                                key="netanal_ml_cv_folds",
                            )
                            max_edges = st.number_input(
                                "Max retained ML edges",
                                min_value=1,
                                max_value=max_possible_edges,
                                value=min(5000, max_possible_edges),
                                step=max(1, min(500, max_possible_edges)),
                                key="netanal_ml_max_edges",
                            )

                        col_v1, col_v2 = st.columns(2)
                        with col_v1:
                            topology_top_n = st.number_input(
                                "Topology top N to validate",
                                min_value=1,
                                max_value=max(1, int(n_variables)),
                                value=min(3, int(n_variables)),
                                step=1,
                                key="netanal_ml_topology_top_n",
                            )
                        with col_v2:
                            ml_top_n = st.number_input(
                                "ML top N candidate list",
                                min_value=1,
                                max_value=max(1, int(n_variables)),
                                value=min(20, int(n_variables)),
                                step=1,
                                key="netanal_ml_top_n",
                            )

                        run_ml_hub = st.form_submit_button(
                            "Run ML Validation",
                            type="primary",
                        )

                    if run_ml_hub:
                        with st.spinner("Running target-wise ElasticNet and Hub comparison ..."):
                            try:
                                ml_result = infer_signed_hub_network(
                                    ml_matrix,
                                    max_variables=int(max_variables),
                                    variable_selection=str(variable_selection),
                                    l1_ratio=float(l1_ratio),
                                    cv_folds=int(cv_folds),
                                    coefficient_threshold=float(coefficient_threshold),
                                    max_edges=int(max_edges),
                                    random_state=123,
                                )
                                ranked_hubs = ml_result["hub_scores"].copy()
                                tie_cols = []
                                for col in [
                                    str(hub_rank_metric),
                                    "out_degree",
                                    "out_strength",
                                    "hub_score",
                                ]:
                                    if col in ranked_hubs.columns and col not in tie_cols:
                                        tie_cols.append(col)
                                ranked_hubs = ranked_hubs.sort_values(
                                    tie_cols,
                                    ascending=False,
                                ).reset_index(drop=True)
                                ranked_hubs.insert(
                                    0,
                                    "hub_rank",
                                    np.arange(1, len(ranked_hubs) + 1),
                                )

                                topology_hubs = None
                                comparison = None
                                if topology_adjs and topology_key in topology_adjs:
                                    topology_hubs = hub_table_from_adjacency(
                                        topology_adjs[topology_key],
                                        edge_threshold=float(topology_edge_threshold),
                                        rank_metric=str(hub_rank_metric),
                                    )
                                    comparison = compare_hub_tables(
                                        ranked_hubs,
                                        topology_hubs,
                                        rank_metric=str(hub_rank_metric),
                                        topology_top_n=int(topology_top_n),
                                        ml_top_n=int(ml_top_n),
                                    )
                            except Exception as e:
                                st.error(f"ML validation failed: {e}")
                            else:
                                st.session_state["netanal_ml_validation_result"] = {
                                    "ml_result": ml_result,
                                    "ranked_hubs": ranked_hubs,
                                    "topology_hubs": topology_hubs,
                                    "comparison": comparison,
                                    "matrix": ml_matrix,
                                    "context": {
                                        "layer": str(selected_layer),
                                        "condition": str(selected_condition),
                                        "cluster": str(selected_cluster),
                                        "data_source": str(data_source),
                                        "topology_key": topology_key,
                                        "rank_metric": str(hub_rank_metric),
                                    },
                                }

        validation_result = st.session_state.get("netanal_ml_validation_result")
        if validation_result is None:
            st.info("Click **Run ML Validation** to generate the ML Hub list and topology comparison.")
        else:
            ml_result = validation_result["ml_result"]
            ranked_hubs = validation_result["ranked_hubs"]
            signed_edges = ml_result["edges"]
            model_scores = ml_result["model_scores"]
            adjacency = ml_result["adjacency"]
            metadata = ml_result["metadata"]
            topology_hubs = validation_result.get("topology_hubs")
            comparison = validation_result.get("comparison")
            context = validation_result["context"]
            hub_rank_metric = context["rank_metric"]

            top_ml_hub = ranked_hubs.iloc[0] if not ranked_hubs.empty else None
            col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
            col_r1.metric(
                "ML top Hub",
                str(top_ml_hub["variable"]) if top_ml_hub is not None else "N/A",
            )
            col_r2.metric(
                "ML rank value",
                f"{float(top_ml_hub[hub_rank_metric]):.4f}" if top_ml_hub is not None else "0.0000",
            )
            col_r3.metric("ML signed edges", f"{metadata['edge_count']:,}")
            col_r4.metric("Median target R2", f"{model_scores['r2'].median():.3f}")
            if comparison is not None:
                summary = comparison["summary"]
                col_r5.metric(
                    "Overlap",
                    f"{summary['overlap_count']}/{summary['topology_top_n']}",
                    help=f"Topology top N vs ML top {summary['ml_top_n']}",
                )
            else:
                col_r5.metric("Overlap", "N/A")

            st.caption(
                "Selection: "
                f"{context['topology_key']} | data={context['data_source']} | "
                f"rank={hub_rank_metric}"
            )

            if metadata["variables_dropped_by_selection"] > 0:
                st.warning(
                    f"{metadata['variables_dropped_by_selection']:,} variables were not modeled "
                    "because of the current Max variables limit."
                )

            if comparison is not None:
                summary = comparison["summary"]
                if summary["overlap_count"] > 0:
                    st.success(
                        "Matched topology Hub candidates in the ML list: "
                        f"{summary['overlap_nodes']}"
                    )
                else:
                    st.warning(
                        "No overlap between the selected topology top N and ML candidate list. "
                        "Try lowering ML sparsity/threshold, increasing ML top N, or validating "
                        "the same layer/data source used by NetRecon."
                    )
                comp_cols = st.columns(3)
                comp_cols[0].metric("Common nodes", f"{summary['common_nodes']:,}")
                comp_cols[1].metric(
                    "Overlap rate",
                    f"{summary['overlap_rate_vs_topology_top_n']:.2%}",
                )
                comp_cols[2].metric(
                    "Spearman",
                    (
                        f"{summary['spearman_metric_correlation']:.3f}"
                        if pd.notna(summary["spearman_metric_correlation"])
                        else "N/A"
                    ),
                )

            st.markdown("#### ML Hub ranking")
            hub_columns = []
            for col in [
                "hub_rank",
                "rank",
                "variable",
                str(hub_rank_metric),
                "hub_score",
                "role",
                "out_strength",
                "in_strength",
                "out_degree",
                "in_degree",
                "promoting_out_strength",
                "inhibiting_out_strength",
                "pagerank",
                "betweenness",
                "target_r2",
            ]:
                if col in ranked_hubs.columns and col not in hub_columns:
                    hub_columns.append(col)
            st.dataframe(ranked_hubs[hub_columns].head(100), width="stretch", height=340)

            if topology_hubs is not None:
                st.markdown("#### Topology Hub ranking from NetRecon adjacency")
                st.dataframe(topology_hubs.head(100), width="stretch", height=300)

            if comparison is not None:
                st.markdown("#### ML vs topology rank comparison")
                st.dataframe(comparison["detail"], width="stretch", height=320)

            st.markdown("#### Signed ML source -> target effects")
            edge_display = signed_edges.copy()
            if not edge_display.empty:
                edge_display["effect_label"] = edge_display["effect"].map(
                    {"promote": "Promote (+)", "inhibit": "Inhibit (-)"}
                )
                edge_columns = [
                    "source",
                    "target",
                    "effect_label",
                    "coefficient",
                    "abs_weight",
                    "target_r2",
                    "alpha",
                ]
                st.dataframe(edge_display[edge_columns].head(200), width="stretch", height=320)
            else:
                st.warning(
                    "No ML edges passed the current coefficient threshold. Try lowering the "
                    "threshold or reducing ElasticNet sparsity."
                )

            dl1, dl2, dl3, dl4 = st.columns(4)
            dl1.download_button(
                label="Download ML Hub CSV",
                data=ranked_hubs.to_csv(index=False).encode("utf-8"),
                file_name="ml_validation_hub_ranking.csv",
                mime="text/csv",
                key="netanal_ml_hub_download",
            )
            dl2.download_button(
                label="Download ML edges CSV",
                data=signed_edges.to_csv(index=False).encode("utf-8"),
                file_name="ml_validation_signed_edges.csv",
                mime="text/csv",
                key="netanal_ml_edges_download",
            )
            dl3.download_button(
                label="Download ML adjacency CSV",
                data=adjacency.to_csv(index=True).encode("utf-8"),
                file_name="ml_validation_signed_adjacency.csv",
                mime="text/csv",
                key="netanal_ml_adjacency_download",
            )
            if comparison is not None:
                dl4.download_button(
                    label="Download comparison CSV",
                    data=comparison["detail"].to_csv(index=False).encode("utf-8"),
                    file_name="ml_vs_topology_hub_comparison.csv",
                    mime="text/csv",
                    key="netanal_ml_comparison_download",
                )

    # ========== Tab 2_3 Network View ==========
    with tab2_3:
        validation_result = st.session_state.get("netanal_ml_validation_result")

        if validation_result is None:
            st.info("Run **ML Validation** first to generate a signed network.")
        else:
            ml_result = validation_result["ml_result"]
            signed_edges = ml_result["edges"]
            hub_scores = validation_result["ranked_hubs"]
            hub_rank_metric = validation_result["context"].get("rank_metric", "out_degree")

            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                plot_node_count = int(len(hub_scores))
                plot_node_max = min(100, plot_node_count)
                if plot_node_max <= 5:
                    plot_top_nodes = max(1, plot_node_count)
                    st.metric("Top nodes in plot", plot_top_nodes)
                else:
                    plot_top_nodes = st.slider(
                        "Top nodes in plot",
                        min_value=5,
                        max_value=plot_node_max,
                        value=min(40, plot_node_max),
                        step=5,
                        key="netanal_ml_plot_top_nodes",
                    )
            with col_v2:
                plot_edge_count = int(len(signed_edges))
                plot_edge_max = min(500, plot_edge_count)
                if plot_edge_max <= 10:
                    plot_top_edges = max(1, plot_edge_count)
                    st.metric("Top edges in plot", plot_top_edges)
                else:
                    plot_top_edges = st.slider(
                        "Top edges in plot",
                        min_value=10,
                        max_value=plot_edge_max,
                        value=min(120, plot_edge_max),
                        step=10,
                        key="netanal_ml_plot_top_edges",
                    )
            with col_v3:
                label_max = min(40, int(len(hub_scores)))
                if label_max <= 0:
                    label_top_n = 0
                    st.metric("Labels for top hubs", label_top_n)
                else:
                    label_top_n = st.slider(
                        "Labels for top hubs",
                        min_value=0,
                        max_value=label_max,
                        value=min(12, label_max),
                        step=1,
                        key="netanal_ml_label_top_n",
                    )

            ml_fig = plot_hub_network(
                signed_edges,
                hub_scores,
                top_nodes=int(plot_top_nodes),
                top_edges=int(plot_top_edges),
                label_top_n=int(label_top_n),
                rank_metric=str(hub_rank_metric),
                random_state=123,
            )
            st.pyplot(ml_fig, width="stretch")

            png_buf = io.BytesIO()
            ml_fig.savefig(png_buf, format="png", dpi=220, bbox_inches="tight")
            pdf_buf = io.BytesIO()
            ml_fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
            plt.close(ml_fig)

            c_png, c_pdf = st.columns(2)
            c_png.download_button(
                label="Download ML network PNG",
                data=png_buf.getvalue(),
                file_name="ml_hub_network.png",
                mime="image/png",
                key="netanal_ml_network_png_download",
            )
            c_pdf.download_button(
                label="Download ML network PDF",
                data=pdf_buf.getvalue(),
                file_name="ml_hub_network.pdf",
                mime="application/pdf",
                key="netanal_ml_network_pdf_download",
            )


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
