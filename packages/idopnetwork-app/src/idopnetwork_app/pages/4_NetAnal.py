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
    module_feature_map_from_labels,
    predict_unknown_condition_samples,
    run_intra_module_feature_importance,
    run_module_classification_validation,
    run_module_single_feature_validation,
    run_module_stability_validation,
    topology_hubs_from_adjacencies,
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


def _clean_condition_label(value: object) -> str:
    """Normalize condition/file labels for loose topology-vs-ML matching."""
    text = os.path.basename(str(value).strip())
    text = os.path.splitext(text)[0]
    return text.casefold()


def _topology_rows_for_classification(
    topology_hubs: pd.DataFrame,
    *,
    positive_label: str,
    module_names: list[str],
) -> tuple[pd.DataFrame, str]:
    """Choose topology Hub rows that should be compared with classification scores."""
    if topology_hubs.empty:
        return topology_hubs, "not uploaded"

    positive_key = _clean_condition_label(positive_label)
    exact_mask = topology_hubs["condition"].map(
        lambda value: _clean_condition_label(value) == positive_key
    )
    exact_rows = topology_hubs[exact_mask].copy()
    if not exact_rows.empty:
        return exact_rows, "matched condition name"

    if len(topology_hubs) == 1:
        return topology_hubs.copy(), "single inter-cluster topology fallback"

    module_set = {str(name) for name in module_names if str(name) != "All"}
    module_rows = topology_hubs[
        topology_hubs["topology_hub"].astype(str).isin(module_set)
    ].copy()
    if not module_rows.empty:
        return module_rows.drop_duplicates("topology_hub"), "matched Hub module name"

    return topology_hubs.iloc[0:0].copy(), "no topology row matched"


def _plot_module_classification_scores(
    scores: pd.DataFrame,
    *,
    expected_hubs: list[str],
    task_label: str,
):
    """Visualize module-level classification scores for Hub validation."""
    if scores.empty or "primary_score_mean" not in scores.columns:
        return None

    plot_df = scores.copy()
    plot_df["module"] = plot_df["module"].astype(str)
    plot_df["score"] = pd.to_numeric(
        plot_df["primary_score_mean"],
        errors="coerce",
    )
    if "primary_score_std" in plot_df.columns:
        plot_df["score_std"] = pd.to_numeric(
            plot_df["primary_score_std"],
            errors="coerce",
        ).fillna(0.0)
    else:
        plot_df["score_std"] = 0.0
    plot_df = plot_df[pd.notna(plot_df["score"])].copy()
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(["score", "module"], ascending=[True, True])
    y_pos = np.arange(len(plot_df))

    metric_name = (
        str(plot_df["primary_metric"].dropna().iloc[0])
        if "primary_metric" in plot_df.columns
        and not plot_df["primary_metric"].dropna().empty
        else "primary score"
    )
    expected_set = {str(hub) for hub in expected_hubs}
    best_single = plot_df[plot_df["module"] != "All"]
    best_single_module = (
        str(best_single.sort_values("score", ascending=False).iloc[0]["module"])
        if not best_single.empty
        else ""
    )

    score_min = float(plot_df["score"].min())
    score_max = float((plot_df["score"] + plot_df["score_std"]).max())
    use_unit_axis = score_min >= -0.02 and score_max <= 1.05
    if use_unit_axis:
        x_left = 0.0
        x_right = 1.0
    else:
        pad = max(0.05, (score_max - score_min) * 0.12)
        x_left = max(0.0, score_min - pad)
        x_right = score_max + pad

    cmap = plt.cm.Blues
    denom = max(1e-9, float(plot_df["score"].max() - plot_df["score"].min()))
    colors = []
    edgecolors = []
    hatches = []
    for _, row in plot_df.iterrows():
        module = str(row["module"])
        if module in expected_set:
            colors.append("#f59e0b")
            edgecolors.append("#92400e")
            hatches.append("")
        elif module == best_single_module:
            colors.append("#2563eb")
            edgecolors.append("#1e3a8a")
            hatches.append("")
        elif module == "All":
            colors.append("#64748b")
            edgecolors.append("#334155")
            hatches.append("//")
        else:
            normalized = 0.35 + 0.55 * (float(row["score"]) - score_min) / denom
            colors.append(cmap(normalized))
            edgecolors.append("#475569")
            hatches.append("")

    fig_height = max(3.6, 0.42 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    bars = ax.barh(
        y_pos,
        plot_df["score"].to_numpy(dtype=float),
        color=colors,
        edgecolor=edgecolors,
        linewidth=1.0,
    )
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    xerr = plot_df["score_std"].to_numpy(dtype=float)
    if np.nanmax(xerr) > 0:
        ax.errorbar(
            plot_df["score"].to_numpy(dtype=float),
            y_pos,
            xerr=xerr,
            fmt="none",
            ecolor="#334155",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=3,
        )

    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        score = float(row["score"])
        label_x = min(x_right - 0.01, score + (x_right - x_left) * 0.012)
        ax.text(
            label_x,
            y,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#0f172a",
        )

    all_rows = plot_df[plot_df["module"] == "All"]
    if not all_rows.empty:
        all_score = float(all_rows.iloc[0]["score"])
        ax.axvline(
            all_score,
            color="#64748b",
            linestyle=":",
            linewidth=1.4,
            label="All-feature baseline",
        )

    chance_line = 0.5 if metric_name == "roc_auc" else None
    if chance_line is not None and x_left <= chance_line <= x_right:
        ax.axvline(
            chance_line,
            color="#94a3b8",
            linestyle="--",
            linewidth=1.0,
            label="Chance level",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["module"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel(metric_name)
    ax.set_title(f"Module classification score ({task_label})")
    ax.grid(axis="x", color="#e5e7eb", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#f59e0b", edgecolor="#92400e", label="Topology Hub"),
        Patch(facecolor="#2563eb", edgecolor="#1e3a8a", label="Best single module"),
        Patch(facecolor="#64748b", edgecolor="#334155", hatch="//", label="All"),
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=legend_handles + line_handles,
        labels=[h.get_label() for h in legend_handles] + line_labels,
        loc="lower right",
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def _plot_single_feature_module_scores(
    feature_scores: pd.DataFrame,
    module_scores: pd.DataFrame,
    *,
    modules: list[str],
    task_label: str,
    max_features_per_module: int,
):
    """Draw Feishu-style single-feature AUC bars against module-level score."""
    if feature_scores.empty or "primary_score_mean" not in feature_scores.columns:
        return None

    plot_source = feature_scores.copy()
    plot_source["module"] = plot_source["module"].astype(str)
    plot_source["feature"] = plot_source["feature"].astype(str)
    plot_source["score"] = pd.to_numeric(
        plot_source["primary_score_mean"],
        errors="coerce",
    )
    if "primary_score_std" in plot_source.columns:
        plot_source["score_std"] = pd.to_numeric(
            plot_source["primary_score_std"],
            errors="coerce",
        ).fillna(0.0)
    else:
        plot_source["score_std"] = 0.0
    plot_source = plot_source[pd.notna(plot_source["score"])].copy()
    if plot_source.empty:
        return None

    module_source = module_scores.copy()
    if not module_source.empty:
        module_source["module"] = module_source["module"].astype(str)
        module_source["module_score"] = pd.to_numeric(
            module_source["primary_score_mean"],
            errors="coerce",
        )
    module_score_lookup = (
        module_source.drop_duplicates("module").set_index("module")["module_score"].to_dict()
        if not module_source.empty and "module_score" in module_source.columns
        else {}
    )

    modules_to_plot = [
        str(module)
        for module in modules
        if not plot_source[plot_source["module"] == str(module)].empty
    ]
    if not modules_to_plot:
        modules_to_plot = plot_source["module"].drop_duplicates().astype(str).tolist()
    if not modules_to_plot:
        return None

    max_plot_features = max(1, int(max_features_per_module))
    metric_name = (
        str(plot_source["primary_metric"].dropna().iloc[0])
        if "primary_metric" in plot_source.columns
        and not plot_source["primary_metric"].dropna().empty
        else "primary score"
    )
    score_max = float(
        max(
            plot_source["score"].max(),
            plot_source["score"].add(plot_source["score_std"]).max(),
            max(
                [value for value in module_score_lookup.values() if pd.notna(value)]
                or [0.0]
            ),
        )
    )
    use_unit_axis = float(plot_source["score"].min()) >= -0.02 and score_max <= 1.05
    x_left = 0.0
    x_right = 1.0 if use_unit_axis else score_max + max(0.05, score_max * 0.08)

    height_ratios = []
    prepared: list[tuple[str, pd.DataFrame]] = []
    for module in modules_to_plot:
        module_df = plot_source[plot_source["module"] == module].copy()
        module_df = module_df.sort_values(
            ["score", "feature"],
            ascending=[False, True],
        ).head(max_plot_features)
        module_df = module_df.sort_values(["score", "feature"], ascending=[True, True])
        if module_df.empty:
            continue
        prepared.append((module, module_df))
        height_ratios.append(max(2.4, 0.34 * len(module_df) + 0.9))
    if not prepared:
        return None

    fig_height = max(3.2, sum(height_ratios) + 0.6)
    fig, axes = plt.subplots(
        len(prepared),
        1,
        figsize=(10.5, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if len(prepared) == 1:
        axes = [axes]

    for ax, (module, module_df) in zip(axes, prepared):
        y_pos = np.arange(len(module_df))
        denom = max(1e-9, float(module_df["score"].max() - module_df["score"].min()))
        colors = [
            plt.cm.Blues(0.35 + 0.55 * (float(score) - float(module_df["score"].min())) / denom)
            for score in module_df["score"].tolist()
        ]
        if colors:
            colors[-1] = "#1d4ed8"

        ax.barh(
            y_pos,
            module_df["score"].to_numpy(dtype=float),
            color=colors,
            edgecolor="#475569",
            linewidth=0.9,
        )
        xerr = module_df["score_std"].to_numpy(dtype=float)
        if np.nanmax(xerr) > 0:
            ax.errorbar(
                module_df["score"].to_numpy(dtype=float),
                y_pos,
                xerr=xerr,
                fmt="none",
                ecolor="#334155",
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                zorder=3,
            )
        for y, (_, row) in zip(y_pos, module_df.iterrows()):
            score = float(row["score"])
            ax.text(
                min(x_right - 0.01, score + (x_right - x_left) * 0.012),
                y,
                f"{score:.3f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#0f172a",
            )

        module_score = module_score_lookup.get(module, np.nan)
        if pd.notna(module_score):
            ax.axvline(
                float(module_score),
                color="#ef4444",
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                label="Module score",
            )
        if metric_name == "roc_auc" and x_left <= 0.5 <= x_right:
            ax.axvline(
                0.5,
                color="#94a3b8",
                linestyle=":",
                linewidth=1.0,
                label="Chance level",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(module_df["feature"].tolist(), fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(x_left, x_right)
        ax.set_title(f"{module} ({task_label})", fontsize=10, fontweight="bold")
        ax.grid(axis="x", color="#e5e7eb", linestyle="-", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel(metric_name)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower right",
            frameon=True,
            fontsize=9,
        )
    fig.tight_layout()
    return fig


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

    tab2_1, tab2_2, tab2_3, tab2_4, tab2_5 = st.tabs(
        [
            "Aligned Inputs",
            "Network Validation",
            "Classification Validation",
            "Sample Prediction",
            "Network View",
        ]
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
        condition_csvs = st.file_uploader(
            label="Upload condition CSV files for classification validation",
            type=["csv"],
            accept_multiple_files=True,
            key="netanal_ml_condition_csv_uploads",
            help=(
                "Each CSV is one condition/label. Rows are samples/subjects; "
                "columns are shared indicators/features."
            ),
        )
        first_column_as_sample_id = st.checkbox(
            "Condition CSV first column contains sample IDs",
            value=True,
            key="netanal_ml_condition_first_col_sample_id",
        )

        if funclu_zip is None:
            st.info("Please upload a FunClu-K export ZIP first.")
        else:
            funclu_signature = (funclu_zip.name, funclu_zip.size)
            if st.session_state.get("netanal_ml_funclu_signature") != funclu_signature:
                st.session_state.pop("netanal_ml_validation_result", None)
                st.session_state.pop("netanal_ml_classification_result", None)
                st.session_state.pop("netanal_ml_intra_result", None)
                st.session_state.pop("netanal_ml_stability_result", None)
                st.session_state.pop("netanal_ml_single_feature_result", None)
                st.session_state.pop("netanal_ml_prediction_result", None)
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

        if condition_csvs:
            condition_signature = tuple((f.name, f.size) for f in condition_csvs) + (
                bool(first_column_as_sample_id),
            )
            if st.session_state.get("netanal_ml_condition_signature") != condition_signature:
                st.session_state.pop("netanal_ml_classification_result", None)
                st.session_state.pop("netanal_ml_intra_result", None)
                st.session_state.pop("netanal_ml_stability_result", None)
                st.session_state.pop("netanal_ml_single_feature_result", None)
                st.session_state.pop("netanal_ml_prediction_result", None)
                st.session_state["netanal_ml_condition_signature"] = condition_signature

            condition_tables = {}
            condition_rows = []
            duplicate_counts = {}
            for uploaded_condition in condition_csvs:
                base_label = os.path.splitext(uploaded_condition.name)[0]
                duplicate_counts[base_label] = duplicate_counts.get(base_label, 0) + 1
                condition_label = (
                    base_label
                    if duplicate_counts[base_label] == 1
                    else f"{base_label}_{duplicate_counts[base_label]}"
                )
                try:
                    raw_condition_df = pd.read_csv(uploaded_condition)
                except Exception as e:
                    st.error(f"Unable to read condition CSV `{uploaded_condition.name}`: {e}")
                    continue
                condition_tables[condition_label] = raw_condition_df
                feature_count = (
                    max(0, int(raw_condition_df.shape[1]) - 1)
                    if first_column_as_sample_id
                    else int(raw_condition_df.shape[1])
                )
                condition_rows.append(
                    {
                        "condition_label": condition_label,
                        "file": uploaded_condition.name,
                        "samples": int(raw_condition_df.shape[0]),
                        "feature_columns": feature_count,
                    }
                )

            if condition_tables:
                st.session_state["netanal_ml_condition_tables"] = condition_tables
                st.markdown("#### Classification condition CSVs")
                st.dataframe(pd.DataFrame(condition_rows), width="stretch", height=220)
        else:
            st.session_state.pop("netanal_ml_condition_tables", None)
            st.session_state.pop("netanal_ml_classification_result", None)
            st.session_state.pop("netanal_ml_intra_result", None)
            st.session_state.pop("netanal_ml_stability_result", None)
            st.session_state.pop("netanal_ml_single_feature_result", None)
            st.session_state.pop("netanal_ml_prediction_result", None)

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

    # ========== Tab 2_3 Classification Validation ==========
    with tab2_3:
        funclu_export = st.session_state.get("netanal_ml_funclu_export")
        condition_tables = st.session_state.get("netanal_ml_condition_tables")
        topology_adjs = st.session_state.get("netanal_ml_topology_adjs", {})

        if funclu_export is None:
            st.info("Please upload a FunClu-K ZIP in **Aligned Inputs** first.")
        elif not condition_tables:
            st.info(
                "Please upload two or more condition CSV files in **Aligned Inputs**. "
                "Each CSV is treated as one class label."
            )
        else:
            condition_options = list(condition_tables.keys())
            if len(condition_options) < 2:
                st.warning("Classification validation needs at least two condition CSVs.")
            else:
                with st.form(key="netanal_ml_classification_form"):
                    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                    with col_c1:
                        classification_task = st.selectbox(
                            "Classification task",
                            options=["one_vs_rest", "multiclass"],
                            format_func=lambda x: {
                                "one_vs_rest": "One condition vs others",
                                "multiclass": "All conditions multiclass",
                            }[x],
                            key="netanal_ml_classification_task",
                        )
                        positive_label = st.selectbox(
                            "Positive condition",
                            options=condition_options,
                            key="netanal_ml_classification_positive_label",
                            disabled=classification_task != "one_vs_rest",
                        )
                    with col_c2:
                        classifier = st.selectbox(
                            "Classifier",
                            options=["logistic_regression", "random_forest"],
                            format_func=lambda x: {
                                "logistic_regression": "Logistic Regression (L2)",
                                "random_forest": "Random Forest",
                            }[x],
                            key="netanal_ml_classification_classifier",
                        )
                        cv_folds = st.number_input(
                            "Cross-validation folds",
                            min_value=2,
                            max_value=10,
                            value=5,
                            step=1,
                            key="netanal_ml_classification_cv_folds",
                        )
                    with col_c3:
                        max_missing_fraction = st.slider(
                            "Max missing fraction per feature",
                            min_value=0.0,
                            max_value=0.95,
                            value=0.50,
                            step=0.05,
                            key="netanal_ml_classification_max_missing",
                        )
                    with col_c4:
                        topology_rank_metric = st.selectbox(
                            "Topology Hub metric",
                            options=["out_degree", "out_strength", "total_degree", "total_strength"],
                            index=0,
                            key="netanal_ml_classification_topology_metric",
                        )
                        topology_edge_threshold = st.number_input(
                            "Min |topology weight| edge",
                            min_value=0.0,
                            max_value=1000000.0,
                            value=0.0,
                            step=0.01,
                            format="%.4f",
                            key="netanal_ml_classification_topology_edge_threshold",
                        )

                    run_classification = st.form_submit_button(
                        "Run Module Classification Validation",
                        type="primary",
                    )

                if run_classification:
                    with st.spinner("Running module-level classification validation ..."):
                        try:
                            classification_result = run_module_classification_validation(
                                condition_tables,
                                funclu_export["labels"],
                                first_column_as_sample_id=bool(
                                    st.session_state.get(
                                        "netanal_ml_condition_first_col_sample_id",
                                        True,
                                    )
                                ),
                                max_missing_fraction=float(max_missing_fraction),
                                task=str(classification_task),
                                positive_label=(
                                    str(positive_label)
                                    if classification_task == "one_vs_rest"
                                    else None
                                ),
                                classifier=str(classifier),
                                cv_folds=int(cv_folds),
                                random_state=123,
                            )
                            topology_hubs = (
                                topology_hubs_from_adjacencies(
                                    topology_adjs,
                                    rank_metric=str(topology_rank_metric),
                                    edge_threshold=float(topology_edge_threshold),
                                )
                                if topology_adjs
                                else pd.DataFrame()
                            )
                        except Exception as e:
                            st.error(f"Module classification validation failed: {e}")
                        else:
                            classification_result["topology_hubs"] = topology_hubs
                            classification_result["context"] = {
                                "task": str(classification_task),
                                "positive_label": (
                                    str(positive_label)
                                    if classification_task == "one_vs_rest"
                                    else ""
                                ),
                                "classifier": str(classifier),
                                "cv_folds": int(cv_folds),
                                "max_missing_fraction": float(max_missing_fraction),
                                "topology_rank_metric": str(topology_rank_metric),
                            }
                            st.session_state[
                                "netanal_ml_classification_result"
                            ] = classification_result
                            st.session_state.pop("netanal_ml_intra_result", None)
                            st.session_state.pop("netanal_ml_stability_result", None)
                            st.session_state.pop("netanal_ml_single_feature_result", None)

        classification_result = st.session_state.get("netanal_ml_classification_result")
        if classification_result is None:
            st.info(
                "Run classification to compare M1, M2, ..., All as predictors of condition labels."
            )
        else:
            scores = classification_result["scores"].copy()
            dataset_info = classification_result["dataset"]
            context = classification_result["context"]
            topology_hubs = classification_result.get("topology_hubs", pd.DataFrame())

            expected_hub = ""
            expected_hubs: list[str] = []
            topology_match_source = ""
            if context.get("task") == "one_vs_rest" and not topology_hubs.empty:
                positive_label = context.get("positive_label", "")
                topo_match, topology_match_source = _topology_rows_for_classification(
                    topology_hubs,
                    positive_label=str(positive_label),
                    module_names=scores["module"].astype(str).tolist(),
                )
                if not topo_match.empty:
                    expected_hubs = [
                        str(value)
                        for value in topo_match["topology_hub"].dropna().astype(str).unique()
                    ]
                    expected_hub = ", ".join(expected_hubs)
                    scores["is_topology_hub"] = scores["module"].astype(str).isin(expected_hubs)
                else:
                    scores["is_topology_hub"] = False
            else:
                scores["is_topology_hub"] = False

            best_row = scores.iloc[0] if not scores.empty else None
            single_scores = scores[scores["module"].astype(str) != "All"]
            best_single_row = single_scores.iloc[0] if not single_scores.empty else None
            topology_row = (
                scores[scores["module"].astype(str).isin(expected_hubs)].iloc[0]
                if expected_hubs and scores["module"].astype(str).isin(expected_hubs).any()
                else None
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Best module",
                str(best_row["module"]) if best_row is not None else "N/A",
            )
            m2.metric(
                "Best score",
                (
                    f"{float(best_row['primary_score_mean']):.3f}"
                    if best_row is not None and pd.notna(best_row["primary_score_mean"])
                    else "N/A"
                ),
                help=(
                    str(best_row["primary_metric"])
                    if best_row is not None and "primary_metric" in best_row
                    else None
                ),
            )
            m3.metric("Topology Hub", expected_hub if expected_hub else "N/A")
            m4.metric(
                "Topology Hub rank",
                (
                    str(int(topology_row["module_rank"]))
                    if topology_row is not None
                    else "N/A"
                ),
            )

            if expected_hub:
                if best_single_row is not None and str(best_single_row["module"]) in expected_hubs:
                    st.success(
                        f"Classification supports topology Hub `{expected_hub}`: "
                        "it is the best single module for this task."
                    )
                else:
                    st.warning(
                        f"Topology Hub `{expected_hub}` is not the best classification module "
                        "under the current settings."
                    )
                st.caption(f"Topology matching rule: {topology_match_source}")

            st.caption(
                f"Task: {context.get('task')} | positive={context.get('positive_label', '')} | "
                f"classifier={context.get('classifier')}"
            )

            st.markdown("#### Module classification scores")
            display_cols = [
                "module_rank",
                "module",
                "is_topology_hub",
                "primary_metric",
                "primary_score_mean",
                "primary_score_std",
                "balanced_accuracy_mean",
                "roc_auc_mean",
                "accuracy_mean",
                "f1_macro_mean",
                "n_features",
                "n_samples",
                "cv_folds_used",
                "status",
            ]
            display_cols = [col for col in display_cols if col in scores.columns]
            st.dataframe(scores[display_cols], width="stretch", height=360)

            st.markdown("#### Module score visualization")
            task_title = (
                f"{context.get('positive_label')} vs Other"
                if context.get("task") == "one_vs_rest"
                and context.get("positive_label")
                else "multiclass"
            )
            score_fig = _plot_module_classification_scores(
                scores,
                expected_hubs=expected_hubs,
                task_label=str(task_title),
            )
            if score_fig is None:
                st.info("No valid module scores are available for visualization.")
            else:
                st.pyplot(score_fig, width="stretch")
                st.caption(
                    "Bars show the cross-validated module score; error bars show the "
                    "cross-validation standard deviation. A topology Hub is supported "
                    "when its highlighted bar is also among the strongest single-module "
                    "classifiers."
                )
                score_png = io.BytesIO()
                score_fig.savefig(
                    score_png,
                    format="png",
                    dpi=220,
                    bbox_inches="tight",
                )
                score_pdf = io.BytesIO()
                score_fig.savefig(score_pdf, format="pdf", bbox_inches="tight")
                plt.close(score_fig)
                dl_score_png, dl_score_pdf = st.columns(2)
                dl_score_png.download_button(
                    label="Download module score PNG",
                    data=score_png.getvalue(),
                    file_name="module_classification_score_visualization.png",
                    mime="image/png",
                    key="netanal_ml_classification_score_png_download",
                )
                dl_score_pdf.download_button(
                    label="Download module score PDF",
                    data=score_pdf.getvalue(),
                    file_name="module_classification_score_visualization.pdf",
                    mime="application/pdf",
                    key="netanal_ml_classification_score_pdf_download",
                )

            st.markdown("#### Single-feature validation inside modules")
            if "n_features" in single_scores.columns:
                single_feature_counts = pd.to_numeric(
                    single_scores["n_features"],
                    errors="coerce",
                ).fillna(0)
            else:
                single_feature_counts = pd.Series(
                    1,
                    index=single_scores.index,
                    dtype=float,
                )
            feature_module_options = (
                single_scores[single_feature_counts > 0]["module"]
                .astype(str)
                .tolist()
            )
            if not feature_module_options:
                st.info("No usable single modules are available for single-feature validation.")
            else:
                default_feature_modules: list[str] = []
                for module in expected_hubs:
                    if module in feature_module_options and module not in default_feature_modules:
                        default_feature_modules.append(module)
                if best_single_row is not None:
                    best_single_module = str(best_single_row["module"])
                    if (
                        best_single_module in feature_module_options
                        and best_single_module not in default_feature_modules
                    ):
                        default_feature_modules.append(best_single_module)
                for module in single_scores["module"].astype(str).head(2).tolist():
                    if module in feature_module_options and module not in default_feature_modules:
                        default_feature_modules.append(module)
                default_feature_modules = default_feature_modules[:2] or [
                    feature_module_options[0]
                ]

                with st.form(key="netanal_ml_single_feature_form"):
                    sf_c1, sf_c2, sf_c3, sf_c4 = st.columns(4)
                    with sf_c1:
                        single_feature_modules = st.multiselect(
                            "Modules to compare",
                            options=feature_module_options,
                            default=default_feature_modules,
                            key="netanal_ml_single_feature_modules",
                            help=(
                                "Feishu-style plot: each selected module gets one panel. "
                                "Each bar is one feature used alone."
                            ),
                        )
                        single_feature_classifier = st.selectbox(
                            "Single-feature model",
                            options=["logistic_regression", "random_forest"],
                            index=(
                                ["logistic_regression", "random_forest"].index(
                                    str(context.get("classifier", "logistic_regression"))
                                )
                                if str(context.get("classifier", "logistic_regression"))
                                in {"logistic_regression", "random_forest"}
                                else 0
                            ),
                            format_func=lambda x: {
                                "logistic_regression": "Logistic Regression (L2)",
                                "random_forest": "Random Forest",
                            }[x],
                            key="netanal_ml_single_feature_classifier",
                        )
                    with sf_c2:
                        single_feature_cv_folds = st.number_input(
                            "Single-feature CV folds",
                            min_value=2,
                            max_value=10,
                            value=int(context.get("cv_folds", 5)),
                            step=1,
                            key="netanal_ml_single_feature_cv_folds",
                        )
                        single_feature_max_missing = st.slider(
                            "Single-feature max missing",
                            min_value=0.0,
                            max_value=0.95,
                            value=float(context.get("max_missing_fraction", 0.5)),
                            step=0.05,
                            key="netanal_ml_single_feature_max_missing",
                        )
                    with sf_c3:
                        single_feature_max_tested = st.number_input(
                            "Max features tested per module",
                            min_value=0,
                            max_value=100000,
                            value=0,
                            step=1,
                            key="netanal_ml_single_feature_max_tested",
                            help="0 means test all usable features in each selected module.",
                        )
                        single_feature_plot_top_n = st.number_input(
                            "Plot top features per module",
                            min_value=1,
                            max_value=200,
                            value=25,
                            step=1,
                            key="netanal_ml_single_feature_plot_top_n",
                        )
                    with sf_c4:
                        st.caption("Feishu interpretation")
                        st.write(
                            "The red dashed line is the whole-module score; bars are "
                            "single-feature scores."
                        )

                    run_single_feature = st.form_submit_button(
                        "Run Single-feature AUC Validation",
                        type="primary",
                    )

                if run_single_feature:
                    if not single_feature_modules:
                        st.warning("Please select at least one module.")
                    else:
                        with st.spinner(
                            "Running Feishu-style single-feature validation ..."
                        ):
                            try:
                                single_feature_result = (
                                    run_module_single_feature_validation(
                                        condition_tables,
                                        funclu_export["labels"],
                                        modules=[str(m) for m in single_feature_modules],
                                        first_column_as_sample_id=bool(
                                            st.session_state.get(
                                                "netanal_ml_condition_first_col_sample_id",
                                                True,
                                            )
                                        ),
                                        max_missing_fraction=float(
                                            single_feature_max_missing
                                        ),
                                        task=str(context.get("task", "one_vs_rest")),
                                        positive_label=(
                                            str(context.get("positive_label", ""))
                                            if context.get("task") == "one_vs_rest"
                                            else None
                                        ),
                                        classifier=str(single_feature_classifier),
                                        cv_folds=int(single_feature_cv_folds),
                                        random_state=123,
                                        max_features_per_module=(
                                            int(single_feature_max_tested)
                                            if int(single_feature_max_tested) > 0
                                            else None
                                        ),
                                    )
                                )
                                single_feature_result["context"][
                                    "plot_top_features_per_module"
                                ] = int(single_feature_plot_top_n)
                            except Exception as e:
                                st.error(f"Single-feature validation failed: {e}")
                            else:
                                st.session_state[
                                    "netanal_ml_single_feature_result"
                                ] = single_feature_result

                single_feature_result = st.session_state.get(
                    "netanal_ml_single_feature_result"
                )
                if single_feature_result is not None:
                    sf_context = single_feature_result["context"]
                    sf_feature_scores = single_feature_result["feature_scores"].copy()
                    sf_module_scores = single_feature_result["module_scores"].copy()
                    valid_sf = sf_feature_scores[
                        pd.notna(
                            pd.to_numeric(
                                sf_feature_scores["primary_score_mean"],
                                errors="coerce",
                            )
                        )
                    ].copy()
                    best_sf = (
                        valid_sf.sort_values(
                            "primary_score_mean",
                            ascending=False,
                        ).iloc[0]
                        if not valid_sf.empty
                        else None
                    )
                    sf_m1, sf_m2, sf_m3, sf_m4 = st.columns(4)
                    sf_m1.metric(
                        "Modules plotted",
                        f"{len(sf_context.get('modules', [])):,}",
                    )
                    sf_m2.metric(
                        "Single-feature tests",
                        f"{len(sf_feature_scores):,}",
                    )
                    sf_m3.metric(
                        "Best feature",
                        (
                            str(best_sf["feature"])
                            if best_sf is not None
                            else "N/A"
                        ),
                    )
                    sf_m4.metric(
                        "Best feature score",
                        (
                            f"{float(best_sf['primary_score_mean']):.3f}"
                            if best_sf is not None
                            and pd.notna(best_sf["primary_score_mean"])
                            else "N/A"
                        ),
                    )

                    summary_rows = []
                    for module in sf_context.get("modules", []):
                        module = str(module)
                        module_row = sf_module_scores[
                            sf_module_scores["module"].astype(str) == module
                        ]
                        module_score = (
                            float(module_row.iloc[0]["primary_score_mean"])
                            if not module_row.empty
                            and pd.notna(module_row.iloc[0]["primary_score_mean"])
                            else np.nan
                        )
                        module_features = valid_sf[
                            valid_sf["module"].astype(str) == module
                        ].copy()
                        if module_features.empty:
                            continue
                        best_feature_row = module_features.sort_values(
                            "primary_score_mean",
                            ascending=False,
                        ).iloc[0]
                        best_feature_score = float(
                            best_feature_row["primary_score_mean"]
                        )
                        summary_rows.append(
                            {
                                "module": module,
                                "module_score": module_score,
                                "best_single_feature": str(best_feature_row["feature"]),
                                "best_single_feature_score": best_feature_score,
                                "module_minus_best_feature": (
                                    module_score - best_feature_score
                                    if pd.notna(module_score)
                                    else np.nan
                                ),
                            }
                        )
                    if summary_rows:
                        sf_summary = pd.DataFrame(summary_rows)
                        st.markdown("##### Module vs best single feature")
                        st.dataframe(sf_summary, width="stretch", height=180)

                    sf_fig = _plot_single_feature_module_scores(
                        sf_feature_scores,
                        sf_module_scores,
                        modules=[str(m) for m in sf_context.get("modules", [])],
                        task_label=str(sf_context.get("task_label", task_title)),
                        max_features_per_module=int(
                            sf_context.get("plot_top_features_per_module", 25)
                        ),
                    )
                    if sf_fig is None:
                        st.info("No valid single-feature scores are available to plot.")
                    else:
                        st.pyplot(sf_fig, width="stretch")
                        st.caption(
                            "Feishu-style validation: each bar is the cross-validated "
                            "score using one feature alone; the red dashed line is the "
                            "score using all selected-module features together."
                        )
                        sf_png = io.BytesIO()
                        sf_fig.savefig(
                            sf_png,
                            format="png",
                            dpi=220,
                            bbox_inches="tight",
                        )
                        sf_pdf = io.BytesIO()
                        sf_fig.savefig(sf_pdf, format="pdf", bbox_inches="tight")
                        plt.close(sf_fig)
                        sf_dl1, sf_dl2 = st.columns(2)
                        sf_dl1.download_button(
                            label="Download single-feature AUC PNG",
                            data=sf_png.getvalue(),
                            file_name="single_feature_module_auc_visualization.png",
                            mime="image/png",
                            key="netanal_ml_single_feature_png_download",
                        )
                        sf_dl2.download_button(
                            label="Download single-feature AUC PDF",
                            data=sf_pdf.getvalue(),
                            file_name="single_feature_module_auc_visualization.pdf",
                            mime="application/pdf",
                            key="netanal_ml_single_feature_pdf_download",
                        )

                    st.markdown("##### Single-feature score table")
                    sf_cols = [
                        "feature_rank_within_module",
                        "module",
                        "feature",
                        "primary_metric",
                        "primary_score_mean",
                        "primary_score_std",
                        "module_primary_score_mean",
                        "roc_auc_mean",
                        "roc_auc_std",
                        "balanced_accuracy_mean",
                        "accuracy_mean",
                        "f1_macro_mean",
                        "cv_folds_used",
                        "status",
                    ]
                    sf_cols = [
                        col for col in sf_cols if col in sf_feature_scores.columns
                    ]
                    st.dataframe(
                        sf_feature_scores[sf_cols],
                        width="stretch",
                        height=320,
                    )
                    sf_t1, sf_t2 = st.columns(2)
                    sf_t1.download_button(
                        label="Download single-feature scores CSV",
                        data=sf_feature_scores.to_csv(index=False).encode("utf-8"),
                        file_name="single_feature_module_scores.csv",
                        mime="text/csv",
                        key="netanal_ml_single_feature_scores_download",
                    )
                    sf_t2.download_button(
                        label="Download selected module scores CSV",
                        data=sf_module_scores.to_csv(index=False).encode("utf-8"),
                        file_name="single_feature_selected_module_scores.csv",
                        mime="text/csv",
                        key="netanal_ml_single_feature_module_scores_download",
                    )

            st.markdown("#### Module stability / robustness")
            if context.get("task") not in {"one_vs_rest", "multiclass"}:
                st.info("Run module classification first to enable stability validation.")
            else:
                with st.form(key="netanal_ml_stability_form"):
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        stability_repeats = st.number_input(
                            "Repeated CV runs",
                            min_value=1,
                            max_value=200,
                            value=20,
                            step=1,
                            key="netanal_ml_stability_repeats",
                        )
                        stability_include_all = st.checkbox(
                            "Include All baseline",
                            value=False,
                            key="netanal_ml_stability_include_all",
                        )
                    with col_s2:
                        stability_cv_folds = st.number_input(
                            "Stability CV folds",
                            min_value=2,
                            max_value=10,
                            value=5,
                            step=1,
                            key="netanal_ml_stability_cv_folds",
                        )
                        stability_classifier = st.selectbox(
                            "Stability model",
                            options=["logistic_regression", "random_forest"],
                            index=(
                                ["logistic_regression", "random_forest"].index(
                                    str(context.get("classifier", "logistic_regression"))
                                )
                                if str(context.get("classifier", "logistic_regression"))
                                in {"logistic_regression", "random_forest"}
                                else 0
                            ),
                            format_func=lambda x: {
                                "logistic_regression": "Logistic Regression (L2)",
                                "random_forest": "Random Forest",
                            }[x],
                            key="netanal_ml_stability_classifier",
                        )
                    with col_s3:
                        stability_max_missing = st.slider(
                            "Stability max missing",
                            min_value=0.0,
                            max_value=0.95,
                            value=0.50,
                            step=0.05,
                            key="netanal_ml_stability_max_missing",
                        )
                    with col_s4:
                        st.caption("Interpretation")
                        st.write("Rank-1 frequency is the share of repeated CV splits where a module ranks first.")

                    run_stability = st.form_submit_button(
                        "Run Module Stability Validation",
                        type="primary",
                    )

                if run_stability:
                    with st.spinner("Running repeated module stability validation ..."):
                        try:
                            stability_result = run_module_stability_validation(
                                condition_tables,
                                funclu_export["labels"],
                                first_column_as_sample_id=bool(
                                    st.session_state.get(
                                        "netanal_ml_condition_first_col_sample_id",
                                        True,
                                    )
                                ),
                                max_missing_fraction=float(stability_max_missing),
                                task=str(context.get("task", "one_vs_rest")),
                                positive_label=(
                                    str(context.get("positive_label", ""))
                                    if context.get("task") == "one_vs_rest"
                                    else None
                                ),
                                classifier=str(stability_classifier),
                                cv_folds=int(stability_cv_folds),
                                n_repeats=int(stability_repeats),
                                include_all=bool(stability_include_all),
                                random_state=123,
                            )
                        except Exception as e:
                            st.error(f"Module stability validation failed: {e}")
                        else:
                            st.session_state[
                                "netanal_ml_stability_result"
                            ] = stability_result

            stability_result = st.session_state.get("netanal_ml_stability_result")
            if stability_result is not None:
                stability_summary = stability_result["summary"].copy()
                stability_summary["is_topology_hub"] = (
                    stability_summary["module"].astype(str).isin(expected_hubs)
                    if expected_hubs
                    else False
                )
                best_stable = (
                    stability_summary.iloc[0]
                    if not stability_summary.empty
                    else None
                )
                topo_stable = (
                    stability_summary[
                        stability_summary["module"].astype(str).isin(expected_hubs)
                    ].iloc[0]
                    if expected_hubs
                    and stability_summary["module"].astype(str).isin(expected_hubs).any()
                    else None
                )
                s_m1, s_m2, s_m3, s_m4 = st.columns(4)
                s_m1.metric(
                    "Most stable module",
                    str(best_stable["module"]) if best_stable is not None else "N/A",
                )
                s_m2.metric(
                    "Rank-1 frequency",
                    (
                        f"{float(best_stable['rank_1_frequency']):.1%}"
                        if best_stable is not None
                        and pd.notna(best_stable["rank_1_frequency"])
                        else "N/A"
                    ),
                )
                s_m3.metric(
                    "Topology Hub rank-1",
                    (
                        f"{float(topo_stable['rank_1_frequency']):.1%}"
                        if topo_stable is not None
                        and pd.notna(topo_stable["rank_1_frequency"])
                        else "N/A"
                    ),
                )
                s_m4.metric(
                    "Repeated splits",
                    f"{int(stability_result['cv_folds_used']) * int(stability_result['n_repeats']):,}",
                )

                if expected_hubs and topo_stable is not None:
                    if float(topo_stable["rank_1_frequency"]) >= 0.5:
                        st.success(
                            f"Topology Hub `{expected_hub}` is stable: it ranks first in "
                            f"{float(topo_stable['rank_1_frequency']):.1%} of repeated CV splits."
                        )
                    else:
                        st.warning(
                            f"Topology Hub `{expected_hub}` is not consistently rank-1 under "
                            "the current repeated CV settings."
                        )

                st.markdown("##### Stability summary")
                stability_cols = [
                    "stability_rank",
                    "module",
                    "is_topology_hub",
                    "primary_metric",
                    "primary_score_mean",
                    "primary_score_std",
                    "primary_score_p025",
                    "primary_score_p975",
                    "mean_rank",
                    "rank_1_frequency",
                    "top_2_frequency",
                    "top_3_frequency",
                    "successful_splits",
                    "total_splits",
                    "n_features",
                    "status",
                ]
                stability_cols = [
                    col for col in stability_cols if col in stability_summary.columns
                ]
                st.dataframe(
                    stability_summary[stability_cols],
                    width="stretch",
                    height=320,
                )
                dl_s1, dl_s2 = st.columns(2)
                dl_s1.download_button(
                    label="Download stability summary CSV",
                    data=stability_summary.to_csv(index=False).encode("utf-8"),
                    file_name="module_stability_summary.csv",
                    mime="text/csv",
                    key="netanal_ml_stability_summary_download",
                )
                dl_s2.download_button(
                    label="Download stability split scores CSV",
                    data=stability_result["split_scores"].to_csv(index=False).encode("utf-8"),
                    file_name="module_stability_split_scores.csv",
                    mime="text/csv",
                    key="netanal_ml_stability_splits_download",
                )

            st.markdown("#### Intra-cluster feature validation")
            if context.get("task") != "one_vs_rest" or not context.get("positive_label"):
                st.info(
                    "Intra-cluster validation is available for **One condition vs others** "
                    "tasks, because feature direction needs one positive condition."
                )
            else:
                feature_summary = dataset_info.get("feature_summary", pd.DataFrame())
                if feature_summary.empty or "module" not in feature_summary.columns:
                    module_options = [
                        str(module)
                        for module in scores["module"].tolist()
                        if str(module) != "All"
                    ]
                else:
                    usable_col = (
                        "usable_features"
                        if "usable_features" in feature_summary.columns
                        else "funclu_features"
                    )
                    module_options = (
                        feature_summary[
                            pd.to_numeric(
                                feature_summary[usable_col],
                                errors="coerce",
                            ).fillna(0)
                            > 0
                        ]["module"]
                        .astype(str)
                        .tolist()
                    )

                if not module_options:
                    st.warning("No usable FunClu modules are available for feature validation.")
                else:
                    best_single_module = ""
                    single_scores = scores[scores["module"].astype(str) != "All"]
                    if not single_scores.empty:
                        best_single_module = str(single_scores.iloc[0]["module"])
                    primary_expected_hub = next(
                        (hub for hub in expected_hubs if hub in module_options),
                        "",
                    )
                    default_module = (
                        primary_expected_hub
                        if primary_expected_hub
                        else best_single_module
                        if best_single_module in module_options
                        else module_options[0]
                    )

                    with st.form(key="netanal_ml_intra_feature_form"):
                        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                        with col_i1:
                            intra_module = st.selectbox(
                                "Module to inspect",
                                options=module_options,
                                index=module_options.index(default_module),
                                key="netanal_ml_intra_module",
                            )
                            intra_repeats = st.number_input(
                                "Permutation repeats",
                                min_value=1,
                                max_value=100,
                                value=20,
                                step=1,
                                key="netanal_ml_intra_repeats",
                            )
                        with col_i2:
                            context_classifier = str(
                                context.get("classifier", "logistic_regression")
                            )
                            intra_classifier = st.selectbox(
                                "Feature model",
                                options=["logistic_regression", "random_forest"],
                                index=(
                                    ["logistic_regression", "random_forest"].index(
                                        context_classifier
                                    )
                                    if context_classifier
                                    in {"logistic_regression", "random_forest"}
                                    else 0
                                ),
                                format_func=lambda x: {
                                    "logistic_regression": "Logistic Regression (L2)",
                                    "random_forest": "Random Forest",
                                }[x],
                                key="netanal_ml_intra_classifier",
                            )
                            intra_cv_folds = st.number_input(
                                "Feature CV folds",
                                min_value=2,
                                max_value=10,
                                value=5,
                                step=1,
                                key="netanal_ml_intra_cv_folds",
                            )
                        with col_i3:
                            intra_max_missing = st.slider(
                                "Max missing fraction",
                                min_value=0.0,
                                max_value=0.95,
                                value=0.50,
                                step=0.05,
                                key="netanal_ml_intra_max_missing",
                            )
                            intra_ml_top_n = st.number_input(
                                "ML top N for overlap",
                                min_value=1,
                                max_value=1000,
                                value=10,
                                step=1,
                                key="netanal_ml_intra_ml_top_n",
                            )
                        with col_i4:
                            intra_topology_metric = st.selectbox(
                                "Intra topology metric",
                                options=[
                                    "out_degree",
                                    "out_strength",
                                    "total_degree",
                                    "total_strength",
                                ],
                                index=0,
                                key="netanal_ml_intra_topology_metric",
                            )
                            intra_topology_threshold = st.number_input(
                                "Min |intra topology edge|",
                                min_value=0.0,
                                max_value=1000000.0,
                                value=0.0,
                                step=0.01,
                                format="%.4f",
                                key="netanal_ml_intra_topology_threshold",
                            )
                            intra_topology_top_n = st.number_input(
                                "Topology top N",
                                min_value=1,
                                max_value=1000,
                                value=3,
                                step=1,
                                key="netanal_ml_intra_topology_top_n",
                            )

                        run_intra = st.form_submit_button(
                            "Run Intra-cluster Feature Validation",
                            type="primary",
                        )

                    if run_intra:
                        positive_label_for_intra = str(context.get("positive_label", ""))
                        with st.spinner("Running intra-cluster feature validation ..."):
                            try:
                                intra_result = run_intra_module_feature_importance(
                                    condition_tables,
                                    funclu_export["labels"],
                                    module=str(intra_module),
                                    first_column_as_sample_id=bool(
                                        st.session_state.get(
                                            "netanal_ml_condition_first_col_sample_id",
                                            True,
                                        )
                                    ),
                                    max_missing_fraction=float(intra_max_missing),
                                    positive_label=positive_label_for_intra,
                                    classifier=str(intra_classifier),
                                    cv_folds=int(intra_cv_folds),
                                    random_state=123,
                                    n_repeats=int(intra_repeats),
                                )

                                topology_key = ""
                                topology_hub_table = pd.DataFrame()
                                if topology_adjs:
                                    exact_candidates: list[str] = []
                                    loose_candidates: list[str] = []
                                    for key in topology_adjs:
                                        parts = str(key).split("/")
                                        if (
                                            len(parts) == 3
                                            and parts[0] == "intra_cluster"
                                            and parts[2] == str(intra_module)
                                        ):
                                            loose_candidates.append(str(key))
                                            if (
                                                _clean_condition_label(parts[1])
                                                == _clean_condition_label(
                                                    positive_label_for_intra
                                                )
                                            ):
                                                exact_candidates.append(str(key))
                                    if exact_candidates:
                                        topology_key = exact_candidates[0]
                                    elif len(loose_candidates) == 1:
                                        topology_key = loose_candidates[0]

                                if topology_key:
                                    topology_hub_table = hub_table_from_adjacency(
                                        topology_adjs[topology_key],
                                        edge_threshold=float(intra_topology_threshold),
                                        rank_metric=str(intra_topology_metric),
                                    )
                                    topology_feature_table = topology_hub_table.rename(
                                        columns={
                                            "variable": "feature",
                                            "rank": "topology_rank",
                                            "out_degree": "topology_out_degree",
                                            "in_degree": "topology_in_degree",
                                            "out_strength": "topology_out_strength",
                                            "in_strength": "topology_in_strength",
                                            "total_degree": "topology_total_degree",
                                            "total_strength": "topology_total_strength",
                                        }
                                    )
                                    merge_cols = [
                                        col
                                        for col in [
                                            "feature",
                                            "topology_rank",
                                            "topology_out_degree",
                                            "topology_in_degree",
                                            "topology_out_strength",
                                            "topology_in_strength",
                                            "topology_total_degree",
                                            "topology_total_strength",
                                        ]
                                        if col in topology_feature_table.columns
                                    ]
                                    intra_result["importance"] = intra_result[
                                        "importance"
                                    ].merge(
                                        topology_feature_table[merge_cols],
                                        on="feature",
                                        how="left",
                                    )
                                    ml_top_n_value = min(
                                        int(intra_ml_top_n),
                                        int(len(intra_result["importance"])),
                                    )
                                    topology_top_n_value = min(
                                        int(intra_topology_top_n),
                                        int(len(topology_hub_table)),
                                    )
                                    intra_result["importance"][
                                        "is_topology_intra_hub"
                                    ] = intra_result["importance"][
                                        "topology_rank"
                                    ].eq(1)
                                    intra_result["importance"]["in_ml_top_n"] = (
                                        intra_result["importance"]["feature_rank"]
                                        <= ml_top_n_value
                                    )
                                    intra_result["importance"]["is_topology_top_n"] = (
                                        pd.to_numeric(
                                            intra_result["importance"]["topology_rank"],
                                            errors="coerce",
                                        )
                                        <= topology_top_n_value
                                    )
                                    overlap_mask = (
                                        intra_result["importance"]["in_ml_top_n"]
                                        & intra_result["importance"]["is_topology_top_n"]
                                    )
                                    overlap_features = (
                                        intra_result["importance"]
                                        .loc[overlap_mask, "feature"]
                                        .astype(str)
                                        .tolist()
                                    )
                                    common_rank = intra_result["importance"].dropna(
                                        subset=["topology_rank"]
                                    )
                                    if (
                                        len(common_rank) >= 2
                                        and common_rank["feature_rank"].nunique() > 1
                                        and common_rank["topology_rank"].nunique() > 1
                                    ):
                                        rank_corr = common_rank["feature_rank"].corr(
                                            common_rank["topology_rank"],
                                            method="spearman",
                                        )
                                    else:
                                        rank_corr = np.nan
                                    topology_top_feature = str(
                                        topology_hub_table.iloc[0]["variable"]
                                    )
                                    top_feature_match = intra_result[
                                        "importance"
                                    ][
                                        intra_result["importance"]["feature"].astype(str)
                                        == topology_top_feature
                                    ]
                                    topology_top_ml_rank = (
                                        int(top_feature_match.iloc[0]["feature_rank"])
                                        if not top_feature_match.empty
                                        else None
                                    )
                                    intra_result["topology_comparison"] = {
                                        "ml_top_n": int(ml_top_n_value),
                                        "topology_top_n": int(topology_top_n_value),
                                        "overlap_count": int(len(overlap_features)),
                                        "overlap_features": ", ".join(overlap_features),
                                        "topology_top_feature": topology_top_feature,
                                        "topology_top_ml_rank": topology_top_ml_rank,
                                        "spearman_rank_correlation": (
                                            float(rank_corr)
                                            if pd.notna(rank_corr)
                                            else np.nan
                                        ),
                                    }
                                else:
                                    intra_result["importance"][
                                        "is_topology_intra_hub"
                                    ] = False
                                    intra_result["importance"]["in_ml_top_n"] = (
                                        intra_result["importance"]["feature_rank"]
                                        <= int(intra_ml_top_n)
                                    )
                                    intra_result["importance"]["is_topology_top_n"] = False
                                    intra_result["topology_comparison"] = {}

                                intra_result["topology_key"] = topology_key
                                intra_result["topology_hubs"] = topology_hub_table
                                intra_result["context"] = {
                                    "module": str(intra_module),
                                    "positive_label": positive_label_for_intra,
                                    "classifier": str(intra_classifier),
                                    "topology_metric": str(intra_topology_metric),
                                    "ml_top_n": int(intra_ml_top_n),
                                    "topology_top_n": int(intra_topology_top_n),
                                }
                            except Exception as e:
                                st.error(f"Intra-cluster feature validation failed: {e}")
                            else:
                                st.session_state["netanal_ml_intra_result"] = intra_result

            intra_result = st.session_state.get("netanal_ml_intra_result")
            if intra_result is not None:
                importance = intra_result["importance"].copy()
                module_score = intra_result["module_score"]
                intra_context = intra_result["context"]
                topology_key = intra_result.get("topology_key", "")
                topology_hub_table = intra_result.get("topology_hubs", pd.DataFrame())
                topology_comparison = intra_result.get("topology_comparison", {})

                top_feature = importance.iloc[0] if not importance.empty else None
                i_m1, i_m2, i_m3, i_m4 = st.columns(4)
                i_m1.metric("Feature top hit", str(top_feature["feature"]) if top_feature is not None else "N/A")
                i_m2.metric(
                    "Module AUC",
                    (
                        f"{float(module_score.get('roc_auc_mean', np.nan)):.3f}"
                        if pd.notna(module_score.get("roc_auc_mean", np.nan))
                        else "N/A"
                    ),
                )
                i_m3.metric(
                    "Features tested",
                    f"{int(module_score.get('n_features', 0)):,}",
                )
                i_m4.metric(
                    "Topology intra Hub",
                    (
                        str(topology_hub_table.iloc[0]["variable"])
                        if topology_hub_table is not None
                        and not topology_hub_table.empty
                        else "N/A"
                    ),
                )

                st.caption(
                    "Selection: "
                    f"module={intra_context.get('module')} | "
                    f"positive={intra_context.get('positive_label')} | "
                    f"classifier={intra_context.get('classifier')} | "
                    f"topology={topology_key if topology_key else 'not matched'}"
                )

                if topology_comparison:
                    c_i1, c_i2, c_i3 = st.columns(3)
                    c_i1.metric(
                        "Topology Hub ML rank",
                        (
                            str(topology_comparison.get("topology_top_ml_rank"))
                            if topology_comparison.get("topology_top_ml_rank")
                            is not None
                            else "N/A"
                        ),
                    )
                    c_i2.metric(
                        "Top-N overlap",
                        (
                            f"{topology_comparison.get('overlap_count', 0)}/"
                            f"{topology_comparison.get('topology_top_n', 0)}"
                        ),
                        help=(
                            f"ML top {topology_comparison.get('ml_top_n')} vs "
                            f"topology top {topology_comparison.get('topology_top_n')}"
                        ),
                    )
                    c_i3.metric(
                        "Rank Spearman",
                        (
                            f"{float(topology_comparison['spearman_rank_correlation']):.3f}"
                            if pd.notna(
                                topology_comparison.get(
                                    "spearman_rank_correlation",
                                    np.nan,
                                )
                            )
                            else "N/A"
                        ),
                    )
                    if topology_comparison.get("overlap_features"):
                        st.success(
                            "Topology top features found in ML top list: "
                            f"{topology_comparison['overlap_features']}"
                        )

                if (
                    topology_hub_table is not None
                    and not topology_hub_table.empty
                    and top_feature is not None
                ):
                    topology_top_feature = str(topology_hub_table.iloc[0]["variable"])
                    if str(top_feature["feature"]) == topology_top_feature:
                        st.success(
                            f"ML feature ranking supports the intra-cluster topology Hub "
                            f"`{topology_top_feature}`."
                        )
                    else:
                        st.warning(
                            "The top ML feature does not match the top intra-cluster "
                            f"topology Hub `{topology_top_feature}` under the current settings."
                        )

                plot_count = min(25, len(importance))
                if plot_count > 0:
                    plot_df = importance.head(plot_count).copy()
                    plot_df["plot_importance"] = pd.to_numeric(
                        plot_df["permutation_importance_mean"],
                        errors="coerce",
                    ).fillna(0.0)
                    plot_df = plot_df.sort_values("plot_importance", ascending=True)
                    fig_i, ax_i = plt.subplots(figsize=(8, max(3, 0.32 * plot_count)))
                    ax_i.barh(plot_df["feature"].astype(str), plot_df["plot_importance"])
                    ax_i.set_xlabel("Permutation importance (AUC drop)")
                    ax_i.set_ylabel("Feature")
                    ax_i.set_title("Top intra-cluster ML features")
                    fig_i.tight_layout()
                    st.pyplot(fig_i, width="stretch")
                    plt.close(fig_i)

                st.markdown("##### Intra-cluster feature ranking")
                intra_cols = [
                    "feature_rank",
                    "feature",
                    "in_ml_top_n",
                    "is_topology_intra_hub",
                    "is_topology_top_n",
                    "topology_rank",
                    "permutation_importance_mean",
                    "permutation_importance_std",
                    "coefficient",
                    "abs_coefficient",
                    "embedded_importance",
                    "mean_positive",
                    "mean_other",
                    "mean_difference",
                    "direction",
                    "missing_fraction",
                ]
                intra_cols = [col for col in intra_cols if col in importance.columns]
                st.dataframe(importance[intra_cols], width="stretch", height=360)

                if topology_hub_table is not None and not topology_hub_table.empty:
                    st.markdown("##### Topology intra-cluster Hub ranking")
                    st.dataframe(topology_hub_table.head(100), width="stretch", height=260)

                dli1, dli2 = st.columns(2)
                dli1.download_button(
                    label="Download intra feature ranking CSV",
                    data=importance.to_csv(index=False).encode("utf-8"),
                    file_name="intra_cluster_feature_importance.csv",
                    mime="text/csv",
                    key="netanal_ml_intra_importance_download",
                )
                if topology_hub_table is not None and not topology_hub_table.empty:
                    dli2.download_button(
                        label="Download intra topology Hub CSV",
                        data=topology_hub_table.to_csv(index=False).encode("utf-8"),
                        file_name="intra_cluster_topology_hubs.csv",
                        mime="text/csv",
                        key="netanal_ml_intra_topology_download",
                    )

            st.markdown("#### Dataset summary")
            c_sum1, c_sum2 = st.columns(2)
            with c_sum1:
                st.dataframe(
                    dataset_info["sample_summary"],
                    width="stretch",
                    height=220,
                )
            with c_sum2:
                st.dataframe(
                    dataset_info["feature_summary"],
                    width="stretch",
                    height=220,
                )
            st.json(dataset_info["diagnostics"])

            if not topology_hubs.empty:
                st.markdown("#### Topology inter-cluster Hub by condition")
                st.dataframe(topology_hubs, width="stretch", height=240)

            dl_c1, dl_c2 = st.columns(2)
            dl_c1.download_button(
                label="Download classification scores CSV",
                data=scores.to_csv(index=False).encode("utf-8"),
                file_name="module_classification_scores.csv",
                mime="text/csv",
                key="netanal_ml_classification_scores_download",
            )
            if not topology_hubs.empty:
                dl_c2.download_button(
                    label="Download topology hubs CSV",
                    data=topology_hubs.to_csv(index=False).encode("utf-8"),
                    file_name="classification_topology_hubs.csv",
                    mime="text/csv",
                    key="netanal_ml_classification_topology_download",
                )

    # ========== Tab 2_4 Sample Prediction ==========
    with tab2_4:
        funclu_export = st.session_state.get("netanal_ml_funclu_export")
        condition_tables = st.session_state.get("netanal_ml_condition_tables")
        classification_result = st.session_state.get("netanal_ml_classification_result")

        if funclu_export is None:
            st.info("Please upload a FunClu-K ZIP in **Aligned Inputs** first.")
        elif not condition_tables:
            st.info(
                "Please upload two or more condition CSV files in **Aligned Inputs**. "
                "Those condition CSVs are used as the labelled training data."
            )
        else:
            condition_options = list(condition_tables.keys())
            if len(condition_options) < 2:
                st.warning("Sample prediction needs at least two condition CSVs.")
            else:
                try:
                    cluster_map = module_feature_map_from_labels(funclu_export["labels"])
                except Exception as e:
                    cluster_map = {}
                    st.warning(
                        "FunClu module labels could not be read, so prediction is limited "
                        f"to all shared features. Details: {e}"
                    )

                prediction_module_options = ["All"] + list(cluster_map.keys())
                prediction_default_module = "All"
                prediction_default_classifier = "logistic_regression"
                prediction_default_positive = condition_options[0]

                if classification_result is not None:
                    classification_context = classification_result.get("context", {})
                    context_classifier = str(
                        classification_context.get(
                            "classifier",
                            prediction_default_classifier,
                        )
                    )
                    if context_classifier in {"logistic_regression", "random_forest"}:
                        prediction_default_classifier = context_classifier

                    context_positive = str(
                        classification_context.get(
                            "positive_label",
                            prediction_default_positive,
                        )
                    )
                    if context_positive in condition_options:
                        prediction_default_positive = context_positive

                    classification_scores = classification_result.get(
                        "scores",
                        pd.DataFrame(),
                    )
                    if not classification_scores.empty:
                        single_scores = classification_scores[
                            classification_scores["module"].astype(str) != "All"
                        ]
                        if not single_scores.empty:
                            best_single_module = str(single_scores.iloc[0]["module"])
                            if best_single_module in prediction_module_options:
                                prediction_default_module = best_single_module

                train_sample_count = sum(
                    int(table.shape[0]) for table in condition_tables.values()
                )
                metric_p1, metric_p2, metric_p3 = st.columns(3)
                metric_p1.metric("Training conditions", f"{len(condition_options):,}")
                metric_p2.metric("Training samples", f"{train_sample_count:,}")
                metric_p3.metric(
                    "Available feature sets",
                    f"{len(prediction_module_options):,}",
                )
                st.caption(
                    "This tab trains a classifier from the uploaded condition CSVs and "
                    "applies it to a new unknown-sample CSV. It does not require running "
                    "Classification Validation first."
                )

                unknown_csv = st.file_uploader(
                    "Upload unknown sample CSV",
                    type=["csv"],
                    key="netanal_ml_unknown_prediction_csv",
                    help=(
                        "Rows are unknown samples; columns should match the training "
                        "condition CSV feature names."
                    ),
                )
                if unknown_csv is None:
                    st.session_state.pop("netanal_ml_unknown_prediction_signature", None)
                else:
                    unknown_signature = (unknown_csv.name, unknown_csv.size)
                    if (
                        st.session_state.get(
                            "netanal_ml_unknown_prediction_signature"
                        )
                        != unknown_signature
                    ):
                        st.session_state.pop("netanal_ml_prediction_result", None)
                        st.session_state[
                            "netanal_ml_unknown_prediction_signature"
                        ] = unknown_signature

                with st.form(key="netanal_ml_prediction_form"):
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    with col_p1:
                        prediction_task = st.selectbox(
                            "Prediction task",
                            options=["multiclass", "one_vs_rest"],
                            index=0,
                            format_func=lambda x: {
                                "multiclass": "Predict condition label",
                                "one_vs_rest": "Positive condition probability",
                            }[x],
                            key="netanal_ml_prediction_task",
                        )
                        prediction_positive = st.selectbox(
                            "Positive condition",
                            options=condition_options,
                            index=condition_options.index(prediction_default_positive),
                            disabled=prediction_task != "one_vs_rest",
                            key="netanal_ml_prediction_positive",
                        )
                    with col_p2:
                        prediction_module = st.selectbox(
                            "Prediction feature set",
                            options=prediction_module_options,
                            index=prediction_module_options.index(
                                prediction_default_module
                            ),
                            key="netanal_ml_prediction_module",
                        )
                        prediction_classifier = st.selectbox(
                            "Prediction model",
                            options=["logistic_regression", "random_forest"],
                            index=["logistic_regression", "random_forest"].index(
                                prediction_default_classifier
                            ),
                            format_func=lambda x: {
                                "logistic_regression": "Logistic Regression (L2)",
                                "random_forest": "Random Forest",
                            }[x],
                            key="netanal_ml_prediction_classifier",
                        )
                    with col_p3:
                        prediction_max_missing = st.slider(
                            "Prediction max missing",
                            min_value=0.0,
                            max_value=0.95,
                            value=0.50,
                            step=0.05,
                            key="netanal_ml_prediction_max_missing",
                        )
                    with col_p4:
                        prediction_first_col = st.checkbox(
                            "Unknown first column is sample ID",
                            value=bool(
                                st.session_state.get(
                                    "netanal_ml_condition_first_col_sample_id",
                                    True,
                                )
                            ),
                            key="netanal_ml_prediction_first_col_sample_id",
                        )

                    run_prediction = st.form_submit_button(
                        "Run Unknown Sample Prediction",
                        type="primary",
                    )

                if run_prediction:
                    if unknown_csv is None:
                        st.warning("Please upload an unknown sample CSV first.")
                    else:
                        with st.spinner(
                            "Training classifier and predicting unknown samples ..."
                        ):
                            try:
                                unknown_table = pd.read_csv(unknown_csv)
                                prediction_result = predict_unknown_condition_samples(
                                    condition_tables,
                                    funclu_export["labels"],
                                    unknown_table,
                                    module=str(prediction_module),
                                    first_column_as_sample_id=bool(
                                        st.session_state.get(
                                            "netanal_ml_condition_first_col_sample_id",
                                            True,
                                        )
                                    ),
                                    unknown_first_column_as_sample_id=bool(
                                        prediction_first_col
                                    ),
                                    max_missing_fraction=float(prediction_max_missing),
                                    task=str(prediction_task),
                                    positive_label=(
                                        str(prediction_positive)
                                        if prediction_task == "one_vs_rest"
                                        else None
                                    ),
                                    classifier=str(prediction_classifier),
                                    random_state=123,
                                )
                            except Exception as e:
                                st.error(f"Unknown sample prediction failed: {e}")
                            else:
                                st.session_state[
                                    "netanal_ml_prediction_result"
                                ] = prediction_result

                prediction_result = st.session_state.get("netanal_ml_prediction_result")
                if prediction_result is None:
                    st.info(
                        "Upload an unknown-sample CSV and click prediction to generate "
                        "condition labels or positive-condition probabilities."
                    )
                else:
                    prediction_context = prediction_result["context"]
                    predictions = prediction_result["predictions"].copy()
                    p_m1, p_m2, p_m3, p_m4 = st.columns(4)
                    p_m1.metric(
                        "Predicted samples",
                        f"{int(prediction_context['n_unknown_samples']):,}",
                    )
                    p_m2.metric(
                        "Feature set",
                        str(prediction_context["module"]),
                    )
                    p_m3.metric(
                        "Features used",
                        f"{int(prediction_context['n_features_used']):,}",
                    )
                    p_m4.metric(
                        "Classes",
                        f"{len(prediction_context.get('classes', [])):,}",
                    )
                    missing_unknown_features = prediction_context.get(
                        "missing_features_in_unknown",
                        [],
                    )
                    extra_unknown_features = prediction_context.get(
                        "extra_features_ignored",
                        [],
                    )
                    if missing_unknown_features:
                        st.warning(
                            f"{len(missing_unknown_features):,} training features are "
                            "missing from the unknown CSV. They were handled by the "
                            "model imputer."
                        )
                    if extra_unknown_features:
                        st.info(
                            f"{len(extra_unknown_features):,} unknown CSV features were "
                            "not part of the selected training feature set and were ignored."
                        )
                    st.caption(
                        "Prediction context: "
                        f"task={prediction_context.get('task')} | "
                        f"classifier={prediction_context.get('classifier')} | "
                        f"module={prediction_context.get('module')}"
                    )
                    st.dataframe(predictions, width="stretch", height=320)
                    st.download_button(
                        label="Download unknown sample predictions CSV",
                        data=predictions.to_csv(index=False).encode("utf-8"),
                        file_name="unknown_sample_predictions.csv",
                        mime="text/csv",
                        key="netanal_ml_prediction_download",
                    )

    # ========== Tab 2_5 Network View ==========
    with tab2_5:
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
