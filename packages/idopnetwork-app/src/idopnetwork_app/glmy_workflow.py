"""Streamlit controls for regular GLMY persistence, recovery and representatives."""
from __future__ import annotations

from hashlib import sha256
import io

import pandas as pd
import streamlit as st

from idopnetwork.analysis.regular_glmy.workflow import DEFAULT_MODE, MODES, run_regular_glmy


@st.cache_data(show_spinner=False, max_entries=4)
def _run(frame: pd.DataFrame, **options) -> dict:
    """Cache completed artifacts across harmless UI reruns."""
    return run_regular_glmy(frame, **options)


def render_glmy_analysis(from_to_df: pd.DataFrame | None, member: str | None) -> None:
    """Render the new GLMY analysis without affecting the other NetAnal tabs."""
    if from_to_df is None:
        st.info("请先在 Uploaded Data 中上传 NetRecon ZIP 并选择子网络。")
        return
    if from_to_df.empty:
        st.warning("当前网络没有可用边。")
        return
    signature = sha256((str(member) + from_to_df.to_csv(index=False)).encode()).hexdigest()
    st.caption("新版 GLMY：按 |weight| ≤ t 过滤，计算 F₂ 上的 regular path homology（β₀–β₂）。正负号用于边的着色或子图筛选，不作为同调系数。")
    with st.form("regular_glmy_form"):
        mode = st.selectbox("GLMY mode", MODES, index=0,
                            key="regular_glmy_mode",
                            help="all_edges_colored：全部边一起计算，正红负蓝；separate_signs：同时导出全边及分别重算的正、负子图。")
        left, right = st.columns(2)
        n_stages = left.number_input("Filtration stages", min_value=2, max_value=12, value=4, step=1)
        max_columns = right.number_input("Representative columns", min_value=1, max_value=8, value=5, step=1)
        threshold_text = st.text_input("Filtration values (optional)",
                                       placeholder="0, 0.3, 0.6, 1.0",
                                       help="留空按边出现时间自动选择；手动输入需非负、有限且严格递增。")
        export_vector = st.checkbox("Export SVG / PDF", value=False)
        submitted = st.form_submit_button("Run GLMY", type="primary")
    if submitted:
        st.session_state.pop("regular_glmy_result", None)
        try:
            thresholds = ([float(value.strip()) for value in threshold_text.split(',')]
                          if threshold_text.strip() else None)
            with st.spinner("计算 regular GLMY、过滤过程和代表元 …"):
                result = _run(from_to_df, mode=mode, n_stages=int(n_stages),
                              max_columns=int(max_columns), thresholds=thresholds,
                              export_vector=export_vector)
            st.session_state["regular_glmy_result"] = (signature, result)
        except Exception as exc:
            st.error(f"GLMY 运行失败：{exc}")
    saved = st.session_state.get("regular_glmy_result")
    if saved is None:
        st.info("点击 Run GLMY 生成条形码、网络过滤过程和 β₁/β₂ 代表元。")
        return
    if saved[0] != signature:
        st.info("输入网络已改变，请重新点击 Run GLMY。")
        return
    result = saved[1]
    st.success(f"GLMY 完成 · {result['mode']}")
    if result["removed_self_loops"]:
        st.caption(f"已排除 {result['removed_self_loops']} 条自环，保留其节点。")
    st.download_button("Download GLMY results ZIP", result["zip"],
                       file_name=f"glmy_{result['mode']}.zip", mime="application/zip")
    st.caption("条形码条数为持续区间数；各过滤阶段的实际 Betti 数见过程图和 snapshots.csv。β₂ 图仅显示路径链支撑边，完整链见 paths.csv。")
    artifacts = result["artifacts"]
    for folder, label in (("01_all_edges_colored", "all_edges_colored"),
                          ("02_separate_signs/positive", "positive"),
                          ("02_separate_signs/negative", "negative")):
        images = [name for name in artifacts if name.startswith(folder + '/') and name.endswith('.png')]
        if not images:
            continue
        st.markdown(f"#### {label}")
        views = st.tabs(["Filtration process", "Barcode", "β₁ representatives", "β₂ representatives", "Tables"])
        for view, filename in zip(views[:4], ("filtration_process.png", "barcode.png",
                                             "homology_representative_beta1.png", "homology_representative_beta2.png")):
            with view:
                name = f"{folder}/{filename}"
                if name in artifacts:
                    st.image(artifacts[name], width="stretch")
                    st.download_button(f"Download {filename}", artifacts[name], filename,
                                       mime="image/png", key=f"regular_download_{name}")
                else:
                    # The comparison implementation includes the barcode in
                    # the recovery plot rather than a standalone figure.
                    st.caption("Barcode 已包含在 Filtration process 图中。")
        with views[4]:
            tables = [name for name in artifacts if name.startswith(folder + '/') and name.endswith('.csv')]
            chosen = st.selectbox("Result table", tables, key=f"regular_table_{folder}")
            st.dataframe(pd.read_csv(io.BytesIO(artifacts[chosen])), width="stretch")
            st.download_button("Download CSV", artifacts[chosen], chosen.rsplit('/', 1)[-1],
                               mime="text/csv", key=f"regular_csv_{folder}")
