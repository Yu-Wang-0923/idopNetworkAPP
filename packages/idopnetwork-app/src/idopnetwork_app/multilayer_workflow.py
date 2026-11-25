"""在 NetRecon 中展示多层 IDOP 网络的矩阵总图。"""

import streamlit as st
import streamlit.components.v1 as components

from idopnetwork.network.interactive import sort_modules
from idopnetwork.network.multilayer_interactive import build_multilayer_html
from idopnetwork_app.i18n import language, t


OVERVIEW_PREFIX = "netrecon.multilayer_overview"
MATRIX_LABELS = (
    "title", "inter", "empty", "noEdges", "positive", "negative", "lock", "reset",
    "nodeSize", "curvature", "labelSize", "zoom", "threshold", "svg", "html", "lockedHint", "editingHint",
)


def render_multilayer_overview(result: dict) -> None:
    """使用本次构网结果绘制多层网总图。"""
    inter = {
        condition: network["adj_df"]
        for condition, network in result["inter_cluster"].items()
    }
    intra = {
        condition: {module: network["adj_df"] for module, network in networks.items()}
        for condition, networks in result["intra_cluster"].items()
        if networks
    }
    options = list(dict.fromkeys([*inter, *intra]))
    if not options:
        st.info(t("netrecon.multilayer_overview.no_results"))
        return

    st.caption(t("netrecon.multilayer_overview.caption"))
    condition_column, module_column = st.columns(2)
    key_prefix = f"ml_overview_{id(result)}"
    conditions = condition_column.multiselect(
        t("netrecon.multilayer_overview.conditions"),
        options,
        default=options,
        key=f"{key_prefix}_conditions",
    )
    available_modules = sort_modules({
        module for condition in conditions for module in intra.get(condition, {})
    })
    modules = module_column.multiselect(
        t("netrecon.multilayer_overview.modules"),
        available_modules,
        default=available_modules,
        key=f"{key_prefix}_modules_{tuple(conditions)!r}",
        disabled=not available_modules,
    )
    request = (tuple(conditions), tuple(modules))
    lang = language()
    cache_key = "ml_overview_render"
    draw_enabled = bool(conditions) and (bool(modules) or any(cond in inter for cond in conditions))
    draw = st.button(t("netrecon.multilayer_overview.draw"), key="ml_overview_draw", disabled=not draw_enabled)
    rendered = st.session_state.get(cache_key)
    same_view = rendered is not None and rendered[0] is result and rendered[1] == request
    if draw or (same_view and rendered[2] != lang):
        try:
            with st.spinner(t("netrecon.multilayer_overview.rendering")):
                labels = {key: t(f"{OVERVIEW_PREFIX}.{key}") for key in MATRIX_LABELS}
                html = build_multilayer_html(
                    inter, intra, conditions=conditions, modules=modules, labels={**labels, "lang": lang},
                )
            st.session_state[cache_key] = (result, request, lang, html)
        except (OSError, ValueError) as exc:
            st.session_state.pop(cache_key, None)
            st.error(t("netrecon.multilayer_overview.failed", error=exc))

    rendered = st.session_state.get(cache_key)
    if rendered is not None and rendered[0] is result and rendered[1] == request:
        embed_html = getattr(st, "iframe", components.html)
        embed_html(rendered[3], height=min(880, 180 + 170 * (len(modules) + 1)))
