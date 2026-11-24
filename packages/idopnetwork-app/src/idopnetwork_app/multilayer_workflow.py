"""在 NetRecon 中展示多层 IDOP 网络的矩阵总图。"""

from io import BytesIO

import streamlit as st

from idopnetwork.network.interactive import sort_modules
from idopnetwork.network.multilayer_plot import build_multilayer_figure
from idopnetwork_app.i18n import t


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
    cache_key = "ml_overview_render"
    draw_enabled = bool(conditions) and (bool(modules) or any(cond in inter for cond in conditions))
    if st.button(t("netrecon.multilayer_overview.draw"), key="ml_overview_draw", disabled=not draw_enabled):
        try:
            with st.spinner(t("netrecon.multilayer_overview.rendering")):
                figure = build_multilayer_figure(inter, intra, conditions=conditions, modules=modules)
                preview = BytesIO()
                vector = BytesIO()
                preview_dpi = min(150, 4200 / max(figure.get_size_inches()))
                figure.savefig(preview, format="png", dpi=preview_dpi)
                figure.savefig(vector, format="svg")
            st.session_state[cache_key] = (result, request, preview.getvalue(), vector.getvalue())
        except (OSError, ValueError) as exc:
            st.session_state.pop(cache_key, None)
            st.error(t("netrecon.multilayer_overview.failed", error=exc))

    rendered = st.session_state.get(cache_key)
    if rendered is not None and rendered[0] is result and rendered[1] == request:
        _, _, png, svg = rendered
        st.image(png, width="stretch")
        st.download_button(
            t("netrecon.multilayer_overview.download"),
            data=svg,
            file_name="idop_multilayer_overview.svg",
            mime="image/svg+xml",
            key="ml_overview_download",
            on_click="ignore",
        )
