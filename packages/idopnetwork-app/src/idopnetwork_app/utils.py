"""
共享工具模块：路径常量、CSS 注入、matplotlib 中文字体配置。
"""
from html import escape
from importlib.resources import files
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError
from idopnetwork_app.i18n import language_switch, t

# ── 静态资源访问（通过 importlib.resources） ──────────────────────────────
_STATIC = files("idopnetwork_app.static")
CSS_PATH = _STATIC / "css" / "custom_style.css"
FONT_PATH = _STATIC / "SimHei.ttf"
IMAGES_DIR = _STATIC / "images"

font_prop = fm.FontProperties(fname=str(FONT_PATH))
plt.rcParams["axes.unicode_minus"] = False

ADMIN_USERS = {"郭佳泽", "11"}


def _is_admin_user() -> bool:
    """判断当前登录用户是否为管理员。"""
    current_user = st.session_state.get("current_user")
    return bool(st.session_state.get("logged_in", False) and current_user in ADMIN_USERS)


def load_css():
    """加载工作区共享样式。"""
    css = CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def upload_guidance(source_page: str, source_label: str, file_name: str, first_step: str) -> None:
    """Empty state for a page driven by an upstream export: what it needs, where it comes from."""
    slug = "".join(ch if ch.isalnum() else "-" for ch in file_name)
    with st.container(key=f"upload-guidance-{slug}"):
        st.markdown(t("guidance.title", file=file_name))
        st.markdown(
            f"1. {first_step}\n"
            f"2. {t('guidance.step2', file=file_name)}\n"
            f"3. {t('guidance.step3')}"
        )
        go = t("guidance.go", label=source_label)
        try:
            st.page_link(source_page, label=go)
        except StreamlitPageNotFoundError:
            # A single page executed on its own (tests, direct run) has no page registry to link into.
            st.caption(go)


def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        with st.container(key="platform-brand", horizontal_alignment="center", gap=None):
            st.image(str(IMAGES_DIR / "TSA.png"), width=112)
            st.markdown("**idopNetwork**")

        language_switch()

        if st.session_state.get("logged_in", False):
            st.page_link("Home.py", label="Home", icon=":material/home:")
            st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon=":material/timeline:")
            st.page_link("pages/2_FunClu.py", label="FunClu", icon=":material/category:")
            st.page_link("pages/3_NetRecon.py", label="NetRecon", icon=":material/hub:")
            st.page_link("pages/4_NetAnal.py", label="NetAnal", icon=":material/insights:")

            st.divider()
            user = escape(str(st.session_state["current_user"]))
            st.markdown(f"<p class='sidebar-user'>{t('sidebar.welcome', user=user)}</p>", unsafe_allow_html=True)
            if st.button(t("sidebar.logout"), key="logout_btn_final"):
                st.session_state["logged_in"] = False
                st.rerun()
        else:
            st.page_link("Home.py", label=t("sidebar.back_home"), icon=":material/login:")
            st.caption(t("sidebar.caption"))


# ── 向核心库注入中文字体 ─────────────────────────────────────────────
import idopnetwork.curve_fitting.plot as cf_plot
import idopnetwork.clustering.plot as cl_plot
import idopnetwork.network.plot as nw_plot

cf_plot.font_prop = font_prop
cl_plot.font_prop = font_prop
nw_plot.font_prop = font_prop
