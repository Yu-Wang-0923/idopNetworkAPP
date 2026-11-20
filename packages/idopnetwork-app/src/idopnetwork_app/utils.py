"""
共享工具模块：路径常量、CSS 注入、matplotlib 中文字体配置。
"""
from html import escape
from importlib.resources import files
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import streamlit as st

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


def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        with st.container(key="platform-brand", horizontal_alignment="center", gap=None):
            st.image(str(IMAGES_DIR / "TSA.png"), width=112)
            st.markdown("**idopNetwork**")

        if st.session_state.get("logged_in", False):
            st.page_link("Home.py", label="Home", icon=":material/home:")
            st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon=":material/timeline:")
            st.page_link("pages/2_FunClu.py", label="FunClu", icon=":material/category:")
            st.page_link("pages/3_NetRecon.py", label="NetRecon", icon=":material/hub:")
            st.page_link("pages/4_NetAnal.py", label="NetAnal", icon=":material/insights:")

            st.divider()
            user = escape(str(st.session_state["current_user"]))
            st.markdown(f"<p class='sidebar-user'>欢迎，{user}</p>", unsafe_allow_html=True)
            if st.button("退出登录", key="logout_btn_final"):
                st.session_state["logged_in"] = False
                st.rerun()
        else:
            st.page_link("Home.py", label="返回首页登录", icon=":material/login:")
            st.caption("登录后可使用分析工具。")


# ── 向核心库注入中文字体 ─────────────────────────────────────────────
import idopnetwork.curve_fitting.plot as cf_plot
import idopnetwork.clustering.plot as cl_plot
import idopnetwork.network.plot as nw_plot

cf_plot.font_prop = font_prop
cl_plot.font_prop = font_prop
nw_plot.font_prop = font_prop
