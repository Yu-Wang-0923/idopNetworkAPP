"""
共享工具模块：路径常量、CSS 注入、matplotlib 中文字体配置。
"""
from pathlib import Path
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
HEADER_TOGGLE_KEY = "show_streamlit_header"


def _is_admin_user() -> bool:
    """判断当前登录用户是否为管理员。"""
    current_user = st.session_state.get("current_user")
    return bool(st.session_state.get("logged_in", False) and current_user in ADMIN_USERS)


def load_css():
    """强制注入的全局核心样式"""
    show_streamlit_header = bool(st.session_state.get(HEADER_TOGGLE_KEY, False))
    header_css = (
        "footer, [data-testid=\"stSidebarNav\"] { display: none !important; }"
        if show_streamlit_header
        else "header, footer, [data-testid=\"stSidebarNav\"] { display: none !important; }"
    )

    css = """
    <style>
    .stApp { background-color: #f8fafc !important; }
    __HEADER_CSS__
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        background-image: none !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 1rem !important;
    }
    .stPageLink a { text-decoration: none !important; padding: 0.3rem 0; transition: all 0.2s; }
    .stPageLink a p { color: #334155 !important; font-size: 1.1rem !important; font-weight: 500; }
    .stPageLink:hover { background-color: rgba(15, 23, 42, 0.05) !important; border-radius: 8px; }
    [data-testid="stSidebar"] img { border-radius: 20px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); background-color: #ffffff; }
    p, li, .stMarkdown { font-size: 1.1rem !important; line-height: 1.6 !important; }

    /* 图标上色 */
    [data-testid="stSidebar"] span[data-testid="stWidgetLabel"] span { font-size: 1.3rem !important; font-weight: bold !important; }
    .stPageLink:nth-of-type(1) span[data-testid="stWidgetLabel"] span { color: #0284c7 !important; }
    .stPageLink:nth-of-type(2) span[data-testid="stWidgetLabel"] span { color: #16a34a !important; }
    .stPageLink:nth-of-type(3) span[data-testid="stWidgetLabel"] span { color: #ea580c !important; }
    .stPageLink:nth-of-type(4) span[data-testid="stWidgetLabel"] span { color: #7c3aed !important; }
    .stPageLink:nth-of-type(5) span[data-testid="stWidgetLabel"] span { color: #e11d48 !important; }
    </style>
    """
    st.markdown(css.replace("__HEADER_CSS__", header_css), unsafe_allow_html=True)


def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3.5, 1])
        with col2:
            tsa_path = str(IMAGES_DIR / "TSA.png")
            st.image(tsa_path, use_container_width=True)
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        if st.session_state.get("logged_in", False):
            st.page_link("Home.py", label="Home", icon=":material/home:")
            st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon=":material/timeline:")
            st.page_link("pages/2_FunClu.py", label="FunClu", icon=":material/category:")
            st.page_link("pages/3_NetRecon.py", label="NetRecon", icon=":material/hub:")
            st.page_link("pages/4_NetAnal.py", label="NetAnal", icon=":material/insights:")

            st.markdown("<hr style='margin: 1.5rem 0; border-color: #cbd5e1;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#334155; font-weight:bold;'>👋 欢迎, {st.session_state['current_user']}</p>", unsafe_allow_html=True)
            if _is_admin_user():
                st.toggle(
                    "显示 Streamlit 顶部菜单",
                    key=HEADER_TOGGLE_KEY,
                    help="开启后可使用右上角 Streamlit 菜单（如 Clear cache）。",
                )
            if st.button("退出登录", key="logout_btn_final"):
                st.session_state["logged_in"] = False
                st.rerun()
        else:
            st.page_link("Home.py", label="返回首页登录", icon=":material/login:")
            st.markdown("<p style='color:#ef4444; font-weight:bold; text-align:center; margin-top:2rem;'>🔒 请先登录以解锁功能</p>", unsafe_allow_html=True)


# ── 向核心库注入中文字体和数据路径 ─────────────────────────────────────
import idopnetwork.curve_fitting.plot as cf_plot
import idopnetwork.clustering.plot as cl_plot
import idopnetwork.network.plot as nw_plot

cf_plot.font_prop = font_prop
cl_plot.font_prop = font_prop
nw_plot.font_prop = font_prop

import idopnetwork.analysis.glmy_test as glmy_test
glmy_test.DEFAULT_M3_CSV = str(files("idopnetwork_app.data") / "M3.csv")
