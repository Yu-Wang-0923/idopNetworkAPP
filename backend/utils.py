"""
共享工具模块：路径常量、CSS 注入、matplotlib 中文字体配置。

所有 static/ 资源的路径引用、页面 CSS 加载以及 matplotlib 中文字体
初始化均集中于此，避免各页面/模块重复硬编码。
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import streamlit as st

# ── 路径常量 ──────────────────────────────────────────────────────────────────

STATIC_DIR: Path = Path(__file__).parent.parent / "static"
CSS_PATH: Path = STATIC_DIR / "css" / "custom_style.css"
FONT_PATH: Path = STATIC_DIR / "SimHei.ttf"

# matplotlib 字体属性对象（供 plot_*.py 直接使用）
font_prop: fm.FontProperties = fm.FontProperties(fname=FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def load_css(path: Path = CSS_PATH) -> None:
    """将自定义 CSS 文件注入 Streamlit 页面。"""
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# def setup_matplotlib_chinese() -> None:
#     """配置 matplotlib 中文字体回退列表与负号显示。"""
#     plt.rcParams["font.sans-serif"] = ["SimHei"]
#     plt.rcParams["axes.unicode_minus"] = False


def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        # 1. 放置统一的 Logo
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True) 
        col1, col2, col3 = st.columns([1, 3.5, 1])
        with col2:
            st.image("TSA.png", use_container_width=True)
        
        # 2. 放置统一的导航按钮
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.page_link("Home.py", label="Home", icon="🏠")
        st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon="📈")
        st.page_link("pages/2_FunClu.py", label="FunClu", icon="🧩")
        st.page_link("pages/3_NetRecon.py", label="NetRecon", icon="🕸️")
        st.page_link("pages/4_NetAnal.py", label="NetAnal", icon="📊")