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


ddef load_css():
    """强制注入的全局核心样式，彻底解决闪烁和颜色失效问题"""
    st.markdown("""
    <style>
    /* 1. 强制页面主背景色为浅灰白 */
    .stApp { background-color: #f8fafc !important; }

    /* 2. 暴力隐藏自带的顶部白边和丑陋的默认导航栏 */
    header, footer, [data-testid="stSidebarNav"] { display: none !important; }

    /* 3. 强制侧边栏变成高级深灰蓝！ */
    [data-testid="stSidebar"] {
        background-color: #334155 !important;
        background-image: none !important;
        padding-top: 1rem !important;
    }

    /* 4. 让侧边栏我们自定义的链接变成亮白色，且悬停有发光效果 */
    .stPageLink a { text-decoration: none !important; padding: 0.25rem 0; transition: all 0.2s;}
    .stPageLink a p { color: #f8fafc !important; font-size: 1.15rem !important; font-weight: 500; }
    .stPageLink:hover { background-color: rgba(255,255,255,0.15) !important; border-radius: 6px; }

    /* 5. TSA Logo 圆角及阴影美化 */
    [data-testid="stSidebar"] img {
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        background-color: #ffffff;
    }

    /* ======================================================== */
    /* 🌟 6. 终极优化：全局字体放大 & 输入框精准白字 🌟 */
    /* ======================================================== */
    
    /* 【字体放大】全局的正文、列表字体稍微放大，提升阅读舒适度 */
    p, li, .stMarkdown {
        font-size: 1.1rem !important; 
        line-height: 1.6 !important;
    }

    /* 【颜色保护】精准定位深色输入框的内部文字变成纯白，绝对不影响外部！ */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] > input, 
    .stTextInput input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important; /* 兼容 Safari/Chrome */
    }
    
    /* 修复下拉菜单展开后的深色背景和白字 */
    ul[data-baseweb="menu"] {
        background-color: #334155 !important;
    }
    ul[data-baseweb="menu"] li {
        color: #ffffff !important;
        font-size: 1.05rem !important; /* 下拉菜单里的字也顺便放大一点 */
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: #475569 !important; /* 鼠标悬停时的亮色反馈 */
    }
    </style>
    """, unsafe_allow_html=True)


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

        # === 下面是你追加粘贴进去的 UI 专属代码 ===

def load_css():
    """强制注入的全局核心样式，彻底解决闪烁和颜色失效问题"""
    st.markdown("""
    <style>
    /* 1. 强制页面主背景色为浅灰白 */
    .stApp { background-color: #f8fafc !important; }

    /* 2. 暴力隐藏自带的顶部白边和丑陋的默认导航栏 */
    header, footer, [data-testid="stSidebarNav"] { display: none !important; }

    /* 3. 强制侧边栏变成高级深灰蓝！ */
    [data-testid="stSidebar"] {
        background-color: #334155 !important;
        background-image: none !important;
        padding-top: 1rem !important;
    }

    /* 4. 让侧边栏我们自定义的链接变成亮白色，且悬停有发光效果 */
    .stPageLink a { text-decoration: none !important; padding: 0.25rem 0; transition: all 0.2s;}
    .stPageLink a p { color: #f8fafc !important; font-size: 1.15rem !important; font-weight: 500; }
    .stPageLink:hover { background-color: rgba(255,255,255,0.15) !important; border-radius: 6px; }

    /* 5. TSA Logo 圆角及阴影美化 */
    [data-testid="stSidebar"] img {
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        # 1. 放置统一的 Logo
        col1, col2, col3 = st.columns([1, 3.5, 1])
        with col2:
            st.image("TSA.png", use_container_width=True)
        
        # 2. 放置我们自己写的带图标导航按钮
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.page_link("Home.py", label="Home", icon="🏠")
        st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon="📈")
        st.page_link("pages/2_FunClu.py", label="FunClu", icon="🧩")
        st.page_link("pages/3_NetRecon.py", label="NetRecon", icon="🕸️")
        st.page_link("pages/4_NetAnal.py", label="NetAnal", icon="📊")