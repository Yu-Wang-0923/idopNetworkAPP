"""
共享工具模块：路径常量、CSS 注入、matplotlib 中文字体配置。
"""
from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import streamlit as st

# ── 路径常量 ──────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent.parent / "static"
CSS_PATH = STATIC_DIR / "css" / "custom_style.css"
FONT_PATH = STATIC_DIR / "SimHei.ttf"

# matplotlib 字体属性对象（供 plot_*.py 直接使用）
font_prop = fm.FontProperties(fname=FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def load_css():
    """强制注入的全局核心样式：浅色高质感侧边栏 + 精细彩色 Material 图标"""
    st.markdown("""
    <style>
    /* 1. 整个页面的背景色：纯净冷灰白 */
    .stApp { background-color: #f8fafc !important; }

    /* 2. 隐藏自带导航、顶部白边和页脚 */
    header, footer, [data-testid="stSidebarNav"] { display: none !important; }

    /* 3. 🌟 锁死浅色侧边栏：使用清爽的亮灰白色，带有右侧浅色分割线 🌟 */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        background-image: none !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 1rem !important;
    }

    /* 4. 侧边栏文字颜色：改用深灰蓝色，保证极佳的可读性 */
    .stPageLink a { text-decoration: none !important; padding: 0.3rem 0; transition: all 0.2s; }
    .stPageLink a p { color: #334155 !important; font-size: 1.1rem !important; font-weight: 500; }
    
    /* 悬停效果：轻微的灰色半透明背景 */
    .stPageLink:hover { background-color: rgba(15, 23, 42, 0.05) !important; border-radius: 8px; }

    /* 5. TSA Logo 圆角及阴影美化：在浅色背景下加上轻微投影，质感无敌 */
    [data-testid="stSidebar"] img {
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        background-color: #ffffff;
    }

    /* 6. 全局字体稍微放大 */
    p, li, .stMarkdown {
        font-size: 1.1rem !important; 
        line-height: 1.6 !important;
    }

    /* ======================================================== */
    /* 🌟 7. 在浅色底色上，给 Material 图标上色（高饱和度、极致吸睛） 🌟 */
    /* ======================================================== */
    
    /* 放大图标，让彩色更明显 */
    [data-testid="stSidebar"] span[data-testid="stWidgetLabel"] span {
        font-size: 1.3rem !important;
        font-weight: bold !important;
    }

    /* 1. Home 按钮：科技亮蓝 */
    .stPageLink:nth-of-type(1) span[data-testid="stWidgetLabel"] span {
        color: #0284c7 !important; 
    }

    /* 2. Curve Fitting 按钮：生机森林绿 */
    .stPageLink:nth-of-type(2) span[data-testid="stWidgetLabel"] span {
        color: #16a34a !important; 
    }

    /* 3. FunClu 按钮：活力暖橙 */
    .stPageLink:nth-of-type(3) span[data-testid="stWidgetLabel"] span {
        color: #ea580c !important; 
    }

    /* 4. NetRecon 按钮：优雅深紫 */
    .stPageLink:nth-of-type(4) span[data-testid="stWidgetLabel"] span {
        color: #7c3aed !important; 
    }

    /* 5. NetAnal 按钮：时尚玫瑰红 */
    .stPageLink:nth-of-type(5) span[data-testid="stWidgetLabel"] span {
        color: #e11d48 !important; 
    }
    </style>
    """, unsafe_allow_html=True)


def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        # 1. 放置统一的 Logo
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True) 
        col1, col2, col3 = st.columns([1, 3.5, 1])
        with col2:
            st.image("TSA.png", use_container_width=True)
        
        # 2. 放置统一的导航按钮（Material Icons 极简风）
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        
        st.page_link("Home.py", label="Home", icon=":material/home:")
        st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon=":material/timeline:")
        st.page_link("pages/2_FunClu.py", label="FunClu", icon=":material/category:") 
        st.page_link("pages/3_NetRecon.py", label="NetRecon", icon=":material/hub:") 
        st.page_link("pages/4_NetAnal.py", label="NetAnal", icon=":material/insights:")