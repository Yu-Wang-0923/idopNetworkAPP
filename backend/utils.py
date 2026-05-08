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

font_prop = fm.FontProperties(fname=FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def load_css():
    """强制注入的全局核心样式"""
    st.markdown("""
    <style>
    .stApp { background-color: #f8fafc !important; }
    header, footer, [data-testid="stSidebarNav"] { display: none !important; }
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
    """, unsafe_allow_html=True)

def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True) 
        col1, col2, col3 = st.columns([1, 3.5, 1])
        with col2:
            st.image("TSA.png", use_container_width=True)
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        
        # 权限控制
        if st.session_state.get("logged_in", False):
            st.page_link("Home.py", label="Home", icon=":material/home:")
            st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon=":material/timeline:")
            st.page_link("pages/2_FunClu.py", label="FunClu", icon=":material/category:") 
            st.page_link("pages/3_NetRecon.py", label="NetRecon", icon=":material/hub:") 
            st.page_link("pages/4_NetAnal.py", label="NetAnal", icon=":material/insights:")
            
            st.markdown("<hr style='margin: 1.5rem 0; border-color: #cbd5e1;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#334155; font-weight:bold;'>👋 欢迎, {st.session_state['current_user']}</p>", unsafe_allow_html=True)
            if st.button("退出登录", key="logout_btn_final"):
                st.session_state["logged_in"] = False
                st.rerun()
        else:
            # 🌟 关键修改：即使没登录，也允许点回 Home 页面去登录！
            st.page_link("Home.py", label="返回首页登录", icon=":material/login:")
            st.markdown("<p style='color:#ef4444; font-weight:bold; text-align:center; margin-top:2rem;'>🔒 请先登录以解锁功能</p>", unsafe_allow_html=True)