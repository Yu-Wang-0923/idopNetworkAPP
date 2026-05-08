
import streamlit as st
from backend.utils import load_css, setup_sidebar
from backend.auth import show_login_ui

# ==========================================
# 1. 基础配置（必须是 Streamlit 命令的第一行）
# ==========================================
st.set_page_config(
    page_title="idopNetwork", 
    page_icon="TSA.png", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 加载全局样式和侧边栏
# ==========================================
# 这一套组合拳，全页只准出现这一次！
load_css()
setup_sidebar()

# ==========================================
# 3. 登录拦截逻辑
# ==========================================
# 如果没登录，显示右上角登录按钮并锁定页面
if not st.session_state.get("logged_in", False):
    show_login_ui()
    
    # 没登录时，主页只显示一个精美的门面
    st.markdown("<h1 style='color:#0f172a;margin-top:0.5rem;font-size:3.2rem;'>idopNetwork：下一代数据分析平台</h1>", unsafe_allow_html=True)
    st.info("💡 请点击右上角的 **[🔑 登录 / 注册]** 按钮，认证成功后解锁全部高级功能。")
    
    # 没登录时，用 st.stop() 强制切断后续代码执行，防止数据泄露
    st.stop()

# ==========================================
# 4. 已登录用户看到的完整内容（全部写在下面）
# ==========================================

# 主页专属 CSS 样式
st.markdown("""
<style>
/* IDOP 静态标签 */
.idop-badges { display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap; }
.badge-item { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; padding: 0.65rem 1.4rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.03); }

/* 工作流卡片样式 */
.card-link {text-decoration: none !important;}
.card { background:#fff; padding:1.2rem; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,0.02); margin-bottom: 0.8rem; transition:0.3s; }
.card:hover { border-color:#3b82f6; box-shadow:0 4px 12px rgba(59,130,246,0.15); transform: translateY(-2px); }
.card-title { font-size: 1.1rem; font-weight: bold; color: #0f172a; margin-bottom: 0.4rem; display: flex; justify-content: space-between; }
.card-desc { font-size: 0.95rem; color: #64748b; line-height: 1.5; margin-bottom:0; }

/* 论文滚动框样式 */
.scroll-box { height:480px; overflow-y:auto; background:#fff; padding:1.2rem; border-radius:8px; border:1px solid #e2e8f0; }
.paper-item { margin-bottom: 1.2rem; padding-bottom: 0.8rem; border-bottom: 1px dashed #e2e8f0; }
.paper-item:last-child {border-bottom: none;}
.paper-title {font-weight: 600; color: #0f172a; font-size: 1.05rem;}
.paper-authors {color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 0.3rem;}
</style>
""", unsafe_allow_html=True)

# 首屏横幅区
col_hero1, col_hero2 = st.columns([1.3, 1], gap="large")
with col_hero1: 
    st.markdown("<h1 style='color:#0f172a;margin-top:0.5rem;font-size:3.2rem;'>idopNetwork：下一代数据分析平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:1.2rem;line-height:1.8;margin-top:1.5rem;'>欢迎回来，<b>{}</b>！您已进入全功能模式。</p>".format(st.session_state['current_user']), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="idop-badges">
        <div class="badge-item">💡 Informative</div>
        <div class="badge-item">🔄 Dynamic</div>
        <div class="badge-item">🌐 Omnidirectional</div>
        <div class="badge-item">🎯 Personalized</div>
    </div>
    """, unsafe_allow_html=True)

with col_hero2: 
    _, img_col, _ = st.columns([1, 6, 1])
    with img_col:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.image("IDOP.png", use_container_width=True)

st.divider()

# 核心内容区 (工作流 + 论文)
col_left, col_right = st.columns([1.15, 1], gap="large")

with col_left:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;font-size:1.3rem;'>平台核心工作流</h4>", unsafe_allow_html=True)
    st.markdown("""
    <a href="Curve_Fitting" target="_self" class="card-link">
        <div class='card'>
            <div class='card-title'><span>📈 曲线拟合 (Curve Fitting)</span> <span style='font-size:0.9rem;color:#3b82f6;'>进入 ➔</span></div>
            <div class='card-desc'>内置高精度拟合算法，针对多源时间序列或纵向观测数据进行特征提取与降噪。</div>
        </div>
    </a>
    <a href="FunClu" target="_self" class="card-link">
        <div class='card'>
            <div class='card-title'><span>🧩 特征聚类 (FunClu)</span> <span style='font-size:0.9rem;color:#3b82f6;'>进入 ➔</span></div>
            <div class='card-desc'>通过函数型聚类识别系统内部具有相似演化轨迹的核心变量模块。</div>
        </div>
    </a>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;font-size:1.3rem;'>代表性学术成果</h4>", unsafe_allow_html=True)
    # (此处省略你之前的 papers_html 内容，直接粘贴即可)
    st.markdown("<div class='scroll-box'>你的论文列表内容...</div>", unsafe_allow_html=True)

# 底部版权
st.markdown(
    """<div style="text-align:center; color:#94a3b8; font-size:0.9rem; margin: 2rem 0;">
        复杂系统拓扑统计理论及应用北京市重点实验室 · 北京雁栖湖应用数学研究院
    </div>""", unsafe_allow_html=True
)