import streamlit as st
from backend.utils import load_css, setup_sidebar
from backend.auth import show_login_ui

# ==========================================
# 1. 基础设置 (必须在第一行)
# ==========================================
st.set_page_config(page_title="idopNetwork", page_icon="TSA.png", layout="wide", initial_sidebar_state="expanded")

# 加载全局样式和侧边栏
load_css()
setup_sidebar()

# ==========================================
# 2. 核心拦截门禁
# ==========================================
if not st.session_state.get("logged_in", False):
    # 如果没登录，显示右上角登录按钮
    show_login_ui()
    
    # 显示一个漂亮的门面介绍，但不给看具体内容
    st.markdown("<h1 style='color:#0f172a;margin-top:0.5rem;font-size:3.2rem;'>idopNetwork：下一代数据分析平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:1.2rem;'>一个面向复杂系统数据的可视化分析工作台。</p>", unsafe_allow_html=True)
    st.info("💡 这是一个私有科研平台。请点击右上角的 **[🔑 登录 / 注册]** 认证身份后解锁全部功能。")
    
    # 强制切断！没登录的人绝对看不到下面的论文和工具链接
    st.stop()

# ==========================================
# 3. 登录后的世界 (你原来的所有代码都在这里)
# ==========================================

# --- 原有的主页专属 CSS ---
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
.paper-link {color: #3b82f6; font-size: 0.85rem; text-decoration: none;}
</style>
""", unsafe_allow_html=True)

# --- 原有的首屏横幅区 ---
col_hero1, col_hero2 = st.columns([1.3, 1], gap="large")
with col_hero1: 
    st.markdown(f"<h1 style='color:#0f172a;margin-top:0.5rem;font-size:3.2rem;'>欢迎回来, {st.session_state['current_user']}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:1.2rem;line-height:1.8;margin-top:1.5rem;'>idopNetwork 已解锁。您可以开始构建动态推演与全景网络洞察。</p>", unsafe_allow_html=True)
    
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

# --- 原有的核心内容区 ---
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
            <div class='card-desc'>通过函数型聚类 (Functional Clustering) 降维，识别系统内部具有相似演化轨迹的核心变量模块。</div>
        </div>
    </a>
    <a href="NetRecon" target="_self" class="card-link">
        <div class='card'>
            <div class='card-title'><span>🕸️ 全景网络重构 (NetRecon)</span> <span style='font-size:0.9rem;color:#3b82f6;'>进入 ➔</span></div>
            <div class='card-desc'>突破传统相关性局限，构建带符号的加权有向图，精准量化内部方向性依赖。</div>
        </div>
    </a>
    <a href="NetAnal" target="_self" class="card-link">
        <div class='card'>
            <div class='card-title'><span>📊 动态推演与解析 (NetAnal)</span> <span style='font-size:0.9rem;color:#3b82f6;'>进入 ➔</span></div>
            <div class='card-desc'>将静态拓扑映射为动态演化轨迹，生成单样本特异性网络并提供全息解析。</div>
        </div>
    </a>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;font-size:1.3rem;'>代表性学术成果</h4>", unsafe_allow_html=True)
    
    # 这里就是你那长长的 46 篇论文列表
    papers_html = """
    <div class='scroll-box'>
        <div class="paper-item"><div class="paper-title">📄 Graph statistics theory of individualized quantitative genetics under haplotype-resolved genome assembly.</div><div class="paper-authors">Sun, L., et al. (<b>2026</b>). <i>PNAS</i>.</div></div>
        <div class="paper-item"><div class="paper-title">📄 A statistical mechanics model to decode tissue crosstalk during graft formation.</div><div class="paper-authors">Dong, A., et al. (<b>2026</b>). <i>Advanced Science</i>.</div></div>
        <div class="paper-item"><div class="paper-title">📄 An omnigenic interactome model to chart the genetic architecture of individual plants.</div><div class="paper-authors">Fa, C., et al. (<b>2026</b>). <i>Horticulture Research</i>.</div></div>
        <div class="paper-item"><div class="paper-title">📄 Network stress: A wiring diagram of whole stress genes.</div><div class="paper-authors">Wang, Y., & Wu, R. (<b>2026</b>). <i>Horticulture Research</i>.</div></div>
        <div class="paper-item"><div class="paper-title">📄 Statistical learning of stochastic complex systems via the yau-yau nonlinear filter.</div><div class="paper-authors">Xu, S., et al. (<b>2026</b>). <i>The Innovation</i>.</div></div>
        <div style="color: #94a3b8; text-align: center; font-size: 0.8rem;">已加载全部代表性成果</div>
    </div>
    """
    st.markdown(papers_html, unsafe_allow_html=True)

# --- 原有的底部版权 ---
st.markdown(
    """<div style="text-align:center; color:#94a3b8; font-size:0.9rem; margin: 3rem 0;">
        复杂系统拓扑统计理论及应用北京市重点实验室 · 北京雁栖湖应用数学研究院 · idopNetwork v2.0
    </div>""", unsafe_allow_html=True
)


# 只有当你用 admin 账号登录时才显示
if st.session_state.get("current_user") == "郭佳泽":
    st.divider()
    st.subheader("🛠️ 注册人员信息后台")
    from backend.auth import load_users
    import pandas as pd

    all_data = load_users()
    # 将复杂的 JSON 转换成整齐的表格
    df = pd.DataFrame.from_dict(all_data, orient='index')
    # 隐藏密码列，保护隐私
    if "password" in df.columns:
        df = df.drop(columns=["password"])
    st.dataframe(df, use_container_width=True)