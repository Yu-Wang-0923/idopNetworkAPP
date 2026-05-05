import streamlit as st
st.set_page_config(page_title="idopNetwork", page_icon="TSA.png", layout="wide", initial_sidebar_state="expanded")

from backend.utils import load_css

load_css()



# 2. 修改版 CSS：修复导航栏隐形问题，并圆角美化 Logo
st.markdown("""
<style>
#MainMenu,header,footer{visibility:hidden;} .stApp{background:#f8fafc;}
[data-testid="stSidebar"] {background-color: #1e293b !important;}

/* 🌟 关键修复 1：把你隐形的原生导航栏文字变成亮色！ */
[data-testid="stSidebarNav"] span {color: #f8fafc !important; font-size: 1.05rem; font-weight: 500;}
[data-testid="stSidebarNav"] a:hover {background-color: rgba(255,255,255,0.1) !important; border-radius: 6px;}

/* 🌟 关键修复 2：把白底方形的 TSA Logo 变成精致的圆角图标 */
[data-testid="stSidebar"] img {border-radius: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);}

.card {background:#fff; padding:1.2rem; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,0.02); height:100%; transition:0.3s;}
.card:hover {border-color:#3b82f6; box-shadow:0 4px 8px rgba(59,130,246,0.12); transform: translateY(-2px);}
.scroll-box {height:320px; overflow-y:auto; background:#fff; padding:1.2rem; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,0.02);}
</style>
""", unsafe_allow_html=True)

# 3. 侧边栏：删掉了不能跳转的假按钮，只放居中优化的 Logo
with st.sidebar:
    # 留出一点空隙，别让 Logo 离你的原生导航太近
    st.markdown("<br><br>", unsafe_allow_html=True)
    # 利用列布局 (1 : 2.5 : 1) 把 Logo 完美居中显示
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        st.image("TSA.png", use_container_width=True)

# 4. 首屏横幅区
col_hero1, col_hero2 = st.columns([1.5, 1], gap="large")
with col_hero1: 
    st.markdown("<h1 style='color:#0f172a;margin-top:1rem;font-size:2.6rem;'>idopNetwork：下一代数据分析平台</h1><p style='color:#475569;font-size:1.1rem;line-height:1.7;'>一个面向复杂系统数据的可视化分析工作台，支持从静态数据出发构建动态推演与全景网络洞察。</p>", unsafe_allow_html=True)
    btn_c1, btn_c2, _ = st.columns([1, 1, 3])
    with btn_c1: st.button("了解更多", type="primary", use_container_width=True)
    with btn_c2: st.button("查阅文档", use_container_width=True)
with col_hero2: 
    st.info("🖼️ 后期可在此处使用 st.image('架构图.png') 放置右侧配图。")

st.divider()

# 5. 核心内容区 (2x2网格的IDOP优势卡片 + 论文滚动区)
col_left, col_right = st.columns([1.6, 1], gap="large")

with col_left:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;'>IDOP 核心技术优势</h4>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: 
        st.markdown("<div class='card'><b style='color:#0f172a;'>I - Informative (富含信息)</b><p style='color:#64748b;font-size:0.9rem;margin-top:0.5rem;'>构建带符号加权图，精准捕获系统内部的<b>方向性依赖 (Directional Dependence)</b> 与调控方向，完整保留结构拓扑信息。</p></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True) 
        st.markdown("<div class='card'><b style='color:#0f172a;'>O - Omnidirectional (全方位)</b><p style='color:#64748b;font-size:0.9rem;margin-top:0.5rem;'>深度融合多源、多层级组学数据与高维特征，消除单一维度的认知偏差，形成全息化的网络视图。</p></div>", unsafe_allow_html=True)
    with c2: 
        st.markdown("<div class='card'><b style='color:#0f172a;'>D - Dynamic (动态演化)</b><p style='color:#64748b;font-size:0.9rem;margin-top:0.5rem;'>基于 qdODEs 理论体系，将静态截面观测数据科学地映射为跨时间与空间的演化轨迹，揭示深层机制。</p></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True) 
        st.markdown("<div class='card'><b style='color:#0f172a;'>P - Personalized (个性化解析)</b><p style='color:#64748b;font-size:0.9rem;margin-top:0.5rem;'>突破传统群体统计局限，构建单样本特异性网络，支持面向特定独立个体的差异解析与精准决策。</p></div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;'>已发表的研究论文</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div class='scroll-box'>
        <b style='color:#0f172a;'>多源组学数据融合的 SLET 动态推演模型</b><br><span style='color:#64748b;font-size:0.85rem;'>2026, 核心期刊</span><hr style='border:none;border-top:1px solid #f1f5f9;margin:0.8rem 0;'>
        <b style='color:#0f172a;'>复杂系统特征信息的方向性依赖网络分析</b><br><span style='color:#64748b;font-size:0.85rem;'>2025, 国际顶级会议</span><hr style='border:none;border-top:1px solid #f1f5f9;margin:0.8rem 0;'>
        <b style='color:#0f172a;'>基于 ST-HGNN 的纵向生理数据演化研究</b><br><span style='color:#64748b;font-size:0.85rem;'>2024, 医学信息学期刊</span><hr style='border:none;border-top:1px solid #f1f5f9;margin:0.8rem 0;'>
        <b style='color:#0f172a;'>基于 qdODE 的环境碳通量时空预测方法</b><br><span style='color:#64748b;font-size:0.85rem;'>2024, 生态学报</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown(
    """
    <div class="affiliation" style="text-align:center; color:#64748b; font-size:0.86rem; line-height:1.7;">
        复杂系统拓扑统计理论及应用北京市重点实验室<br/>
        北京雁栖湖应用数学研究院
    </div>
    <p style='text-align:center; color:#94a3b8; font-size:0.82rem; margin-top:0.5rem;'>idopNetwork v2.0</p>
    """,
    unsafe_allow_html=True,
)





st.divider()
st.markdown(
    """
    <div class="affiliation">
        复杂系统拓扑统计理论及应用北京市重点实验室<br/>
        北京雁栖湖应用数学研究院
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<p class='home-footer'>idopNetwork v2.0</p>", unsafe_allow_html=True)
