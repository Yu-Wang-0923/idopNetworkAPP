import streamlit as st

# 1. 基础设置
st.set_page_config(page_title="idopNetwork", page_icon="TSA.png", layout="wide", initial_sidebar_state="expanded")

# (删除了死板的 st.logo，我们在下面的 sidebar 里自己画一个大的)

# 2. 核心 CSS 样式（已大幅优化版面结构）
st.markdown("""
<style>
#MainMenu,header,footer{visibility:hidden;} .stApp{background:#f8fafc;}

/* 侧边栏底色 */
[data-testid="stSidebar"] {background-color: #334155 !important;}

/* 🌟 强行把导航菜单往下推，给 Logo 留出宽裕的空间 */
[data-testid="stSidebarNav"] {margin-top: 1.5rem !important;} 
[data-testid="stSidebarNav"] span {color: #f8fafc !important; font-size: 1.15rem !important; font-weight: 500;}
[data-testid="stSidebarNav"] a:hover {background-color: rgba(255,255,255,0.15) !important; border-radius: 6px;}

/* 增加整体的正文字号 */
p, div {font-size: 1.05rem;}

/* Logo 圆角与阴影美化 */
[data-testid="stSidebar"] img {border-radius: 22px; box-shadow: 0 6px 16px rgba(0,0,0,0.25);}

/* 🌟 放大 IDOP 静态标签，填补空白，增加质感 */
.idop-badges {display: flex; gap: 1rem; margin-top: 2.5rem; flex-wrap: wrap;}
.badge-item {
    background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; 
    padding: 0.65rem 1.4rem; /* 大幅增加内边距让标签变胖 */
    border-radius: 8px; font-size: 1.1rem; /* 增大字号 */
    font-weight: 600; cursor: default;
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
}

/* 内容区卡片样式 */
.card {background:#fff; padding:1.4rem; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,0.02); height:100%; transition:0.3s;}
.card:hover {border-color:#3b82f6; box-shadow:0 4px 8px rgba(59,130,246,0.12);}

/* 论文滚动框调整 */
.scroll-box {height:450px; overflow-y:auto; background:#fff; padding:1.2rem; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,0.02);}
.paper-item {margin-bottom: 1.2rem;}
.paper-title {font-weight: bold; color: #0f172a; font-size: 1rem;}
.paper-authors {color: #64748b; font-size: 0.9rem; line-height: 1.5;}
.paper-link {color: #3b82f6; font-size: 0.9rem; text-decoration: none; font-weight: 500;}
.paper-link:hover {text-decoration: underline;}

/* 功能模块的标题和小图标 */
.app-title {font-size: 1.15rem; font-weight: bold; color: #0f172a; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;}
.app-desc {font-size: 0.95rem; color: #64748b; line-height: 1.6;}
</style>
""", unsafe_allow_html=True)

# 3. 🌟 侧边栏：重新绘制巨大且居中的 Logo
with st.sidebar:
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True) # 顶部留白
    # 用 1:3.5:1 的比例，让中间的 Logo 足够大且完美居中
    col1, col2, col3 = st.columns([1, 3.5, 1])
    with col2:
        st.image("TSA.png", use_container_width=True)

# 4. 首屏横幅区
col_hero1, col_hero2 = st.columns([1.3, 1], gap="large")
with col_hero1: 
    # 加大标题和副标题字号与行距
    st.markdown("<h1 style='color:#0f172a;margin-top:2rem;font-size:3.2rem;'>idopNetwork：下一代数据分析平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:1.2rem;line-height:1.8;margin-top:1.5rem;'>一个面向复杂系统数据的可视化分析工作台，支持从静态数据出发构建动态推演与全景网络洞察。</p>", unsafe_allow_html=True)
    
    # 放大的饱满静态标签
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
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.image("IDOP.png", use_container_width=True)

st.divider()

# 5. 核心内容区 (功能模块 + 论文区)
# 🌟 优化比例，从 1.6:1 调整为 1.15:1，让左右更加匀称
col_left, col_right = st.columns([1.15, 1], gap="large")

with col_left:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;font-size:1.3rem;'>平台核心工作流</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card' style='margin-bottom: 1rem;'>
        <div class='app-title'>🧩 曲线拟合与特征聚类 (Curve Fitting & FunClu)</div>
        <div class='app-desc'>
            内置高精度拟合算法，针对多源时间序列或纵向观测数据进行特征提取。通过函数型聚类 (Functional Clustering) 降维，识别系统内部具有相似演化轨迹的核心变量模块。
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card' style='margin-bottom: 1rem;'>
        <div class='app-title'>🕸️ 全景网络重构 (NetRecon)</div>
        <div class='app-desc'>
            突破传统相关性局限，构建带符号的加权有向图。精准量化复杂系统内部的<b>方向性依赖 (Directional Dependence)</b>，支持跨时空、多层级数据的全息互作网络重构。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='app-title'>📊 动态推演与个性化解析 (NetAnal)</div>
        <div class='app-desc'>
            依托 qdODEs 理论模型，将静态拓扑映射为动态演化轨迹。支持突破群体统计平均的局限，生成单样本特异性网络，为精准调控与个性化决策提供数理依据。
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;font-size:1.3rem;'>代表性学术成果</h4>", unsafe_allow_html=True)
    
    papers_html = """
    <div class='scroll-box'>
        <div class="paper-item">
            <div class="paper-title">IdopNetwork as a genomic predictor of drug response.</div>
            <div class="paper-authors">Che, J., Jin, Y., Gragnoli, C., Yau, S.-T., & Wu, R. (2025). <i>Drug Discovery Today</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1016/j.drudis.2024.104252" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">Hypernetwork modeling and topology of high-order interactions for complex systems.</div>
            <div class="paper-authors">Feng, L., Gong, H., Zhang, S., et al. (2024). <i>PNAS</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1073/pnas.2412220121" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">Network modeling and topology of aging.</div>
            <div class="paper-authors">Feng, L., Yang, D., Wu, S., et al. (2025). <i>Physics Reports</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1016/j.physrep.2024.10.006" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">idopNetwork: A network tool to dissect spatial community ecology.</div>
            <div class="paper-authors">Dong, A., Wu, S., Che, J., Wang, Y., & Wu, R. (2023). <i>Methods in Ecology and Evolution</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1111/2041-210X.14172" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">The metabolomic physics of complex diseases.</div>
            <div class="paper-authors">Wu, S., Liu, X., Dong, A., et al. (2023). <i>PNAS</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1073/pnas.2308496120" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">Graph statistics theory of individualized quantitative genetics under haplotype-resolved genome assembly.</div>
            <div class="paper-authors">Sun, L., Bian, Y., Yang, D., et al. (2026). <i>PNAS</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1073/pnas.2600004123" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">Recovering dynamic networks in big static datasets.</div>
            <div class="paper-authors">Wu, R., & Jiang, L. (2021). <i>Physics Reports</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1016/j.physrep.2021.01.003" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">Complex network approaches to nonlinear time series analysis.</div>
            <div class="paper-authors">Zou, Y., Donner, R. V., Marwan, N., Donges, J. F., & Kurths, J. (2019). <i>Physics Reports</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1016/j.physrep.2018.10.005" target="_blank">🔗 点击跳转原文</a></div>
        </div>
        <div class="paper-item">
            <div class="paper-title">Topological change of soil microbiota networks for forest resilience under global warming.</div>
            <div class="paper-authors">Gong, H., Wang, H., Wang, Y., et al. (2024). <i>Physics of Life Reviews</i>. <br>
            <a class="paper-link" href="https://doi.org/10.1016/j.plrev.2024.08.001" target="_blank">🔗 点击跳转原文</a></div>
        </div>
    </div>
    """
    st.markdown(papers_html, unsafe_allow_html=True)







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
