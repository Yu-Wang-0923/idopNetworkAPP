import streamlit as st
st.set_page_config(page_title="idopNetwork", page_icon="TSA.png", layout="wide", initial_sidebar_state="expanded")

from backend.utils import load_css

load_css()


# 2. 核心 CSS 样式
st.markdown("""
<style>
#MainMenu,header,footer{visibility:hidden;} .stApp{background:#f8fafc;}
[data-testid="stSidebar"] {background-color: #2D3748 !important;}

/* 侧边栏导航文字颜色和 Hover 效果 */
[data-testid="stSidebarNav"] span {color: #f8fafc !important; font-size: 1.05rem; font-weight: 500;}
[data-testid="stSidebarNav"] a:hover {background-color: rgba(255,255,255,0.1) !important; border-radius: 6px;}

/* Logo 圆角美化 */
[data-testid="stSidebar"] img {border-radius: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);}

/* 新增：Hero区域的 IDOP 小标签样式 */
.idop-badges {display: flex; gap: 0.8rem; margin-top: 1.5rem;}
.badge-item {background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem; font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: 0.2s;}
.badge-item:hover {background: #3b82f6; color: white; transform: translateY(-1px);}

/* 内容区卡片样式 */
.card {background:#fff; padding:1.2rem; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,0.02); height:100%; transition:0.3s;}
.card:hover {border-color:#3b82f6; box-shadow:0 4px 8px rgba(59,130,246,0.12); transform: translateY(-2px);}
.scroll-box {height:360px; overflow-y:auto; background:#fff; padding:1.2rem; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,0.02);}

/* 应用场景的标题和小图标 */
.app-title {font-size: 1.05rem; font-weight: bold; color: #0f172a; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.5rem;}
.app-desc {font-size: 0.9rem; color: #64748b; line-height: 1.6;}
</style>
""", unsafe_allow_html=True)

# 3. 侧边栏
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        st.image("TSA.png", use_container_width=True)

# 4. 首屏横幅区
col_hero1, col_hero2 = st.columns([1.5, 1], gap="large")
with col_hero1: 
    st.markdown("<h1 style='color:#0f172a;margin-top:1rem;font-size:2.6rem;'>idopNetwork：下一代数据分析平台</h1><p style='color:#475569;font-size:1.1rem;line-height:1.7;'>一个面向复杂系统数据的可视化分析工作台，支持从静态数据出发构建动态推演与全景网络洞察。</p>", unsafe_allow_html=True)
    
    # 用精致的徽章替换掉原来的大按钮
    st.markdown("""
    <div class="idop-badges">
        <div class="badge-item">💡 Informative</div>
        <div class="badge-item">🔄 Dynamic</div>
        <div class="badge-item">🌐 Omnidirectional</div>
        <div class="badge-item">🎯 Personalized</div>
    </div>
    """, unsafe_allow_html=True)

with col_hero2: 
    # 这里直接调用你新上传的 IDOP 图片
    # 注意：如果你的图片后缀是 .jpg，请把下面的 .png 改成 .jpg
    st.image("IDOP.png", use_container_width=True)

st.divider()

# 5. 核心内容区
col_left, col_right = st.columns([1.6, 1], gap="large")

with col_left:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;'>多学科交叉应用场景</h4>", unsafe_allow_html=True)
    
    # 第一个场景：生态与环境
    st.markdown("""
    <div class='card' style='margin-bottom: 0.8rem;'>
        <div class='app-title'>🌱 宏观生态与环境驱动</div>
        <div class='app-desc'>
            突破传统相关性分析的局限，精确量化多重环境因子对生态系统（如荒漠草原碳通量）的<b>方向性依赖 (Directional Dependence)</b>。结合 qdODE 推演环境驱动力在不同时空尺度下的动态演化机制。
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 第二个场景：临床与分子医学
    st.markdown("""
    <div class='card' style='margin-bottom: 0.8rem;'>
        <div class='app-title'>🧬 临床疾病推演与分子靶点</div>
        <div class='card-content app-desc'>
            深度融合多源组学与高维临床特征，构建单样本特异性网络。支持复杂退行性疾病或免疫系统疾病（如 SLET）的动态轨迹重构，为发掘关键致病模块与个性化诊疗提供计算生物学证据。
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 第三个场景：生理信号与时间序列
    st.markdown("""
    <div class='card'>
        <div class='app-title'>🧠 复杂纵向生理信号拓扑</div>
        <div class='app-desc'>
            针对高维、纵向观测的时序生理数据（如 fMRI 脑网络信号、长程心电图序列），构建具有时空一致性的全息网络视图，解析系统状态突变的前兆网络特征。
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    <div class="affiliation">
        复杂系统拓扑统计理论及应用北京市重点实验室<br/>
        北京雁栖湖应用数学研究院
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<p class='home-footer'>idopNetwork v2.0</p>", unsafe_allow_html=True)
