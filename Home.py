import streamlit as st

# 添加自定义CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
# 调用函数，加载我们创建的CSS文件
local_css("static/css/custom_style.css")

st.set_page_config(
    page_title="idopNetwork",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        font-family: "Roboto", -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        background:
            radial-gradient(circle at 10% 12%, rgba(56, 189, 248, 0.08) 0%, transparent 30%),
            radial-gradient(circle at 86% 10%, rgba(37, 99, 235, 0.08) 0%, transparent 34%),
            linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        animation: bgFlow 14s ease-in-out infinite alternate;
    }
    @keyframes bgFlow {
        0% {
            background-position: 0% 0%, 100% 0%, 0% 0%;
        }
        100% {
            background-position: 8% 6%, 92% 8%, 0% 100%;
        }
    }
    .home-hero {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 2.6rem 2.4rem;
        margin: 0.8rem 0 1.8rem 0;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 48%, #1d4ed8 100%);
        color: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.28);
        box-shadow: 0 28px 60px -18px rgba(15, 23, 42, 0.45);
        text-align: center;
    }
    .home-hero::before {
        content: "";
        position: absolute;
        top: -52%;
        left: -12%;
        width: 52%;
        height: 208%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.2) 0%, rgba(56, 189, 248, 0.05) 48%, transparent 68%);
        transform: rotate(26deg);
        pointer-events: none;
        animation: heroGlow 7s ease-in-out infinite alternate;
    }
    .home-hero::after {
        content: "";
        position: absolute;
        right: -120px;
        top: -120px;
        width: 330px;
        height: 330px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(147, 197, 253, 0.36) 0%, rgba(147, 197, 253, 0.08) 52%, transparent 72%);
        pointer-events: none;
        animation: heroPulse 6.5s ease-in-out infinite;
    }
    @keyframes heroGlow {
        0% {
            opacity: 0.7;
            transform: rotate(24deg) translateX(0);
        }
        100% {
            opacity: 1;
            transform: rotate(30deg) translateX(12px);
        }
    }
    @keyframes heroPulse {
        0%, 100% {
            opacity: 0.6;
            transform: scale(0.95);
        }
        50% {
            opacity: 1;
            transform: scale(1.04);
        }
    }
    .home-title {
        position: relative;
        z-index: 2;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.9rem;
        background: linear-gradient(to right, #ffffff, #bfdbfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .home-subtitle {
        position: relative;
        z-index: 2;
        font-size: 1.04rem;
        color: #cbd5e1;
        line-height: 1.74;
        margin: 0 0 1.4rem 0;
        max-width: 840px;
        font-weight: 400;
        margin-left: auto;
        margin-right: auto;
    }
    .hero-badges {
        position: relative;
        z-index: 2;
        display: flex;
        flex-wrap: wrap;
        gap: 0.62rem;
        justify-content: center;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.3rem 0.76rem;
        background: rgba(148, 163, 184, 0.18);
        border: 1px solid rgba(148, 163, 184, 0.35);
        font-size: 0.77rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #e2e8f0;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 1.45rem;
        border: 1px solid rgba(226, 232, 240, 0.95);
        box-shadow: 0 8px 18px -10px rgba(15, 23, 42, 0.18), 0 1px 3px rgba(15, 23, 42, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 228px;
        display: flex;
        flex-direction: column;
        margin-bottom: 0.45rem;
        animation: cardAura 4.8s ease-in-out infinite;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #bfdbfe;
        box-shadow: 0 22px 28px -14px rgba(37, 99, 235, 0.18), 0 8px 14px -10px rgba(37, 99, 235, 0.18);
    }
    @keyframes cardAura {
        0%, 100% {
            box-shadow: 0 8px 18px -10px rgba(15, 23, 42, 0.18), 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        50% {
            box-shadow: 0 14px 24px -12px rgba(37, 99, 235, 0.22), 0 4px 10px -8px rgba(37, 99, 235, 0.16);
        }
    }
    .feature-icon-wrapper {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: #eff6ff;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        color: #2563eb;
        border: 1px solid #dbeafe;
    }
    .feature-icon-wrapper svg {
        width: 23px;
        height: 23px;
    }
    .feature-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin-bottom: 0.48rem;
        color: #0f172a;
    }
    .feature-desc {
        font-size: 0.94rem;
        color: #64748b;
        line-height: 1.64;
        margin-bottom: 1rem;
        flex-grow: 1;
    }
    .feature-tip {
        align-self: flex-start;
        display: inline-flex;
        align-items: center;
        border-radius: 8px;
        padding: 0.3rem 0.62rem;
        font-size: 0.75rem;
        color: #2563eb;
        font-weight: 600;
        background: #f0fdfa;
        border: 1px solid #ccfbf1;
        letter-spacing: 0.02em;
    }
    .section-label {
        margin: 0.2rem 0 0.72rem 0.2rem;
        color: #475569;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .home-footer {
        color: #94a3b8;
        text-align: center;
        font-size: 0.82rem;
    }
    .affiliation {
        margin: 0.2rem auto 0.55rem auto;
        text-align: center;
        color: #64748b;
        font-size: 0.86rem;
        line-height: 1.7;
        letter-spacing: 0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="home-hero">
        <div class="home-title">idopNetwork 数据分析平台</div>
        <p class="home-subtitle">
            <span style="display:block; text-align:center;">一个面向复杂系统数据的可视化分析工作台，支持从静态数据出发构建动态推演与全景网络洞察。</span>
        </p>
        <div class="hero-badges">
            <span class="hero-badge">Informative</span>
            <span class="hero-badge">Dynamic</span>
            <span class="hero-badge">Omnidirectional</span>
            <span class="hero-badge">Personalized</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("<p class='section-label'>IDOP 核心框架（Informative · Dynamic · Omnidirectional · Personalized）</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 17.5V6.8a1.3 1.3 0 0 1 2.1-1l3.8 3a1.3 1.3 0 0 0 1.6 0l3.8-3a1.3 1.3 0 0 1 2.1 1v10.7"/>
                    <path d="M4 17.5h16"/>
                    <path d="m9.5 14.2 2.5-2.1 2.5 2.1"/>
                </svg>
            </div>
            <div class="feature-title">Informative（富含信息）</div>
            <div class="feature-desc">以双向带符号加权图表达关系强度与方向，尽可能完整保留结构信息。</div>
            <div class="feature-tip">Bidirectional • Signed • Weighted</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="13" r="7.5"/>
                    <path d="M12 13V9.5"/>
                    <path d="m12 13 2.8 1.8"/>
                    <path d="M9 3h6"/>
                </svg>
            </div>
            <div class="feature-title">Dynamic（动态）</div>
            <div class="feature-desc">qdODEs 将静态观测推演为跨时间与空间的演化轨迹，揭示系统变化过程。</div>
            <div class="feature-tip">Static-to-Dynamic Inference</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

col3, col4 = st.columns(2, gap="large")
with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="8"/>
                    <path d="M4.5 9h15"/>
                    <path d="M4.5 15h15"/>
                    <path d="M12 4c2.5 2.2 2.5 13.8 0 16"/>
                    <path d="M12 4c-2.5 2.2-2.5 13.8 0 16"/>
                </svg>
            </div>
            <div class="feature-title">Omnidirectional（全方位）</div>
            <div class="feature-desc">融合多源、多层级组学与特征信息，形成全息化网络视图与一致性解释。</div>
            <div class="feature-tip">Multi-Omics Integrated Network</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 13.2a4.1 4.1 0 1 0 0-8.2 4.1 4.1 0 0 0 0 8.2Z"/>
                    <path d="M5 19.4a7.2 7.2 0 0 1 14 0"/>
                    <path d="M17.8 7.6 19.5 9.3l2.4-2.4"/>
                </svg>
            </div>
            <div class="feature-title">Personalized（个性化）</div>
            <div class="feature-desc">构建单样本特异性网络，支持面向个体对象的差异解析与精准决策。</div>
            <div class="feature-tip">Sample-Specific Network</div>
        </div>
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
