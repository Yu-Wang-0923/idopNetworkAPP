

import streamlit as st
from backend.utils import load_css, setup_sidebar
from backend.auth import show_login_ui

# ==========================================
# 1. 基础设置 (必须在第一行)
# ==========================================
st.set_page_config(
    page_title="idopNetwork", 
    page_icon="TSA.png", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 加载全局样式和侧边栏
load_css()
setup_sidebar()

# ==========================================
# 2. 核心拦截门禁
# ==========================================
if not st.session_state.get("logged_in", False):
    # 如果没登录，显示右上角登录按钮
    show_login_ui()
    
    # 显示一个漂亮的门面介绍
    st.markdown("<h1 style='color:#0f172a;margin-top:0.5rem;font-size:3.2rem;'>idopNetwork：个性化全景动态网络推演平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:1.2rem;'>一个面向复杂系统数据的可视化分析工作台。</p>", unsafe_allow_html=True)
    st.info("💡 请点击右上角的 **[🔑 登录 / 注册]** 认证身份后解锁全部功能。")
    
    # 强制切断！
    st.stop()

# ==========================================
# 3. 登录后的世界 (只有认证用户可见)
# ==========================================
else:
    # --- 主页专属 CSS 样式 ---
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

    /* 个人主页按钮样式 */
    .profile-btn {
        display: inline-block;
        margin-top: 10px;
        padding: 6px 20px;
        background-color: #3b82f6;
        color: white !important;
        text-decoration: none !important;
        border-radius: 5px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .profile-btn:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 10px rgba(59,130,246,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # --- 首屏横幅区 ---
    col_hero1, col_hero2 = st.columns([1.3, 1], gap="large")
    with col_hero1: 
        st.markdown(f"<h1 style='color:#0f172a;margin-top:0.5rem;font-size:3.2rem;'>欢迎回来, {st.session_state['current_user']}</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#475569;font-size:1.15rem;line-height:1.7;margin-top:1.2rem;'>
            idopNetwork 是一套基于 <b>邬荣领教授</b> 提出的统计力学框架构建的复杂系统分析体系。
            通过将静态的观测数据映射到高维拓扑空间，我们能够突破传统统计学的局限，实现对系统内部运作机理的深度还原。
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="idop-badges">
            <div class="badge-item" title="将复杂数据转化为可解释的生物/物理逻辑">💡 Informative</div>
            <div class="badge-item" title="刻画系统随时间与环境变化的演化轨迹">🔄 Dynamic</div>
            <div class="badge-item" title="覆盖从分子到表型的全尺度互作网络">🌐 Omnidirectional</div>
            <div class="badge-item" title="生成针对单样本或个体的特异性精准分析">🎯 Personalized</div>
        </div>
        """, unsafe_allow_html=True)

    with col_hero2: 
        # 右侧：邬教授专家名片
        _, img_col, _ = st.columns([1.5, 4, 1.5])
        with img_col:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            _, center_img_col, _ = st.columns([1, 2, 1])
            with center_img_col:
                st.image("wu.jpg", width=150)  # 通过子列居中，确保与下方文字同轴
                st.markdown("<p style='text-align:center; font-weight:bold; font-size:1.15rem; color:#0f172a; margin-top:4px; margin-bottom:2px;'>邬荣领 教授</p>", unsafe_allow_html=True)
                st.markdown("<p style='text-align:center; color:#475569; font-size:0.95rem; margin-top:0; margin-bottom:4px;'>BIMSA 副院长 / 首席科学家</p>", unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align:center;">
                    <a href="https://www.bimsa.cn/zh-CN/detail/ronglingwu.html" target="_blank" class="profile-btn">
                        查看个人主页 &rarr;
                    </a>
                </div>
                """, unsafe_allow_html=True)
       

    st.divider()

 # --- 核心内容区 ---
    col_left, col_right = st.columns([1.15, 1], gap="large")

    with col_left:
        st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;font-size:1.3rem;'>核心计算模块</h4>", unsafe_allow_html=True)
        st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet" />
        
        <a href=" " target="_self" class="card-link">
            <div class='card'>
                <div class='card-title'>
                    <span style='display:flex; align-items:center; gap:6px;'>
                        <span class="material-symbols-rounded" style="color:#16a34a; font-size:1.4rem;">timeline</span> 
                        曲线拟合 (Curve Fitting)
                    </span> 
                    <span style='font-size:0.9rem;color:#3b82f6;'>进入 &rarr;</span>
                </div>
                <div class='card-desc'>通过异速生长定律将多源静态离散数据转化为连续的<b>拟动态（Quasi-dynamic）</b>演化曲线，为后续微分方程的构建提供底层数学基础。</div>
            </div>
        </a >
        
        <a href="FunClu" target="_self" class="card-link">
            <div class='card'>
                <div class='card-title'>
                    <span style='display:flex; align-items:center; gap:6px;'>
                        <span class="material-symbols-rounded" style="color:#ea580c; font-size:1.4rem;">category</span> 
                        功能聚类 (FunClu)
                    </span> 
                    <span style='font-size:0.9rem;color:#3b82f6;'>进入 &rarr;</span>
                </div>
                <div class='card-desc'>运用函数型聚类降维，识别系统内部具有相似动力学轨迹的核心变量模块，极大降低高维全景网络重构的计算复杂度。</div>
            </div>
        </a >
        
        <a href="NetRecon" target="_self" class="card-link">
            <div class='card'>
                <div class='card-title'>
                    <span style='display:flex; align-items:center; gap:6px;'>
                        <span class="material-symbols-rounded" style="color:#7c3aed; font-size:1.4rem;">hub</span> 
                        idop网络重构 (NetRecon)
                    </span> 
                    <span style='font-size:0.9rem;color:#3b82f6;'>进入 &rarr;</span>
                </div>
                <div class='card-desc'>基于拟动态常微分方程（qdODEs），重构带符号的加权有向图，精准量化各变量之间的“促进”与“抑制”因果依赖关系。</div>
            </div>
        </a >
        
        <a href="NetAnal" target="_self" class="card-link">
            <div class='card'>
                <div class='card-title'>
                    <span style='display:flex; align-items:center; gap:6px;'>
                        <span class="material-symbols-rounded" style="color:#e11d48; font-size:1.4rem;">insights</span> 
                        idop网络解析 (NetAnal)
                    </span> 
                    <span style='font-size:0.9rem;color:#3b82f6;'>进入 &rarr;</span>
                </div>
                <div class='card-desc'>追踪网络拓扑结构（Topology）随时间与环境的时空演变规律，并支持提取单样本特异性网络，实现微观级别的个性化洞察。</div>
            </div>
        </a >
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("<h4 style='color:#0f172a;margin-bottom:1rem;font-size:1.3rem;'>代表性学术成果</h4>", unsafe_allow_html=True)
    
    # 这里就是你那长长的 46 篇论文列表
    papers_html = """
    <div class='scroll-box'>
        <div class="scroll-box">
    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2026</h4>
    <div class="paper-item"><div class="paper-title">📄 Graph statistics theory of individualized quantitative genetics under haplotype-resolved genome assembly.</div><div class="paper-authors">Sun, L., Bian, Y., Yang, D., et al. (<b>2026</b>). <i>Proceedings of the National Academy of Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.1073/pnas.2600004123" target="_blank">🔗 doi: 10.1073/pnas.2600004123</a></div></div>
    <div class="paper-item"><div class="paper-title">📄 A statistical mechanics model to decode tissue crosstalk during graft formation.</div><div class="paper-authors">Dong, A., Meng, Y., Yau, S. S.-T., et al. (<b>2026</b>). <i>Advanced Science</i>.<br><a class="paper-link" href="https://doi.org/10.1002/advs.202523373" target="_blank">🔗 doi: 10.1002/advs.202523373</a></div></div>
    <div class="paper-item"><div class="paper-title">📄 An omnigenic interactome model to chart the genetic architecture of individual plants.</div><div class="paper-authors">Fa, C., Wang, G., Pan, W., et al. (<b>2026</b>). <i>Horticulture Research</i>.<br><a class="paper-link" href="https://doi.org/10.1093/hr/uhaf345" target="_blank">🔗 doi: 10.1093/hr/uhaf345</a></div></div>
    <div class="paper-item"><div class="paper-title">📄 Network stress: A wiring diagram of whole stress genes.</div><div class="paper-authors">Wang, Y., & Wu, R. (<b>2026</b>). <i>Horticulture Research</i>.<br><a class="paper-link" href="https://doi.org/10.1093/hr/uhaf302" target="_blank">🔗 doi: 10.1093/hr/uhaf302</a></div></div>
    <div class="paper-item"><div class="paper-title">📄 Statistical learning of stochastic complex systems via the yau-yau nonlinear filter.</div><div class="paper-authors">Xu, S., Wang, Y., Wu, S., et al. (<b>2026</b>). <i>The Innovation</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.xinn.2026.101267" target="_blank">🔗 doi: 10.1016/j.xinn.2026.101267</a></div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2025</h4>
    <div class="paper-item"><div class="paper-title">📄 IdopNetwork as a genomic predictor of drug response.</div><div class="paper-authors">Che, J., Jin, Y., Gragnoli, C., et al. (<b>2025</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2024.104252" target="_blank">🔗 doi: 10.1016/j.drudis.2024.104252</a></div></div>
    <div class="paper-item"><div class="paper-title">📄 High-order interaction modeling of tumor-microenvironment crosstalk for tumor growth.</div><div class="paper-authors">Che, J., Wang, Y., Feng, L., et al. (<b>2025</b>). <i>Physics of Life Reviews</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.plrev.2025.05.007" target="_blank">🔗 doi: 10.1016/j.plrev.2025.05.007</div></div>
    <div class="paper-item"><div class="paper-title">📄 Network modeling and topology of aging.</div><div class="paper-authors">Feng, L., Yang, D., Wu, S., et al. (<b>2025</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2024.10.006" target="_blank">🔗 doi: 10.1016/j.physrep.2024.10.006</div></div>
    <div class="paper-item"><div class="paper-title">📄 idopNetwork analysis of salt-responsive transcriptomes reveals hub regulatory modules and genes in populus euphratica.</div><div class="paper-authors">Wu, S., Pan, W., & Dong, A. (<b>2025</b>). <i>International Journal of Molecular Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.3390/ijms26094091" target="_blank">🔗 doi: 10.3390/ijms26094091</div></div>
    <div class="paper-item"><div class="paper-title">📄 Disentangling complex systems: IdopNetwork meets GLMY homology theory.</div><div class="paper-authors">Wu, S., & Zhang, M. (<b>2025</b>). <i>Data Analytics and Topology</i>.</div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2024</h4>
    <div class="paper-item"><div class="paper-title">📄 Hypernetwork modeling and topology of high-order interactions for complex systems.</div><div class="paper-authors">Feng, L., Gong, H., Zhang, S., et al. (<b>2024</b>). <i>Proceedings of the National Academy of Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.1073/pnas.2412220121" target="_blank">🔗 doi: 10.1073/pnas.2412220121</div></div>
    <div class="paper-item"><div class="paper-title">📄 Topological change of soil microbiota networks for forest resilience under global warming.</div><div class="paper-authors">Gong, H., Wang, H., Wang, Y., et al. (<b>2024</b>). <i>Physics of Life Reviews</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.plrev.2024.08.001" target="_blank">🔗 doi: 10.1016/j.plrev.2024.08.001</div></div>
    <div class="paper-item"><div class="paper-title">📄 Genome-wide network analysis of above- and below-ground Co-growth in populus euphratica.</div><div class="paper-authors">Lu, K., Gong, H., Yang, D., et al. (<b>2024</b>). <i>Plant Phenomics</i>.<br><a class="paper-link" href="https://doi.org/10.34133/plantphenomics.0131" target="_blank">🔗 doi: 10.34133/plantphenomics.0131</div></div>
    <div class="paper-item"><div class="paper-title">📄 Mapping the influence of light intensity on the transgenerational genetic architecture of arabidopsis thaliana.</div><div class="paper-authors">Mei, J., Che, J., Shi, Y., et al. (<b>2024</b>). <i>Current Issues in Molecular Biology</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cimb46080482" target="_blank">🔗 doi: 10.3390/cimb46080482</div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2023</h4>
    <div class="paper-item"><div class="paper-title">📄 idopNetwork: A network tool to dissect spatial community ecology.</div><div class="paper-authors">Dong, A., Wu, S., Che, J., et al. (<b>2023</b>). <i>Methods in Ecology and Evolution</i>.<br><a class="paper-link" href="https://doi.org/10.1111/2041-210X.14172" target="_blank">🔗 doi: 10.1111/2041-210X.14172</div></div>
    <div class="paper-item"><div class="paper-title">📄 A personalized pharmaco-epistatic network model of precision medicine.</div><div class="paper-authors">Feng, L., Yang, W., Ding, M., et al. (<b>2023</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2023.103608" target="_blank">🔗 doi: 10.1016/j.drudis.2023.103608</div></div>
    <div class="paper-item"><div class="paper-title">📄 Competition-cooperation mechanism between escherichia coli and staphylococcus aureus based on systems mapping.</div><div class="paper-authors">Li, C., Yin, L., He, X., et al. (<b>2023</b>). <i>Frontiers in Microbiology</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fmicb.2023.1192574" target="_blank">🔗 doi: 10.3389/fmicb.2023.1192574</div></div>
    <div class="paper-item"><div class="paper-title">📄 The genetic architecture of trait covariation in populus euphratica, a desert tree.</div><div class="paper-authors">Lu, K., Wang, X., Gong, H., et al. (<b>2023</b>). <i>Frontiers in Plant Science</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fpls.2023.1149879" target="_blank">🔗 doi: 10.3389/fpls.2023.1149879</div></div>
    <div class="paper-item"><div class="paper-title">📄 The genomic physics of tumor–microenvironment crosstalk.</div><div class="paper-authors">Sang, M., Feng, L., Dong, A., et al. (<b>2023</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2023.07.006" target="_blank">🔗 doi: 10.1016/j.physrep.2023.07.006</div></div>
    <div class="paper-item"><div class="paper-title">📄 A pleiotropic–epistatic entangelement model of drug response.</div><div class="paper-authors">Wang, Y., Sang, M., Feng, L., et al. (<b>2023</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2023.103790" target="_blank">🔗 doi: 10.1016/j.drudis.2023.103790</div></div>
    <div class="paper-item"><div class="paper-title">📄 The metabolomic physics of complex diseases.</div><div class="paper-authors">Wu, S., Liu, X., Dong, A., et al. (<b>2023</b>). <i>Proceedings of the National Academy of Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.1073/pnas.2308496120" target="_blank">🔗 doi: 10.1073/pnas.2308496120</div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2022</h4>
    <div class="paper-item"><div class="paper-title">📄 Modeling spatial interaction networks of the gut microbiota.</div><div class="paper-authors">Cao, X., Dong, A., Kang, G., et al. (<b>2022</b>). <i>Gut Microbes</i>.<br><a class="paper-link" href="https://doi.org/10.1080/19490976.2022.2106103" target="_blank">🔗 doi: 10.1080/19490976.2022.2106103</div></div>
    <div class="paper-item"><div class="paper-title">📄 An eco-evo-devo genetic network model of stress response.</div><div class="paper-authors">Feng, L., Dong, T., Jiang, P., et al. (<b>2022</b>). <i>Horticulture Research</i>.<br><a class="paper-link" href="https://doi.org/10.1093/hr/uhac135" target="_blank">🔗 doi: 10.1093/hr/uhac135</div></div>
    <div class="paper-item"><div class="paper-title">📄 Disentangling leaf-microbiome interactions in arabidopsis thaliana by network mapping.</div><div class="paper-authors">Li, K., Cheng, K., Wang, H., et al. (<b>2022</b>). <i>Frontiers in Plant Science</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fpls.2022.996121" target="_blank">🔗 doi: 10.3389/fpls.2022.996121</div></div>
    <div class="paper-item"><div class="paper-title">📄 A graph model of combination therapies.</div><div class="paper-authors">Sang, M., Dong, A., Wu, S., et al. (<b>2022</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2022.02.001" target="_blank">🔗 doi: 10.1016/j.drudis.2022.02.001</div></div>
    <div class="paper-item"><div class="paper-title">📄 A single-cell omics network model of cell crosstalk during the formation of primordial follicles.</div><div class="paper-authors">Wang, Q., Dong, A., Jiang, L., et al. (<b>2022</b>). <i>Cells</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cells11030332" target="_blank">🔗 doi: 10.3390/cells11030332</div></div>
    <div class="paper-item"><div class="paper-title">📄 Vaginal microbiota networks as a mechanistic predictor of aerobic vaginitis.</div><div class="paper-authors">Wang, Q., Dong, A., Zhao, J., et al. (<b>2022</b>). <i>Frontiers in Microbiology</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fmicb.2022.998813" target="_blank">🔗 doi: 10.3389/fmicb.2022.998813</div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2021</h4>
    <div class="paper-item"><div class="paper-title">📄 FunGraph: A statistical protocol to reconstruct omnigenic multilayer interactome networks for complex traits.</div><div class="paper-authors">Dong, A., Feng, L., Yang, D., et al. (<b>2021</b>). <i>STAR Protocols</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.xpro.2021.100985" target="_blank">🔗 doi: 10.1016/j.xpro.2021.100985</div></div>
    <div class="paper-item"><div class="paper-title">📄 The genomic physics of COVID-19 pathogenesis and spread.</div><div class="paper-authors">Dong, A., Zhao, J., Griffin, C., & Wu, R. (<b>2021</b>). <i>Cells</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cells11010080" target="_blank">🔗 doi: 10.3390/cells11010080</div></div>
    <div class="paper-item"><div class="paper-title">📄 Genetic dissection of growth trajectories in forest trees: From FunMap to FunGraph.</div><div class="paper-authors">Feng, L., Jiang, P., Li, C., et al. (<b>2021</b>). <i>Forestry Research</i>.<br><a class="paper-link" href="https://doi.org/10.48130/FR-2021-0019" target="_blank">🔗 doi: 10.48130/FR-2021-0019</div></div>
    <div class="paper-item"><div class="paper-title">📄 Genetic architecture of multiphasic growth covariation as revealed by a nonlinear mixed mapping framework.</div><div class="paper-authors">Gong, H., Zhang, X.-Y., Zhu, S., et al. (<b>2021</b>). <i>Frontiers in Plant Science</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fpls.2021.711219" target="_blank">🔗 doi: 10.3389/fpls.2021.711219</div></div>
    <div class="paper-item"><div class="paper-title">📄 A multilayer interactome network constructed in a forest poplar population mediates the pleiotropic control of complex traits.</div><div class="paper-authors">Gong, H., Zhu, S., Zhu, X., et al. (<b>2021</b>). <i>Frontiers in Genetics</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fgene.2021.769688" target="_blank">🔗 doi: 10.3389/fgene.2021.769688</div></div>
    <div class="paper-item"><div class="paper-title">📄 Network mapping of root–microbe interactions in arabidopsis thaliana.</div><div class="paper-authors">He, X., Zhang, Q., Li, B., et al. (<b>2021</b>). <i>Npj Biofilms and Microbiomes</i>.<br><a class="paper-link" href="https://doi.org/10.1038/s41522-021-00241-4" target="_blank">🔗 doi: 10.1038/s41522-021-00241-4</div></div>
    <div class="paper-item"><div class="paper-title">📄 A behavioral model for mapping the genetic architecture of gut-microbiota networks.</div><div class="paper-authors">Jiang, L., Liu, X., He, X., et al. (<b>2021</b>). <i>Gut Microbes</i>.<br><a class="paper-link" href="https://doi.org/10.1080/19490976.2020.1820847" target="_blank">🔗 doi: 10.1080/19490976.2020.1820847</div></div>
    <div class="paper-item"><div class="paper-title">📄 Adaptive sparse group LASSO in quantile regression.</div><div class="paper-authors">Mendez-Civieta, A., Aguilera-Morillo, M. C., & Lillo, R. E. (<b>2021</b>). <i>Advances in Data Analysis and Classification</i>.<br><a class="paper-link" href="https://doi.org/10.1007/s11634-020-00413-8" target="_blank">🔗 doi: 10.1007/s11634-020-00413-8</div></div>
    <div class="paper-item"><div class="paper-title">📄 Statistical mechanics of clock gene networks underlying circadian rhythms.</div><div class="paper-authors">Sun, L., Dong, A., Griffin, C., & Wu, R. (<b>2021</b>). <i>Applied Physics Reviews</i>.<br><a class="paper-link" href="https://doi.org/10.1063/5.0029993" target="_blank">🔗 doi: 10.1063/5.0029993</div></div>
    <div class="paper-item"><div class="paper-title">📄 Modeling genome-wide by environment interactions through omnigenic interactome networks.</div><div class="paper-authors">Wang, H., Ye, M., Fu, Y., et al. (<b>2021</b>). <i>Cell Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.celrep.2021.109114" target="_blank">🔗 doi: 10.1016/j.celrep.2021.109114</div></div>
    <div class="paper-item"><div class="paper-title">📄 Recovering dynamic networks in big static datasets.</div><div class="paper-authors">Wu, R., & Jiang, L. (<b>2021</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2021.01.003" target="_blank">🔗 doi: 10.1016/j.physrep.2021.01.003</div></div>
    <div class="paper-item"><div class="paper-title">📄 Inferring multilayer interactome networks shaping phenotypic plasticity and evolution.</div><div class="paper-authors">Yang, D., Jin, Y., He, X., et al. (<b>2021</b>). <i>Nature Communications</i>.<br><a class="paper-link" href="https://doi.org/10.1038/s41467-021-25086-5" target="_blank">🔗 doi: 10.1038/s41467-021-25086-5</div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2020</h4>
    <div class="paper-item"><div class="paper-title">📄 SEGN: Inferring real-time gene networks mediating phenotypic plasticity.</div><div class="paper-authors">Jiang, L., Griffin, C. H., & Wu, R. (<b>2020</b>). <i>Computational and Structural Biotechnology Journal</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.csbj.2020.08.029" target="_blank">🔗 doi: 10.1016/j.csbj.2020.08.029</div></div>
    <div class="paper-item"><div class="paper-title">📄 Computational identification of gene networks as a biomarker of neuroblastoma risk.</div><div class="paper-authors">Sun, L., Jiang, L., Grant, C. N., et al. (<b>2020</b>). <i>Cancers</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cancers12082086" target="_blank">🔗 doi: 10.3390/cancers12082086</div></div>
    <div class="paper-item"><div class="paper-title">📄 An omnidirectional visualization model of personalized gene regulatory networks.</div><div class="paper-authors">Chen, C., Jiang, L., Fu, G., et al. (<b>2020</b>). <i>Npj Systems Biology and Applications</i>.<br><a class="paper-link" href="https://doi.org/10.1038/s41540-019-0116-1" target="_blank">🔗 doi: 10.1038/s41540-019-0116-1</div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2019</h4>
    <div class="paper-item"><div class="paper-title">📄 A drive to driven model of mapping intraspecific interaction networks.</div><div class="paper-authors">Jiang, L., Xu, J., Sang, M., et al. (<b>2019</b>). <i>iScience</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.isci.2019.11.002" target="_blank">🔗 doi: 10.1016/j.isci.2019.11.002</div></div>
    <div class="paper-item"><div class="paper-title">📄 Interrogation of internal workings in microbial community assembly: Play a game through a behavioral network?</div><div class="paper-authors">Wang, Q., Liu, X., Jiang, L., et al. (<b>2019</b>). <i>mSystems</i>.<br><a class="paper-link" href="https://doi.org/10.1128/mSystems.00550-19" target="_blank">🔗 doi: 10.1128/mSystems.00550-19</div></div>
    <div class="paper-item"><div class="paper-title">📄 Complex network approaches to nonlinear time series analysis.</div><div class="paper-authors">Zou, Y., Donner, R. V., Marwan, N., et al. (<b>2019</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2018.10.005" target="_blank">🔗 doi: 10.1016/j.physrep.2018.10.005</div></div>
    <div class="paper-item"><div class="paper-title">📄 A computational-experimental framework for mapping plant coexistence.</div><div class="paper-authors">Jiang, L., Shi, C., Ye, M., et al. (<b>2019</b>). <i>Methods in Ecology and Evolution</i>.<br><a class="paper-link" href="https://doi.org/10.1111/2041-210X.12981" target="_blank">🔗 doi: 10.1111/2041-210X.12981</div></div>

    <h4 style="margin:0.6rem 0 0.3rem 0; color:#2563eb; border-bottom:1px solid #e2e8f0; padding-bottom:0.3rem;">2018</h4>
    <div class="paper-item"><div class="paper-title">📄 A computational-experimental framework for mapping plant coexistence.</div><div class="paper-authors">Jiang, L., Shi, C., Ye, M., et al. (<b>2018</b>). <i>Methods in Ecology and Evolution</i>.<br><a class="paper-link" href="https://doi.org/10.1111/2041-210X.12981" target="_blank">🔗 doi: 10.1111/2041-210X.12981</div></div>
    </div>
    
    """
    st.markdown(papers_html, unsafe_allow_html=True)

# --- 原有的底部版权 ---
st.markdown(
    """<div style="text-align:center; color:#94a3b8; font-size:0.9rem; margin: 3rem 0;">
        复杂系统拓扑统计理论及应用北京市重点实验室 
               北京雁栖湖应用数学研究院 
                 idopNetwork v2.0
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