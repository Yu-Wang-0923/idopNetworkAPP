

import streamlit as st
from html import escape
from importlib.resources import files
from idopnetwork_app.utils import HEADER_TOGGLE_KEY, _is_admin_user, load_css, setup_sidebar
from idopnetwork_app.auth import show_auth_modal

_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")
_WU = str(files("idopnetwork_app.static.images") / "wu.jpg")

# ==========================================
# 1. 基础设置 (必须在第一行)
# ==========================================
st.set_page_config(
    page_title="idopNetwork",
    page_icon=_ICON,
    layout="wide",
    initial_sidebar_state="auto"
)

# 加载全局样式和侧边栏
load_css()
setup_sidebar()

logged_in = st.session_state.get("logged_in", False)
top_padding = "4rem" if st.session_state.get(HEADER_TOGGLE_KEY) else "1rem"
home_css = (files("idopnetwork_app.static") / "css" / "home.css").read_text(encoding="utf-8")
st.html(f"<style>:root {{ --home-top: {top_padding}; }}{home_css}</style>")

# --- 首屏横幅区 ---
col_hero1, col_hero2 = st.container(key="home-hero").columns([2.2, 1], gap="medium", vertical_alignment="center")
with col_hero1:
    greeting = f"欢迎回来，{escape(str(st.session_state['current_user']))}" if logged_in else "复杂系统在线计算平台"
    st.markdown(f"<p class='home-greeting'>{greeting}</p><h1 class='home-title'>idopNetwork</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p class='home-intro'>
        idopNetwork 是一套基于 <b>邬荣领教授</b> 提出的统计力学框架构建的复杂系统分析体系。
        通过将静态的观测数据映射到高维拓扑空间，我们能够突破传统统计学的局限，实现对系统内部运作机理的深度还原。
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="idop-badges">
        <div class="badge-item" title="将复杂数据转化为可解释的生物/物理逻辑">Informative</div>
        <div class="badge-item" title="刻画系统随时间与环境变化的演化轨迹">Dynamic</div>
        <div class="badge-item" title="覆盖从分子到表型的全尺度互作网络">Omnidirectional</div>
        <div class="badge-item" title="生成针对单样本或个体的特异性精准分析">Personalized</div>
    </div>
    """, unsafe_allow_html=True)

    if not logged_in and st.button("登录 / 注册", type="primary"):
        show_auth_modal()

with col_hero2.container(key="home-profile"):
    st.image(_WU, width="stretch")
    st.markdown("""
    <div class="profile-details">
    <p><strong>邬荣领 教授</strong></p>
    <p>BIMSA 副院长 / 首席科学家</p>
    <a href="https://www.bimsa.cn/zh-CN/detail/ronglingwu.html"
       target="_blank" rel="noopener noreferrer" class="profile-btn">查看个人主页 &rarr;</a>
    </div>
    """, unsafe_allow_html=True)


if not logged_in:
    st.stop()

# --- 核心内容区 ---
col_left, col_right = st.columns([1.15, 1], gap="large")

with col_left:
    st.markdown("<h2 class='home-section-title'>核心计算模块</h2>", unsafe_allow_html=True)
    modules = [
        ("pages/1_Curve Fitting.py", "曲线拟合 (Curve Fitting)", "timeline",
         "基于生态位理论与异速生长定律，实现静态数据的拟动态幂律曲线转化，为后续微分方程的构建提供底层数学基础。"),
        ("pages/2_FunClu.py", "功能聚类 (FunClu)", "category",
         "通过多元高斯混合模型实现功能模块聚类，降低高维网络解析复杂度。"),
        ("pages/3_NetRecon.py", "idop网络重构 (NetRecon)", "hub",
         "基于进化博弈论构建微分方程，通过带约束多任务稀疏回归，求解变量自身表达与交互效应，构建动态个性化复杂网络。"),
        ("pages/4_NetAnal.py", "idop网络解析 (NetAnal)", "insights",
         "采用 GLMY 拓扑同调方法解析动态网络高阶结构，也可通过网络深度与“网络中位数”理论，刻画群体共性拓扑特征。"),
    ]
    for page, label, icon, description in modules:
        with st.container(key=f"home-module-{icon}", border=True):
            heading, entry = st.columns([4, 1], vertical_alignment="center")
            heading.markdown(f"**:material/{icon}: {label}**")
            # Native page navigation preserves the authenticated session.
            entry.page_link(page, label="进入 →", use_container_width=True)
            st.caption(description)

with col_right:
    st.markdown("<h2 class='home-section-title'>代表性学术成果</h2>", unsafe_allow_html=True)
    
    # 这里就是你那长长的 46 篇论文列表
    papers_html = """
    <div class='scroll-box' role='region' aria-label='按年份浏览学术成果' tabindex='0'>

    <h3>2026</h3>
    <div class="paper-item"><div class="paper-title">Graph statistics theory of individualized quantitative genetics under haplotype-resolved genome assembly.</div><div class="paper-authors">Sun, L., Bian, Y., Yang, D., et al. (<b>2026</b>). <i>Proceedings of the National Academy of Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.1073/pnas.2600004123" target="_blank">DOI 10.1073/pnas.2600004123</a></div></div>
    <div class="paper-item"><div class="paper-title">A statistical mechanics model to decode tissue crosstalk during graft formation.</div><div class="paper-authors">Dong, A., Meng, Y., Yau, S. S.-T., et al. (<b>2026</b>). <i>Advanced Science</i>.<br><a class="paper-link" href="https://doi.org/10.1002/advs.202523373" target="_blank">DOI 10.1002/advs.202523373</a></div></div>
    <div class="paper-item"><div class="paper-title">An omnigenic interactome model to chart the genetic architecture of individual plants.</div><div class="paper-authors">Fa, C., Wang, G., Pan, W., et al. (<b>2026</b>). <i>Horticulture Research</i>.<br><a class="paper-link" href="https://doi.org/10.1093/hr/uhaf345" target="_blank">DOI 10.1093/hr/uhaf345</a></div></div>
    <div class="paper-item"><div class="paper-title">Network stress: A wiring diagram of whole stress genes.</div><div class="paper-authors">Wang, Y., & Wu, R. (<b>2026</b>). <i>Horticulture Research</i>.<br><a class="paper-link" href="https://doi.org/10.1093/hr/uhaf302" target="_blank">DOI 10.1093/hr/uhaf302</a></div></div>
    <div class="paper-item"><div class="paper-title">Statistical learning of stochastic complex systems via the yau-yau nonlinear filter.</div><div class="paper-authors">Xu, S., Wang, Y., Wu, S., et al. (<b>2026</b>). <i>The Innovation</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.xinn.2026.101267" target="_blank">DOI 10.1016/j.xinn.2026.101267</a></div></div>

    <h3>2025</h3>
    <div class="paper-item"><div class="paper-title">IdopNetwork as a genomic predictor of drug response.</div><div class="paper-authors">Che, J., Jin, Y., Gragnoli, C., et al. (<b>2025</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2024.104252" target="_blank">DOI 10.1016/j.drudis.2024.104252</a></div></div>
    <div class="paper-item"><div class="paper-title">High-order interaction modeling of tumor-microenvironment crosstalk for tumor growth.</div><div class="paper-authors">Che, J., Wang, Y., Feng, L., et al. (<b>2025</b>). <i>Physics of Life Reviews</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.plrev.2025.05.007" target="_blank">DOI 10.1016/j.plrev.2025.05.007</a></div></div>
    <div class="paper-item"><div class="paper-title">Network modeling and topology of aging.</div><div class="paper-authors">Feng, L., Yang, D., Wu, S., et al. (<b>2025</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2024.10.006" target="_blank">DOI 10.1016/j.physrep.2024.10.006</a></div></div>
    <div class="paper-item"><div class="paper-title">idopNetwork analysis of salt-responsive transcriptomes reveals hub regulatory modules and genes in populus euphratica.</div><div class="paper-authors">Wu, S., Pan, W., & Dong, A. (<b>2025</b>). <i>International Journal of Molecular Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.3390/ijms26094091" target="_blank">DOI 10.3390/ijms26094091</a></div></div>
    <div class="paper-item"><div class="paper-title">Disentangling complex systems: IdopNetwork meets GLMY homology theory.</div><div class="paper-authors">Wu, S., & Zhang, M. (<b>2025</b>). <i>Data Analytics and Topology</i>.</div></div>

    <h3>2024</h3>
    <div class="paper-item"><div class="paper-title">Hypernetwork modeling and topology of high-order interactions for complex systems.</div><div class="paper-authors">Feng, L., Gong, H., Zhang, S., et al. (<b>2024</b>). <i>Proceedings of the National Academy of Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.1073/pnas.2412220121" target="_blank">DOI 10.1073/pnas.2412220121</a></div></div>
    <div class="paper-item"><div class="paper-title">Topological change of soil microbiota networks for forest resilience under global warming.</div><div class="paper-authors">Gong, H., Wang, H., Wang, Y., et al. (<b>2024</b>). <i>Physics of Life Reviews</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.plrev.2024.08.001" target="_blank">DOI 10.1016/j.plrev.2024.08.001</a></div></div>
    <div class="paper-item"><div class="paper-title">Genome-wide network analysis of above- and below-ground Co-growth in populus euphratica.</div><div class="paper-authors">Lu, K., Gong, H., Yang, D., et al. (<b>2024</b>). <i>Plant Phenomics</i>.<br><a class="paper-link" href="https://doi.org/10.34133/plantphenomics.0131" target="_blank">DOI 10.34133/plantphenomics.0131</a></div></div>
    <div class="paper-item"><div class="paper-title">Mapping the influence of light intensity on the transgenerational genetic architecture of arabidopsis thaliana.</div><div class="paper-authors">Mei, J., Che, J., Shi, Y., et al. (<b>2024</b>). <i>Current Issues in Molecular Biology</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cimb46080482" target="_blank">DOI 10.3390/cimb46080482</a></div></div>

    <h3>2023</h3>
    <div class="paper-item"><div class="paper-title">idopNetwork: A network tool to dissect spatial community ecology.</div><div class="paper-authors">Dong, A., Wu, S., Che, J., et al. (<b>2023</b>). <i>Methods in Ecology and Evolution</i>.<br><a class="paper-link" href="https://doi.org/10.1111/2041-210X.14172" target="_blank">DOI 10.1111/2041-210X.14172</a></div></div>
    <div class="paper-item"><div class="paper-title">A personalized pharmaco-epistatic network model of precision medicine.</div><div class="paper-authors">Feng, L., Yang, W., Ding, M., et al. (<b>2023</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2023.103608" target="_blank">DOI 10.1016/j.drudis.2023.103608</a></div></div>
    <div class="paper-item"><div class="paper-title">Competition-cooperation mechanism between escherichia coli and staphylococcus aureus based on systems mapping.</div><div class="paper-authors">Li, C., Yin, L., He, X., et al. (<b>2023</b>). <i>Frontiers in Microbiology</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fmicb.2023.1192574" target="_blank">DOI 10.3389/fmicb.2023.1192574</a></div></div>
    <div class="paper-item"><div class="paper-title">The genetic architecture of trait covariation in populus euphratica, a desert tree.</div><div class="paper-authors">Lu, K., Wang, X., Gong, H., et al. (<b>2023</b>). <i>Frontiers in Plant Science</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fpls.2023.1149879" target="_blank">DOI 10.3389/fpls.2023.1149879</a></div></div>
    <div class="paper-item"><div class="paper-title">The genomic physics of tumor–microenvironment crosstalk.</div><div class="paper-authors">Sang, M., Feng, L., Dong, A., et al. (<b>2023</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2023.07.006" target="_blank">DOI 10.1016/j.physrep.2023.07.006</a></div></div>
    <div class="paper-item"><div class="paper-title">A pleiotropic–epistatic entangelement model of drug response.</div><div class="paper-authors">Wang, Y., Sang, M., Feng, L., et al. (<b>2023</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2023.103790" target="_blank">DOI 10.1016/j.drudis.2023.103790</a></div></div>
    <div class="paper-item"><div class="paper-title">The metabolomic physics of complex diseases.</div><div class="paper-authors">Wu, S., Liu, X., Dong, A., et al. (<b>2023</b>). <i>Proceedings of the National Academy of Sciences</i>.<br><a class="paper-link" href="https://doi.org/10.1073/pnas.2308496120" target="_blank">DOI 10.1073/pnas.2308496120</a></div></div>

    <h3>2022</h3>
    <div class="paper-item"><div class="paper-title">Modeling spatial interaction networks of the gut microbiota.</div><div class="paper-authors">Cao, X., Dong, A., Kang, G., et al. (<b>2022</b>). <i>Gut Microbes</i>.<br><a class="paper-link" href="https://doi.org/10.1080/19490976.2022.2106103" target="_blank">DOI 10.1080/19490976.2022.2106103</a></div></div>
    <div class="paper-item"><div class="paper-title">An eco-evo-devo genetic network model of stress response.</div><div class="paper-authors">Feng, L., Dong, T., Jiang, P., et al. (<b>2022</b>). <i>Horticulture Research</i>.<br><a class="paper-link" href="https://doi.org/10.1093/hr/uhac135" target="_blank">DOI 10.1093/hr/uhac135</a></div></div>
    <div class="paper-item"><div class="paper-title">Disentangling leaf-microbiome interactions in arabidopsis thaliana by network mapping.</div><div class="paper-authors">Li, K., Cheng, K., Wang, H., et al. (<b>2022</b>). <i>Frontiers in Plant Science</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fpls.2022.996121" target="_blank">DOI 10.3389/fpls.2022.996121</a></div></div>
    <div class="paper-item"><div class="paper-title">A graph model of combination therapies.</div><div class="paper-authors">Sang, M., Dong, A., Wu, S., et al. (<b>2022</b>). <i>Drug Discovery Today</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.drudis.2022.02.001" target="_blank">DOI 10.1016/j.drudis.2022.02.001</a></div></div>
    <div class="paper-item"><div class="paper-title">A single-cell omics network model of cell crosstalk during the formation of primordial follicles.</div><div class="paper-authors">Wang, Q., Dong, A., Jiang, L., et al. (<b>2022</b>). <i>Cells</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cells11030332" target="_blank">DOI 10.3390/cells11030332</a></div></div>
    <div class="paper-item"><div class="paper-title">Vaginal microbiota networks as a mechanistic predictor of aerobic vaginitis.</div><div class="paper-authors">Wang, Q., Dong, A., Zhao, J., et al. (<b>2022</b>). <i>Frontiers in Microbiology</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fmicb.2022.998813" target="_blank">DOI 10.3389/fmicb.2022.998813</a></div></div>

    <h3>2021</h3>
    <div class="paper-item"><div class="paper-title">FunGraph: A statistical protocol to reconstruct omnigenic multilayer interactome networks for complex traits.</div><div class="paper-authors">Dong, A., Feng, L., Yang, D., et al. (<b>2021</b>). <i>STAR Protocols</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.xpro.2021.100985" target="_blank">DOI 10.1016/j.xpro.2021.100985</a></div></div>
    <div class="paper-item"><div class="paper-title">The genomic physics of COVID-19 pathogenesis and spread.</div><div class="paper-authors">Dong, A., Zhao, J., Griffin, C., & Wu, R. (<b>2021</b>). <i>Cells</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cells11010080" target="_blank">DOI 10.3390/cells11010080</a></div></div>
    <div class="paper-item"><div class="paper-title">Genetic dissection of growth trajectories in forest trees: From FunMap to FunGraph.</div><div class="paper-authors">Feng, L., Jiang, P., Li, C., et al. (<b>2021</b>). <i>Forestry Research</i>.<br><a class="paper-link" href="https://doi.org/10.48130/FR-2021-0019" target="_blank">DOI 10.48130/FR-2021-0019</a></div></div>
    <div class="paper-item"><div class="paper-title">Genetic architecture of multiphasic growth covariation as revealed by a nonlinear mixed mapping framework.</div><div class="paper-authors">Gong, H., Zhang, X.-Y., Zhu, S., et al. (<b>2021</b>). <i>Frontiers in Plant Science</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fpls.2021.711219" target="_blank">DOI 10.3389/fpls.2021.711219</a></div></div>
    <div class="paper-item"><div class="paper-title">A multilayer interactome network constructed in a forest poplar population mediates the pleiotropic control of complex traits.</div><div class="paper-authors">Gong, H., Zhu, S., Zhu, X., et al. (<b>2021</b>). <i>Frontiers in Genetics</i>.<br><a class="paper-link" href="https://doi.org/10.3389/fgene.2021.769688" target="_blank">DOI 10.3389/fgene.2021.769688</a></div></div>
    <div class="paper-item"><div class="paper-title">Network mapping of root–microbe interactions in arabidopsis thaliana.</div><div class="paper-authors">He, X., Zhang, Q., Li, B., et al. (<b>2021</b>). <i>Npj Biofilms and Microbiomes</i>.<br><a class="paper-link" href="https://doi.org/10.1038/s41522-021-00241-4" target="_blank">DOI 10.1038/s41522-021-00241-4</a></div></div>
    <div class="paper-item"><div class="paper-title">A behavioral model for mapping the genetic architecture of gut-microbiota networks.</div><div class="paper-authors">Jiang, L., Liu, X., He, X., et al. (<b>2021</b>). <i>Gut Microbes</i>.<br><a class="paper-link" href="https://doi.org/10.1080/19490976.2020.1820847" target="_blank">DOI 10.1080/19490976.2020.1820847</a></div></div>
    <div class="paper-item"><div class="paper-title">Adaptive sparse group LASSO in quantile regression.</div><div class="paper-authors">Mendez-Civieta, A., Aguilera-Morillo, M. C., & Lillo, R. E. (<b>2021</b>). <i>Advances in Data Analysis and Classification</i>.<br><a class="paper-link" href="https://doi.org/10.1007/s11634-020-00413-8" target="_blank">DOI 10.1007/s11634-020-00413-8</a></div></div>
    <div class="paper-item"><div class="paper-title">Statistical mechanics of clock gene networks underlying circadian rhythms.</div><div class="paper-authors">Sun, L., Dong, A., Griffin, C., & Wu, R. (<b>2021</b>). <i>Applied Physics Reviews</i>.<br><a class="paper-link" href="https://doi.org/10.1063/5.0029993" target="_blank">DOI 10.1063/5.0029993</a></div></div>
    <div class="paper-item"><div class="paper-title">Modeling genome-wide by environment interactions through omnigenic interactome networks.</div><div class="paper-authors">Wang, H., Ye, M., Fu, Y., et al. (<b>2021</b>). <i>Cell Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.celrep.2021.109114" target="_blank">DOI 10.1016/j.celrep.2021.109114</a></div></div>
    <div class="paper-item"><div class="paper-title">Recovering dynamic networks in big static datasets.</div><div class="paper-authors">Wu, R., & Jiang, L. (<b>2021</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2021.01.003" target="_blank">DOI 10.1016/j.physrep.2021.01.003</a></div></div>
    <div class="paper-item"><div class="paper-title">Inferring multilayer interactome networks shaping phenotypic plasticity and evolution.</div><div class="paper-authors">Yang, D., Jin, Y., He, X., et al. (<b>2021</b>). <i>Nature Communications</i>.<br><a class="paper-link" href="https://doi.org/10.1038/s41467-021-25086-5" target="_blank">DOI 10.1038/s41467-021-25086-5</a></div></div>

    <h3>2020</h3>
    <div class="paper-item"><div class="paper-title">SEGN: Inferring real-time gene networks mediating phenotypic plasticity.</div><div class="paper-authors">Jiang, L., Griffin, C. H., & Wu, R. (<b>2020</b>). <i>Computational and Structural Biotechnology Journal</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.csbj.2020.08.029" target="_blank">DOI 10.1016/j.csbj.2020.08.029</a></div></div>
    <div class="paper-item"><div class="paper-title">Computational identification of gene networks as a biomarker of neuroblastoma risk.</div><div class="paper-authors">Sun, L., Jiang, L., Grant, C. N., et al. (<b>2020</b>). <i>Cancers</i>.<br><a class="paper-link" href="https://doi.org/10.3390/cancers12082086" target="_blank">DOI 10.3390/cancers12082086</a></div></div>
    <div class="paper-item"><div class="paper-title">An omnidirectional visualization model of personalized gene regulatory networks.</div><div class="paper-authors">Chen, C., Jiang, L., Fu, G., et al. (<b>2020</b>). <i>Npj Systems Biology and Applications</i>.<br><a class="paper-link" href="https://doi.org/10.1038/s41540-019-0116-1" target="_blank">DOI 10.1038/s41540-019-0116-1</a></div></div>

    <h3>2019</h3>
    <div class="paper-item"><div class="paper-title">A drive to driven model of mapping intraspecific interaction networks.</div><div class="paper-authors">Jiang, L., Xu, J., Sang, M., et al. (<b>2019</b>). <i>iScience</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.isci.2019.11.002" target="_blank">DOI 10.1016/j.isci.2019.11.002</a></div></div>
    <div class="paper-item"><div class="paper-title">Interrogation of internal workings in microbial community assembly: Play a game through a behavioral network?</div><div class="paper-authors">Wang, Q., Liu, X., Jiang, L., et al. (<b>2019</b>). <i>mSystems</i>.<br><a class="paper-link" href="https://doi.org/10.1128/mSystems.00550-19" target="_blank">DOI 10.1128/mSystems.00550-19</a></div></div>
    <div class="paper-item"><div class="paper-title">Complex network approaches to nonlinear time series analysis.</div><div class="paper-authors">Zou, Y., Donner, R. V., Marwan, N., et al. (<b>2019</b>). <i>Physics Reports</i>.<br><a class="paper-link" href="https://doi.org/10.1016/j.physrep.2018.10.005" target="_blank">DOI 10.1016/j.physrep.2018.10.005</a></div></div>
    <div class="paper-item"><div class="paper-title">A computational-experimental framework for mapping plant coexistence.</div><div class="paper-authors">Jiang, L., Shi, C., Ye, M., et al. (<b>2019</b>). <i>Methods in Ecology and Evolution</i>.<br><a class="paper-link" href="https://doi.org/10.1111/2041-210X.12981" target="_blank">DOI 10.1111/2041-210X.12981</a></div></div>

    <h3>2018</h3>
    <div class="paper-item"><div class="paper-title">A computational-experimental framework for mapping plant coexistence.</div><div class="paper-authors">Jiang, L., Shi, C., Ye, M., et al. (<b>2018</b>). <i>Methods in Ecology and Evolution</i>.<br><a class="paper-link" href="https://doi.org/10.1111/2041-210X.12981" target="_blank">DOI 10.1111/2041-210X.12981</a></div></div>
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
if _is_admin_user():
    st.divider()
    st.subheader("🛠️ 注册人员信息后台")
    from idopnetwork_app.auth import load_users
    import pandas as pd

    all_data = load_users()
    # 将复杂的 JSON 转换成整齐的表格
    df = pd.DataFrame.from_dict(all_data, orient='index')
    # 隐藏密码列，保护隐私
    if "password" in df.columns:
        df = df.drop(columns=["password"])
    st.dataframe(df, use_container_width=True)
