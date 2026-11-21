"""Two-language UI switch (中文 / English).

The choice lives in one segmented control in the sidebar and is mirrored into the
`lang` query parameter, so a shared URL keeps its language.

Module names (Curve Fitting / FunClu / NetRecon / NetAnal) are proper nouns and stay
untranslated; everything descriptive goes through `t()`.
"""
from __future__ import annotations

import streamlit as st

DEFAULT = "zh"
OPTIONS = {"zh": "中文", "en": "English"}

STRINGS: dict[str, dict[str, str]] = {
    "zh": {
        "sidebar.caption": "登录后可使用分析工具。",
        "sidebar.back_home": "返回首页登录",
        "sidebar.welcome": "欢迎，{user}",
        "sidebar.logout": "退出登录",
        "auth.login_required": "请先返回首页登录后使用。",
        "home.hero.suffix": "计算平台",
        "home.hero.intro": (
            "idopNetwork 是一套基于 <b>邬荣领教授</b> 提出的统计力学框架构建的复杂系统分析体系。"
            "通过将静态的观测数据映射到高维拓扑空间，我们能够突破传统统计学的局限，"
            "实现对系统内部运作机理的深度还原。"
        ),
        "home.section.modules": "核心计算模块",
        "home.section.papers": "代表性学术成果",
        "home.module.entry": "进入 →",
        "home.module.curve_fitting.title": "曲线拟合 (Curve Fitting)",
        "home.module.curve_fitting.desc": (
            "基于生态位理论与异速生长定律，实现静态数据的拟动态幂律曲线转化，"
            "为后续微分方程的构建提供底层数学基础。"
        ),
        "home.module.funclu.title": "功能聚类 (FunClu)",
        "home.module.funclu.desc": "通过多元高斯混合模型实现功能模块聚类，降低高维网络解析复杂度。",
        "home.module.netrecon.title": "idop网络重构 (NetRecon)",
        "home.module.netrecon.desc": (
            "基于进化博弈论构建微分方程，通过带约束多任务稀疏回归，求解变量自身表达与交互效应，"
            "构建动态个性化复杂网络。"
        ),
        "home.module.netanal.title": "idop网络解析 (NetAnal)",
        "home.module.netanal.desc": (
            "采用 GLMY 拓扑同调方法解析动态网络高阶结构，追踪连通分量、环、腔等拓扑特征随过滤阈值的变化。"
        ),
        "home.profile.role": "BIMSA 副院长 / 首席科学家",
        "home.profile.name": "邬荣领 教授",
        "home.profile.link": "查看个人主页",
        "home.login": "登录 / 注册",
        "home.admin.title": "🛠️ 注册人员信息后台",
        "home.footer": "复杂系统拓扑统计理论及应用北京市重点实验室 北京雁栖湖应用数学研究院 idopNetwork v2.0",
        "guidance.title": "**这一步需要 `{file}`**",
        "guidance.step2": "在该页下载 `{file}`",
        "guidance.step3": "回到本页，把它拖进上面的上传框",
        "guidance.go": "去 {label} →",
        "funclu.need.step1": "在 Curve Fitting 页上传观测数据，点击 Run Fitting 完成拟合",
        "netrecon.single.step1": "在 Curve Fitting 页上传观测数据，点击 Run Fitting 完成拟合",
        "netrecon.multi.step1": "在 FunClu 页的 FunClu-K 标签完成聚类并导出",
        "netanal.step1": "在 NetRecon 页完成网络构建（单层，或多层），两种导出都可上传",
        "fitting.intake.title": "导入观测数据",
        "fitting.example.expander": "下载示例数据与使用说明",
        "fitting.intake.caption": "支持多个 CSV 文件。第一列为样本或时间索引，其余列为分析特征。",
        "fitting.settings.title": "分析设置",
        "fitting.example.education": "教育数据已匿名化。解压后上传 CSV，选择 Quasi-dynamic 进行拟合。",
        "fitting.example.ecg": "12 导联、1000 个采样点。选择 Dynamic 去噪，建网时选择 idopECG(4)，采样率保持 100 Hz。",
        "fitting.mode.quasi": "静态数据 → 列变换 → 按行和排序成拟动态序列 → 幂律拟合。",
        "fitting.mode.dynamic": (
            "时间序列（如 12 导联 ECG）→ 逐通道 SWT 平稳小波软阈值去噪；"
            "去噪后的波形可直接用于建网页的「Fourier + Legendre（动态 idop）」算法。"
        ),
        "fitting.transforms.hint": (
            "变换按顺序执行，None 跳过该步。Z_min_add1 把每列抬到最小值为 1；"
            "行按 log1p 之和排序、过滤出正索引，再按比例截去前几行。"
        ),
    },
    "en": {
        "sidebar.caption": "Sign in to use the analysis tools.",
        "sidebar.back_home": "Back to Home to sign in",
        "sidebar.welcome": "Signed in as {user}",
        "sidebar.logout": "Sign out",
        "auth.login_required": "Please return to Home and sign in first.",
        "home.hero.suffix": "Computing Platform",
        "home.hero.intro": (
            "idopNetwork is a complex-systems framework built on the statistical mechanics proposed by "
            "<b>Prof. Rongling Wu</b>. Mapping static observations into a high-dimensional topological space "
            "lets us reconstruct the internal workings of a system beyond the limits of conventional statistics."
        ),
        "home.section.modules": "Core computing modules",
        "home.section.papers": "Selected publications",
        "home.module.entry": "Open →",
        "home.module.curve_fitting.title": "Curve Fitting",
        "home.module.curve_fitting.desc": (
            "Converts static data into quasi-dynamic power-law curves via niche theory and allometric growth, "
            "providing the mathematical base for the differential equations that follow."
        ),
        "home.module.funclu.title": "FunClu",
        "home.module.funclu.desc": (
            "Clusters traits into functional modules with a multivariate Gaussian mixture model, "
            "reducing the complexity of high-dimensional network reconstruction."
        ),
        "home.module.netrecon.title": "NetRecon",
        "home.module.netrecon.desc": (
            "Builds differential equations from evolutionary game theory and solves self- and interaction "
            "effects with constrained multi-task sparse regression, reconstructing a dynamic personalized network."
        ),
        "home.module.netanal.title": "NetAnal",
        "home.module.netanal.desc": (
            "Uses GLMY path homology to dissect the high-order structure of dynamic networks, tracking "
            "connected components, cycles and cavities across filtration thresholds."
        ),
        "home.profile.role": "Deputy Director & Chief Scientist, BIMSA",
        "home.profile.name": "Prof. Rongling Wu",
        "home.profile.link": "Personal page",
        "home.login": "Sign in / Register",
        "home.admin.title": "🛠️ Registered users",
        "home.footer": (
            "Beijing Key Laboratory of Topological Statistics and Applications for Complex Systems · "
            "Beijing Institute of Mathematical Sciences and Applications · idopNetwork v2.0"
        ),
        "guidance.title": "**This step needs `{file}`**",
        "guidance.step2": "Download `{file}` on that page",
        "guidance.step3": "Come back here and drop it into the upload box above",
        "guidance.go": "Go to {label} →",
        "funclu.need.step1": "Upload the observations on the Curve Fitting page and click Run Fitting",
        "netrecon.single.step1": "Upload the observations on the Curve Fitting page and click Run Fitting",
        "netrecon.multi.step1": "Cluster and export on the FunClu page, FunClu-K tab",
        "netanal.step1": "Build the network on NetRecon (single- or multi-layer); either export works",
        "fitting.intake.title": "Import observations",
        "fitting.example.expander": "Sample data & instructions",
        "fitting.intake.caption": "Multiple CSV files are supported. The first column is the sample or time index; the rest are features.",
        "fitting.settings.title": "Analysis settings",
        "fitting.example.education": "Anonymized education data. Unzip, upload the CSVs, and choose Quasi-dynamic.",
        "fitting.example.ecg": "12 leads, 1000 samples. Choose Dynamic for denoising and keep 100 Hz when building the network.",
        "fitting.mode.quasi": "Static data → column transforms → rows sorted into a quasi-dynamic sequence → power-law fit.",
        "fitting.mode.dynamic": (
            "Time series (e.g. a 12-lead ECG) → per-channel SWT soft-threshold denoising; "
            "the denoised traces feed the dynamic idop construction directly."
        ),
        "fitting.transforms.hint": (
            "Transforms run in order; None skips a step. Z_min_add1 shifts each column to a minimum of 1. "
            "Rows are sorted by log1p of their sums, filtered to positive indices, then trimmed."
        ),
    },
}


def language() -> str:
    """Current language code, seeded from the URL on first view and kept in session state."""
    if "lang" not in st.session_state:
        query = st.query_params.get("lang")
        st.session_state["lang"] = query if query in OPTIONS else DEFAULT
    return st.session_state["lang"]


def t(key: str, **kwargs: object) -> str:
    """Look up a UI string, falling back to the default language, then to the key."""
    lang = language()
    text = STRINGS.get(lang, {}).get(key) or STRINGS[DEFAULT].get(key, key)
    return text.format(**kwargs) if kwargs else text


def _apply_language() -> None:
    st.session_state["lang"] = st.session_state["lang_widget"]
    st.query_params["lang"] = st.session_state["lang"]


def language_switch() -> None:
    """One-click 中文 / English switch, shown at the top of the sidebar."""
    lang = language()
    st.segmented_control(
        "Language",
        options=list(OPTIONS),
        format_func=lambda code: OPTIONS[code],
        default=lang,
        key="lang_widget",
        required=True,
        label_visibility="collapsed",
        on_change=_apply_language,
    )
