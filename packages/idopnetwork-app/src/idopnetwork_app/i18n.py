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
        "sidebar.logo.alt": "idopNetwork 标识",
        "sidebar.back_home": "返回首页登录",
        "sidebar.welcome": "欢迎，{user}",
        "sidebar.logout": "退出登录",
        "auth.login_required": "请先返回首页登录后使用。",
        "home.hero.suffix": "计算平台",
        "home.hero.figure.alt": "多层网络示意图：M1、M2、M3 三层节点与有向边",
        "home.profile.photo.alt": "邬荣领教授肖像",
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
        "funclu.overview": "数据概览",
        "funclu.prepare_failed": "数据准备失败：{error}",
        "funclu.metric.conditions": "条件数",
        "funclu.metric.features": "共有特征数",
        "funclu.metric.timepoints": "时间点总数",
        "funclu.per_condition": "各条件汇总",
        "funclu.shared_features": "各条件共有特征（{n}）",
        "funclu.select_condition": "选择要预览的条件：",
        "funclu.preview_caption": "`{cond}` 的前 5 行 × 5 列：",
        "funclu.settings": "FunClu 参数设置",
        "funclu.run_em": "运行 EM 拟合",
        "funclu.em_failed": "EM 拟合失败：{error}",
        "funclu.em_done": "完成：converged={c}，iters={i}/{m}，log-lik={l}，BIC={b}",
        "funclu.cluster_sizes_weights": "**簇规模与权重（EM 后）**",
        "funclu.cluster_profiles": "**簇轮廓**",
        "funclu.plot_settings": "簇轮廓绘图设置",
        "funclu.profile_layout": "排布方式",
        "funclu.subplot_cols": "每行子图数",
        "funclu.subplot_cols_help": "combined 布局每行放几个子图；per_cluster 拼图时作为网格列数，行数按簇数自动推算。",
        "funclu.member_source": "成员数据来源",
        "funclu.member_source_help": "curve：每个成员一条细线（基于 curve_sample）；qd_df：每个时间点 × 每成员的散点（适合配 quasi_dynamic）。",
        "funclu.show_members": "显示成员曲线",
        "funclu.show_mean": "显示均值曲线",
        "funclu.show_ci": "显示置信带（±1.96 SE）",
        "funclu.log_y": "纵轴取对数",
        "funclu.log_x": "横轴取对数",
        "funclu.show_legend": "显示图例",
        "funclu.apply_plot": "应用绘图设置",
        "funclu.need_em": "请先在相邻标签页运行 EM 拟合。",
        "funclu.export_failed": "FunClu-K 导出失败：{error}",
        "funclu.export_contents": "**导出内容**",
        "funclu.export_list": "- `labels.csv`：特征到簇的标签。\n- `cluster_sizes.csv`：每个簇的特征数量。\n- `cluster_members/<condition>/M*_curve_sample.csv`：每个簇、每个条件的全部成员曲线。\n- `cluster_members/<condition>/M*_quasi_dynamic.csv`：供多层 NetRecon 使用的成员响应变量。\n- `cluster_centers/<condition>/cluster_center_curve_sample.csv`：由 `FunClu.get_cluster_curves()` 生成的簇中心曲线。\n- `cluster_centers/<condition>/cluster_center_quasi_dynamic.csv`：由成员 quasi_dynamic 聚合得到的簇级响应变量。",
        "funclu.download_zip": "下载 FunClu-K 导出 ZIP",
        "funclu.bic_settings": "BIC 扫描设置",
        "funclu.advanced": "高级设置",
        "funclu.help.scan": "从 2 扫到 int(sqrt(n))，寻找类似肘部的 BIC 局部极小值，取最左侧候选作为最优 K。",
        "funclu.help.seeds": "每个 K 用多个连续随机种子运行并汇总 BIC/SI，降低局部最优噪声。",
        "funclu.help.aggregation": "中位数对偶发失败或较差局部最优更稳健；均值对全部运行更敏感。",
        "funclu.help.min_size": "簇规模小于该阈值的 K 仍保留在表中，但会尽量排除在主 BIC 推荐之外。",
        "funclu.try_default": "尝试默认最优 K",
        "funclu.run_bic": "运行 BIC 分析",
        "funclu.no_elbow": "未找到肘部形态的 BIC 局部极小值，因此不会标记「尝试默认最优 K」。可再运行一次 BIC 分析。",
        "funclu.min_size_violated": "没有 K 满足最小簇规模约束，改为展示无约束的 BIC 最小值。",
        "funclu.bic_failed": "BIC 扫描失败：{error}",
        "funclu.best_k": "最优 K 建议",
        "funclu.metric.recommended": "推荐 K",
        "funclu.metric.bic": "BIC",
        "funclu.metric.raw_bic_k": "原始 BIC 的 K",
        "funclu.metric.si": "SI @ 推荐 K",
        "funclu.metric.min_size": "最小簇规模",
        "funclu.metric.converged": "是否收敛",
        "funclu.best_si_caption": "在合格行中按 SI 的最优 K：{label}（{value}）",
        "funclu.elbow_plot": "BIC 肘部图",
        "funclu.seed_note": "此结果与随机种子有关，不同种子可能得到不同结果。",
        "funclu.results_table": "结果表",
        "funclu.need_bic": "请先在相邻标签页运行 BIC 分析。",
        "funclu.export_csv": "导出 BIC 结果为 CSV",
        "funclu.download_bic": "下载 BIC 结果（CSV）",
        "funclu.tab.summary": "概览",
        "funclu.strip": "共 {n} 个条件 · {f} 个特征 · {tp} 个时间点",
        "funclu.em.empty": "运行后在此显示：聚类规模与权重、mu/cov 参数表、各条件中心曲线。",
        "funclu.bic.empty": "运行后在此显示：BIC 曲线、最优 K 与各 K 的拟合状态。",
        "funclu.status.title": "运行状态",
        "funclu.status.dataset": "数据集",
        "funclu.status.shape": "条件 × 特征",
        "funclu.status.k": "EM 聚类数",
        "funclu.status.bic": "BIC 最优 K",
        "funclu.status.none": "尚未运行",
        "funclu.metric.per_condition": "每条件：{v}",
        "funclu.tab.data": "数据",
        "funclu.upload.change": "更换文件",
        "funclu.tab.detail": "明细",
        "funclu.preview.title": "数据预览",
        "funclu.tab.features": "共有特征",
        "funclu.tab.preview": "数据预览",
        "funclu.tab.em": "EM 拟合",
        "funclu.tab.export": "导出",
        "funclu.tab.bic": "BIC 分析",
        "funclu.col.condition": "条件",
        "funclu.col.timepoints": "时间点",
        "funclu.col.shape": "形状（特征 × 时间）",
        "funclu.col.time_min": "时间最小值",
        "funclu.col.time_max": "时间最大值",
        "funclu.col.feature": "特征名",
        "guidance.title": "**这一步需要 `{file}`**",
        "guidance.step2": "在该页下载 `{file}`",
        "guidance.step3": "回到本页，把它拖进上面的上传框",
        "guidance.go": "去 {label} →",
        "funclu.tab.uploaded": "上传数据",
        "funclu.init.title": "数据初始化",
        "funclu.upload.label": "请上传文件",
        "funclu.upload.help": "上传 Curve Fitting 导出的 ZIP：Export Result → Export ZIP → curve_fitting_export.zip（单个文件）。",
        "funclu.upload.ok": "文件已载入。",
        "funclu.upload.failed": "读取 ZIP 失败：{error}",
        "funclu.need_upload": "请先在「上传数据」标签页上传数据。",
        "funclu.need.step1": "在 Curve Fitting 页上传观测数据，点击 Run Fitting 完成拟合",
        "netrecon.single.step1": "在 Curve Fitting 页上传观测数据，点击 Run Fitting 完成拟合",
        "netrecon.multi.step1": "在 FunClu 页的 FunClu-K 标签完成聚类并导出",
        "netrecon.multilayer_overview.tab": "多层网总图",
        "netrecon.multilayer_overview.caption": "每列对应一个条件，首行显示模块间网络，下方显示所选模块的内部网络。",
        "netrecon.multilayer_overview.conditions": "显示的条件",
        "netrecon.multilayer_overview.modules": "显示的模块",
        "netrecon.multilayer_overview.draw": "绘制多层网总图",
        "netrecon.multilayer_overview.rendering": "正在绘制多层网络总图…",
        "netrecon.multilayer_overview.download": "下载总图 SVG",
        "netrecon.multilayer_overview.no_results": "本次构网没有可绘制的网络。",
        "netrecon.multilayer_overview.failed": "绘制总图失败：{error}",
        "netanal.step1": "在 NetRecon 页完成网络构建（单层，或多层），两种导出都可上传",
        "fitting.intake.title": "导入观测数据",
        "fitting.example.expander": "下载示例数据与使用说明",
        "fitting.intake.caption": "支持多个 CSV 文件。第一列为样本或时间索引，其余列为分析特征。",
        "fitting.settings.title": "分析设置",
        "fitting.results.title": "分析结果",
        "fitting.example.education": "教育数据已匿名化。解压后上传 CSV，选择 Quasi-dynamic 进行拟合。",
        "fitting.example.ecg": "12 导联、1000 个采样点。选择 Dynamic 去噪，建网时选择 idopECG(4)，采样率保持 100 Hz。",
        "fitting.mode.quasi": "静态数据 → 列变换 → 按行和排序成拟动态序列 → 幂律拟合。",
        "fitting.mode.dynamic": (
            "时间序列（如 12 导联 ECG）→ 逐通道 SWT 平稳小波软阈值去噪；"
            "去噪后的波形可直接用于建网页的「Fourier + Legendre（动态 idop）」算法。"
        ),
        "fitting.dynamic.note": "Dynamic 模式下导出沿用同一套三表契约，但含义为：quasi_dynamic = 原始波形，curve_sample = SWT 去噪后波形，curve_params = 每个通道的小波/层数/sigma/阈值诊断。",
        "fitting.results.empty": "上传数据并点击 Run Fitting 后，结果会出现在这里。",
        "fitting.metrics.compact": "{fitted} 特征 · {rows} 行 · {samples} 个采样点",
        "fitting.status.quasi": "{files} 个文件 · 变换 {first} → {second} · {samples} 个采样点 · 去除前 {trim}%",
        "fitting.status.dynamic": "{files} 个文件 · SWT {wavelet}，{level} 层，α={alpha}",
        "fitting.results.for": "结果文件",
        "fitting.download.zip": "下载拟合结果 ZIP",
        "fitting.transforms.hint": (
            "变换按顺序执行，None 跳过该步。Z_min_add1 把每列抬到最小值为 1；"
            "行按 log1p 之和排序、过滤出正索引，再按比例截去前几行。"
        ),
    },
    "en": {
        "sidebar.caption": "Sign in to use the analysis tools.",
        "sidebar.logo.alt": "idopNetwork logo",
        "sidebar.back_home": "Back to Home to sign in",
        "sidebar.welcome": "Signed in as {user}",
        "sidebar.logout": "Sign out",
        "auth.login_required": "Please return to Home and sign in first.",
        "home.hero.suffix": "Computing Platform",
        "home.hero.figure.alt": "Multi-layer network diagram with M1, M2 and M3 nodes and directed edges",
        "home.profile.photo.alt": "Portrait of Prof. Rongling Wu",
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
        "funclu.overview": "Data Overview",
        "funclu.prepare_failed": "Data preparation failed: {error}",
        "funclu.metric.conditions": "Conditions",
        "funclu.metric.features": "Common Features",
        "funclu.metric.timepoints": "Time Points (Total)",
        "funclu.per_condition": "Per-Condition Summary",
        "funclu.shared_features": "Shared Features Across All Conditions ({n})",
        "funclu.select_condition": "Select Condition to Preview:",
        "funclu.preview_caption": "Top 5 rows × 5 cols for `{cond}`:",
        "funclu.settings": "FunClu parameter settings",
        "funclu.run_em": "Run EM Fitting",
        "funclu.em_failed": "EM fitting failed: {error}",
        "funclu.em_done": "Done. converged={c}, iters={i}/{m}, log-lik={l}, BIC={b}",
        "funclu.cluster_sizes_weights": "**Cluster sizes & weights (after EM)**",
        "funclu.cluster_profiles": "**Cluster profiles**",
        "funclu.plot_settings": "Cluster profiles plot settings",
        "funclu.profile_layout": "Profile layout",
        "funclu.subplot_cols": "Subplot cols",
        "funclu.subplot_cols_help": "combined: subplots per row; per_cluster: grid columns, rows derived from the cluster count.",
        "funclu.member_source": "Member source",
        "funclu.member_source_help": "curve: one thin line per member (curve_sample); qd_df: scatter per time point × member (good with quasi_dynamic).",
        "funclu.show_members": "Show members",
        "funclu.show_mean": "Show mean curve",
        "funclu.show_ci": "Show CI band (±1.96 SE)",
        "funclu.log_y": "log-scale Y",
        "funclu.log_x": "log-scale X",
        "funclu.show_legend": "Show legend",
        "funclu.apply_plot": "Apply plot settings",
        "funclu.need_em": "Please run EM Fitting in the adjacent tab first.",
        "funclu.export_failed": "FunClu-K export failed: {error}",
        "funclu.export_contents": "**Export contents**",
        "funclu.export_list": "- `labels.csv`: feature-to-cluster labels.\n- `cluster_sizes.csv`: number of features in each cluster.\n- `cluster_members/<condition>/M*_curve_sample.csv`: all member curves for each cluster and condition.\n- `cluster_members/<condition>/M*_quasi_dynamic.csv`: member response variables for multi-layer NetRecon.\n- `cluster_centers/<condition>/cluster_center_curve_sample.csv`: cluster-center curves generated by `FunClu.get_cluster_curves()`.\n- `cluster_centers/<condition>/cluster_center_quasi_dynamic.csv`: cluster-level response variables aggregated from member quasi_dynamic.",
        "funclu.download_zip": "Download FunClu-K export ZIP",
        "funclu.bic_settings": "BIC Scan Settings",
        "funclu.advanced": "Advanced Settings",
        "funclu.help.scan": "Scan K from 2 to int(sqrt(n)), find checkmark-like local BIC minima, and select the leftmost candidate as Best K.",
        "funclu.help.seeds": "Run each K with multiple consecutive random seeds and aggregate BIC/SI to reduce local-optimum noise.",
        "funclu.help.aggregation": "Median is more robust to occasional failed or poor local optima; mean is more sensitive to all runs.",
        "funclu.help.min_size": "K values with clusters smaller than this threshold are kept in the table but excluded from the primary BIC recommendation when possible.",
        "funclu.try_default": "Try Default Best K",
        "funclu.run_bic": "Run BIC Analysis",
        "funclu.no_elbow": "No elbow-like local BIC minimum was found; no Try Default best K will be marked. Try Run BIC Analysis function.",
        "funclu.min_size_violated": "No K satisfies the minimum cluster-size constraint; showing the unconstrained BIC minimum instead.",
        "funclu.bic_failed": "BIC scan failed: {error}",
        "funclu.best_k": "Best K Recommendation",
        "funclu.metric.recommended": "Recommended K",
        "funclu.metric.bic": "BIC",
        "funclu.metric.raw_bic_k": "Raw BIC K",
        "funclu.metric.si": "SI @ Rec K",
        "funclu.metric.min_size": "Min Size",
        "funclu.metric.converged": "Converged",
        "funclu.best_si_caption": "Best K by SI among eligible rows: {label} ({value})",
        "funclu.elbow_plot": "BIC Elbow Plot",
        "funclu.seed_note": "These results depend on the random seed; different seeds may differ.",
        "funclu.results_table": "Results Table",
        "funclu.need_bic": "Run BIC Analysis first in the adjacent tab.",
        "funclu.export_csv": "Export BIC Results as CSV",
        "funclu.download_bic": "Download BIC results (CSV)",
        "funclu.tab.summary": "Summary",
        "funclu.strip": "{n} conditions · {f} features · {tp} time points",
        "funclu.em.empty": "After a run: cluster sizes and weights, mu/cov tables, and centre curves per condition.",
        "funclu.bic.empty": "After a scan: the BIC curve, the best K, and per-K fit status.",
        "funclu.status.title": "Run status",
        "funclu.status.dataset": "Dataset",
        "funclu.status.shape": "Conditions × features",
        "funclu.status.k": "EM clusters",
        "funclu.status.bic": "BIC best K",
        "funclu.status.none": "Not run yet",
        "funclu.metric.per_condition": "Per condition: {v}",
        "funclu.tab.data": "Data",
        "funclu.upload.change": "Change file",
        "funclu.tab.detail": "Details",
        "funclu.preview.title": "Data preview",
        "funclu.tab.features": "Common Features",
        "funclu.tab.preview": "Data Preview",
        "funclu.tab.em": "EM Fitting",
        "funclu.tab.export": "Export",
        "funclu.tab.bic": "BIC Analysis",
        "funclu.col.condition": "Condition",
        "funclu.col.timepoints": "Time Points",
        "funclu.col.shape": "Shape (Features × Times)",
        "funclu.col.time_min": "Time Min",
        "funclu.col.time_max": "Time Max",
        "funclu.col.feature": "Feature Name",
        "guidance.title": "**This step needs `{file}`**",
        "guidance.step2": "Download `{file}` on that page",
        "guidance.step3": "Come back here and drop it into the upload box above",
        "guidance.go": "Go to {label} →",
        "funclu.tab.uploaded": "Uploaded Data",
        "funclu.init.title": "Data Initialization",
        "funclu.upload.label": "Please upload your file",
        "funclu.upload.help": "Upload the ZIP from Curve Fitting: Export Result → Export ZIP → curve_fitting_export.zip (one file).",
        "funclu.upload.ok": "File loaded.",
        "funclu.upload.failed": "Could not read the ZIP: {error}",
        "funclu.need_upload": "Please upload data in the 'Uploaded Data' tab first.",
        "funclu.need.step1": "Upload the observations on the Curve Fitting page and click Run Fitting",
        "netrecon.single.step1": "Upload the observations on the Curve Fitting page and click Run Fitting",
        "netrecon.multi.step1": "Cluster and export on the FunClu page, FunClu-K tab",
        "netrecon.multilayer_overview.tab": "Multilayer network overview",
        "netrecon.multilayer_overview.caption": "Columns compare conditions. The first row shows inter-module networks, followed by the selected module subnetworks.",
        "netrecon.multilayer_overview.conditions": "Conditions to show",
        "netrecon.multilayer_overview.modules": "Modules to show",
        "netrecon.multilayer_overview.draw": "Draw multilayer network overview",
        "netrecon.multilayer_overview.rendering": "Drawing the multilayer network overview…",
        "netrecon.multilayer_overview.download": "Download overview SVG",
        "netrecon.multilayer_overview.no_results": "This run produced no networks to plot.",
        "netrecon.multilayer_overview.failed": "Could not draw the overview: {error}",
        "netanal.step1": "Build the network on NetRecon (single- or multi-layer); either export works",
        "fitting.intake.title": "Import observations",
        "fitting.example.expander": "Sample data & instructions",
        "fitting.intake.caption": "Multiple CSV files are supported. The first column is the sample or time index; the rest are features.",
        "fitting.settings.title": "Analysis settings",
        "fitting.results.title": "Results",
        "fitting.example.education": "Anonymized education data. Unzip, upload the CSVs, and choose Quasi-dynamic.",
        "fitting.example.ecg": "12 leads, 1000 samples. Choose Dynamic for denoising and keep 100 Hz when building the network.",
        "fitting.mode.quasi": "Static data → column transforms → rows sorted into a quasi-dynamic sequence → power-law fit.",
        "fitting.mode.dynamic": (
            "Time series (e.g. a 12-lead ECG) → per-channel SWT soft-threshold denoising; "
            "the denoised traces feed the dynamic idop construction directly."
        ),
        "fitting.dynamic.note": "Dynamic mode keeps the same three-table export contract: quasi_dynamic = raw waveform, curve_sample = SWT-denoised waveform, curve_params = per-channel wavelet/level/sigma/threshold diagnostics.",
        "fitting.results.empty": "Results appear here after you upload data and click Run Fitting.",
        "fitting.metrics.compact": "{fitted} features · {rows} rows · {samples} samples",
        "fitting.status.quasi": "{files} file(s) · {first} → {second} · {samples} samples · first {trim}% removed",
        "fitting.status.dynamic": "{files} file(s) · SWT {wavelet}, {level} levels, α={alpha}",
        "fitting.results.for": "Results for",
        "fitting.download.zip": "Download fitting results ZIP",
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
