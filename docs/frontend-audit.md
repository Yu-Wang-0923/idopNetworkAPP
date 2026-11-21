# idopNetwork 前端审计（技术质量）

日期：2026-11-21 · 对象：`packages/idopnetwork-app/src/idopnetwork_app`（Home + 4 个功能页）
方法：impeccable `audit` + 检测器 `detect` + 无头 Edge 截图（1440×1800 / 390×1400）
证据：`scratch/shots/*.png`、`scratch/detect-ui.json`、`scratch/detect-url.json`

---

## 总分

| # | 维度 | 分数 | 关键发现 |
|---|------|:---:|----------|
| 1 | 可访问性 A11y | 2 | 3 处小字对比度低于 4.5:1；5 张图无 alt；标题层级从 h1 跳到 h3 |
| 2 | 性能 | 2 | 首屏图片约 11.6 MB（5.3 MB 肖像渲染到约 200px）；25 处已弃用 API，每次重跑刷多条警告 |
| 3 | 主题化 | 2 | 有 `:root` 变量，但两个 CSS 文件里仍有约 40 处硬编码颜色；无暗色模式方案 |
| 4 | 响应式 | 3 | 390px 下正常堆叠、截图未见横向溢出；页头按钮 28×28px，低于 44px 舒适值 |
| 5 | 实现一致性 | 3 | 检测器 1 条告警（圆角卡片叠粗色边）；5 个页面的上传区容器、文案语言不统一 |
| **合计** | | **12/20** | **Acceptable（需要成体系地修，不是零星补）** |

## 实现一致性判定（先看这条）

**通过（有条件）。** 这套 UI 有明确且产品相关的视觉语言：Georgia 衬线标题 + 深蓝墨色 + 浅蓝纸面 + 白色卡片，配合“复杂网络”主题的多层网络插图；不是通用模板换皮。扣分来自**页面之间的执行不一致**：Curve Fitting 的上传区包在带蓝色顶边的卡片里（`custom_style.css:96` `.st-key-fit-intake`），FunClu / NetRecon / NetAnal 的上传区是裸露的表单控件；文案语言在 5 个页面里各走各的。

---

## 主要问题（按严重度）

### P1 · 三处小字对比度不达标（WCAG 1.4.3）

| 位置 | 前景 / 背景 | 对比度 | 要求 |
|---|---|---|---|
| 首页卷首“BIMSA 副院长 / 首席科学家” `home.css:50` | `#456a97` / 渐变 `#c8dff5` | **4.08:1** | 4.5:1 |
| 页脚机构信息 `Home.py:180` | `#94a3b8` / `#f7f9fd` | **2.43:1** | 4.5:1 |
| 侧栏“登录后可使用分析工具” `utils.py:56`（Streamlit 默认 caption，60% 不透明度） | 约 `#7c7f8a` / `#ecf2fa` | **≈3.3:1** | 4.5:1 |

检测器独立命中同一处（`detect-url.json`：`low-contrast 4.1:1 (need 4.5:1) — #456a97 on #c8dff5`）。
影响：低视力用户与投影/强光环境下这行字基本读不出来；页脚是版权与机构署名，属关键信息。
建议：这三处前景统一压到 `#3a6191`（4.65:1 已达标）或更深；页脚建议 `#5b6b80`。

### P1 · 25 处 `use_container_width` 已弃用

`packages/idopnetwork-app/src/idopnetwork_app/**/*.py`，共 25 处。每次页面重跑都向终端打印 4 行弃用警告（见 `pty-2` 日志，一次会话刷了上百行），官方声明该参数将被移除。
影响：下一次 Streamlit 大版本升级会直接抛错；开发时真实报错被警告淹没。
建议：机械替换为 `width="stretch"`（`True`）或 `width="content"`（`False`）。

### P2 · 首屏图片体积过大

`static/images/`：`wu.jpg` 5.3 MB、`IDOP.png` 3.1 MB、`TSA.png` 2.7 MB、`idopn.png` 0.5 MB。首页把 5.3 MB 的肖像渲染成约 200px 宽的圆角图，侧栏把 2.7 MB 的 PNG 渲染成 112px。
影响：首屏在多数学术网络环境下加载明显变慢（本地无感，线上 `streamlit.app` 有感）。
建议：导出 WebP 或压缩后的 JPEG/PNG（目标：肖像 <80 KB、logo <40 KB），`IDOP.png` 若未使用则移除。

### P2 · 空状态只有一行提示

FunClu / NetRecon / NetAnal 未上传文件时，页面只剩标题 + 上传控件 + 一行 `Please upload … first.`，信息区约 400px 高、其余全空。
影响：新用户不知道上一步该做什么（这三个页面都依赖上游 ZIP），也不知道该用哪个导出文件。
建议：空状态给出“上游页面 → 导出文件 → 本页”的三步指引 + 一个示例数据下载入口（Curve Fitting 已有 `下载示例数据与使用说明`，其余页面没有）。

### P2 · 语言与术语不统一

Home 中文；Curve Fitting 中英混排（`导入观测数据` / `Fitting parameters` / `Run Fitting`）；FunClu、NetRecon 全英文；NetAnal 中文长段落。
影响：同一产品里切换语境，读者需要重新判断“这是给谁看的”；对外演示时尤其明显。
建议：定一条规则（推荐“中文为主、英文术语用括号随附”），然后逐页套用；这一步属于文案统一，不是翻译。

### P2 · 标题层级 h1 → h3

功能页用 `st.title`（h1）后直接 `st.subheader`（Streamlit 渲染为 h3），中间缺 h2；首页则用自定义 `h2.home-section-title`。
影响：屏幕阅读器按标题跳转时层级断裂；`outline` 语义不完整。
建议：统一改用 `<h2>` 容器（`st.markdown("<h2 …>")`）或在自定义 CSS 里把 `st.subheader` 的视觉与语义分离处理。

### P2 · 图片无替代文本

5 处 `st.image`（`TSA.png`、`wu.jpg`、`idopn.png` 等）都没有描述性 alt；Streamlit 的 `st.image` 只支持 `caption`，会渲染成可见文字。
影响：读屏用户听到的是文件名或空白；网络插图是核心内容之一（多层网络示意图）。
建议：用 `<img alt="…">` 走 `st.markdown(unsafe_allow_html=True)`，或给图配 `caption`。

### P3 · 圆角卡片配粗色边（检测器告警）

`custom_style.css:100`：`.st-key-fit-intake { border-radius: 12px; border-top: 3px solid var(--workspace-accent); }`
检测器规则 `border-accent-on-rounded`：粗色边与圆角相冲。
影响：视觉上“贴了一条胶带”，与其余纯 1px 边框卡片不同源（这也是上一条“页面不一致”的一部分）。
建议：二选一——去掉 3px 顶边改用 1px 描边 + 阴影，或把该卡片改成直角。

### P3 · 页头小按钮 28×28px

移动端 `stExpandSidebarButton` 与 `stMainMenuButton` 均为 28×28px（WCAG 2.5.8 AA 最低 24px 可过，但低于 44px 舒适值）。
建议：移动端用 CSS 放大到 40–44px（`@media (max-width: 640px)`）。

---

## 系统性问题

1. **“卡片”语义只用于首页与 Curve Fitting**：`st-key-fit-intake`、`home-module-*` 有卡片容器，其余页面的上传/参数区没有 → 4 个页面看起来像两套产品。
2. **颜色 token 只覆盖一半**：`:root` 里 4 个变量，两个 CSS 文件里仍有 54 处十六进制字面量（`custom_style.css` 21 处、`home.css` 33 处，含少量渐变与描边色）→ 改主色时要手工搜。
3. **空状态与引导缺失**：4 个功能页都靠上游导出文件驱动，但没有一处把这条流水线画出来。

## 做得好的地方（保持）

- 键盘可达性基础扎实：`button:focus-visible`、`.stPageLink a:focus-visible` 有 3px 描边；侧栏链接与 tab 都有 44px 最小高度。
- 论文列表容器带 `role="region"`、`aria-label="按年份浏览学术成果"`、`tabindex="0"`——这是少见的主动可访问性处理。
- 配色克制：深蓝墨色 + 浅蓝纸面，标题用衬线、正文用无衬线，层级清楚。
- 页面在 390px 宽度下正常单列堆叠，无横向滚动。
- 无动画，因此不存在 `prefers-reduced-motion` 缺失的问题。

## 建议的执行顺序

1. **[P1] `$impeccable typeset`** —— 修三处对比度（页脚、侧栏 caption、卷首副标题），顺带统一小字灰度。
2. **[P1] `$impeccable harden`** —— 25 处 `use_container_width` 迁移 + 空状态与错误文案的兜底。
3. **[P1] `$impeccable clarify`** —— 统一中英文案与术语（含 4 个功能页的空状态指引）。
4. **[P2] `$impeccable optimize`** —— 压缩 11.6 MB 图片、清理未使用资产。
5. **[P2] `$impeccable layout`** —— 上传区容器统一、嵌套 tab 的层级区分、空状态版式。
6. **[P3] `$impeccable polish`** —— 收尾（圆角卡片色边、移动端按钮尺寸）。

> 审计范围外但值得单独处理：`auth.py` 的默认口令 `admin/admin` 仍可用，且 `~/.idopnetwork/users.json` 以**明文**保存密码。这是安全问题，不属于前端审计，建议尽快单独修（改哈希 + 改默认口令）。

## Token 建议（供 `$impeccable document` 沉淀成 DESIGN.md）

```css
:root {
  /* 色板：现有 4 个 + 补齐语义层 */
  --ink:        #193454;   /* 标题/正文主色 */
  --ink-muted:  #3a6191;   /* 次级文字（≥4.5:1 于浅蓝底） */
  --accent:     #245db4;   /* 主操作/高亮 */
  --line:       #d4dfed;   /* 分隔线/卡片描边 */
  --paper:      #f7f9fd;   /* 页面底色 */
  --card:       #ffffff;   /* 卡片面 */
  --sidebar:    #ecf2fa;   /* 侧栏底 */
  /* 字号阶梯：1 套，替代现在的 0.85/0.9/1/1.12/1.35/… 散值 */
  --text-xs: .82rem; --text-sm: .92rem; --text-base: 1rem;
  --text-lg: 1.2rem; --text-xl: 1.6rem; --text-display: clamp(2.2rem, 4.6vw, 4.6rem);
  /* 间距阶梯：4 的倍数 */
  --space-1: .25rem; --space-2: .5rem; --space-3: .75rem;
  --space-4: 1rem;   --space-6: 1.5rem; --space-8: 2rem;
  /* 圆角：只留两档 */
  --radius-sm: 7px; --radius-lg: 12px;
}
```

配套三条规则：① 上传/参数区一律用 `--card` + `--line` 1px 卡片，不再使用 3px 色边；② 所有小字最低用 `--ink-muted`，禁用 `#94a3b8` 一类灰；③ 每个页面只允许一个 `--accent` 实心按钮（主操作），其余用次级样式。

---

## 修复记录（2026-11-21 第一轮）

| 项 | 状态 | 改法 / 证据 |
|---|---|---|
| P1 三处对比度 | 已修 | 新增 `--workspace-ink-soft: #4a5f7d`；caption 全局覆盖 `color + opacity:1`（原为 60% 不透明度）；卷首副标题 `#456a97 → #2f5784`（4.08 → **5.44:1**）；页脚 `#94a3b8 → #4a5f7d`（2.43 → **6.18:1**） |
| P1 25 处弃用 API | 已修 | `use_container_width=True → width="stretch"`，全仓 0 残留；终端不再刷警告 |
| P2 图片体积 | 已修 | `wu.jpg` 5476 → **62 KB**（2368px → 640px）；`TSA.png` 2777 → **10 KB**（量化 192 色 / 336px）；`idopn.png` 555 → **45 KB**（128 色 / 900px）。首屏图片合计约 11.6 MB → **117 KB** |
| P3 圆角卡片粗色边 | 已修 | `.st-key-fit-intake` 顶边 3px → 1px accent；检测器 `border-accent-on-rounded` 消失 |
| P3 移动端页头按钮 | 已修 | `@media (max-width: 640px)` 下 `[data-testid="stHeader"] button` 最小 40×40px |
| P2 空状态引导 | 已修 | 三页的裸提示换成 `upload_guidance()`：写明所需 ZIP、三步从哪来、给出去上游页面的链接（`utils.py` + `.st-key-upload-guidance*` 卡片样式） |
| 回归验证 | 通过 | `pytest`：**114 passed**；检测器：CSS **0 findings**、线上 URL **0 findings** |
| P2 语言/术语统一 | 进行中 | 空状态已统一为“中文为主 + 英文标识符随附”，其余控件文案仍待逐页套用 |
| P2 标题层级 | 已修 | Curve Fitting 三个区块、FunClu `Data Initialization`、NetAnal `GLMY` 改为 `##`（h2），Home 后台标题同步；CSS 把 markdown h2 拉回原 subheader 尺度，视觉不变 |
| P2 图片 alt | 已修 | 更正审计结论：Streamlit 1.67 的 `st.image` **支持** `alt`。侧栏 logo、卷首网络图、导师肖像走 i18n 词条；ECG 与 GLMY 生成图按图表名补 alt |
| P3 `IDOP.png` 3.1 MB | 未动 | 全仓无引用；删除属资产决策，留给作者确认 |

### 修复后自评（检测器 + 截图，未做全量重审）

| 维度 | 前 | 后 | 依据 |
|---|:---:|:---:|---|
| 可访问性 | 2 | 4 | 对比度、alt、标题层级均已处理 |
| 性能 | 2 | 3 | 图片与弃用 API 已解决；图片仍无懒加载 |
| 主题化 | 2 | 2 | token 覆盖面未变 |
| 响应式 | 3 | 3 | 页头触控目标已放大 |
| 一致性 | 3 | 4 | 检测器 0 findings |
| **合计** | **12** | **16 / 20（Good）** | |

---

## 语言切换（i18n）

**机制**：`packages/idopnetwork-app/src/idopnetwork_app/i18n.py` 提供 `t(key)`、`language()`、`language_switch()`。

- 语言存 `st.session_state["lang"]`，侧栏顶部的 `中文 | English` 分段控件一键切换，`on_change` 把选择写进 `?lang=` 查询参数，刷新或分享链接都保持语言；
- 兜底顺序：当前语言 → 默认语言（zh）→ 字面 key，缺词不会渲染成空白；
- **模块名不翻译**（Curve Fitting / FunClu / NetRecon / NetAnal 是专有名词），侧栏导航与 h1 在两版里保持一致，这也是截图脚本能继续按英文名点击的前提。

**已迁移**：首页全部文案（卷首标题/导语、两个区块标题、四张模块卡标题与描述、导师信息、页脚）、侧栏（欢迎语、退出、未登录提示、语言开关）、四个空状态引导、Curve Fitting 的区块标题与说明文字、登录提示，以及 FunClu 全页（标签页、数据概览、EM 拟合与绘图设置、FunClu-K 导出、BIC 扫描/建议/结果/导出，含表头与按钮）。

**尚未迁移**：NetRecon 与 NetAnal 的页面内部文案（目前本身就是英文，切到中文时保持英文）。技术参数名（`k_min`、`tol`、`n_components` 等）与 LaTeX 公式标签不做翻译。另外 Streamlit 内置的上传控件文案（`Upload`、`500MB per file · ZIP`）不开放本地化。新增词条只需在 `STRINGS["zh"]` 与 `STRINGS["en"]` 各加一行，页面里用 `t("key")` 即可。

**设计取舍**：Curve Fitting 的数据类型 tab 标签改成语言中立的 `Quasi-dynamic` / `Dynamic`，把中文解释移到 tab 下方的 caption —— 既缩短了长标签（审计遗留项），也避免切换语言时 tab 的 key 状态指向一个已不存在的标签。

---

## Curve Fitting 布局改造（方案 A）

诊断来自三档宽度的实测几何 + 一次独立评审，改造后复测同一组数字：

| 指标 | 改前 | 改后 |
|---|---|---|
| h1 → 上传卡间距 | **0 px**（h1 下边框与卡片 accent 顶边叠缝） | **32 px** |
| 上传卡 → 分析设置 | 19 px | **39 px** |
| 设置区结构 | h2 + tab 条 + 说明 + 卡片 + 重复的「Fitting parameters」 | 一张设置卡：分段控件 + 一行说明 + 字段 + Run |
| 字段网格 | 两列各 466（宽屏过宽） | 两行两列各约 316，左轴与卡片对齐（x=405） |
| Run 按钮 | 90×40，孤立在卡内左下 | 右栏 44px 高，与末行字段对齐 |
| 结果区图表起点 | **1695 px** | **1379 px**（−316） |
| 页面总高（3 文件） | 2309 px | **2031 px**（−278） |
| 未运行时的结果区 | 直接 `st.stop()`，页面戛然而止 | 虚线结果槽 + 一行说明 |
| 390 下字段顺序 | 1-3-2-4（按列展开） | 1-2-3-4（按行重排） |

结果区删掉的三块重复信息：绿色「Fitting complete」通栏、`Displayed results: …` 参数回显、占满 980px 的 `Results for` 单行下拉；现在压成一行状态 + 选择/下载并排。

**文件**：`pages/1_Curve Fitting.py`、`static/css/custom_style.css`、`i18n.py`（6 个新词条）。
**回归**：`pytest` 114 passed；`impeccable detect --scope layout` 0 findings。

**未解决（R-2，下一轮）**：390 宽下图表仍是 358×145、6 子图不可读；3 张 metric 竖排堆叠约 360px；结果 tab 条被裁切无滚动提示。这三项都要求图表按容器宽度自适应（改 `plot_curve_fitting` 的 figsize/列数）与移动端媒体查询，单独一轮处理。
## R-2 图表自适应

上一轮遗留的移动端问题，这一轮处理。核心是**按设备能力切换图表网格**，而不是等比例缩小：

| 项 | 前 | 后 |
|---|---|---|
| 桌面图表 | 980×398（2×3 子图，每格约 170 高） | **948×516**（cell_height 1.5 → 2.0） |
| 移动端图表 | 358×145，2×3 子图不可读 | **358×536，1 列 2 行**，每格约 250 高 |
| 移动端指标 | 3 张 metric 竖排 360px | 一行紧凑状态「32 / 32 特征 · 5 行 · 30 个采样点」 |
| 移动端图表起点 | 2361px | **1939px** |
| 结果 tab 条 | 裁切且不可滚 | `overflow-x: auto`，可横向滑动 |

实现：`st.context.headers` 读 User-Agent，命中 `Android/iPhone/iPad/Mobile` 时走紧凑分支（每页 2 张图、1 列；指标压成一行）。`plot_curve_fitting` 新增 `cell_height` 参数（默认 1.5，行为不变），页面按模式传 1.8 / 2.0。

注意：**库模块改动需要重启 Streamlit 进程**才生效（`runOnSave` 只重跑页面脚本，不重载已导入的 `idopnetwork` 包）。这一轮重启过一次服务。

## R-3 出版级图表（HTML/SVG 渲染器）

Curve Fitting 的绘图从 matplotlib 的 `st.pyplot` 改为组件内的 HTML/SVG 渲染器（`idopnetwork_app/charts/curve_fitting_html.py`），默认参数面向期刊投稿：

| 项 | 值 |
|---|---|
| 画布 | 子图尺寸驱动（41 × 25 mm），行列间距 0 / 5 mm，自适应留白（外边距 1.5 pt） |
| 排布 | 3 × 4；共享 Y 轴，只在第一列标数字；X 刻度数字只在最底一行 |
| 排版 | Times 11 pt；面板字母 a–l；四向数据边距 20/20/30/10 % |
| 轴标题 | X 距子图 10 pt、Y 距子图 0 pt，均居中于子图网格 |
| 导出 | 组件内 SVG（物理尺寸，pt）与 300/600 dpi PNG |

控制面板在组件内（`特征`、`画布与子图`、`刻度与范围`、`排版与轴标题`、`图元`、`导出`，均可折叠），改参数即时重绘，不再触发服务端重跑；页面只保留一行说明，`Features to plot` 与 `Feature page` 已移除。

**文件**：`charts/curve_fitting_html.py`（新增）、`pages/1_Curve Fitting.py`、`tests/test_curve_fitting_page.py`。
**回归**：`pytest` 114 passed。
**Compare fitted curves** 也改用同一渲染器：每个面板一个特征、每个文件一条序列（同色系空心点 + 曲线），图例在画布顶部。
**仍用 matplotlib 的地方**：仅 Dynamic 模式的 SWT 波形；PDF/EPS 导出尚未接入。
