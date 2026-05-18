# Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单体 Streamlit 应用拆分为 `idopnetwork`（纯算法库）和 `idopnetwork-app`（Streamlit 界面）两个 pip 可安装包。

**Architecture:** src-layout 双包结构。`idopnetwork` 零 Streamlit 依赖，torch 可选；`idopnetwork-app` 依赖 `idopnetwork`，通过 importlib.resources 管理静态资源。旧 `backend.xxx` 导入统一迁移到新包名。

**Tech Stack:** Python >= 3.10, setuptools, importlib.resources, Streamlit, matplotlib

---

### Task 1: 创建 idopnetwork 包骨架

**Files:**
- Create: `packages/idopnetwork/pyproject.toml`
- Create: `packages/idopnetwork/LICENSE`
- Create: `packages/idopnetwork/README.md`
- Create: `packages/idopnetwork/src/idopnetwork/__init__.py`
- Create: `packages/idopnetwork/src/idopnetwork/py.typed`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p packages/idopnetwork/src/idopnetwork
```

- [ ] **Step 2: 编写 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "idopnetwork"
version = "0.1.0"
description = "idopNetwork: complex systems analysis toolkit — curve fitting, functional clustering, network reconstruction, and topological data analysis"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [
    { name = "Yu Wang" },
]
dependencies = [
    "numpy>=1.26",
    "pandas>=2.0",
    "scipy>=1.13",
    "scikit-learn>=1.7",
    "matplotlib>=3.8",
    "cvxpy>=1.6",
]

[project.optional-dependencies]
ml = ["torch>=2.4"]

[project.urls]
Homepage = "https://idopnetworkapp-bimsa-statistics.streamlit.app/"
Repository = "https://github.com/Yu-Wang-0923/idopNetworkAPP"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: 编写 LICENSE（MIT）**

```
MIT License

Copyright (c) 2026 Yu Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: 编写 README.md**

```markdown
# idopNetwork

A Python toolkit for complex systems data analysis, implementing:

- **Curve Fitting**: Power-law fitting `y = a·x^b` with data transformation and quasi-dynamic ranking
- **Functional Clustering (FunClu)**: EM Gaussian mixture clustering with power-mean models
- **Network Reconstruction (NetRecon)**: Legendre basis expansion + constrained sparse regression (IDOPRegressor)
- **Network Analysis (NetAnal)**: GLMY persistent path homology and network depth analysis

## Installation

```bash
pip install idopnetwork
```

For machine learning extras:

```bash
pip install idopnetwork[ml]
```

## Usage

```python
from idopnetwork.curve_fitting import fit_power_loglinear, get_power_function_sample
from idopnetwork.clustering import FunClu
from idopnetwork.network import IDOPRegressor
from idopnetwork.analysis import run_glmy
```

## Citation

If you use idopNetwork in your research, please cite:

> Wang, Y. et al. "idopNetwork: An integrative platform for complex systems analysis." (in preparation)
```

- [ ] **Step 5: 编写包 \_\_init\_\_.py**

```python
"""idopNetwork: complex systems analysis toolkit."""

__version__ = "0.1.0"
```

- [ ] **Step 6: 创建 py.typed 标记文件**

```bash
touch packages/idopnetwork/src/idopnetwork/py.typed
```

- [ ] **Step 7: 提交**

```bash
git add packages/idopnetwork/
git commit -m "feat: add idopnetwork core package skeleton"
```

---

### Task 2: 迁移核心算法模块到 idopnetwork

**Files:**
- Create: `packages/idopnetwork/src/idopnetwork/curve_fitting/__init__.py`
- Create: `packages/idopnetwork/src/idopnetwork/curve_fitting/fitting.py`
- Create: `packages/idopnetwork/src/idopnetwork/curve_fitting/plot.py`
- Create: `packages/idopnetwork/src/idopnetwork/clustering/__init__.py`
- Create: `packages/idopnetwork/src/idopnetwork/clustering/funclu.py`
- Create: `packages/idopnetwork/src/idopnetwork/clustering/plot.py`
- Create: `packages/idopnetwork/src/idopnetwork/network/__init__.py`
- Create: `packages/idopnetwork/src/idopnetwork/network/construction.py`
- Create: `packages/idopnetwork/src/idopnetwork/network/plot.py`
- Create: `packages/idopnetwork/src/idopnetwork/analysis/__init__.py`
- Create: `packages/idopnetwork/src/idopnetwork/analysis/digraph.py`
- Create: `packages/idopnetwork/src/idopnetwork/analysis/glmy.py`
- Create: `packages/idopnetwork/src/idopnetwork/analysis/glmy_test.py`
- Create: `packages/idopnetwork/src/idopnetwork/analysis/network_analysis.py`
- Create: `packages/idopnetwork/src/idopnetwork/analysis/plot_analysis.py`

- [ ] **Step 1: 复制所有源文件并批量替换导入路径**

```bash
# 创建子包目录
mkdir -p packages/idopnetwork/src/idopnetwork/{curve_fitting,clustering,network,analysis,ml}

# 复制文件（不包括 utils.py 和 auth.py —— 它们属于 app 层）
cp backend/curve_fitting/fitting.py packages/idopnetwork/src/idopnetwork/curve_fitting/fitting.py
cp backend/curve_fitting/plot.py packages/idopnetwork/src/idopnetwork/curve_fitting/plot.py
cp backend/clustering/funclu.py packages/idopnetwork/src/idopnetwork/clustering/funclu.py
cp backend/clustering/plot.py packages/idopnetwork/src/idopnetwork/clustering/plot.py
cp backend/network/construction.py packages/idopnetwork/src/idopnetwork/network/construction.py
cp backend/network/plot.py packages/idopnetwork/src/idopnetwork/network/plot.py
cp backend/analysis/digraph.py packages/idopnetwork/src/idopnetwork/analysis/digraph.py
cp backend/analysis/glmy.py packages/idopnetwork/src/idopnetwork/analysis/glmy.py
cp backend/analysis/glmy_test.py packages/idopnetwork/src/idopnetwork/analysis/glmy_test.py
cp backend/analysis/network_analysis.py packages/idopnetwork/src/idopnetwork/analysis/network_analysis.py
cp backend/analysis/plot_analysis.py packages/idopnetwork/src/idopnetwork/analysis/plot_analysis.py

# 批量替换所有 Python 文件中的 backend. 为 idopnetwork.
find packages/idopnetwork/src/idopnetwork -name "*.py" -exec sed -i '' 's/from backend\./from idopnetwork./g' {} \;
find packages/idopnetwork/src/idopnetwork -name "*.py" -exec sed -i '' 's/import backend\./import idopnetwork./g' {} \;
```

- [ ] **Step 2: 修复 curve_fitting/fitting.py —— 移除 Streamlit 依赖**

在 `packages/idopnetwork/src/idopnetwork/curve_fitting/fitting.py` 中：

删除第 6 行 `import streamlit as st`

删除所有 `@st.cache_data` 装饰器（第 12, 31, 82, 169, 196 行附近），替换为空行。

删除 `_show_overview` 函数（约第 210-218 行）—— 该函数全部使用 `st.expander`/`st.dataframe`/`st.write`，属于 UI 层逻辑。

- [ ] **Step 3: 修复 curve_fitting/plot.py —— 移除 Streamlit 调用**

在 `packages/idopnetwork/src/idopnetwork/curve_fitting/plot.py` 中：

删除 `import streamlit as st`（第 5 行）。

删除所有 `st.pyplot(fig)` 调用（保留 `fig` 作为返回值）。

删除 `st.markdown(...)` + `st.columns(...)` + `st.download_button(...)` 段（约第 165-173 行）。

将 `from backend.utils import font_prop`（已被 sed 替换为 `from idopnetwork.utils import font_prop`）替换为：
```python
font_prop = None  # 由应用层设置为 matplotlib FontProperties（用于 CJK 字体）
```

然后将所有 `fontproperties=font_prop` 的调用改为条件使用：
```python
# 模式：所有 fontproperties=font_prop 替换为 fontproperties=font_prop if font_prop else None
```
执行 sed 替换：
```bash
cd packages/idopnetwork/src/idopnetwork/curve_fitting
# 对于 kwarg 形式，font_prop 为 None 时 matplotlib 会忽略 fontproperties=None
# 无需额外条件判断，直接保持 fontproperties=font_prop，matplotlib 可接受 None
```

- [ ] **Step 4: 修复 clustering/plot.py —— 条件化 Streamlit + font_prop**

在 `packages/idopnetwork/src/idopnetwork/clustering/plot.py` 中：

替换 `import streamlit as st` 为函数内按需导入。删除顶部 `import streamlit as st`（第 21 行）。

将 `from backend.utils import font_prop`（已被 sed 替换为 `from idopnetwork.utils import font_prop`）替换为：
```python
font_prop = None  # 由应用层设置为 matplotlib FontProperties（用于 CJK 字体）
```

找到所有 `show_in_streamlit` 为 True 时调用 `st.pyplot(fig, ...)` 的地方（约第 256-257 行、691-692 行），改为：
```python
if show_in_streamlit:
    import streamlit as st
    st.pyplot(fig, use_container_width=True)
```

`show_in_streamlit` 参数默认值从 `True` 改为 `False`。

- [ ] **Step 5: 修复 network/plot.py —— 移除 Streamlit 调用 + font_prop**

在 `packages/idopnetwork/src/idopnetwork/network/plot.py` 中：

删除 `import streamlit as st`（第 18 行）。

将 `from backend.utils import font_prop`（已被 sed 替换为 `from idopnetwork.utils import font_prop`）替换为：
```python
font_prop = None  # 由应用层设置为 matplotlib FontProperties（用于 CJK 字体）
```

将所有 `st.pyplot(fig, ...)` 替换为仅返回 `fig`（删除 `st.pyplot` 调用，确保 `fig` 作为返回值保留）。

- [ ] **Step 6: 修复 analysis/glmy_test.py —— 路径解析**

将第 40-41 行：
```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_M3_CSV: Path = REPO_ROOT / "data" / "M3.csv"
```
改为：
```python
DEFAULT_M3_CSV: Path | None = None  # 由应用层设置路径
```

同时检查 sed 已将 `from backend.analysis.digraph import Digraph` 改为 `from idopnetwork.analysis.digraph import Digraph`。

- [ ] **Step 7: 清理空占位文件**

删除以下 0 行占位文件：
```bash
rm packages/idopnetwork/src/idopnetwork/analysis/center.py
rm packages/idopnetwork/src/idopnetwork/analysis/plot_center.py
rm packages/idopnetwork/src/idopnetwork/analysis/plot_glmy.py
```

`ml/core.py` 和 `ml/plot.py` 保留（未来计划有内容）。如果它们为空，先写入占位 docstring：
```python
# packages/idopnetwork/src/idopnetwork/ml/core.py
"""Machine learning module (placeholder)."""
```
```python
# packages/idopnetwork/src/idopnetwork/ml/plot.py
"""Machine learning visualization (placeholder)."""
```

- [ ] **Step 8: 处理 font_prop —— 应用层注入方案**

核心库的三个 plot 模块（`curve_fitting/plot.py`、`clustering/plot.py`、`network/plot.py`）已将 `font_prop` 设为模块级变量并默认 `None`。matplotlib 的 `fontproperties=None` 会使用默认英文字体。

`idopnetwork-app` 的 `utils.py` 在初始化时向其注入中文字体：

```python
# 在 idopnetwork_app/utils.py 末尾添加：
import idopnetwork.curve_fitting.plot as cf_plot
import idopnetwork.clustering.plot as cl_plot
import idopnetwork.network.plot as nw_plot

cf_plot.font_prop = font_prop
cl_plot.font_prop = font_prop
nw_plot.font_prop = font_prop
```

同理，`glmy_test.py` 的 `DEFAULT_M3_CSV` 在 app 层设置：
```python
# 在 idopnetwork_app/utils.py 末尾添加：
from importlib.resources import files
import idopnetwork.analysis.glmy_test as glmy_test
glmy_test.DEFAULT_M3_CSV = str(files("idopnetwork_app.data") / "M3.csv")
```

- [ ] **Step 9: 编写各子包的 \_\_init\_\_.py（导入路径已更新）**

`packages/idopnetwork/src/idopnetwork/curve_fitting/__init__.py`:
```python
from idopnetwork.curve_fitting.fitting import (
    _numeric_frame_for_transform,
    data_transformation,
    fit_power_loglinear,
    get_power_function_params,
    get_power_function_sample,
    get_quasi_dynamic_df,
    load_csv,
)
```

`packages/idopnetwork/src/idopnetwork/clustering/__init__.py`:
```python
from idopnetwork.clustering.funclu import FunClu
from idopnetwork.clustering.plot import plot_cluster_profiles
```

`packages/idopnetwork/src/idopnetwork/network/__init__.py`:
```python
from idopnetwork.network.construction import (
    IDOPRegressor,
    align_response_to_design,
    polynomial_basis_expansion,
)
from idopnetwork.network.plot import (
    plot_effect,
    plot_network,
    plot_adjusted_matrix_heatmap,
)
```

`packages/idopnetwork/src/idopnetwork/analysis/__init__.py`:
```python
from idopnetwork.analysis.network_analysis import (
    list_from_to_members,
    member_display_label,
    load_from_to_from_zip,
    run_glmy,
    suggest_max_x,
    sanitize_name,
)
from idopnetwork.analysis.plot_analysis import plot_glmy_barcode
from idopnetwork.analysis.glmy_test import (
    DEFAULT_DIM as M3_DEFAULT_DIM,
    DEFAULT_M3_CSV,
    DEFAULT_MAX_X as M3_DEFAULT_MAX_X,
    DEFAULT_WEIGHT_OFFSET as M3_DEFAULT_WEIGHT_OFFSET,
    betti_summary as m3_betti_summary,
    load_m3_dataframe,
    paper_3_2_dataframe,
    run_digraph_on_m3,
)
```

`packages/idopnetwork/src/idopnetwork/ml/__init__.py`:
```python
try:
    from idopnetwork.ml.core import *  # noqa: F403
except ImportError:
    pass
```

- [ ] **Step 10: 处理 ml/ 子包**

复制 `backend/ml/` 下的文件到 `packages/idopnetwork/src/idopnetwork/ml/`。目前 ml/core.py 和 ml/plot.py 为空。保留空的占位文件供未来填充。

- [ ] **Step 11: 安装测试核心库**

```bash
cd packages/idopnetwork && pip install -e .
```

然后验证导入：
```bash
python -c "from idopnetwork.curve_fitting import fit_power_loglinear; print('OK')"
python -c "from idopnetwork.clustering import FunClu; print('OK')"
python -c "from idopnetwork.network import IDOPRegressor; print('OK')"
python -c "from idopnetwork.analysis import run_glmy; print('OK')"
```

- [ ] **Step 12: 提交**

---

### Task 3: 创建 idopnetwork-app 包骨架

**Files:**
- Create: `packages/idopnetwork-app/pyproject.toml`
- Create: `packages/idopnetwork-app/README.md`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/__init__.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p packages/idopnetwork-app/src/idopnetwork_app/{pages,static/{css,images},data}
```

- [ ] **Step 2: 编写 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "idopnetwork-app"
version = "0.1.0"
description = "idopNetwork Streamlit web application"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
dependencies = [
    "idopnetwork>=0.1.0",
    "streamlit>=1.52",
]

[project.scripts]
idopnetwork-app = "idopnetwork_app.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
idopnetwork_app = [
    "static/css/*.css",
    "static/images/*",
    "static/SimHei.ttf",
    "data/*.csv",
    "data/*.R",
]
```

- [ ] **Step 3: 编写 README.md**

```markdown
# idopNetwork App

Streamlit-based web application for the idopNetwork complex systems analysis platform.

## Installation

```bash
pip install idopnetwork-app
```

## Usage

```bash
idopnetwork-app
```

Or run directly:

```bash
streamlit run $(python -c "from importlib.resources import files; print(files('idopnetwork_app') / 'Home.py')")
```
```

- [ ] **Step 4: 编写 \_\_init\_\_.py**

```python
"""idopNetwork Streamlit application."""
```

- [ ] **Step 5: 提交**

---

### Task 4: 迁移 Streamlit 应用代码

**Files:**
- Create: `packages/idopnetwork-app/src/idopnetwork_app/Home.py`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/utils.py`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/auth.py`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/cli.py`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/pages/1_Curve Fitting.py`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/pages/2_FunClu.py`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/pages/3_NetRecon.py`
- Create: `packages/idopnetwork-app/src/idopnetwork_app/pages/4_NetAnal.py`

- [ ] **Step 1: 复制静态资源和数据**

```bash
cp -r static/* packages/idopnetwork-app/src/idopnetwork_app/static/
cp -r data/* packages/idopnetwork-app/src/idopnetwork_app/data/
```

- [ ] **Step 2: 迁移 utils.py —— 用 importlib.resources 替代硬编码路径**

复制 `backend/utils.py` 到 `packages/idopnetwork-app/src/idopnetwork_app/utils.py`，做以下修改：

```python
"""
共享工具模块：路径常量、CSS 注入、matplotlib 中文字体配置。
"""
from pathlib import Path
from importlib.resources import files
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import streamlit as st

# ── 静态资源访问（通过 importlib.resources） ──────────────────────────────
_STATIC = files("idopnetwork_app.static")
CSS_PATH = _STATIC / "css" / "custom_style.css"
FONT_PATH = _STATIC / "SimHei.ttf"
IMAGES_DIR = _STATIC / "images"

font_prop = fm.FontProperties(fname=str(FONT_PATH))
plt.rcParams["axes.unicode_minus"] = False

ADMIN_USERS = {"郭佳泽", "11"}
HEADER_TOGGLE_KEY = "show_streamlit_header"

# ── 辅助函数 ──────────────────────────────────────────────────────────

def _is_admin_user() -> bool:
    """判断当前登录用户是否为管理员。"""
    current_user = st.session_state.get("current_user")
    return bool(st.session_state.get("logged_in", False) and current_user in ADMIN_USERS)


def load_css():
    """强制注入的全局核心样式"""
    show_streamlit_header = bool(st.session_state.get(HEADER_TOGGLE_KEY, False))
    header_css = (
        "footer, [data-testid=\"stSidebarNav\"] { display: none !important; }"
        if show_streamlit_header
        else "header, footer, [data-testid=\"stSidebarNav\"] { display: none !important; }"
    )

    css = """
    <style>
    .stApp { background-color: #f8fafc !important; }
    __HEADER_CSS__
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        background-image: none !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 1rem !important;
    }
    .stPageLink a { text-decoration: none !important; padding: 0.3rem 0; transition: all 0.2s; }
    .stPageLink a p { color: #334155 !important; font-size: 1.1rem !important; font-weight: 500; }
    .stPageLink:hover { background-color: rgba(15, 23, 42, 0.05) !important; border-radius: 8px; }
    [data-testid="stSidebar"] img { border-radius: 20px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); background-color: #ffffff; }
    p, li, .stMarkdown { font-size: 1.1rem !important; line-height: 1.6 !important; }
    
    /* 图标上色 */
    [data-testid="stSidebar"] span[data-testid="stWidgetLabel"] span { font-size: 1.3rem !important; font-weight: bold !important; }
    .stPageLink:nth-of-type(1) span[data-testid="stWidgetLabel"] span { color: #0284c7 !important; }
    .stPageLink:nth-of-type(2) span[data-testid="stWidgetLabel"] span { color: #16a34a !important; }
    .stPageLink:nth-of-type(3) span[data-testid="stWidgetLabel"] span { color: #ea580c !important; }
    .stPageLink:nth-of-type(4) span[data-testid="stWidgetLabel"] span { color: #7c3aed !important; }
    .stPageLink:nth-of-type(5) span[data-testid="stWidgetLabel"] span { color: #e11d48 !important; }
    </style>
    """
    st.markdown(css.replace("__HEADER_CSS__", header_css), unsafe_allow_html=True)


def setup_sidebar():
    """全局统一的侧边栏组件"""
    with st.sidebar:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3.5, 1])
        with col2:
            tsa_path = str(IMAGES_DIR / "TSA.png")
            st.image(tsa_path, use_container_width=True)
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        # 权限控制
        if st.session_state.get("logged_in", False):
            st.page_link("Home.py", label="Home", icon=":material/home:")
            st.page_link("pages/1_Curve Fitting.py", label="Curve Fitting", icon=":material/timeline:")
            st.page_link("pages/2_FunClu.py", label="FunClu", icon=":material/category:")
            st.page_link("pages/3_NetRecon.py", label="NetRecon", icon=":material/hub:")
            st.page_link("pages/4_NetAnal.py", label="NetAnal", icon=":material/insights:")

            st.markdown("<hr style='margin: 1.5rem 0; border-color: #cbd5e1;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#334155; font-weight:bold;'>👋 欢迎, {st.session_state['current_user']}</p>", unsafe_allow_html=True)
            if _is_admin_user():
                st.toggle(
                    "显示 Streamlit 顶部菜单",
                    key=HEADER_TOGGLE_KEY,
                    help="开启后可使用右上角 Streamlit 菜单（如 Clear cache）。",
                )
            if st.button("退出登录", key="logout_btn_final"):
                st.session_state["logged_in"] = False
                st.rerun()
        else:
            st.page_link("Home.py", label="返回首页登录", icon=":material/login:")
            st.markdown("<p style='color:#ef4444; font-weight:bold; text-align:center; margin-top:2rem;'>🔒 请先登录以解锁功能</p>", unsafe_allow_html=True)
```

- [ ] **Step 3: 迁移 auth.py —— 用户数据写入 ~/.idopnetwork/**

复制 `backend/auth.py` 到 `packages/idopnetwork-app/src/idopnetwork_app/auth.py`，修改用户数据路径：

```python
import streamlit as st
import json
import os
from pathlib import Path

# 用户数据保存路径 —— 写入用户家目录，不再存储在包内
_USER_DATA_DIR = Path.home() / ".idopnetwork"
_USER_DATA_FILE = _USER_DATA_DIR / "users.json"

def _ensure_data_dir():
    """确保数据目录存在，并创建默认管理员账号。"""
    _USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _USER_DATA_FILE.exists():
        default_users = {
            "admin": {
                "password": "admin",
                "real_name": "Administrator",
                "phone": "",
                "organization": "",
                "research_direction": "",
            }
        }
        save_users(default_users)

def load_users():
    """读取用户信息"""
    _ensure_data_dir()
    if not _USER_DATA_FILE.exists():
        return {}
    with open(_USER_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    """保存用户信息"""
    _ensure_data_dir()
    with open(_USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def show_login_ui():
    """右上角登录按钮"""
    col_title, col_btn = st.columns([8.5, 1.5])
    with col_btn:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔑 登录 / 注册", type="primary", use_container_width=True):
            show_auth_modal()

@st.dialog("🔐 统一身份认证中心", width="large")
def show_auth_modal():
    """弹窗内部：登录与详细信息注册"""
    tab_login, tab_register = st.tabs(["🔑 登录已有账户", "📝 注册新账户"])
    users = load_users()

    with tab_login:
        log_user = st.text_input("用户名", key="log_user")
        log_pwd = st.text_input("密码", type="password", key="log_pwd")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("验证并进入系统", type="primary", use_container_width=True):
            user_info = users.get(log_user)
            if user_info and isinstance(user_info, dict) and user_info.get("password") == log_pwd:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = log_user
                st.session_state["user_details"] = user_info
                st.rerun()
            elif user_info == log_pwd:  # 兼容旧的字符串存法
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = log_user
                st.rerun()
            else:
                st.error("用户名或密码错误")

    with tab_register:
        st.markdown("##### 请填写您的真实研究信息")
        reg_user = st.text_input("用户名 *", placeholder="用于登录的唯一ID", key="reg_user")
        reg_pwd = st.text_input("设置密码 *", type="password", key="reg_pwd")
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            reg_real_name = st.text_input("真实姓名", placeholder="张三", key="reg_real_name")
            reg_phone = st.text_input("电话号", placeholder="138XXXXXXXX", key="reg_phone")
        with col_reg2:
            reg_org = st.text_input("所属单位", placeholder="XX大学/XX研究院", key="reg_org")
            reg_field = st.text_input("研究方向", placeholder="复杂网络/生物信息等", key="reg_field")
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("ℹ️ **声明**：您填写的个人信息仅用于学术交流及课题组内部成员身份核验，平台将严格保护您的隐私。")

        if st.button("提交注册申请", use_container_width=True, type="primary"):
            if not reg_user or not reg_pwd:
                st.warning("请至少填写用户名和密码")
            elif reg_user in users:
                st.error("该用户名已存在，请更换")
            else:
                users[reg_user] = {
                    "password": reg_pwd,
                    "real_name": reg_real_name,
                    "phone": reg_phone,
                    "organization": reg_org,
                    "research_direction": reg_field,
                }
                save_users(users)
                st.success("🎉 注册成功！请切换到『登录』页进行验证。")
```

- [ ] **Step 4: 编写 cli.py 入口**

```python
"""idopNetwork-app CLI entry point."""
import sys
from streamlit.web import cli as stcli
from importlib.resources import files


def main():
    home = str(files("idopnetwork_app") / "Home.py")
    sys.argv = ["streamlit", "run", home] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 迁移页面文件 —— 批量替换导入路径**

```bash
# 复制页面文件
cp Home.py packages/idopnetwork-app/src/idopnetwork_app/Home.py
cp pages/1_Curve\ Fitting.py "packages/idopnetwork-app/src/idopnetwork_app/pages/1_Curve Fitting.py"
cp pages/2_FunClu.py "packages/idopnetwork-app/src/idopnetwork_app/pages/2_FunClu.py"
cp pages/3_NetRecon.py "packages/idopnetwork-app/src/idopnetwork_app/pages/3_NetRecon.py"
cp pages/4_NetAnal.py "packages/idopnetwork-app/src/idopnetwork_app/pages/4_NetAnal.py"

# 批量替换导入路径
find packages/idopnetwork-app/src/idopnetwork_app -name "*.py" -exec sed -i '' 's/from backend\.utils/from idopnetwork_app.utils/g' {} \;
find packages/idopnetwork-app/src/idopnetwork_app -name "*.py" -exec sed -i '' 's/from backend\.auth/from idopnetwork_app.auth/g' {} \;
find packages/idopnetwork-app/src/idopnetwork_app -name "*.py" -exec sed -i '' 's/from backend\./from idopnetwork./g' {} \;
```

- [ ] **Step 6: 修复页面文件中的静态资源引用**

所有页面文件中的 `page_icon="static/images/TSA.png"` 需要改为通过 importlib.resources 解析。在每个页面文件顶部添加：

```python
from importlib.resources import files
PAGE_ICON = str(files("idopnetwork_app.static.images") / "TSA.png")
```

然后 `st.set_page_config(..., page_icon=PAGE_ICON, ...)`

`Home.py` 中的 `st.image("static/images/wu.jpg", ...)` 也需要改为：
```python
from importlib.resources import files
WU_IMG = str(files("idopnetwork_app.static.images") / "wu.jpg")
st.image(WU_IMG, width=150)
```

- [ ] **Step 7: 修复页面中 import 的差异**

各页面文件中导入 `backend.curve_fitting.fitting` 等的 import 语句已被 sed 替换为 `idopnetwork.curve_fitting.fitting`。需要手动检查每处替换是否正确。

- [ ] **Step 8: 安装测试 app 包**

```bash
cd packages/idopnetwork-app && pip install -e .
```

验证：
```bash
python -c "from idopnetwork_app.utils import load_css, font_prop; print('OK')"
python -c "from idopnetwork_app.auth import load_users; print('OK')"
```

- [ ] **Step 9: 提交**

---

### Task 5: 清理旧文件和更新根目录配置

- [ ] **Step 1: 删除旧的 backend/ 目录**

```bash
rm -rf backend/
```

- [ ] **Step 2: 删除旧的 pages/ 目录、Home.py 和 static/ data/**

```bash
rm -rf pages/ Home.py static/ data/
```

注意：部署配置（`.streamlit/config.toml`）保留在原位。

- [ ] **Step 3: 更新根目录 pyproject.toml 或删除**

将 `pyproject.toml` 替换为一个 workspace 级别的说明，或直接删除（因为现在有两个独立的包）。

如果保留根 `pyproject.toml` 仅用于本地开发便利，内容为：
```toml
# 本地开发便利文件，指向两个子包
# pip install -e packages/idopnetwork
# pip install -e packages/idopnetwork-app
```

- [ ] **Step 4: 更新 .gitignore**

添加：
```
dist/
build/
*.egg-info/
*.whl
__pycache__/
*.pyc
```

添加 `backend/users.json`（如果它还存在的话）。

- [ ] **Step 5: 从 git 历史中清除 users.json**

```bash
# 从所有历史提交中移除（需要 force push，谨慎操作）
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch backend/users.json" \
  --prune-empty --tag-name-filter cat -- --all
```

注意：此操作会重写 git 历史。需要在执行前与用户确认。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "refactor: split into idopnetwork + idopnetwork-app dual packages"
```

---

### Task 6: 端到端测试

- [ ] **Step 1: 在全新虚拟环境中测试安装**

```bash
python -m venv /tmp/test_idop
source /tmp/test_idop/bin/activate
pip install -e packages/idopnetwork
pip install -e packages/idopnetwork-app
```

- [ ] **Step 2: 验证核心库导入**

```bash
python -c "
from idopnetwork.curve_fitting import (
    load_csv, data_transformation, fit_power_loglinear,
    get_power_function_sample, get_power_function_params, get_quasi_dynamic_df,
)
from idopnetwork.clustering import FunClu, plot_cluster_profiles
from idopnetwork.network import (
    IDOPRegressor, polynomial_basis_expansion, align_response_to_design,
    plot_effect, plot_network, plot_adjusted_matrix_heatmap,
)
from idopnetwork.analysis import (
    run_glmy, plot_glmy_barcode, load_from_to_from_zip,
    m3_betti_summary, load_m3_dataframe, run_digraph_on_m3,
)
print('All imports successful')
"
```

- [ ] **Step 3: 验证 Streamlit 应用可启动（headless）**

```bash
timeout 10 streamlit run $(python -c "from importlib.resources import files; print(files('idopnetwork_app') / 'Home.py')") --server.headless true 2>&1 | head -5
```

Expected: Streamlit 启动日志，无 ImportError。

- [ ] **Step 4: 清理测试环境**

```bash
deactivate
rm -rf /tmp/test_idop
```
