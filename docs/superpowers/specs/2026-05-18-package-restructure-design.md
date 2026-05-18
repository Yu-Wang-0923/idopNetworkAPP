# 包结构重构设计

## 目标

将 idopNetworkAPP 从单体 Streamlit 应用拆分为两个独立的 pip 可安装包：
- `idopnetwork` — 纯算法库，论文可引用
- `idopnetwork-app` — Streamlit 交互界面，依赖前者

## 最终结构

```
idopNetworkAPP/
├── packages/
│   ├── idopnetwork/                   # 核心算法库
│   │   ├── pyproject.toml
│   │   ├── LICENSE                    # MIT
│   │   ├── README.md
│   │   └── src/
│   │       └── idopnetwork/
│   │           ├── __init__.py
│   │           ├── py.typed
│   │           ├── curve_fitting/
│   │           │   ├── __init__.py
│   │           │   ├── fitting.py
│   │           │   └── plot.py        # 仅绘图函数，不调 st.*
│   │           ├── clustering/
│   │           │   ├── __init__.py
│   │           │   ├── funclu.py
│   │           │   └── plot.py
│   │           ├── network/
│   │           │   ├── __init__.py
│   │           │   ├── construction.py
│   │           │   └── plot.py
│   │           ├── analysis/
│   │           │   ├── __init__.py
│   │           │   ├── digraph.py
│   │           │   ├── glmy.py
│   │           │   ├── glmy_test.py
│   │           │   ├── network_analysis.py
│   │           │   ├── plot_analysis.py
│   │           │   ├── center.py
│   │           │   ├── plot_center.py
│   │           │   └── plot_glmy.py
│   │           └── ml/
│   │               ├── __init__.py
│   │               ├── core.py
│   │               └── plot.py
│   │
│   └── idopnetwork-app/              # Streamlit 应用
│       ├── pyproject.toml            # 依赖 idopnetwork
│       ├── README.md
│       └── src/
│           └── idopnetwork_app/
│               ├── __init__.py
│               ├── cli.py
│               ├── Home.py
│               ├── utils.py           # CSS注入、侧边栏、中文字体配置
│               ├── auth.py            # 用户认证（users.json -> ~/.idopnetwork/）
│               ├── pages/
│               │   ├── 1_Curve Fitting.py
│               │   ├── 2_FunClu.py
│               │   ├── 3_NetRecon.py
│               │   └── 4_NetAnal.py
│               ├── static/
│               │   ├── css/
│               │   │   └── custom_style.css
│               │   ├── images/
│               │   │   ├── IDOP.png
│               │   │   ├── TSA.png
│               │   │   └── wu.jpg
│               │   └── SimHei.ttf
│               └── data/
│                   ├── a.csv
│                   ├── M3.csv
│                   └── tiger_demo.R

## 依赖关系

```
idopnetwork
├── numpy, pandas, scipy, scikit-learn, matplotlib, cvxpy
└── torch (仅 [ml] 可选依赖)

idopnetwork-app
├── idopnetwork
└── streamlit
```

## 关键变更

### 1. 路径解析（核心库）

`idopnetwork` 不包含 static/ 和 data/，所有需要这些资源的路径：
- 由 `idopnetwork-app` 通过 importlib.resources 读取后注入
- 绘图函数增加 `font_prop` 参数，由调用方传入

`idopnetwork-app` 内部：
- `static/` 随包分发，通过 `importlib.resources.files("idopnetwork_app.static")` 访问
- `data/` 示例数据同理，通过 `importlib.resources` 访问
- 不再使用 `Path(__file__).parent.parent` 硬编码

### 2. 中文字体

- `SimHei.ttf` 放入 app 包的 static 中
- `setup_matplotlib_chinese()` 从 app 包中读取字体文件
- 核心库的 plot 函数接受 `font_prop` 参数，默认英文

### 3. 用户认证

- `users.json` 写入 `~/.idopnetwork/users.json`
- 从 git 历史中永久清除（不再跟踪）
- app 包首次运行时自动创建（含默认 admin 账号）

### 4. torch 可选

- 移至 `[project.optional-dependencies]` 的 `[ml]` 组
- `ml/` 子包 import 用 try/except 优雅降级

### 5. CLI 入口

- `idopnetwork-app` 提供 `idopnetwork-app` 命令
- 等价于 `streamlit run` 启动内置页面

### 6. 导入路径迁移

- `backend.xxx` → `idopnetwork.xxx`（核心库内导入）
- `backend.xxx` → `idopnetwork.xxx`（app 层从核心库导入）
- 图像/字体 → `importlib.resources.files("idopnetwork_app.static")`
- 示例数据 → `importlib.resources.files("idopnetwork_app.data")`

## 不变内容

- 页面功能逻辑完全不变
- 外部 API（所有核心函数的签名）不变
- 数据流（Curve Fitting → FunClu → NetRecon → NetAnal）不变
- matplotlib 绘图代码不变

## 从 git 历史清除的文件

- `backend/users.json`（含明文密码）

## 新增文件

- `packages/idopnetwork/LICENSE`（MIT）
- `packages/idopnetwork/src/idopnetwork/py.typed`（PEP 561）
- `packages/idopnetwork/README.md`
- `packages/idopnetwork-app/README.md`

## 删除的空占位文件

- `backend/ml/core.py`（0 行）
- `backend/ml/plot.py`（0 行）
- `backend/analysis/center.py`（0 行）
- `backend/analysis/plot_center.py`（0 行）
- `backend/analysis/plot_glmy.py`（0 行）
- `backend/__init__.py`（1 行空）

## 风险

- Streamlit 多页面要求 `pages/` 目录与入口脚本在同一层级，放入 `src/` 布局后需要处理（使用 `streamlit run` 的 `--` 参数或创建软链接/包装脚本）
