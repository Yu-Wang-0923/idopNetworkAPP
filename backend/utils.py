"""
共享工具模块：路径常量、CSS 注入、matplotlib 中文字体配置。

所有 static/ 资源的路径引用、页面 CSS 加载以及 matplotlib 中文字体
初始化均集中于此，避免各页面/模块重复硬编码。
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import streamlit as st

# ── 路径常量 ──────────────────────────────────────────────────────────────────

STATIC_DIR: Path = Path(__file__).parent.parent / "static"
CSS_PATH: Path = STATIC_DIR / "css" / "custom_style.css"
FONT_PATH: Path = STATIC_DIR / "SimHei.ttf"

# matplotlib 字体属性对象（供 plot_*.py 直接使用）
font_prop: fm.FontProperties = fm.FontProperties(fname=FONT_PATH)


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def load_css(path: Path = CSS_PATH) -> None:
    """将自定义 CSS 文件注入 Streamlit 页面。"""
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def setup_matplotlib_chinese() -> None:
    """配置 matplotlib 中文字体回退列表与负号显示。"""
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
