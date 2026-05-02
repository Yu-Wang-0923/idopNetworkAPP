
import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = Path(__file__).parent.parent / "static" / "SimHei.ttf"
font_prop = fm.FontProperties(fname=font_path)


def plot_curve_fitting(
    df_scatter, 
    df_curve,
    show_curve=True,
    nrow=2,
    ncol=2,
    nsubfig=4,  # 总子图数量
):
    selected_cols = df_scatter.columns.tolist()
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 4), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()
    for i, col in enumerate(selected_cols[:nsubfig]):  # 最多画 nsubfig 个
        axes[i].scatter(df_scatter.index, df_scatter[col], alpha=1, s=100, facecolors='none', edgecolors='#F9B3AD', linewidth=1)
        if show_curve:
            axes[i].plot(df_curve.index, df_curve[col], color='#F8A09B', linewidth=4)
        axes[i].set_title(col, fontsize=11, fontproperties=font_prop)
        axes[i].margins(x=0.2, y=0.25)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    #, facecolors='none', edgecolors='#4285F4', linewidth=1


# # 浅色系列（适合填充、背景）
# light_colors = [
#     "#F9B3AD",  # 浅蜜桃粉
#     "#EDC66A",  # 奶油黄
#     "#9FDAF7",  # 婴儿蓝
#     "#C9A1CA",  # 淡薰衣草紫
#     "#D0DEE7",  # 浅雾灰蓝
#     "#E59A9A"   # 浅豆沙红（对应第一排最右）
# ]

# # 深色系列（适合线条、描边、强调）
# dark_colors = [
#     "#F8A09B",  # 深蜜桃粉
#     "#E9C060",  # 深奶油黄
#     "#89CFF0",  # 天蓝色
#     "#BC8FC1",  # 深薰衣草紫
#     "#C8D7E0",  # 深雾灰蓝
#     "#C13A3B"   # 深砖红
# ]