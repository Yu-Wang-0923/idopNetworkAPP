
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(6, 3), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()
    for i, col in enumerate(selected_cols[:nsubfig]):  # 最多画 nsubfig 个
        axes[i].scatter(df_scatter.index, df_scatter[col], alpha=0.95, s=100, facecolors='none', edgecolors='#F9B3AD', linewidth=1)
        if show_curve:
            axes[i].plot(df_curve.index, df_curve[col], color='#F8A09B', linewidth=4)
        axes[i].set_title(col, fontsize=11, fontproperties=font_prop)
        axes[i].margins(x=0.2, y=0.3)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
        for label in axes[i].get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in axes[i].get_yticklabels():
            label.set_fontproperties(font_prop)
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


import matplotlib.pyplot as plt
import streamlit as st

def plot_curve_fitting_compare(
    df_scatter_list,
    df_curve_list,
    label_list,
    show_curve=True,
    nrow=2,
    ncol=2,
    nsubfig=4,
):
    scatter_colors = ['#F9B3AD', '#C9A1CA', '#76C2AF', '#E5C68F', '#C59FCE', '#A0D8E7']
    curve_colors = ['#F8A09B', '#BC8FC1', '#52B793', '#DDB866', '#B488C2', '#79C6DF']
    
    selected_cols = df_scatter_list[0].columns.tolist()
    fig, axes = plt.subplots(nrow, ncol, figsize=(6, 3), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()

    # 用来存放所有图例句柄，统一画在顶部
    all_handles = []
    all_labels = []

    for i, col in enumerate(selected_cols[:nsubfig]):
        ax = axes[i]
        
        for idx, (df_scatter, df_curve) in enumerate(zip(df_scatter_list, df_curve_list)):
            color_scatter = scatter_colors[idx % len(scatter_colors)]
            color_curve = curve_colors[idx % len(scatter_colors)]
            label = label_list[idx]

            # 散点
            scatter = ax.scatter(
                df_scatter.index, df_scatter[col],
                alpha=0.95, s=100, facecolors='none',
                edgecolors=color_scatter, linewidth=1,
            )
            # 曲线
            curve = None
            if show_curve:
                curve = ax.plot(
                    df_curve.index, df_curve[col],
                    color=color_curve, linewidth=3,
                )[0]

            # 只在第一个子图收集图例
            if i == 0:
                all_handles.append(scatter)
                all_labels.append(f"{label} 原始")
                if show_curve:
                    all_handles.append(curve)
                    all_labels.append(f"{label} 拟合")
        
        # 样式
        ax.set_title(col, fontsize=11, fontproperties=font_prop)
        ax.margins(x=0.1, y=0.1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)

    # 图例放在【整张图顶部、居中、横向排列】
    fig.legend(
        all_handles, all_labels,
        loc='upper center',    # 顶部居中
        bbox_to_anchor=(0.5, 1.15),  # 稍微往上一点，不遮挡图
        ncol=4, 
        fontsize=8,
        frameon=False,
        prop=font_prop
    )

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)