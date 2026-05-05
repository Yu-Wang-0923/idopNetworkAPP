
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from backend.utils import font_prop


def plot_curve_fitting(
    # 数据
    df_scatter,
    df_curve,
    plot_scatter_type="line",
    show_curve=True, # 是否显示曲线


    # 布局
    nrow=4,
    ncol=3,
    nsubfig=4,
    scatter_x="sequence",
    scatter_size=100,
    scatter_linewidth=2,
    
    margins_x=0.1,
    margins_y=0.2,


    # 标签
    # title="",

    # 颜色
    color_scatter="#F9B3AD",
    color_curve="#A06EA5",
    subfig_background_color="#FFFFFF",
):
    GOLDEN_RATIO = 1.618  # 黄金分割比
    base_height = 1.25       # 单行基准高度（英寸）
    base_width = base_height * GOLDEN_RATIO  # 单列宽度更宽
    
    selected_cols = df_scatter.columns.tolist()
    fig, axes = plt.subplots(
        nrow, 
        ncol, 
        figsize=(base_width * ncol, base_height * nrow),  # 黄金分割
        sharex=True, 
        # sharey=True, 
        dpi=300)
    axes = np.array(axes).flatten()
    for i, col in enumerate(selected_cols[:nsubfig]):  # 最多画 nsubfig 个
        if scatter_x == "index":
            x = df_scatter.index
        elif scatter_x == "sequence":
            x = np.arange(len(df_scatter))
        else:
            x = np.arange(len(df_scatter))  # fallback
        
        if plot_scatter_type == "scatter":
            axes[i].scatter(
                x, 
                df_scatter[col], 
                alpha=0.95, 
                s=scatter_size, 
                facecolors='none', 
                edgecolors=color_scatter, 
                linewidth=1,
            )
        elif plot_scatter_type == "line":
            axes[i].plot(
                x, 
                df_scatter[col], 
                alpha=0.95,
                color=color_scatter, 
                linewidth=scatter_linewidth,
            )

        if show_curve and df_curve is not None and col in df_curve.columns:
            axes[i].plot(
                df_curve.index, 
                df_curve[col], 
                color=color_curve, 
                linewidth=4
            )
        axes[i].set_title(col, fontsize=11, fontproperties=font_prop)
        axes[i].margins(x=margins_x, y=margins_y)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
        for label in axes[i].get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in axes[i].get_yticklabels():
            label.set_fontproperties(font_prop)
        axes[i].set_facecolor(subfig_background_color)
    # 隐藏多余子图
    n_plotted = min(len(selected_cols), nsubfig)
    for j in range(n_plotted, len(axes)):
        axes[j].set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



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
    curve_colors = ['#9E2223', '#A06EA5', '#52B793', '#DDB866', '#B488C2', '#79C6DF']
    
    selected_cols = df_scatter_list[0].columns.tolist()
    fig, axes = plt.subplots(nrow, ncol, figsize=(6, 3), sharex=True, sharey=True, dpi=300)
    axes = np.array(axes).flatten()

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
    n_plotted = min(len(selected_cols), nsubfig)
    for j in range(n_plotted, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


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

# dark_colors = [
#     "#E87974",  # 加深蜜桃粉
#     "#D8A840",  # 加深奶油黄
#     "#5BA8D1",  # 加深天蓝色
#     "#A06EA5",  # 加深薰衣草紫
#     "#9FB4C2",  # 加深雾灰蓝
#     "#9E2223"   # 加深砖红
# ]