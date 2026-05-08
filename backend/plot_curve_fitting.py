
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
    color_scatter="#1F77B4",
    color_curve="#D62728",
    subfig_background_color="#FFFFFF",
):
    import io  # 用于将图片保存到内存以供下载

    # ==========================================
    # 🌟 1. 学术级全局参数注入 (SCI Publication Settings)
    # ==========================================
    plt.rcParams.update({
        "axes.linewidth": 1.2,        # 坐标轴边框加粗
        "xtick.direction": "in",      # X轴刻度朝内 (经典学术规范)
        "ytick.direction": "in",      # Y轴刻度朝内
        "xtick.major.width": 1.0,     # X轴刻度线加粗
        "ytick.major.width": 1.0,     # Y轴刻度线加粗
        "xtick.labelsize": 9,         # 刻度字体大小
        "ytick.labelsize": 9,
        "font.family": "sans-serif",  # 无衬线字体更具现代科技感
    })

    GOLDEN_RATIO = 1.618  # 黄金分割比
    base_height = 1.5     # 略微调大单行基准高度，让空间更充裕
    base_width = base_height * GOLDEN_RATIO 
    
    selected_cols = df_scatter.columns.tolist()
    fig, axes = plt.subplots(
        nrow, 
        ncol, 
        figsize=(base_width * ncol, base_height * nrow),
        sharex=True, 
        dpi=300
    )
    
    # 兼容处理：当只有 1x1 时，axes 不是数组，需要转一下
    if nrow * ncol == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, col in enumerate(selected_cols[:nsubfig]):  # 最多画 nsubfig 个
        if scatter_x == "index":
            x = df_scatter.index
        elif scatter_x == "sequence":
            x = np.arange(len(df_scatter))
        else:
            x = np.arange(len(df_scatter))  # fallback
        
        # --- 散点或线条绘制 ---
        if plot_scatter_type == "scatter":
            axes[i].scatter(
                x, 
                df_scatter[col], 
                alpha=0.75,          # 降低透明度，解决密集数据点糊成一团的问题
                s=scatter_size, 
                facecolors='none', 
                edgecolors=color_scatter, 
                linewidth=0.8,       # 散点边缘稍细，显得精致
                zorder=1             # 设置图层在底层
            )
        elif plot_scatter_type == "line":
            axes[i].plot(
                x, 
                df_scatter[col], 
                alpha=0.8,           # 线条增加层次感
                color=color_scatter, 
                linewidth=scatter_linewidth,
                zorder=1
            )

        # --- 拟合曲线绘制 ---
        if show_curve and df_curve is not None and col in df_curve.columns:
            axes[i].plot(
                df_curve.index, 
                df_curve[col], 
                color=color_curve, 
                linewidth=2.5,       # 线宽2.5刚刚好，既醒目又不过分粗糙
                zorder=5             # zorder=5 确保拟合线绝对压在散点上方
            )
        
        # 标题与刻度调整
        axes[i].set_title(col, fontsize=11, fontweight='bold', fontproperties=font_prop, pad=6)
        axes[i].margins(x=margins_x, y=margins_y)
        
        # 🌟 精简刻度数量：X轴和Y轴最多只保留 4 个刻度，避免刻度文字拥挤
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        
        # 开启顶部和右侧的内向刻度线 (顶级期刊的标准画法)
        axes[i].tick_params(top=True, right=True, direction='in')

        # 字体属性（兼容中英文）
        for label in axes[i].get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in axes[i].get_yticklabels():
            label.set_fontproperties(font_prop)
            
        axes[i].set_facecolor(subfig_background_color)

    # 隐藏多余子图
    n_plotted = min(len(selected_cols), nsubfig)
    for j in range(n_plotted, len(axes)):
        axes[j].set_visible(False)
        
    # 🌟 使用 tight_layout 替代手动调整，排版更智能、不重叠
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
    
    # 在 Streamlit 中渲染展示
    st.pyplot(fig)

    # ==========================================
    # 🌟 2. 学术级多格式导出引擎
    # ==========================================
    # 将图像保存到内存缓冲中 (不产生本地垃圾文件)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
    buf_png.seek(0)
    
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
    buf_pdf.seek(0)
    
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches='tight')
    buf_svg.seek(0)

    # 在界面上并排生成三个优雅的下载按钮
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    col_dl1, col_dl2, col_dl3, _ = st.columns([1.2, 1.2, 1.2, 2])
    with col_dl1:
        st.download_button("📥 高清 PNG (300dpi)", data=buf_png, file_name="idopNetwork_plot.png", mime="image/png")
    with col_dl2:
        st.download_button("📥 矢量 PDF", data=buf_pdf, file_name="idopNetwork_plot.pdf", mime="application/pdf")
    with col_dl3:
        st.download_button("📥 矢量 SVG", data=buf_svg, file_name="idopNetwork_plot.svg", mime="image/svg+xml")

    # 清理内存
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
        ax.margins(x=0.1, y=0.3)
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