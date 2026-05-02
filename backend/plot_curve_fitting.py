
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
    show_scatter=True,
    nrow=2,
    ncol=2,
    nsubfig=4,  # 总子图数量
):
    selected_cols = df_scatter.columns.tolist()
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 4), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()
    for i, col in enumerate(selected_cols[:nsubfig]):  # 最多画 nsubfig 个
        if show_scatter:
            axes[i].scatter(df_scatter.index, df_scatter[col], alpha=0.6, s=200, facecolors='none', edgecolors='#F9B3AD', linewidth=1)
        axes[i].plot(df_curve.index, df_curve[col], color='#F8A09B', linewidth=2.5)
        axes[i].set_title(col, fontsize=11, fontproperties=font_prop)
        axes[i].margins(x=0.2, y=0.25)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()
    st.pyplot(fig, dpi)
    plt.close(fig)

    #, facecolors='none', edgecolors='#4285F4', linewidth=1