
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*5, nrow*4))
    axes = axes.flatten()
    for i, col in enumerate(selected_cols[:nsubfig]):  # 最多画 nsubfig 个
        x = df_scatter.index
        if show_scatter:
            axes[i].scatter(x, df_scatter[col], alpha=0.6, s=25)
        axes[i].plot(x, df_curve[col], linewidth=2.5)
        axes[i].set_title(col, fontsize=11)
        axes[i].margins(x=0.2, y=0.3)
    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)