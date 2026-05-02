
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



import io
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit








font_path = Path(__file__).parent.parent / "static" / "SimHei.ttf"
font_prop = fm.FontProperties(fname=font_path)



def plot_scatter_matrix(df, use_seq, n_cols, max_plots):
    x = np.arange(1, len(df)+1) if use_seq else df.index
    cols = st.columns(n_cols)
    selected_cols = df.columns[:max_plots]
    for i, col in enumerate(selected_cols):
        with cols[i % n_cols]:
            fig, ax = plt.subplots(figsize=(4,3)) # , dpi=300
            ax.scatter(x, df[col], s=150, alpha=0.7, facecolors='none', edgecolors='#4285F4', linewidth=1)
            ax.set_title(col, fontproperties=font_prop)
            ax.set_xlabel("Sequence" if use_seq else "Index", fontproperties=font_prop)
            ax.set_ylabel(col, fontproperties=font_prop)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_prop)
            ax.grid(alpha=0.3)
            ax.margins(x=0.2, y=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            st.pyplot(fig)
            plt.close()

def plot_curve_fitting(
    df_quasi_dynamic, 
    df_curve_sample,
    use_seq, 
    n_cols, 
    max_plots
):
    x = np.arange(1, len(df_quasi_dynamic)+1) if use_seq else df_quasi_dynamic.index
    a_index = df_curve_sample.index
    cols = st.columns(n_cols)
    selected_cols = df_quasi_dynamic.columns[:max_plots]
    for i, col in enumerate(selected_cols):
        with cols[i % n_cols]:
            fig, ax = plt.subplots(figsize=(4,3)) # , dpi=300
            ax.scatter(a_index, df_quasi_dynamic[col], s=150, alpha=0.7, facecolors='none', edgecolors='#4285F4', linewidth=1)
            ax.plot(a_index, df_curve_sample[col], alpha=0.7, color='#ff0000', linewidth=4)
            ax.set_title(col, fontproperties=font_prop)
            ax.set_xlabel("Sequence" if use_seq else "Index", fontproperties=font_prop)
            ax.set_ylabel(col, fontproperties=font_prop)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_prop)
            ax.grid(alpha=0.3)
            ax.margins(x=0.2, y=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            st.pyplot(fig)
            plt.close()