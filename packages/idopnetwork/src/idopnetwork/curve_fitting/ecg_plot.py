# -*- coding: utf-8 -*-
"""
@author: Yu Wang (yuwang@bimsa.cn)
plot fitting
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_fitting(
    data_original: pd.DataFrame,
    data_fitted: pd.DataFrame,
    output_path: Path,
    sampling_rate: float = 100.0,
    fit_label: str = "SWT fit",
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """
    Plot each lead in a separate figure with full-record and zoom panels.

    Parameters
    ----------
    data_original : (n_timepoints, n_channels) | pd.DataFrame
        columns = ["time", "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    data_fitted : (n_timepoints, n_channels) | pd.DataFrame
        columns 同 data_original
    output_path : Path
        图片输出目录
    sampling_rate : float
        采样率 (Hz)，没有以秒为单位的 "time" 列时用于生成时间轴。

    Returns
    -------
    figure_paths : list[Path]
        各导联保存的图片路径，按输入列顺序排列。
    """
    lead_columns = [
        column for column in data_original.columns if column != "time"
    ]
    if data_original.empty or not lead_columns:
        raise ValueError("Original data must contain samples and ECG leads.")
    if len(data_original) != len(data_fitted):
        raise ValueError("Original and fitted data must have the same length.")
    missing = [lead for lead in lead_columns if lead not in data_fitted.columns]
    if missing:
        raise ValueError(f"Fitted data is missing leads: {missing}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if "time" in data_original.columns:
        time = data_original["time"].to_numpy(dtype=float)
    else:
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be finite and positive.")
        time = np.arange(len(data_original)) / sampling_rate
    zoom_count = min(250, len(data_original))
    figure_paths = []

    for lead in lead_columns:
        observed = data_original[lead].to_numpy(dtype=float)
        fitted = data_fitted[lead].to_numpy(dtype=float)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        try:
            axes[0].axvspan(time[0], time[zoom_count - 1], color="0.92", zorder=0)
            for ax, count, title in zip(
                axes,
                (len(time), zoom_count),
                (f"a  Lead {lead} · Full record",
                 f"b  Lead {lead} · First {zoom_count} samples"),
            ):
                ax.plot(
                    time[:count], observed[:count],
                    linestyle="none", marker="o", markersize=4.8,
                    markerfacecolor="none", markeredgecolor="0.3",
                    markeredgewidth=0.8, label="Observed", zorder=2,
                )
                ax.plot(
                    time[:count], fitted[:count],
                    color="#d73027", linewidth=1.4, label=fit_label, zorder=3,
                )
                ax.set_title(title, loc="left", fontsize=16, pad=10)
                ax.set_xlabel("Time (s)", fontsize=14)
                ax.set_ylabel("Signal (stored units)", fontsize=14)
                if count > 1:
                    ax.set_xlim(time[0], time[count - 1])
                ax.margins(y=0.12)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="both", labelsize=12, direction="out")
            axes[0].legend(
                loc="lower right", bbox_to_anchor=(1, 1.02),
                borderaxespad=0, frameon=False, ncol=2, fontsize=13,
            )
            figure_path = output_path / f"fitting_{len(figure_paths) + 1}.png"
            for extension in formats:
                if extension not in {"png", "pdf", "svg"}:
                    raise ValueError("formats must contain png, pdf or svg")
                fig.savefig(figure_path.with_suffix(f".{extension}"), dpi=300)
            figure_paths.append(figure_path)
        finally:
            plt.close(fig)
    return figure_paths

