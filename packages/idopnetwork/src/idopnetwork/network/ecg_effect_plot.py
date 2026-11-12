# -*- coding: utf-8 -*-
"""
@author: Yu Wang (yuwang@bimsa.cn)
plot effect decomposition
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# --- 期刊排版契约 -----------------------------------------------------------
# 双栏宽 183 mm；正文/刻度 6.5-7 pt，最小字形 5 pt；文字保持可编辑
FIGURE_WIDTH_MM = 183.0
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def _panel_title(
    ax: plt.Axes,
    letter: str,
    text: str,
    letter_size: float,
    text_size: float,
) -> None:
    """左上角混排标题：小写粗体面板字母 + 常规字重的说明文字。"""
    # 用空标题占位，向 constrained_layout 预留与面板字母相当的高度
    ax.set_title("\u00a0", loc="left", fontsize=letter_size, pad=6)
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points",
                fontsize=letter_size, fontweight="bold", va="bottom", ha="left")
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0.78 * letter_size + 3, 3), textcoords="offset points",
                fontsize=text_size, va="bottom", ha="left")


def plot_effect_decomposition(
    effects: dict[str, pd.DataFrame],
    output_path: Path,
    start_time: float = 0.0,
    duration: float | None = 2.5,
    selected_leads: list[str] | None = None,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
) -> list[Path]:
    """Plot independent and separate cross effects over a short time window.

    Parameters
    ----------
    effects : dict[str, pd.DataFrame]
        get_effects 的返回值，至少包含 reconstruction、self_effects 和
        interaction_effects；也支持读取这三个 CSV 后构建的字典。
    output_path : Path
        图片目录。每个导联保存 300 DPI PNG 和可编辑文字的矢量 PDF/SVG。
    start_time : float
        相对记录起点的展示起始秒数，默认 0；横轴保留原始时间坐标。
    duration : float | None
        展示时长（秒），默认 2.5，None 表示展示到记录末尾。

    Notes
    -----
    上方全宽总览图：黑色 SWT 拟合信号、蓝色总效应、红色独立效应及
    各条绿色交互效应叠加；绿色交互项以不同线型区分。
    下方并列子图：独立效应（含记录初值）和每条交互效应各占一个子图，
    每行最多四个。各项均为积分后的信号贡献，单位与拟合信号一致。
    仅截取展示窗口，不在短窗重新拟合、不重置积分初值、不归一化幅值。
    窗口内保留全部样本；R²/RMSE 仅针对展示窗口计算。各子图独立设置纵轴
    范围以展示较小的交互效应，比较幅值时应结合纵轴刻度。

    Returns
    -------
    figure_paths : list[Path]
        按 reconstruction 中导联出现顺序排列的 PNG 路径。
    """
    if not np.isfinite(start_time) or start_time < 0:
        raise ValueError("start_time must be finite and nonnegative.")
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive, or None.")
    required_columns = {
        "reconstruction": [
            "time", "target", "smoothed", "self_trajectory", "interaction_sum",
            "reconstructed", "fitted_rate",
        ],
        "self_effects": [
            "time", "target", "rate", "cumulative_effect", "self_trajectory",
        ],
        "interaction_effects": ["time", "target", "source", "rate", "cumulative_effect"],
    }
    for name, columns in required_columns.items():
        if name not in effects or not set(columns).issubset(effects[name].columns):
            raise ValueError(f"{name} must contain columns: {columns}")
        table = effects[name]
        identifiers = [column for column in ("target", "source") if column in columns]
        numeric = [column for column in columns if column not in identifiers]
        if table[identifiers].isna().any().any() or not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains missing identifiers or non-finite values.")

    reconstruction = effects["reconstruction"]
    self_effects = effects["self_effects"]
    interactions = effects["interaction_effects"]
    leads = reconstruction["target"].unique().tolist()
    if not leads or set(self_effects["target"]) != set(leads):
        raise ValueError("Reconstruction and self effects must contain the same nonempty target set.")
    if not set(interactions["target"]).issubset(leads) or not set(interactions["source"]).issubset(leads):
        raise ValueError("Interaction effects contain an unknown source or target.")
    if (interactions["target"] == interactions["source"]).any():
        raise ValueError("Interaction effects must not contain self edges.")

    def align_time(table: pd.DataFrame, time: np.ndarray, label: str) -> pd.DataFrame:
        ordered = table.sort_values("time")
        if len(ordered) != len(time) or ordered["time"].duplicated().any():
            raise ValueError(f"{label} must have exactly one row per reconstruction time.")
        if not np.allclose(ordered["time"], time, rtol=0, atol=1e-10):
            raise ValueError(f"{label} time points do not match reconstruction.")
        return ordered

    # 先验证所有导联的时间对齐及加和关系，避免把错配的 CSV 画成有效分解。
    prepared = []
    for lead in leads:
        record = reconstruction.loc[reconstruction["target"] == lead].sort_values("time")
        time = record["time"].to_numpy(dtype=float)
        if len(time) < 2 or np.any(np.diff(time) <= 0):
            raise ValueError(f"Lead {lead} must have at least two distinct, increasing time points.")
        own = align_time(self_effects.loc[self_effects["target"] == lead], time, f"Self {lead}")
        edges = [
            (source, align_time(table, time, f"{source} -> {lead}"))
            for source, table in interactions.loc[interactions["target"] == lead].groupby("source", sort=False)
        ]
        cumulative_sum = np.zeros(len(time))
        rate_sum = own["rate"].to_numpy(dtype=float).copy()
        for _, edge in edges:
            cumulative_sum += edge["cumulative_effect"].to_numpy(dtype=float)
            rate_sum += edge["rate"].to_numpy(dtype=float)
        checks = (
            (own["self_trajectory"], record["self_trajectory"]),
            (own["self_trajectory"], record["smoothed"].iloc[0] + own["cumulative_effect"]),
            (cumulative_sum, record["interaction_sum"]),
            (own["self_trajectory"].to_numpy() + cumulative_sum, record["reconstructed"]),
            (rate_sum, record["fitted_rate"]),
        )
        if not all(np.allclose(left, right, rtol=1e-7, atol=1e-8) for left, right in checks):
            raise ValueError(f"Effect sums do not match reconstruction for lead {lead}.")
        window_start = time[0] + start_time
        window_end = min(time[-1], window_start + duration) if duration is not None else time[-1]
        mask = (time >= window_start - 1e-10) & (time <= window_end + 1e-10)
        if mask.sum() < 2:
            raise ValueError(f"Display window contains fewer than two samples for lead {lead}.")
        prepared.append((
            lead, record.iloc[mask], own.iloc[mask],
            [(source, edge.iloc[mask]) for source, edge in edges],
            window_start, window_end,
        ))

    if selected_leads is not None:
        if not selected_leads or not set(selected_leads).issubset(leads):
            raise ValueError("selected_leads must contain known targets")
        prepared = [item for item in prepared if item[0] in selected_leads]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    independent_color = "#D73027"
    total_color = "#2166D1"
    # 交互效应各自取色, 避开红/蓝/黑以免与独立效应、总和、观测混淆
    cross_colors = ["#1B7837", "#E08214", "#7B3294", "#8C510A", "#C51B7D"]
    figure_paths = []
    with plt.rc_context(STYLE):
        for lead, record, own, edges, window_start, window_end in prepared:
            time = record["time"].to_numpy(dtype=float)
            n_components = len(edges) + 1
            n_columns = min(4, n_components)
            n_component_rows = int(np.ceil(n_components / n_columns))
            # 双栏 183 mm；总览 + 一行分量约 88 mm，多一行再加 42 mm
            fig = plt.figure(
                figsize=(FIGURE_WIDTH_MM / 25.4,
                         (88.0 + 42.0 * (n_component_rows - 1)) / 25.4),
                constrained_layout=True,
            )
            grid = fig.add_gridspec(
                1 + n_component_rows, n_columns,
                height_ratios=[2.2] + [1.25] * n_component_rows,
            )
            edge_colors = [cross_colors[index % len(cross_colors)]
                           for index in range(len(edges))]
            try:
                overview = fig.add_subplot(grid[0, :])
                axes = [overview]
                overview.plot(
                    time, record["smoothed"], color="black", linewidth=0.7,
                    label="SWT Fitted Signal", zorder=4,
                )
                overview.plot(
                    time, record["reconstructed"], color=total_color, linestyle="--",
                    linewidth=0.9, label="Total Effect", zorder=5,
                )
                overview.plot(
                    time, own["self_trajectory"], color=independent_color,
                    linewidth=0.9, label="Independent Effect", zorder=3,
                )
                line_styles = ("-", "--", "-.", ":")
                for index, (source, edge) in enumerate(edges):
                    overview.plot(
                        time, edge["cumulative_effect"], color=edge_colors[index],
                        linestyle=line_styles[index % len(line_styles)], linewidth=0.8,
                        label=f"Cross {source} → {lead}", zorder=2,
                    )
                _panel_title(overview, "a",
                             f"Lead {lead} · Combined Decomposition", 10, 9)
                overview.set_ylabel("Signal", fontsize=9)
                overview.set_xlabel("Time (s)", fontsize=9)

                components = [("Independent Effect", own["self_trajectory"],
                               independent_color)]
                components.extend(
                    (f"Cross {source} → {lead}", edge["cumulative_effect"],
                     edge_colors[index])
                    for index, (source, edge) in enumerate(edges)
                )
                for index, (label, values, color) in enumerate(components):
                    row, column = divmod(index, n_columns)
                    ax = fig.add_subplot(grid[1 + row, column])
                    axes.append(ax)
                    ax.plot(time, values, color=color, linewidth=0.8)
                    _panel_title(ax, chr(98 + index), label, 10, 9)
                    ax.set_xlabel("Time (s)", fontsize=8.5)
                    ax.set_ylabel("Signal", fontsize=8.5)
                for ax in axes:
                    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, zorder=0)
                    ax.set_xlim(window_start, window_end)
                    ax.margins(y=0.12)
                    ax.tick_params(axis="both", labelsize=8, direction="out",
                                   width=0.6, length=2.5)
                handles, labels = overview.get_legend_handles_labels()
                fig.legend(
                    handles, labels, loc="outside lower center",
                    ncol=min(3, len(labels)), frameon=False, fontsize=8.5,
                )
                figure_path = output_path / f"effect_decomposition_{len(figure_paths) + 1}.png"
                # 不裁剪, 保证导出物理尺寸与声明的 183 mm 完全一致
                for extension in formats:
                    if extension not in {"png", "pdf", "svg", "tiff"}:
                        raise ValueError("Unsupported figure format")
                    fig.savefig(figure_path.with_suffix(f".{extension}"), dpi=600)
                figure_paths.append(figure_path)
            finally:
                plt.close(fig)
    return figure_paths

