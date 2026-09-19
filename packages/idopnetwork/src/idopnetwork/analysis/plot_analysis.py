"""Network Analysis 绘图模块：GLMY barcode（新版风格）。

相对旧版绘图的变化：

- x 轴**以 0 为对称中心**（旧版由 ``max_x`` 直接给出左右端，右端不对称）；
- 无穷条在右端用黑色 ``>`` 箭头标记（旧版用 ``FancyArrow`` patch）；
- y 轴范围由 ``shared_bar_counts`` 决定，可在多个文件 / 多个子图之间共享高度；
- 新增正 / 负权重拆分的 ``plot_glmy_barcode_split``（4 行 × 2 列）。

兼容性
------
``plot_glmy_barcode`` 仍接受 ``max_x``：给定它就作为对称半轴长（保留页面与
M3 / Paper §3.2 自检既有的横轴控件语义），不给则由数据自动推算。
````-1`` 与 ``None`` 都被识别为无穷哨兵``，因此新版 compute（``None``）与自检
路径（``-1``）产出的 homology 都能直接绘制。
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from idopnetwork.analysis.glmy import (
    DIMENSION_COLORS,
    DIMENSION_KEYS,
    DIMENSION_LABELS,
    bar_counts,
    bar_counts_split,
)

# 旧名保留，避免外部按旧常量导入时断裂
_DIM_KEYS = DIMENSION_KEYS
_DIM_COLORS = DIMENSION_COLORS
_DIM_LABELS = DIMENSION_LABELS


def _sorted_bars(
    homology: dict[str, list[list[Any]]],
    key: str,
) -> list[list[float | None]]:
    """取某一维度的条，统一无穷哨兵并排序。

    ``-1``（旧版自检路径）与 ``None``（新版 compute）都归一化成 ``None``。
    排序规则与新版一致：先按 birth 升序（无穷最小），再按 death 降序（无穷最小）。
    """
    bars: list[list[float | None]] = []
    for bar in homology.get(key, []):
        if not bar:
            continue
        birth, death = bar[0], bar[1]
        bars.append(
            [
                None if birth is None or birth == -1 else float(birth),
                None if death is None or death == -1 else float(death),
            ]
        )

    bars.sort(
        key=lambda bar: (
            float("-inf") if bar[0] is None else bar[0],
            float("-inf") if bar[1] is None else -bar[1],
        )
    )
    return bars


def _finite_endpoints(bars: list[list[float | None]]) -> list[float]:
    return [
        value
        for birth, death in bars
        for value in (birth, death)
        if value is not None
    ]


def _draw_bars(
    ax: plt.Axes,
    bars: list[list[float | None]],
    *,
    color: str,
    start_for_infinite: float,
    infinite_x: float,
) -> None:
    """把一维的条画到 ``ax`` 上；无穷条末端带黑色 ``>`` 箭头。"""
    for index, (birth, death) in enumerate(bars):
        start_x = start_for_infinite if birth is None else birth
        if death is None:
            ax.plot(
                [start_x, infinite_x],
                [index, index],
                color=color,
                linewidth=3,
                alpha=0.8,
            )
            ax.plot(
                infinite_x,
                index,
                marker=">",
                markersize=12,
                color="black",
                markeredgecolor="black",
            )
        else:
            ax.plot(
                [start_x, death],
                [index, index],
                color=color,
                linewidth=2.5,
            )


def plot_glmy_barcode(
    homology: dict[str, list[list[Any]]],
    shared_bar_counts: dict[str, int] | None = None,
    *,
    max_x: float | None = None,
) -> plt.Figure:
    """绘制 GLMY barcode（β₃–β₀ 自上而下共 4 个子图，x 轴以 0 为中心）。

    Parameters
    ----------
    homology: ``{dim: [[birth, death], ...]}``；``None`` 或 ``-1`` 表示无穷区间。
    shared_bar_counts: 每个维度的 y 轴高度；``None`` 时由 ``homology`` 自推。
    max_x: 可选的对称半轴长；给定则横轴为 ``[-max_x*1.1, max_x*1.1]``，
        否则由数据端点的最大绝对值自动推算。

    Returns
    -------
    matplotlib.figure.Figure
        调用方负责 ``st.pyplot(fig)`` 与 ``plt.close(fig)``。
    """
    if shared_bar_counts is None:
        shared_bar_counts = bar_counts(homology)

    bars_per_dimension = [
        _sorted_bars(homology, key) for key in DIMENSION_KEYS
    ]

    if max_x is not None and float(max_x) > 0:
        half_span = float(max_x)
    else:
        half_span = max(
            (abs(value) for bars in bars_per_dimension
             for value in _finite_endpoints(bars)),
            default=0.0,
        )
        if half_span == 0.0:
            half_span = 1.0

    left_x = -half_span * 1.10
    right_x = half_span * 1.10
    infinite_x = half_span * 1.07

    fig, axes = plt.subplots(
        len(DIMENSION_KEYS),
        1,
        figsize=(7, 10),
        sharex=True,
    )
    if len(DIMENSION_KEYS) == 1:
        axes = [axes]

    for key, bars, ax, color, label in zip(
        DIMENSION_KEYS,
        bars_per_dimension,
        axes,
        DIMENSION_COLORS,
        DIMENSION_LABELS,
    ):
        maximum_y = max(shared_bar_counts.get(key, len(bars)), 5)
        y_margin = max(1.0, maximum_y * 0.05)
        ax.set_ylim(-y_margin, maximum_y - 1 + y_margin)
        ax.set_ylabel(label, fontsize=24, labelpad=20)
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(integer=True, nbins=3)
        )
        ax.tick_params(axis="both", which="major", labelsize=20)

        _draw_bars(
            ax,
            bars,
            color=color,
            start_for_infinite=left_x,
            infinite_x=infinite_x,
        )

    axes[-1].set_xlim(left_x, right_x)
    fig.tight_layout()
    return fig


def plot_glmy_barcode_split(
    homologies: dict[str, dict[str, list[list[Any]]]],
    shared_bar_counts: dict[str, dict[str, int]] | None = None,
) -> plt.Figure:
    """把正 / 负权重两组的 barcode 并排绘制（4 行 × 2 列，共 8 个子图）。

    Parameters
    ----------
    homologies: 键为 ``"positive"`` / ``"negative"``，值为各自的 homology。
    shared_bar_counts: ``{sign: {dim: count}}``；``None`` 时由入参自推。

    Returns
    -------
    matplotlib.figure.Figure
    """
    if shared_bar_counts is None:
        shared_bar_counts = bar_counts_split(homologies)

    figure, axes = plt.subplots(
        len(DIMENSION_KEYS),
        2,
        figsize=(7, 10),
        sharex=False,
        constrained_layout=True,
    )
    if len(DIMENSION_KEYS) == 1:
        axes = [axes]

    for row, key in enumerate(DIMENSION_KEYS):
        for column, sign in enumerate(("positive", "negative")):
            ax = axes[row][column]
            color = DIMENSION_COLORS[row]

            bars = _sorted_bars(homologies.get(sign, {}), key)

            finite_endpoints = _finite_endpoints(bars)
            max_x = max(finite_endpoints, default=1.0)
            if max_x == 0.0:
                max_x = 1.0

            counts = shared_bar_counts.get(sign, {}).get(key, len(bars))
            maximum_y = max(counts, 5)
            y_margin = max(1.0, maximum_y * 0.05)
            ax.set_ylim(-y_margin, maximum_y - 1 + y_margin)
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(integer=True, nbins=3)
            )
            ax.tick_params(axis="both", which="major", labelsize=20)

            if column == 0:
                ax.set_ylabel(
                    DIMENSION_LABELS[row],
                    fontsize=24,
                    labelpad=20,
                )
            if row == 0:
                ax.set_title(sign, fontsize=28)

            # 两侧渲染口径一致：横轴 0 在最左、|w| 最大值在最右，箭头向右。
            ax.set_xlim(0.0, max_x * 1.10)

            _draw_bars(
                ax,
                bars,
                color=color,
                start_for_infinite=0.0,
                infinite_x=max_x * 1.07,
            )

    return figure
