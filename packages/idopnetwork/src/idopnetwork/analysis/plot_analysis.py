"""Network Analysis 绘图模块：GLMY barcode。"""
from __future__ import annotations

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# β₀..β₃ 自下而上排列（与原始 GLMY1.py 风格一致：顶部 β₃，底部 β₀）。
_DIM_KEYS: tuple[str, ...] = ("3", "2", "1", "0")
_DIM_COLORS: tuple[str, ...] = ("purple", "g", "r", "b")
_DIM_LABELS: tuple[str, ...] = (r"$\beta_3$", r"$\beta_2$", r"$\beta_1$", r"$\beta_0$")


def plot_glmy_barcode(
    homology: dict[str, list[list[Any]]],
    *,
    max_x: float,
) -> plt.Figure:
    """绘制 GLMY barcode 图（β₀–β₃ 共 4 个子图）。

    Parameters
    ----------
    homology: 已减去 +100 偏移的 ``{dim: [[birth, death], ...]}``；``death == -1``
        代表无穷区间，绘制为带箭头的延伸线。
    max_x: 横轴右端位置；用于无穷条的延伸长度与坐标范围。

    Returns
    -------
    matplotlib.figure.Figure
        调用方负责 ``st.pyplot(fig)`` 与 ``plt.close(fig)``。
    """
    bars_per_dim: list[list[list[Any]]] = []
    for key in _DIM_KEYS:
        raw = list(homology.get(key, []))
        raw.sort(
            key=lambda x: (
                x[0],
                -(x[1] if x[1] != -1 else float("-inf")),
            )
        )
        bars_per_dim.append(raw)

    fig, axes = plt.subplots(
        len(_DIM_KEYS),
        1,
        figsize=(7, len(_DIM_KEYS) * 2.5),
        sharex=True,
    )
    if len(_DIM_KEYS) == 1:
        axes = [axes]

    safe_max_x = float(max_x) if max_x and max_x > 0 else 1.0
    visible_endpoints: list[float] = []

    for bars, ax, color, label in zip(bars_per_dim, axes, _DIM_COLORS, _DIM_LABELS):
        max_y = max(len(bars), 5)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
        ax.set_ylim(-1, max_y)
        ax.yaxis.labelpad = 20
        ax.tick_params(axis="y", which="major", labelsize=20)
        ax.tick_params(axis="x", which="major", labelsize=20)
        ax.set_ylabel(label, fontsize=24)

        for j, bar in enumerate(bars):
            if not bar:
                continue
            birth = float(bar[0])
            death = bar[1]
            if birth >= -safe_max_x:
                visible_endpoints.append(birth)
            if death == -1:
                ax.plot(
                    [birth, safe_max_x],
                    [j, j],
                    color=color,
                    linewidth=3,
                    alpha=0.8,
                )
                arrow = patches.FancyArrow(
                    safe_max_x,
                    j,
                    1e-6,
                    0,
                    width=0.5,
                    head_width=max_y / 8,
                    head_length=safe_max_x / 10,
                    color="black",
                    alpha=0.8,
                )
                ax.add_patch(arrow)
            else:
                finite_death = float(death)
                if finite_death >= -safe_max_x:
                    visible_endpoints.append(finite_death)
                ax.plot(
                    [birth, finite_death],
                    [j, j],
                    color=color,
                    linewidth=2.5,
                )

    min_visible_x = min(visible_endpoints, default=0.0)
    left_x = min(0.0, min_visible_x)
    if left_x < 0:
        left_x *= 1.1
    axes[-1].set_xlim([left_x, safe_max_x * 1.1])
    fig.tight_layout()
    return fig
