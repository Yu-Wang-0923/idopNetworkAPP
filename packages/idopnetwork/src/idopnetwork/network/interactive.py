# -*- coding: utf-8 -*-
"""
plot_network.py

独立版「CSV → 交互式 HTML 网络图」模块。

代码主体按原样逐行移植自 `src/蛋白.py` 的交互式 HTML 网络机制：

    - 常量区   ：原文件第 374-517 行（环形布局 / 颜色 / 阈值 / HTML 显示参数）
    - 函数     ：原文件第 4636-9764 行的 15 个函数
                 (_stable_integer … write_interactive_network_html)
    - 辅助函数 ：module_number / sort_modules / make_degree

本文件在其上新增了入口函数 `plot_network()` 与命令行用法，输入接口与
`network_data.py` 的输出一致（source / target / weight / sign 四列，
也兼容源脚本 07_FINAL_network_edges.csv 的三列边表）。

与源脚本 14_network.html 相同的功能
----------------------------------
    - Hub-centered concentric-ring IDOP 网络（单细胞同款环形布局）
    - 与弱边过滤严格同步的 in/out-degree 图
    - 交互控件：弱边 slider、节点 / 标签 / 边缩放与透明度
    - Effect decomposition 下载入口：仅当输出目录中存在源脚本生成的
      15_effect_decomposition 数据时提供，否则自动跳过（不影响主图）

用法
----
与 `network_data.py` 无缝衔接：

    import network_data
    import plot_network

    df = network_data.generate_network_data(n=12, p=0.2)   # DataFrame
    plot_network.plot_network(df, "14_network.html")

也可以直接吃 `network_data.py` 生成的 CSV（运行 `python network_data.py`
会写出 `output/network.csv`）：

    plot_network.plot_network(
        "output/network.csv",
        "output/network.html",
        title="Network | Functional Modules",
    )

命令行：

    无参数直接运行即可（默认读取 network_data.py 生成的 output/network.csv，
    写出 output/network.html）：

        python plot_network.py

    需要自定义时：

        python plot_network.py --csv output/network.csv --out output/network.html \\
            [--title ...] [--group-name ...] [--group-dir ...]

输入接口
--------
    `plot_network()` 的第一个参数既可以是 CSV 路径（str / os.PathLike），
    也可以是 pandas.DataFrame —— 即 `network_data.generate_network_data()`
    的返回值。

    要求至少包含三列：source、target、weight。
    `network_data` 额外输出的 sign 列（positive / negative）与 weight 的
    符号冗余，会被自动忽略，不影响结果。

    模块集合自动取 source ∪ target，并按源脚本规则排序后写入模块级全局
    MODULES（节点名统一按字符串处理，"0".."11" 会按数值顺序排列）。
"""


import hashlib
import json
import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_CENTER_HUBS = 3

LAYOUT_METRIC = "weighted_out_abs"


CENTER_HUB_RADIUS = 2.70

FIRST_NONHUB_RING_RADIUS = 6.50

SECOND_NONHUB_RING_RADIUS = 10.20


RING_GAP_BASE = 2.25

RING_GAP_GROWTH = 0.20


RING_CAPACITY_START = 8

RING_CAPACITY_STEP = 4

RING_CAPACITY_MAX = 40


RING_ROTATION_DEG = 23.0

EVEN_RING_HALF_STEP = True


LAYOUT_SEED = 20260903

ANGLE_JITTER_FRACTION = 0.004

RADIUS_JITTER = 0.0015


# -----------------------------------------------------------------------------
# 节点
# -----------------------------------------------------------------------------

HUB_NODE_DIAMETER = 35.0

FIRST_RING_DIAMETER = 28.0

RING_DIAMETER_DECAY = 4.6

MIN_NODE_DIAMETER = 7.5


RING_COLOR_PALETTE = [

    "#D73027",
    "#F46D43",
    "#FDAE61",
    "#FEE08B",
    "#D9EF8B",
    "#A6D96A",
    "#66C2A5",
    "#74ADD1",
    "#4575B4",
    "#7B68EE",
    "#9E9AC8",

]


NODE_EDGE_COLOR = "#555555"

TEXT_COLOR = "#202020"


# -----------------------------------------------------------------------------
# 边
# -----------------------------------------------------------------------------

POSITIVE_EDGE_COLOR = "#F08A5D"

NEGATIVE_EDGE_COLOR = "#4EA3F1"


TOPK_PER_TARGET = 8

TOPK_PER_SOURCE = 5

EDGE_QUANTILE = 0.60


MIN_DISPLAY_EDGES = 260

MAX_DISPLAY_EDGES = 700


EDGE_WIDTH_STRONG = 1.55

EDGE_WIDTH_MEDIUM = 0.95

EDGE_WIDTH_WEAK = 0.40


RECIPROCAL_CURVATURE = 0.085

CURVATURE_BASE = 0.018

CURVATURE_STEP = 0.008


# -----------------------------------------------------------------------------
# HTML 默认显示
# -----------------------------------------------------------------------------

HTML_HEIGHT = 950

POSITION_SCALE = 78.0


DEFAULT_NODE_SCALE = 1.5

DEFAULT_LABEL_SCALE = 1.5

DEFAULT_EDGE_SCALE = 1.8

DEFAULT_EDGE_OPACITY = 1.0

# 注意：边的高亮色用的是 min(1.0, edgeOpacity + 0.30)。默认取 1.0 时边已完全不透明，
# 因此悬停不会再改变边的透明度——这是"已经完全不透明就没法再提亮"的固有限制，
# 若需要明显的悬停反馈可把默认调低（例如 0.85）。

DEFAULT_EDGE_PERCENTILE = 0


# =============================================================================
# 一体化 HTML：Network + filter-synchronized Degree + Effect downloads
# =============================================================================

DEGREE_PANEL_HEIGHT = 950

IN_DEGREE_COLOR = "#4EA3F1"

OUT_DEGREE_COLOR = "#F08A5D"


# Effect decomposition 图不直接塞进 HTML 主区域。
# 构网完成后预先输出 PNG/PDF，并在 HTML 中提供下载入口。
SAVE_EFFECT_PNG = True

SAVE_EFFECT_PDF = True

EFFECT_DPI = 300


# =============================================================================
# 运行时模块列表
#
# 移植说明：源脚本中 MODULES 由 labels 数据在运行时推导
# （MODULES = sort_modules(labels["module"].unique())）。
#
# 独立版本在调用 plot_network() 时，会依据输入 CSV 中的节点集合
# （source ∪ target）自动计算并赋值，一般无需手动修改。
# =============================================================================

MODULES = []


def _resolve_modules(modules=None):
    """返回本次调用使用的模块顺序。

    原文件是独立脚本，用模块级全局 ``MODULES`` 在各函数之间传递状态。本包在
    Streamlit 中会被多会话并发调用，若继续依赖全局变量，后一个会话会覆盖模块
    列表，导致前一个会话的 HTML 用错节点集合（跨会话串数据）。因此改为显式传
    参，仅在未传参时回退到全局变量，独立脚本的原有用法仍然可用。
    """
    if modules is not None:
        return list(modules)
    return list(MODULES)


def module_number(x):

    s = str(x).strip()

    if s.upper().startswith("M"):

        try:
            return int(float(s[1:]))

        except Exception:
            return 10**9

    try:
        return int(float(s))

    except Exception:
        return 10**9


def sort_modules(x):

    return sorted(
        [str(i) for i in x],
        key=module_number
    )


def make_degree(
    edges,
    modules=None,
):

    rows = []


    for module in _resolve_modules(modules):

        outgoing = edges[
            edges[
                "source"
            ]
            ==
            module
        ]


        incoming = edges[
            edges[
                "target"
            ]
            ==
            module
        ]


        rows.append(
            {
                "module":
                    module,

                "out_degree":
                    len(
                        outgoing
                    ),

                "in_degree":
                    len(
                        incoming
                    ),

                "total_degree":
                    len(
                        outgoing
                    )
                    +
                    len(
                        incoming
                    ),

                "positive_out":
                    int(
                        (
                            outgoing[
                                "weight"
                            ]
                            >
                            0
                        ).sum()
                    ),

                "negative_out":
                    int(
                        (
                            outgoing[
                                "weight"
                            ]
                            <
                            0
                        ).sum()
                    ),

                "positive_in":
                    int(
                        (
                            incoming[
                                "weight"
                            ]
                            >
                            0
                        ).sum()
                    ),

                "negative_in":
                    int(
                        (
                            incoming[
                                "weight"
                            ]
                            <
                            0
                        ).sum()
                    ),

                "weighted_out_abs":
                    float(
                        outgoing[
                            "weight"
                        ]
                        .abs()
                        .sum()
                    ),

                "weighted_in_abs":
                    float(
                        incoming[
                            "weight"
                        ]
                        .abs()
                        .sum()
                    ),
            }
        )


    return (

        pd.DataFrame(
            rows
        )

        .sort_values(

            [
                "out_degree",
                "total_degree",
                "weighted_out_abs",
            ],

            ascending=[
                False,
                False,
                False
            ]
        )

        .reset_index(
            drop=True
        )
    )


def _stable_integer(
    text,
):

    digest = hashlib.md5(
        str(text).encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[
            :8
        ],
        16
    )


def _ring_capacity(
    ring_index,
):

    return int(

        min(

            RING_CAPACITY_START

            +

            ring_index
            *
            RING_CAPACITY_STEP,

            RING_CAPACITY_MAX
        )
    )


def _split_nodes_to_rings(
    nodes,
    start_ring_index=0,
):

    nodes = list(
        nodes
    )

    rings = []

    cursor = 0

    ring_index = int(
        start_ring_index
    )


    while cursor < len(
        nodes
    ):

        capacity = (
            _ring_capacity(
                ring_index
            )
        )


        current = nodes[
            cursor:
            cursor
            +
            capacity
        ]


        rings.append(
            {
                "ring_index":
                    ring_index,

                "nodes":
                    current,
            }
        )


        cursor += len(
            current
        )


        ring_index += 1


    return rings


def _ring_radius(
    ring_number,
):

    ring_number = int(
        ring_number
    )


    if ring_number == 0:

        return 0.0


    if ring_number == 1:

        return float(
            FIRST_NONHUB_RING_RADIUS
        )


    if ring_number == 2:

        return float(
            SECOND_NONHUB_RING_RADIUS
        )


    radius = float(
        SECOND_NONHUB_RING_RADIUS
    )


    for current_ring in range(
        3,
        ring_number
        +
        1
    ):

        gap = (

            RING_GAP_BASE

            +

            (
                current_ring
                -
                3
            )

            *

            RING_GAP_GROWTH
        )


        radius += gap


    return float(
        radius
    )


def _node_diameter(
    ring_number,
):

    ring_number = int(
        ring_number
    )


    if ring_number == 0:

        return float(
            HUB_NODE_DIAMETER
        )


    diameter = (

        FIRST_RING_DIAMETER

        -

        (
            ring_number
            -
            1
        )

        *

        RING_DIAMETER_DECAY
    )


    return float(

        max(
            diameter,
            MIN_NODE_DIAMETER
        )
    )


def _ring_color(
    ring_number,
):

    return RING_COLOR_PALETTE[

        int(
            ring_number
        )

        %

        len(
            RING_COLOR_PALETTE
        )
    ]


def _select_display_edges(
    edges,
):

    if edges.empty:

        return edges.copy()


    work = edges.copy()


    work[
        "abs_weight"
    ] = (
        work[
            "weight"
        ]
        .abs()
    )


    threshold = float(

        work[
            "abs_weight"
        ]
        .quantile(
            EDGE_QUANTILE
        )
    )


    selected_parts = []


    for _, g in work.groupby(
        "target",
        sort=False
    ):

        selected_parts.append(

            g
            .sort_values(
                "abs_weight",
                ascending=False
            )
            .head(
                TOPK_PER_TARGET
            )
        )


    for _, g in work.groupby(
        "source",
        sort=False
    ):

        selected_parts.append(

            g
            .sort_values(
                "abs_weight",
                ascending=False
            )
            .head(
                TOPK_PER_SOURCE
            )
        )


    selected_parts.append(

        work.loc[
            work[
                "abs_weight"
            ]
            >=
            threshold
        ]
    )


    selected = (

        pd.concat(
            selected_parts,
            ignore_index=True
        )

        .drop_duplicates(
            subset=[
                "source",
                "target",
            ]
        )
    )


    min_required = min(

        MIN_DISPLAY_EDGES,

        len(
            work
        )
    )


    if len(
        selected
    ) < min_required:

        supplement = (

            work

            .sort_values(
                "abs_weight",
                ascending=False
            )

            .head(
                min_required
            )
        )


        selected = (

            pd.concat(
                [
                    selected,
                    supplement,
                ],
                ignore_index=True
            )

            .drop_duplicates(
                subset=[
                    "source",
                    "target",
                ]
            )
        )


    selected = (

        selected

        .sort_values(
            "abs_weight",
            ascending=False
        )

        .head(
            MAX_DISPLAY_EDGES
        )

        .reset_index(
            drop=True
        )
    )


    return selected


def _build_ring_layout(
    degree_df,
    group_name,
    modules=None,
):

    work = (
        degree_df
        .copy()
    )


    work[
        "module"
    ] = (
        work[
            "module"
        ]
        .astype(str)
    )


    work = (

        work

        .set_index(
            "module"
        )

        .reindex(
            _resolve_modules(modules)
        )

        .fillna(
            0.0
        )
    )


    # -------------------------------------------------------------
    # 与单细胞图一致：
    # weighted out-degree 最大的节点优先进入中心/内圈
    # -------------------------------------------------------------

    nonzero_df = (

        work.loc[
            work[
                "weighted_out_abs"
            ]
            >
            0
        ]

        .copy()

        .sort_values(

            [
                "weighted_out_abs",
                "out_degree",
                "total_degree",
            ],

            ascending=[
                False,
                False,
                False,
            ]
        )
    )


    hub_nodes = (

        nonzero_df

        .head(
            N_CENTER_HUBS
        )

        .index
        .astype(str)
        .tolist()
    )


    remaining_nonzero = [

        node

        for node in (
            nonzero_df
            .index
            .astype(str)
            .tolist()
        )

        if node not in hub_nodes
    ]


    zero_nodes = (

        work.loc[
            work[
                "weighted_out_abs"
            ]
            <=
            0
        ]

        .index
        .astype(str)
        .tolist()
    )


    rng = np.random.default_rng(

        LAYOUT_SEED

        +

        (
            _stable_integer(
                group_name
            )
            %
            10000
        )
    )


    # zero out-degree 的节点最外圈随机放
    rng.shuffle(
        zero_nodes
    )


    nonzero_rings = (
        _split_nodes_to_rings(
            remaining_nonzero,
            start_ring_index=0,
        )
    )


    ring_records = []


    for item in nonzero_rings:

        ring_records.append(
            {
                "ring_number":
                    item[
                        "ring_index"
                    ]
                    +
                    1,

                "nodes":
                    item[
                        "nodes"
                    ],

                "zero_ring":
                    False,
            }
        )


    next_ring_index = (

        ring_records[
            -1
        ][
            "ring_number"
        ]

        if ring_records

        else 0
    )


    zero_rings = (
        _split_nodes_to_rings(
            zero_nodes,
            start_ring_index=
                next_ring_index,
        )
    )


    for item in zero_rings:

        ring_records.append(
            {
                "ring_number":
                    item[
                        "ring_index"
                    ]
                    +
                    1,

                "nodes":
                    item[
                        "nodes"
                    ],

                "zero_ring":
                    True,
            }
        )


    positions = {}

    ring_map = {}


    # -------------------------------------------------------------
    # center hubs
    # -------------------------------------------------------------

    if len(
        hub_nodes
    ) == 1:

        node = hub_nodes[
            0
        ]


        positions[
            node
        ] = (
            0.0,
            0.0
        )


        ring_map[
            node
        ] = 0


    elif len(
        hub_nodes
    ) > 1:

        for i, node in enumerate(
            hub_nodes
        ):

            angle = (

                np.pi
                /
                2

                -

                2
                *
                np.pi
                *
                i
                /
                len(
                    hub_nodes
                )
            )


            positions[
                node
            ] = (

                CENTER_HUB_RADIUS
                *
                np.cos(
                    angle
                ),

                CENTER_HUB_RADIUS
                *
                np.sin(
                    angle
                ),
            )


            ring_map[
                node
            ] = 0


    # -------------------------------------------------------------
    # outer rings
    # -------------------------------------------------------------

    for record in ring_records:

        ring_number = int(
            record[
                "ring_number"
            ]
        )


        nodes = list(
            record[
                "nodes"
            ]
        )


        n_nodes = len(
            nodes
        )


        if n_nodes == 0:

            continue


        radius = (
            _ring_radius(
                ring_number
            )
        )


        base_angle = (

            np.pi
            /
            2

            -

            np.deg2rad(

                RING_ROTATION_DEG

                *

                (
                    ring_number
                    -
                    1
                )
            )
        )


        if (
            EVEN_RING_HALF_STEP
            and
            ring_number
            %
            2
            ==
            0
        ):

            base_angle -= (

                np.pi

                /
                n_nodes
            )


        angle_step = (

            2
            *
            np.pi
            /
            n_nodes
        )


        for i, node in enumerate(
            nodes
        ):

            angle = (

                base_angle

                -

                i
                *
                angle_step
            )


            angle += rng.normal(

                0.0,

                ANGLE_JITTER_FRACTION
                *
                angle_step
            )


            current_radius = (

                radius

                *

                (
                    1.0

                    +

                    rng.normal(
                        0.0,
                        RADIUS_JITTER
                    )
                )
            )


            positions[
                node
            ] = (

                current_radius
                *
                np.cos(
                    angle
                ),

                current_radius
                *
                np.sin(
                    angle
                ),
            )


            ring_map[
                node
            ] = (
                ring_number
            )


    rows = []


    for module in _resolve_modules(modules):

        if module not in positions:

            # 理论上不会进入这里，兜底而已
            positions[
                module
            ] = (
                0.0,
                0.0
            )

            ring_map[
                module
            ] = 0


        rows.append(
            {
                "group":
                    group_name,

                "module":
                    module,

                "x":
                    float(
                        positions[
                            module
                        ][0]
                    ),

                "y":
                    float(
                        positions[
                            module
                        ][1]
                    ),

                "ring_number":
                    int(
                        ring_map[
                            module
                        ]
                    ),

                "is_hub":
                    bool(
                        module
                        in
                        hub_nodes
                    ),

                "weighted_out_abs":
                    float(
                        work.loc[
                            module,
                            "weighted_out_abs"
                        ]
                    ),

                "out_degree":
                    int(
                        work.loc[
                            module,
                            "out_degree"
                        ]
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


def _edge_width_levels(
    display_edges,
):

    if display_edges.empty:

        return (
            0.0,
            0.0
        )


    values = (

        display_edges[
            "weight"
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )


    return (
        float(
            np.quantile(
                values,
                1 / 3
            )
        ),

        float(
            np.quantile(
                values,
                2 / 3
            )
        ),
    )


def _edge_width(
    abs_weight,
    q1,
    q2,
):

    if abs_weight >= q2:

        return float(
            EDGE_WIDTH_STRONG
        )


    if abs_weight >= q1:

        return float(
            EDGE_WIDTH_MEDIUM
        )


    return float(
        EDGE_WIDTH_WEAK
    )


def _edge_curvature(
    source,
    target,
    reciprocal,
):

    integer = _stable_integer(
        f"{source}->{target}"
    )


    sign = (

        -1

        if integer % 2 == 0

        else 1
    )


    if reciprocal:

        magnitude = float(
            RECIPROCAL_CURVATURE
        )


    else:

        magnitude = (

            CURVATURE_BASE

            +

            CURVATURE_STEP

            *

            (
                integer
                %
                4
            )
        )


    return float(
        sign
        *
        magnitude
    )


def _hex_to_rgba(
    color,
    alpha,
):

    color = (
        str(
            color
        )
        .replace(
            "#",
            ""
        )
    )


    r = int(
        color[
            0:2
        ],
        16
    )


    g = int(
        color[
            2:4
        ],
        16
    )


    b = int(
        color[
            4:6
        ],
        16
    )


    return (
        f"rgba({r},{g},{b},{float(alpha):.4f})"
    )


def generate_effect_download_artifacts(
    group_dir,
):
    """
    从 run_group() 已经保存的：

        15_effect_decomposition/
            predicted.csv
            intercept.csv
            effect/
                M1.csv
                M2.csv
                ...

    生成：

        15_effect_decomposition/
            plots/
                M1_effect.png
                M1_effect.pdf
                ...

        16_effect_decomposition_download.zip

    HTML 不直接渲染这些 effect 图，只提供下载入口。
    """

    effect_root = os.path.join(
        group_dir,
        "15_effect_decomposition"
    )


    predicted_file = os.path.join(
        effect_root,
        "predicted.csv"
    )


    intercept_file = os.path.join(
        effect_root,
        "intercept.csv"
    )


    effect_dir = os.path.join(
        effect_root,
        "effect"
    )


    if not (
        os.path.exists(
            predicted_file
        )
        and
        os.path.exists(
            intercept_file
        )
        and
        os.path.isdir(
            effect_dir
        )
    ):

        print(
            "[Effect warning] effect decomposition 文件不完整："
        )

        print(
            effect_root
        )


        return {
            "effect_root":
                effect_root,

            "plot_dir":
                None,

            "zip_path":
                None,

            "targets":
                [],
        }


    plot_dir = os.path.join(
        effect_root,
        "plots"
    )


    os.makedirs(
        plot_dir,
        exist_ok=True
    )


    predicted = pd.read_csv(
        predicted_file,
        index_col=0
    )


    predicted.index = pd.to_numeric(
        predicted.index,
        errors="coerce"
    )


    predicted = predicted.loc[
        np.isfinite(
            predicted.index
        )
    ].copy()


    predicted = predicted.apply(
        pd.to_numeric,
        errors="coerce"
    )


    intercept = pd.read_csv(
        intercept_file,
        index_col=0
    )


    intercept.index = (
        intercept.index
        .astype(str)
    )


    effect_files = [

        x

        for x in os.listdir(
            effect_dir
        )

        if x.lower().endswith(
            ".csv"
        )
    ]


    effect_files = sorted(
        effect_files,
        key=module_number
    )


    targets = []


    for filename in effect_files:

        target = os.path.splitext(
            filename
        )[0]


        effect_file = os.path.join(
            effect_dir,
            filename
        )


        effect_df = pd.read_csv(
            effect_file,
            index_col=0
        )


        effect_df.index = pd.to_numeric(
            effect_df.index,
            errors="coerce"
        )


        effect_df = effect_df.loc[
            np.isfinite(
                effect_df.index
            )
        ].copy()


        effect_df = effect_df.apply(
            pd.to_numeric,
            errors="coerce"
        )


        if target not in predicted.columns:

            print(
                f"[Effect warning] {target} 不在 predicted.csv 中，跳过画图。"
            )

            continue


        targets.append(
            target
        )


        x_effect = effect_df.index.to_numpy(
            dtype=float
        )


        x_pred = predicted.index.to_numpy(
            dtype=float
        )


        predicted_y = predicted[
            target
        ].to_numpy(
            dtype=float
        )


        intercept_value = 0.0


        if target in intercept.index:

            intercept_value = float(

                pd.to_numeric(

                    intercept.loc[
                        target
                    ].iloc[
                        0
                    ],

                    errors="coerce"
                )
            )


        fig, ax = plt.subplots(
            figsize=(
                9.2,
                5.5
            )
        )


        # ---------------------------------------------------------------------
        # Predicted
        # ---------------------------------------------------------------------

        ax.plot(

            x_pred,

            predicted_y,

            linewidth=2.8,

            label="Predicted",

            zorder=10,
        )


        # ---------------------------------------------------------------------
        # Self + intercept
        # ---------------------------------------------------------------------

        if target in effect_df.columns:

            self_plus_intercept = (

                effect_df[
                    target
                ].to_numpy(
                    dtype=float
                )

                +

                intercept_value
            )


            ax.plot(

                x_effect,

                self_plus_intercept,

                linewidth=2.1,

                linestyle="--",

                label="Self + intercept",

                zorder=9,
            )


        # ---------------------------------------------------------------------
        # Individual non-self effects
        # ---------------------------------------------------------------------

        nonself_sources = [

            str(
                c
            )

            for c in effect_df.columns

            if str(
                c
            )
            !=
            target
        ]


        for source in nonself_sources:

            ax.plot(

                x_effect,

                effect_df[
                    source
                ].to_numpy(
                    dtype=float
                ),

                linewidth=1.35,

                alpha=0.88,

                label=source,
            )


        ax.axhline(

            0.0,

            linewidth=0.7,

            alpha=0.30,
        )


        ax.set_xlabel(
            "Quasi index"
        )


        ax.set_ylabel(
            "Effect / reconstructed state"
        )


        ax.set_title(
            f"{target} effect decomposition"
        )


        ax.grid(
            alpha=0.15
        )


        ax.legend(

            frameon=False,

            fontsize=7.5,

            ncol=2,

            bbox_to_anchor=(
                1.02,
                1.0
            ),

            loc="upper left",
        )


        fig.tight_layout()


        if SAVE_EFFECT_PNG:

            fig.savefig(

                os.path.join(
                    plot_dir,
                    f"{target}_effect.png"
                ),

                dpi=EFFECT_DPI,

                bbox_inches="tight",

                facecolor="white"
            )


        if SAVE_EFFECT_PDF:

            fig.savefig(

                os.path.join(
                    plot_dir,
                    f"{target}_effect.pdf"
                ),

                bbox_inches="tight",

                facecolor="white"
            )


        plt.close(
            fig
        )


    # =========================================================================
    # ZIP: raw effect + predicted/intercept + generated plots
    # =========================================================================

    zip_path = os.path.join(
        group_dir,
        "16_effect_decomposition_download.zip"
    )


    with zipfile.ZipFile(

        zip_path,

        mode="w",

        compression=
            zipfile.ZIP_DEFLATED

    ) as zf:


        for root, _, files in os.walk(
            effect_root
        ):

            for filename in files:

                full_path = os.path.join(
                    root,
                    filename
                )


                arcname = os.path.relpath(

                    full_path,

                    start=effect_root
                )


                zf.write(

                    full_path,

                    arcname=arcname
                )


    return {

        "effect_root":
            effect_root,

        "plot_dir":
            plot_dir,

        "zip_path":
            zip_path,

        "targets":
            targets,

        "predicted_file":
            predicted_file,

        "intercept_file":
            intercept_file,

        "effect_dir":
            effect_dir,
    }


def _relative_href(
    target_path,
    html_path,
):

    if target_path is None:

        return None


    rel = os.path.relpath(

        target_path,

        start=
            os.path.dirname(
                html_path
            )
    )


    return rel.replace(
        "\\",
        "/"
    )


def write_interactive_network_html(
    edges,
    degree_df,
    output_path,
    title,
    group_name,
    group_dir,
    modules=None,
):

    """
    一体化 HTML。

    左：
        单细胞同款 Hub-centered concentric-ring IDOP network

    右：
        与当前弱边过滤严格同步的 in/out-degree 图

    底部折叠区：
        Effect decomposition 下载入口

    关键：
        网络 weak-edge slider 和 Degree 使用完全相同的 visible edges。
    """

    if edges.empty:

        print(
            "[HTML] edges empty, skip."
        )

        return


    # =========================================================================
    # display edges
    # =========================================================================

    display_edges = (
        _select_display_edges(
            edges
        )
    )


    display_edges.to_csv(

        os.path.join(
            group_dir,
            "13B_display_edges.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )


    # =========================================================================
    # ring layout
    # =========================================================================

    layout_df = (
        _build_ring_layout(
            degree_df,
            group_name,
            modules=modules,
        )
    )


    layout_df.to_csv(

        os.path.join(
            group_dir,
            "13A_ring_layout.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )


    layout_index = (

        layout_df

        .set_index(
            "module"
        )
    )


    degree_index = (

        degree_df

        .copy()

        .set_index(
            "module"
        )
    )


    # =========================================================================
    # effect files/plots/ZIP
    # =========================================================================

    effect_info = (
        generate_effect_download_artifacts(
            group_dir
        )
    )


    # =========================================================================
    # nodes JSON
    # =========================================================================

    nodes_json = []

    initial_positions = {}


    for module in _resolve_modules(modules):

        row = layout_index.loc[
            module
        ]


        ring_number = int(
            row[
                "ring_number"
            ]
        )


        is_hub = bool(
            row[
                "is_hub"
            ]
        )


        x = float(
            row[
                "x"
            ]
        ) * POSITION_SCALE


        # browser y axis points downward
        y = -float(
            row[
                "y"
            ]
        ) * POSITION_SCALE


        diameter = (
            _node_diameter(
                ring_number
            )
        )


        node_size = float(
            diameter
            *
            0.78
        )


        font_size = (

            14.0

            if is_hub

            else max(
                8.0,
                12.0
                -
                0.7
                *
                ring_number
            )
        )


        if module in degree_index.index:

            drow = degree_index.loc[
                module
            ]


            out_degree = int(
                drow[
                    "out_degree"
                ]
            )


            in_degree = int(
                drow[
                    "in_degree"
                ]
            )


            weighted_out = float(
                drow[
                    "weighted_out_abs"
                ]
            )


            weighted_in = float(
                drow[
                    "weighted_in_abs"
                ]
            )


        else:

            out_degree = 0

            in_degree = 0

            weighted_out = 0.0

            weighted_in = 0.0


        node_color = (
            _ring_color(
                ring_number
            )
        )


        nodes_json.append(
            {
                "id":
                    module,

                "label":
                    module,

                "x":
                    x,

                "y":
                    y,

                "baseX":
                    x,

                "baseY":
                    y,

                "ring":
                    ring_number,

                "isHub":
                    is_hub,

                "baseSize":
                    node_size,

                "size":
                    node_size,

                "baseFont":
                    font_size,

                "font":
                    {
                        "size":
                            font_size,

                        "color":
                            TEXT_COLOR,

                        "face":
                            "Arial",
                    },

                "shape":
                    "dot",

                "color":
                    {
                        "background":
                            node_color,

                        "border":
                            NODE_EDGE_COLOR,

                        "highlight":
                            {
                                "background":
                                    node_color,

                                "border":
                                    "#111111",
                            },
                    },

                "borderWidth":
                    1.4
                    if is_hub
                    else
                    0.8,

                "title":
                    (
                        f"<b>{module}</b>"
                        f"<br>Group: {group_name}"
                        f"<br>Ring: {ring_number}"
                        f"<br>Hub: {is_hub}"
                        f"<br>Full out-degree: {out_degree}"
                        f"<br>Full in-degree: {in_degree}"
                        f"<br>Full weighted out: {weighted_out:.6g}"
                        f"<br>Full weighted in: {weighted_in:.6g}"
                    ),
            }
        )


        initial_positions[
            module
        ] = {
            "x":
                x,

            "y":
                y,
        }


    # =========================================================================
    # edges JSON
    # =========================================================================

    q1, q2 = (
        _edge_width_levels(
            display_edges
        )
    )


    pair_set = {

        (
            str(
                r.source
            ),
            str(
                r.target
            ),
        )

        for r in display_edges.itertuples(
            index=False
        )
    }


    edges_json = []


    for edge_id, r in enumerate(
        display_edges.itertuples(
            index=False
        )
    ):

        source = str(
            r.source
        )


        target = str(
            r.target
        )


        weight = float(
            r.weight
        )


        abs_weight = abs(
            weight
        )


        reciprocal = (

            (
                target,
                source
            )

            in

            pair_set
        )


        curvature = (
            _edge_curvature(
                source,
                target,
                reciprocal,
            )
        )


        smooth_type = (

            "curvedCW"

            if curvature >= 0

            else "curvedCCW"
        )


        roundness = float(

            np.clip(

                abs(
                    curvature
                )
                *
                3.2,

                0.045,

                0.34
            )
        )


        base_width = (
            _edge_width(
                abs_weight,
                q1,
                q2,
            )
        )


        edge_color = (

            POSITIVE_EDGE_COLOR

            if weight > 0

            else NEGATIVE_EDGE_COLOR
        )


        edges_json.append(
            {
                "id":
                    f"e{edge_id}",

                "from":
                    source,

                "to":
                    target,

                "weight":
                    weight,

                "absWeight":
                    abs_weight,

                "baseWidth":
                    base_width,

                "baseColor":
                    edge_color,

                "arrows":
                    {
                        "to":
                            {
                                "enabled":
                                    True,

                                "scaleFactor":
                                    0.62,
                            }
                    },

                "smooth":
                    {
                        "enabled":
                            True,

                        "type":
                            smooth_type,

                        "roundness":
                            roundness,
                    },

                "arrowStrikethrough":
                    False,

                "title":
                    (
                        f"{source} → {target}"
                        f"<br>weight={weight:.6g}"
                    ),
            }
        )


    # =========================================================================
    # Effect download data
    # =========================================================================

    effect_targets = {}


    if (
        effect_info.get(
            "plot_dir"
        )
        is not None
    ):

        for target in effect_info.get(
            "targets",
            []
        ):

            raw_csv = os.path.join(
                effect_info[
                    "effect_dir"
                ],
                f"{target}.csv"
            )


            png = os.path.join(
                effect_info[
                    "plot_dir"
                ],
                f"{target}_effect.png"
            )


            pdf = os.path.join(
                effect_info[
                    "plot_dir"
                ],
                f"{target}_effect.pdf"
            )


            effect_targets[
                target
            ] = {

                "csv":
                    _relative_href(
                        raw_csv,
                        output_path
                    ),

                "png":
                    (
                        _relative_href(
                            png,
                            output_path
                        )
                        if os.path.exists(
                            png
                        )
                        else None
                    ),

                "pdf":
                    (
                        _relative_href(
                            pdf,
                            output_path
                        )
                        if os.path.exists(
                            pdf
                        )
                        else None
                    ),
            }


    effect_download_data = {

        "zip":
            (
                _relative_href(
                    effect_info.get(
                        "zip_path"
                    ),
                    output_path
                )
                if effect_info.get(
                    "zip_path"
                )
                else None
            ),

        "predicted":
            (
                _relative_href(
                    effect_info.get(
                        "predicted_file"
                    ),
                    output_path
                )
                if effect_info.get(
                    "predicted_file"
                )
                else None
            ),

        "intercept":
            (
                _relative_href(
                    effect_info.get(
                        "intercept_file"
                    ),
                    output_path
                )
                if effect_info.get(
                    "intercept_file"
                )
                else None
            ),

        "targets":
            effect_targets,
    }


    # =========================================================================
    # JSON
    # =========================================================================

    nodes_text = json.dumps(
        nodes_json,
        ensure_ascii=False
    )


    edges_text = json.dumps(
        edges_json,
        ensure_ascii=False
    )


    initial_text = json.dumps(
        initial_positions,
        ensure_ascii=False
    )


    effect_text = json.dumps(
        effect_download_data,
        ensure_ascii=False
    )


    storage_key = (
        f"ProteinIDOP|{group_name}|coarse"
    )


    # =========================================================================
    # HTML
    # =========================================================================

    html = f"""<!DOCTYPE html>

<html lang="zh-CN">

<head>

<meta charset="utf-8"/>

<meta name="viewport"
      content="width=device-width, initial-scale=1.0"/>

<title>{title}</title>


<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>


<style>

* {{
    box-sizing: border-box;
}}


html,
body {{

    margin: 0;

    padding: 0;

    background: #f5f6f8;

    color: #202020;

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}}


#toolbar {{

    position: sticky;

    top: 0;

    z-index: 999;

    padding: 9px 13px;

    background:
        rgba(
            255,
            255,
            255,
            0.98
        );

    border-bottom:
        1px solid #d7d7d7;

    box-shadow:
        0 1px 8px
        rgba(
            0,
            0,
            0,
            0.05
        );
}}


#title {{

    font-size: 17px;

    font-weight: 700;

    margin-bottom: 8px;
}}


.controls {{

    display: flex;

    flex-wrap: wrap;

    align-items: center;

    gap: 7px 12px;

    font-size: 12px;
}}


.control {{

    display: inline-flex;

    align-items: center;

    gap: 5px;
}}


.control input[type="range"] {{

    width: 100px;
}}


.value {{

    min-width: 36px;

    text-align: right;

    font-variant-numeric:
        tabular-nums;
}}


button,
.download-link {{

    border:
        1px solid #bdbdbd;

    border-radius: 5px;

    background: white;

    padding: 5px 8px;

    color: #202020;

    text-decoration: none;

    cursor: pointer;

    font-size: 12px;
}}


button:hover,
.download-link:hover {{

    background: #eeeeee;
}}


.legend {{

    margin-top: 7px;

    display: flex;

    flex-wrap: wrap;

    gap: 10px 16px;

    color: #666666;

    font-size: 11px;
}}


.legend-line {{

    display: inline-block;

    width: 25px;

    height: 3px;

    vertical-align: middle;

    margin-right: 4px;
}}


.main-grid {{

    display: grid;

    grid-template-columns:
        minmax(
            0,
            1.65fr
        )
        minmax(
            390px,
            0.85fr
        );

    gap: 8px;

    padding: 8px;
}}


.card {{

    min-width: 0;

    background: white;

    border:
        1px solid #d8d8d8;

    border-radius: 7px;

    overflow: hidden;
}}


.card-title {{

    height: 36px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-bottom:
        1px solid #e5e5e5;

    font-size: 13px;

    font-weight: 700;

    background: #fbfbfb;
}}


#network {{

    width: 100%;

    height: {HTML_HEIGHT}px;

    background: white;
}}


.degree-toolbar {{

    padding: 7px 9px;

    border-bottom:
        1px solid #e8e8e8;

    display: flex;

    flex-wrap: wrap;

    align-items: center;

    gap: 7px;

    font-size: 11px;
}}


#degreeScroll {{

    height: {DEGREE_PANEL_HEIGHT - 50}px;

    overflow-y: auto;

    overflow-x: hidden;

    background: white;
}}


#degreeSvg {{

    width: 100%;

    display: block;
}}


.effect-panel {{

    margin:
        0
        8px
        12px
        8px;

    background: white;

    border:
        1px solid #d8d8d8;

    border-radius: 7px;

    padding: 9px 11px;
}}


.effect-panel summary {{

    cursor: pointer;

    font-weight: 700;

    font-size: 13px;
}}


.effect-links {{

    margin-top: 10px;

    display: flex;

    flex-wrap: wrap;

    gap: 7px;

    align-items: center;
}}


.effect-note {{

    margin-top: 7px;

    color: #666666;

    font-size: 11px;
}}


.status {{

    margin-left: auto;

    color: #666666;

    font-size: 11px;
}}


@media (
    max-width: 1150px
) {{

    .main-grid {{

        grid-template-columns:
            1fr;
    }}
}}

</style>

</head>


<body>


<div id="toolbar">

    <div id="title">
        {title}
    </div>


    <div class="controls">


        <label class="control">

            节点

            <input
                id="nodeScale"
                type="range"
                min="0.45"
                max="5.00"
                step="0.05"
                value="{DEFAULT_NODE_SCALE}"
            >

            <span
                id="nodeScaleValue"
                class="value"
            >
                {DEFAULT_NODE_SCALE:.2f}
            </span>

        </label>


        <label class="control">

            标签

            <input
                id="labelScale"
                type="range"
                min="0.40"
                max="6.00"
                step="0.05"
                value="{DEFAULT_LABEL_SCALE}"
            >

            <span
                id="labelScaleValue"
                class="value"
            >
                {DEFAULT_LABEL_SCALE:.2f}
            </span>

        </label>


        <label class="control">

            边宽

            <input
                id="edgeScale"
                type="range"
                min="0.25"
                max="8.00"
                step="0.05"
                value="{DEFAULT_EDGE_SCALE}"
            >

            <span
                id="edgeScaleValue"
                class="value"
            >
                {DEFAULT_EDGE_SCALE:.2f}
            </span>

        </label>


        <label class="control">

            边透明度

            <input
                id="edgeOpacity"
                type="range"
                min="0.05"
                max="1.00"
                step="0.05"
                value="{DEFAULT_EDGE_OPACITY}"
            >

            <span
                id="edgeOpacityValue"
                class="value"
            >
                {DEFAULT_EDGE_OPACITY:.2f}
            </span>

        </label>


        <label class="control">

            弱边过滤

            <input
                id="edgePercentile"
                type="range"
                min="0"
                max="95"
                step="1"
                value="{DEFAULT_EDGE_PERCENTILE}"
            >

            <span
                id="edgePercentileValue"
                class="value"
            >
                {DEFAULT_EDGE_PERCENTILE}%
            </span>

        </label>


        <label class="control">

            <input
                id="showLabels"
                type="checkbox"
                checked
            >

            标签

        </label>


        <button id="fitBtn">
            Fit
        </button>


        <button id="resetBtn">
            恢复环形
        </button>


        <button id="saveBtn">
            保存布局
        </button>


        <button id="loadBtn">
            读取布局
        </button>


        <button id="exportPngBtn">
            导出网络 PNG
        </button>


        <span
            id="status"
            class="status"
        >
            Ready
        </span>

    </div>


    <div class="legend">

        <span>

            <span
                class="legend-line"
                style="background:{POSITIVE_EDGE_COLOR};"
            ></span>

            Positive edge

        </span>


        <span>

            <span
                class="legend-line"
                style="background:{NEGATIVE_EDGE_COLOR};"
            ></span>

            Negative edge

        </span>


        <span>

            <span
                class="legend-line"
                style="background:{IN_DEGREE_COLOR};"
            ></span>

            In-degree

        </span>


        <span>

            <span
                class="legend-line"
                style="background:{OUT_DEGREE_COLOR};"
            ></span>

            Out-degree

        </span>


        <span>
            Degree 始终使用当前网络可见边重新计算
        </span>

    </div>

</div>


<div class="main-grid">


    <section class="card">

        <div
            class="card-title"
        >
            IDOP Network
        </div>


        <div id="network"></div>

    </section>


    <section class="card">

        <div
            id="degreeTitle"
            class="card-title"
        >
            Filter-synchronized In / Out Degree
        </div>


        <div class="degree-toolbar">


            <label>

                Degree metric

                <select id="degreeMetric">

                    <option value="count">
                        Count
                    </option>

                    <option value="weighted">
                        Weighted |weight|
                    </option>

                </select>

            </label>


            <button id="degreeCsvBtn">
                导出当前 Degree CSV
            </button>


            <button id="degreeSvgBtn">
                导出当前 Degree SVG
            </button>


        </div>


        <div id="degreeScroll">

            <svg
                id="degreeSvg"
                role="img"
                aria-label="Filter synchronized in and out degree chart"
            ></svg>

        </div>

    </section>


</div>


<details class="effect-panel">

    <summary>
        Effect decomposition 下载
        （默认不在 HTML 主画面显示）
    </summary>


    <div class="effect-links">


        <a
            id="effectZipLink"
            class="download-link"
            href="#"
            download
        >
            下载全部 Effect ZIP
        </a>


        <a
            id="predictedLink"
            class="download-link"
            href="#"
            download
        >
            predicted.csv
        </a>


        <a
            id="interceptLink"
            class="download-link"
            href="#"
            download
        >
            intercept.csv
        </a>


        <label>

            Target

            <select id="effectTargetSelect">
            </select>

        </label>


        <a
            id="effectPngLink"
            class="download-link"
            href="#"
            download
        >
            下载 Target PNG
        </a>


        <a
            id="effectPdfLink"
            class="download-link"
            href="#"
            download
        >
            下载 Target PDF
        </a>


        <a
            id="effectCsvLink"
            class="download-link"
            href="#"
            download
        >
            下载 Target Effect CSV
        </a>


    </div>


    <div class="effect-note">

        Effect 图已经在本地预生成。
        页面本身不渲染几十张 effect 曲线，避免网络图、Degree 图和大量 effect 图同时占用浏览器资源。

    </div>

</details>


<script>


const BASE_NODES = {nodes_text};

const BASE_EDGES = {edges_text};

const INITIAL_POSITIONS = {initial_text};

const EFFECT_DATA = {effect_text};

const STORAGE_KEY = {json.dumps(storage_key)};


const nodeSet = new vis.DataSet(
    BASE_NODES
);


const edgeSet = new vis.DataSet(
    BASE_EDGES
);


const container = document.getElementById(
    "network"
);


const network = new vis.Network(

    container,

    {{
        nodes:
            nodeSet,

        edges:
            edgeSet
    }},

    {{
        autoResize:
            true,

        layout:
            {{
                improvedLayout:
                    false
            }},

        physics:
            {{
                enabled:
                    false
            }},

        interaction:
            {{
                hover:
                    true,

                hoverConnectedEdges:
                    true,

                dragNodes:
                    true,

                dragView:
                    true,

                zoomView:
                    true,

                multiselect:
                    true,

                navigationButtons:
                    true,

                keyboard:
                    true,

                tooltipDelay:
                    80
            }},

        edges:
            {{
                arrowStrikethrough:
                    false,

                selectionWidth:
                    1.8,

                hoverWidth:
                    0.8
            }}
    }}
);


let currentVisibleEdges = [];

let currentDegreeRows = [];


// =============================================================================
// JS helpers
// =============================================================================

function setStatus(
    text
) {{

    document
    .getElementById(
        "status"
    )
    .textContent = text;
}}


function hexToRgba(
    hex,
    alpha
) {{

    const h = String(
        hex
    ).replace(
        "#",
        ""
    );


    const r = parseInt(
        h.substring(
            0,
            2
        ),
        16
    );


    const g = parseInt(
        h.substring(
            2,
            4
        ),
        16
    );


    const b = parseInt(
        h.substring(
            4,
            6
        ),
        16
    );


    return (
        `rgba(${{r}},${{g}},${{b}},${{alpha}})`
    );
}}


function percentile(
    values,
    p
) {{

    if (
        values.length === 0
    ) {{

        return 0;
    }}


    const x = values
        .slice()
        .sort(
            (a, b) => a - b
        );


    const index = (

        (
            x.length
            -
            1
        )

        *

        p

        /

        100.0
    );


    const lo = Math.floor(
        index
    );


    const hi = Math.ceil(
        index
    );


    if (
        lo === hi
    ) {{

        return x[
            lo
        ];
    }}


    const frac = (
        index
        -
        lo
    );


    return (

        x[
            lo
        ]
        *
        (
            1
            -
            frac
        )

        +

        x[
            hi
        ]
        *
        frac
    );
}}


function downloadBlob(
    filename,
    blob
) {{

    const url = URL.createObjectURL(
        blob
    );


    const a = document.createElement(
        "a"
    );


    a.href = url;

    a.download = filename;


    document.body.appendChild(
        a
    );


    a.click();


    document.body.removeChild(
        a
    );


    setTimeout(
        () => URL.revokeObjectURL(
            url
        ),
        1000
    );
}}


// =============================================================================
// Degree recomputation using CURRENT VISIBLE EDGES
// =============================================================================

function calculateCurrentDegree() {{

    const rows = {{}};


    BASE_NODES.forEach(
        node => {{

            rows[
                node.id
            ] = {{

                module:
                    node.id,

                in_degree:
                    0,

                out_degree:
                    0,

                weighted_in:
                    0,

                weighted_out:
                    0,
            }};
        }}
    );


    currentVisibleEdges.forEach(
        edge => {{

            if (
                rows[
                    edge.from
                ]
            ) {{

                rows[
                    edge.from
                ].out_degree += 1;


                rows[
                    edge.from
                ].weighted_out += Number(
                    edge.absWeight
                );
            }}


            if (
                rows[
                    edge.to
                ]
            ) {{

                rows[
                    edge.to
                ].in_degree += 1;


                rows[
                    edge.to
                ].weighted_in += Number(
                    edge.absWeight
                );
            }}
        }}
    );


    currentDegreeRows = Object.values(
        rows
    );


    return currentDegreeRows;
}}


function renderDegreeChart() {{

    const rows = calculateCurrentDegree();


    const metric = (
        document.getElementById(
            "degreeMetric"
        ).value
    );


    const inKey = (

        metric === "weighted"

        ? "weighted_in"

        : "in_degree"
    );


    const outKey = (

        metric === "weighted"

        ? "weighted_out"

        : "out_degree"
    );


    const sorted = rows
        .slice()
        .sort(
            (a, b) => {{

                if (
                    b[
                        outKey
                    ]
                    !==
                    a[
                        outKey
                    ]
                ) {{

                    return (
                        b[
                            outKey
                        ]
                        -
                        a[
                            outKey
                        ]
                    );
                }}


                if (
                    b[
                        inKey
                    ]
                    !==
                    a[
                        inKey
                    ]
                ) {{

                    return (
                        b[
                            inKey
                        ]
                        -
                        a[
                            inKey
                        ]
                    );
                }}


                return (
                    String(
                        a.module
                    )
                    .localeCompare(
                        String(
                            b.module
                        ),
                        undefined,
                        {{
                            numeric:
                                true
                        }}
                    )
                );
            }}
        );


    const svg = document.getElementById(
        "degreeSvg"
    );


    while (
        svg.firstChild
    ) {{

        svg.removeChild(
            svg.firstChild
        );
    }}


    const panelWidth = Math.max(

        430,

        document.getElementById(
            "degreeScroll"
        ).clientWidth
        -
        8
    );


    const rowHeight = 24;

    const topMargin = 34;

    const bottomMargin = 34;


    const height = (

        topMargin

        +

        sorted.length
        *
        rowHeight

        +

        bottomMargin
    );


    svg.setAttribute(
        "viewBox",
        `0 0 ${{panelWidth}} ${{height}}`
    );


    svg.setAttribute(
        "height",
        String(
            height
        )
    );


    const center = (
        panelWidth
        /
        2
    );


    const labelHalfWidth = 43;

    const labelGap = 6;

    const sidePadding = 25;


    const maxBarWidth = Math.max(

        70,

        center
        -
        labelHalfWidth
        -
        labelGap
        -
        sidePadding
    );


    const maxValue = Math.max(

        1e-12,

        ...sorted.map(
            row => Math.max(

                Number(
                    row[
                        inKey
                    ]
                ),

                Number(
                    row[
                        outKey
                    ]
                )
            )
        )
    );


    const ns = (
        "http://www.w3.org/2000/svg"
    );


    function svgElement(
        name,
        attrs
    ) {{

        const el = document.createElementNS(
            ns,
            name
        );


        Object.entries(
            attrs
        ).forEach(
            ([key, value]) => {{

                el.setAttribute(
                    key,
                    String(
                        value
                    )
                );
            }}
        );


        return el;
    }}


    const headerIn = svgElement(

        "text",

        {{
            x:
                center
                -
                labelHalfWidth
                -
                10,

            y:
                20,

            "text-anchor":
                "end",

            "font-size":
                11,

            fill:
                "#555555",
        }}
    );


    headerIn.textContent = (

        metric === "weighted"

        ? "Weighted in"

        : "In-degree"
    );


    svg.appendChild(
        headerIn
    );


    const headerOut = svgElement(

        "text",

        {{
            x:
                center
                +
                labelHalfWidth
                +
                10,

            y:
                20,

            "text-anchor":
                "start",

            "font-size":
                11,

            fill:
                "#555555",
        }}
    );


    headerOut.textContent = (

        metric === "weighted"

        ? "Weighted out"

        : "Out-degree"
    );


    svg.appendChild(
        headerOut
    );


    svg.appendChild(

        svgElement(

            "line",

            {{
                x1:
                    center,

                x2:
                    center,

                y1:
                    topMargin
                    -
                    8,

                y2:
                    height
                    -
                    bottomMargin
                    +
                    4,

                stroke:
                    "#bdbdbd",

                "stroke-width":
                    1,
            }}
        )
    );


    sorted.forEach(
        (row, index) => {{

            const y = (

                topMargin

                +

                index
                *
                rowHeight

                +

                rowHeight
                /
                2
            );


            const inValue = Number(
                row[
                    inKey
                ]
            );


            const outValue = Number(
                row[
                    outKey
                ]
            );


            const inWidth = (

                inValue

                /
                maxValue

                *
                maxBarWidth
            );


            const outWidth = (

                outValue

                /
                maxValue

                *
                maxBarWidth
            );


            svg.appendChild(

                svgElement(

                    "rect",

                    {{
                        x:
                            center
                            -
                            labelHalfWidth
                            -
                            labelGap
                            -
                            inWidth,

                        y:
                            y
                            -
                            6,

                        width:
                            inWidth,

                        height:
                            12,

                        fill:
                            "{IN_DEGREE_COLOR}",

                        opacity:
                            0.84,
                    }}
                )
            );


            svg.appendChild(

                svgElement(

                    "rect",

                    {{
                        x:
                            center
                            +
                            labelHalfWidth
                            +
                            labelGap,

                        y:
                            y
                            -
                            6,

                        width:
                            outWidth,

                        height:
                            12,

                        fill:
                            "{OUT_DEGREE_COLOR}",

                        opacity:
                            0.84,
                    }}
                )
            );


            const label = svgElement(

                "text",

                {{
                    x:
                        center,

                    y:
                        y
                        +
                        3.5,

                    "text-anchor":
                        "middle",

                    "font-size":
                        10,

                    fill:
                        "#202020",
                }}
            );


            label.textContent = (
                row.module
            );


            svg.appendChild(
                label
            );


            const formatter = (

                metric === "weighted"

                ? value => (
                    value.toFixed(
                        3
                    )
                )

                : value => (
                    String(
                        Math.round(
                            value
                        )
                    )
                )
            );


            const inText = svgElement(

                "text",

                {{
                    x:
                        center
                        -
                        labelHalfWidth
                        -
                        labelGap
                        -
                        inWidth
                        -
                        4,

                    y:
                        y
                        +
                        3.5,

                    "text-anchor":
                        "end",

                    "font-size":
                        8.5,

                    fill:
                        "#555555",
                }}
            );


            inText.textContent = formatter(
                inValue
            );


            svg.appendChild(
                inText
            );


            const outText = svgElement(

                "text",

                {{
                    x:
                        center
                        +
                        labelHalfWidth
                        +
                        labelGap
                        +
                        outWidth
                        +
                        4,

                    y:
                        y
                        +
                        3.5,

                    "text-anchor":
                        "start",

                    "font-size":
                        8.5,

                    fill:
                        "#555555",
                }}
            );


            outText.textContent = formatter(
                outValue
            );


            svg.appendChild(
                outText
            );
        }}
    );


    document
    .getElementById(
        "degreeTitle"
    )
    .textContent = (

        "Filter-synchronized In / Out Degree"
        +
        ` | visible edges: ${{currentVisibleEdges.length}} / ${{BASE_EDGES.length}}`
    );
}}


// =============================================================================
// Network controls
// =============================================================================

function applyControls() {{

    const nodeScale = parseFloat(
        document.getElementById(
            "nodeScale"
        ).value
    );


    const labelScale = parseFloat(
        document.getElementById(
            "labelScale"
        ).value
    );


    const edgeScale = parseFloat(
        document.getElementById(
            "edgeScale"
        ).value
    );


    const edgeOpacity = parseFloat(
        document.getElementById(
            "edgeOpacity"
        ).value
    );


    const edgePercentile = parseFloat(
        document.getElementById(
            "edgePercentile"
        ).value
    );


    const showLabels = (
        document.getElementById(
            "showLabels"
        ).checked
    );


    document
    .getElementById(
        "nodeScaleValue"
    )
    .textContent = (
        nodeScale.toFixed(
            2
        )
    );


    document
    .getElementById(
        "labelScaleValue"
    )
    .textContent = (
        labelScale.toFixed(
            2
        )
    );


    document
    .getElementById(
        "edgeScaleValue"
    )
    .textContent = (
        edgeScale.toFixed(
            2
        )
    );


    document
    .getElementById(
        "edgeOpacityValue"
    )
    .textContent = (
        edgeOpacity.toFixed(
            2
        )
    );


    document
    .getElementById(
        "edgePercentileValue"
    )
    .textContent = (
        Math.round(
            edgePercentile
        )
        +
        "%"
    );


    nodeSet.update(

        BASE_NODES.map(
            n => (
                {{

                    id:
                        n.id,

                    size:
                        n.baseSize
                        *
                        nodeScale,

                    label:
                        showLabels
                        ?
                        n.id
                        :
                        "",

                    font:
                        {{

                            size:
                                n.baseFont
                                *
                                labelScale,

                            color:
                                "{TEXT_COLOR}",

                            face:
                                "Arial"
                        }}
                }}
            )
        )
    );


    // -------------------------------------------------------------------------
    // ONE threshold controls both network and degree.
    // -------------------------------------------------------------------------

    const absValues = BASE_EDGES
        .map(
            e => Number(
                e.absWeight
            )
        )
        .filter(
            x => Number.isFinite(
                x
            )
        );


    const threshold = percentile(
        absValues,
        edgePercentile
    );


    currentVisibleEdges = BASE_EDGES.filter(
        e => (
            Number(
                e.absWeight
            )
            >=
            threshold
        )
    );


    const visibleIds = new Set(

        currentVisibleEdges.map(
            e => e.id
        )
    );


    edgeSet.update(

        BASE_EDGES.map(
            e => (
                {{

                    id:
                        e.id,

                    width:
                        e.baseWidth
                        *
                        edgeScale,

                    hidden:
                        !visibleIds.has(
                            e.id
                        ),

                    color:
                        {{

                            color:
                                hexToRgba(
                                    e.baseColor,
                                    edgeOpacity
                                ),

                            highlight:
                                hexToRgba(

                                    e.baseColor,

                                    Math.min(
                                        1.0,
                                        edgeOpacity
                                        +
                                        0.30
                                    )
                                )
                        }}
                }}
            )
        )
    );


    renderDegreeChart();


    setStatus(

        `visible edges ${{currentVisibleEdges.length}} / ${{BASE_EDGES.length}}`
        +
        ` | |w| threshold = ${{threshold.toPrecision(4)}}`
    );
}}


// =============================================================================
// Effect downloads
// =============================================================================

function setDownloadLink(
    id,
    href
) {{

    const link = document.getElementById(
        id
    );


    if (
        href
    ) {{

        link.href = href;

        link.style.display = "inline-block";

    }} else {{

        link.href = "#";

        link.style.display = "none";
    }}
}}


function updateTargetEffectLinks() {{

    const select = document.getElementById(
        "effectTargetSelect"
    );


    const target = select.value;


    const info = (

        EFFECT_DATA.targets[
            target
        ]

        ||
        {{}}
    );


    setDownloadLink(
        "effectPngLink",
        info.png
    );


    setDownloadLink(
        "effectPdfLink",
        info.pdf
    );


    setDownloadLink(
        "effectCsvLink",
        info.csv
    );
}}


function initializeEffectDownloads() {{

    setDownloadLink(
        "effectZipLink",
        EFFECT_DATA.zip
    );


    setDownloadLink(
        "predictedLink",
        EFFECT_DATA.predicted
    );


    setDownloadLink(
        "interceptLink",
        EFFECT_DATA.intercept
    );


    const select = document.getElementById(
        "effectTargetSelect"
    );


    Object.keys(
        EFFECT_DATA.targets
        ||
        {{}}
    )
    .sort(
        (a, b) => (
            a.localeCompare(
                b,
                undefined,
                {{
                    numeric:
                        true
                }}
            )
        )
    )
    .forEach(
        target => {{

            const option = document.createElement(
                "option"
            );


            option.value = target;

            option.textContent = target;


            select.appendChild(
                option
            );
        }}
    );


    updateTargetEffectLinks();
}}


// =============================================================================
// Exports
// =============================================================================

function downloadBlob(
    filename,
    blob
) {{

    const url = URL.createObjectURL(
        blob
    );


    const a = document.createElement(
        "a"
    );


    a.href = url;

    a.download = filename;


    document.body.appendChild(
        a
    );


    a.click();


    document.body.removeChild(
        a
    );


    setTimeout(
        () => URL.revokeObjectURL(
            url
        ),
        1000
    );
}}


function exportDegreeCsv() {{

    calculateCurrentDegree();


    const lines = [

        [
            "module",
            "in_degree",
            "out_degree",
            "weighted_in",
            "weighted_out",
        ].join(
            ","
        )
    ];


    currentDegreeRows.forEach(
        row => {{

            lines.push(

                [
                    row.module,
                    row.in_degree,
                    row.out_degree,
                    row.weighted_in,
                    row.weighted_out,
                ].join(
                    ","
                )
            );
        }}
    );


    const blob = new Blob(

        [
            "\\ufeff"
            +
            lines.join(
                "\\n"
            )
        ],

        {{
            type:
                "text/csv;charset=utf-8"
        }}
    );


    downloadBlob(

        "{group_name}_CURRENT_FILTER_degree.csv",

        blob
    );
}}


function exportDegreeSvg() {{

    const svg = document.getElementById(
        "degreeSvg"
    );


    const source = (
        new XMLSerializer()
        .serializeToString(
            svg
        )
    );


    const blob = new Blob(

        [
            source
        ],

        {{
            type:
                "image/svg+xml;charset=utf-8"
        }}
    );


    downloadBlob(

        "{group_name}_CURRENT_FILTER_degree.svg",

        blob
    );
}}


// =============================================================================
// Events
// =============================================================================

[
    "nodeScale",
    "labelScale",
    "edgeScale",
    "edgeOpacity",
    "edgePercentile",
].forEach(

    id => {{

        document
        .getElementById(
            id
        )
        .addEventListener(
            "input",
            applyControls
        );
    }}
);


document
.getElementById(
    "showLabels"
)
.addEventListener(
    "change",
    applyControls
);


document
.getElementById(
    "degreeMetric"
)
.addEventListener(
    "change",
    renderDegreeChart
);


document
.getElementById(
    "effectTargetSelect"
)
.addEventListener(
    "change",
    updateTargetEffectLinks
);


document
.getElementById(
    "fitBtn"
)
.addEventListener(

    "click",

    () => {{

        network.fit(
            {{
                animation:
                    {{
                        duration:
                            300
                    }}
            }}
        );
    }}
);


document
.getElementById(
    "resetBtn"
)
.addEventListener(

    "click",

    () => {{

        nodeSet.update(

            Object.entries(
                INITIAL_POSITIONS
            )
            .map(

                ([id, p]) => (
                    {{

                        id:
                            id,

                        x:
                            p.x,

                        y:
                            p.y
                    }}
                )
            )
        );


        network.fit(
            {{
                animation:
                    {{
                        duration:
                            250
                    }}
            }}
        );


        setStatus(
            "Initial ring layout restored"
        );
    }}
);


document
.getElementById(
    "saveBtn"
)
.addEventListener(

    "click",

    () => {{

        localStorage.setItem(

            STORAGE_KEY,

            JSON.stringify(
                {{

                    positions:
                        network.getPositions(),

                    savedAt:
                        new Date()
                        .toISOString()
                }}
            )
        );


        setStatus(
            "Current layout saved"
        );
    }}
);


document
.getElementById(
    "loadBtn"
)
.addEventListener(

    "click",

    () => {{

        const raw = localStorage.getItem(
            STORAGE_KEY
        );


        if (
            !raw
        ) {{

            setStatus(
                "No saved layout"
            );

            return;
        }}


        const payload = JSON.parse(
            raw
        );


        nodeSet.update(

            Object.entries(
                payload.positions
                ||
                {{}}
            )
            .map(

                ([id, p]) => (
                    {{

                        id:
                            id,

                        x:
                            p.x,

                        y:
                            p.y
                    }}
                )
            )
        );


        network.fit(
            {{
                animation:
                    {{
                        duration:
                            250
                    }}
            }}
        );


        setStatus(
            "Saved layout loaded"
        );
    }}
);


document
.getElementById(
    "exportPngBtn"
)
.addEventListener(

    "click",

    () => {{

        network.redraw();


        const canvas = (
            network
            .canvas
            .frame
            .canvas
        );


        const a = document.createElement(
            "a"
        );


        a.download = (
            "{group_name}_CURRENT_FILTER_network.png"
        );


        a.href = canvas.toDataURL(
            "image/png"
        );


        document.body.appendChild(
            a
        );


        a.click();


        document.body.removeChild(
            a
        );
    }}
);


document
.getElementById(
    "degreeCsvBtn"
)
.addEventListener(
    "click",
    exportDegreeCsv
);


document
.getElementById(
    "degreeSvgBtn"
)
.addEventListener(
    "click",
    exportDegreeSvg
);


window.addEventListener(

    "resize",

    () => {{

        renderDegreeChart();
    }}
);


// =============================================================================
// Restore saved layout if present
// =============================================================================

const saved = localStorage.getItem(
    STORAGE_KEY
);


if (
    saved
) {{

    try {{

        const payload = JSON.parse(
            saved
        );


        nodeSet.update(

            Object.entries(
                payload.positions
                ||
                {{}}
            )
            .map(

                ([id, p]) => (
                    {{

                        id:
                            id,

                        x:
                            p.x,

                        y:
                            p.y
                    }}
                )
            )
        );


    }} catch(
        error
    ) {{

        console.error(
            error
        );
    }}
}}


// =============================================================================
// Initial render
// =============================================================================

network.once(

    "afterDrawing",

    () => {{

        network.fit(
            {{
                animation:
                    false
            }}
        );
    }}
);


initializeEffectDownloads();

applyControls();


</script>


</body>

</html>
"""


    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            html
        )


    print(
        "[HTML overwritten]",
        output_path
    )


# =============================================================================
# 独立入口：network_data / CSV -> HTML
# =============================================================================


def plot_network(
    data,
    output_path,
    title=None,
    group_name=None,
    group_dir=None,
    modules=None,
):
    """
    把一张边表（DataFrame 或 CSV 路径）画成交互式 HTML 网络图。

    输入接口与 `network_data.generate_network_data()` 的输出一致：
    至少需要 source / target / weight 三列；network_data 额外输出的
    sign 列（positive / negative，与 weight 符号冗余）会被自动忽略。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        边表。可以是 CSV 路径（如 `python network_data.py` 生成的
        output/network.csv），也可以是
        `network_data.generate_network_data()` 直接返回的 DataFrame。
    output_path : str
        输出 HTML 路径。
    title : str, optional
        页面标题；默认 f"{group_name} | Functional Modules Original IDOP"。
    group_name : str, optional
        分组名（用于环形布局 hub 标注等）；默认取 CSV 文件名（去扩展名），
        DataFrame 输入时默认 "network"。
    group_dir : str, optional
        工作目录。HTML 所需的 13A_ring_layout.csv、13B_display_edges.csv
        以及 effect 下载文件写在这里；默认取 output_path 所在目录。
    modules : sequence, optional
        节点（功能模块）全集及其显示顺序；默认由输入边表的 source ∪ target
        自动推导。显式传入可避免并发会话之间互相覆盖模块列表
        （原脚本用模块级全局 MODULES 传递状态，本包已改为传参）。

    Returns
    -------
    str
        output_path
    """
    global MODULES

    if isinstance(data, (str, os.PathLike)):

        # CSV 路径：network_data.py 的输出为 utf-8（无 BOM），
        # 用 utf-8-sig 读取对两者均兼容。
        edges = pd.read_csv(
            data,
            encoding="utf-8-sig",
        )

        source_name = (
            os.path.splitext(
                os.path.basename(
                    data
                )
            )[0]
        )

    elif isinstance(data, pd.DataFrame):

        edges = data.copy()

        source_name = None

    else:

        raise TypeError(
            "data 必须是 CSV 路径（str / os.PathLike）或 pandas.DataFrame，"
            f"实际为 {type(data).__name__}。"
        )

    missing = [
        col
        for col in [
            "source",
            "target",
            "weight",
        ]
        if col not in edges.columns
    ]

    if missing:

        raise ValueError(
            "输入缺少列："
            + ", ".join(
                repr(col)
                for col in missing
            )
            + "。需要 source / target / weight；"
            "network_data 输出的 sign 列会被自动忽略。"
        )

    edges = (
        edges[
            [
                "source",
                "target",
                "weight",
            ]
        ]
        .copy()
    )

    edges["source"] = (
        edges["source"]
        .astype(str)
        .str.strip()
    )

    edges["target"] = (
        edges["target"]
        .astype(str)
        .str.strip()
    )

    edges["weight"] = pd.to_numeric(
        edges["weight"],
        errors="coerce",
    )

    edges = (
        edges.dropna(
            subset=["weight"]
        )
    )

    if edges.empty:

        raise ValueError(
            "CSV 中没有有效的边记录。"
        )

    # 模块全集 = source ∪ target（与源脚本一致的排序规则）；
    # 显式传入 modules 时以调用方为准，避免并发会话互相覆盖。
    resolved_modules = (
        list(modules)
        if modules is not None
        else
        sort_modules(
            set(
                edges["source"]
            )
            |
            set(
                edges["target"]
            )
        )
    )

    # 兼容独立脚本用法：同时刷新模块级全局
    MODULES = resolved_modules

    degree = (
        make_degree(
            edges,
            modules=resolved_modules,
        )
    )

    if group_name is None:

        group_name = (
            source_name
            or
            "network"
        )

    if group_dir is None:

        group_dir = os.path.dirname(
            os.path.abspath(
                output_path
            )
        )

    os.makedirs(
        group_dir,
        exist_ok=True
    )

    if title is None:

        title = (
            f"{group_name} | Functional Modules Original IDOP"
        )

    write_interactive_network_html(
        edges,

        degree_df=
            degree,

        output_path=
            output_path,

        title=
            title,

        group_name=
            group_name,

        group_dir=
            group_dir,

        modules=
            resolved_modules,
    )

    return output_path


def network_edges_from_adjacency(
    adj_df,
    target_node=None,
    top_edges=None,
):
    """把邻接矩阵转成 ``source`` / ``target`` / ``weight`` 边表。

    约定与 :func:`idopnetwork.network.plot.plot_network` 一致：
    ``adj_df.loc[source, target]`` 表示 ``source -> target`` 的权重。
    本函数是静态图管线与新版交互式 HTML 之间的桥：池子里的 ``adj_df`` 先转成
    边表，才能喂给 :func:`plot_network`。

    Args:
        adj_df: 邻接矩阵，index = source，columns = target。
        target_node: 非空时只保留**指向**该节点的边，与静态图的口径一致。
        top_edges: 非空时按 ``|weight|`` 从大到小只保留前 N 条边。

    Returns:
        含 ``source`` / ``target`` / ``weight`` 三列的 DataFrame；无可用边时返回空表。
    """
    frame = pd.DataFrame(adj_df).copy()
    if frame.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    frame.index = pd.Index([str(v).strip() for v in frame.index])
    frame.columns = pd.Index([str(c).strip() for c in frame.columns])
    frame = frame.apply(pd.to_numeric, errors="coerce")

    # 合并重复的 source / target 标签，否则 stack 会因重复轴标签报错
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0, sort=False).sum()
    if frame.columns.has_duplicates:
        frame = frame.T.groupby(level=0, sort=False).sum().T

    frame.index.name = "source"
    frame.columns.name = "target"

    if target_node:
        node = str(target_node).strip()
        if node not in frame.columns:
            return pd.DataFrame(columns=["source", "target", "weight"])
        frame = frame[[node]]

    # 注意：新版 pandas 移除了 stack(dropna=...) 参数，这里改为 stack() 之后
    # 统一按 weight 列 dropna，新旧 pandas 行为一致。
    edges = (
        frame
        .stack()
        .rename("weight")
        .reset_index()
    )
    edges.columns = ["source", "target", "weight"]
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    edges = edges.dropna(subset=["weight"])
    edges = edges[edges["source"] != edges["target"]]
    edges = edges[edges["weight"].abs() > 0.0]

    if top_edges is not None and int(top_edges) > 0:
        order = edges["weight"].abs().sort_values(ascending=False).index
        edges = edges.loc[order].head(int(top_edges))

    return edges.reset_index(drop=True)


if __name__ == "__main__":

    import argparse
    import sys

    parser = (
        argparse.ArgumentParser(
            description=
                "把 network_data.py 生成的边表画成交互式 HTML 网络图。"
                "无参数直接运行即可：默认读取 output/network.csv，"
                "写出 output/network.html。"
        )
    )

    parser.add_argument(
        "--csv",
        default="output/network.csv",
        help="输入边表 CSV 路径（默认 output/network.csv）",
    )

    parser.add_argument(
        "--out",
        default="output/network.html",
        help="输出 HTML 路径（默认 output/network.html）",
    )

    parser.add_argument(
        "--title",
        default=None,
        help="页面标题（可选）",
    )

    parser.add_argument(
        "--group-name",
        default=None,
        help="分组名（可选，默认取 CSV 文件名）",
    )

    parser.add_argument(
        "--group-dir",
        default=None,
        help="工作目录（可选，默认取输出 HTML 所在目录）",
    )

    args = (
        parser.parse_args()
    )

    if not os.path.isfile(args.csv):

        print(
            "找不到输入边表：",
            args.csv,
        )

        print(
            "请先运行：python network_data.py"
        )

        print(
            "或通过 --csv 指定边表路径。"
        )

        sys.exit(1)

    plot_network(
        args.csv,
        args.out,
        title=args.title,
        group_name=args.group_name,
        group_dir=args.group_dir,
    )