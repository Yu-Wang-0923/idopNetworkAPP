"""绘制 IDOP 多层网总图。

输入是 NetRecon 的 Multi-Layer 导出 ZIP 或同结构的解压目录。每列一个
条件，第一行是模块间网络，后续每行是同一模块的内部网络。每个面板的
中心节点按本面板完整网络的出度排序。同行共用边线宽阈值和坐标范围。

示例:
    python -m idopnetwork.network.multilayer_plot
    python -m idopnetwork.network.multilayer_plot multilayer_export.zip --out idop_multilayer.svg
    python -m idopnetwork.network.multilayer_plot multilayer_export.zip --out matrix.png --conditions HC UC

不带参数时打开桌面窗口。选择 NetRecon 导出 ZIP 后，可以勾选要画的模块。

在 Python 中可调用 plot_multilayer(inter_cluster, intra_cluster, output_path)。
inter_cluster 为 {condition: adjacency_df}，intra_cluster 为
{condition: {module: adjacency_df}}。矩阵的行节点指向列节点。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from textwrap import fill
from zipfile import BadZipFile, ZipFile

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.patheffects import withStroke
from matplotlib.transforms import Bbox

from . import interactive as idop


PANEL_BORDER = "#D5DCE4"
HEADER_COLOR = "#F2F5F8"
LABEL_COLOR = "#192936"
DEFAULT_EDGE_CURVATURE_SCALE = 1.5


def _read_adjacency(stream):
    frame = pd.read_csv(stream, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return frame.set_index(frame.columns[0])


def read_netrecon_export(path: str | Path) -> tuple[dict, dict]:
    """读取 NetRecon 导出的层内邻接矩阵及其条件、模块名称。"""
    path = Path(path)
    inter, intra = {}, {}

    def add(parts, stream):
        if len(parts) == 3 and parts[0] == "inter_cluster":
            inter[parts[1]] = _read_adjacency(stream)
        elif len(parts) == 4 and parts[0] == "intra_cluster":
            intra.setdefault(parts[1], {})[parts[2]] = _read_adjacency(stream)

    if path.is_dir():
        for file in sorted(path.rglob("adjacency_matrix.csv")):
            parts = file.relative_to(path).parts
            with file.open("rb") as stream:
                add(parts, stream)
    else:
        with ZipFile(path) as archive:
            for member in archive.infolist():
                if PurePosixPath(member.filename).name == "adjacency_matrix.csv":
                    parts = PurePosixPath(member.filename).parts
                    with archive.open(member) as stream:
                        add(parts, stream)

    if not inter and not intra:
        raise ValueError("找不到 NetRecon 多层导出中的 adjacency_matrix.csv。")
    return inter, intra


def _prepare_panel(matrix, layout_key, max_edges):
    frame = matrix.copy()
    frame.index = frame.index.map(lambda node: str(node).strip())
    frame.columns = frame.columns.map(lambda node: str(node).strip())
    if (
        frame.empty or not frame.index.is_unique or not frame.columns.is_unique
        or "" in frame.index or set(frame.index) != set(frame.columns)
    ):
        raise ValueError(f"{layout_key} 的邻接矩阵须为方阵，且行列使用相同的唯一节点名。")

    modules = frame.index.tolist()
    edges = idop.network_edges_from_adjacency(frame)
    edges = edges.loc[np.isfinite(edges["weight"])]
    degree = idop.make_degree(edges, modules=modules)
    layout = idop._build_ring_layout(degree, layout_key, modules=modules)
    display = idop._select_display_edges(edges)
    if max_edges is not None:
        display = display.head(max_edges)
    return layout, display, edges


def _draw_panel(ax, panel, radius, width_levels, curvature_scale):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color(PANEL_BORDER)
        spine.set_linewidth(0.55)

    if panel is None:
        ax.set_facecolor("#F8FAFC")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#7890A0")
        return

    layout, display, edges = panel
    positions = {row.module: (row.x, row.y) for row in layout.itertuples(index=False)}
    node_scale = min(0.30, 1.15 / np.sqrt(len(layout)))
    diameters = {
        row.module: node_scale * idop._node_diameter(row.ring_number)
        for row in layout.itertuples(index=False)
    }
    pairs = set(zip(display["source"], display["target"]))
    for edge in display.iloc[::-1].itertuples(index=False):
        color = idop.POSITIVE_EDGE_COLOR if edge.weight > 0 else idop.NEGATIVE_EDGE_COLOR
        curvature = curvature_scale * idop._edge_curvature(
            edge.source, edge.target, (edge.target, edge.source) in pairs,
        )
        width = idop._edge_width(abs(edge.weight), *width_levels)
        arrow = FancyArrowPatch(
            positions[edge.source], positions[edge.target],
            connectionstyle=f"arc3,rad={curvature}",
            shrinkA=diameters[edge.source] / 2, shrinkB=diameters[edge.target] / 2,
            arrowstyle="-|>", mutation_scale=6.5, linewidth=0.8 * width,
            color=color, alpha=0.82, zorder=2,
        )
        ax.add_patch(arrow)

    ax.scatter(
        layout["x"], layout["y"],
        s=[diameters[node] ** 2 for node in layout["module"]],
        c=[idop._ring_color(ring) for ring in layout["ring_number"]],
        edgecolors=idop.NODE_EDGE_COLOR, linewidths=0.45, zorder=3,
    )
    _draw_node_labels(ax, layout, diameters)
    if edges.empty:
        ax.text(0.96, 0.04, "No edges", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6, color="#7890A0")


def _draw_node_labels(ax, layout, diameters):
    """给长节点名换行，并在节点周围选择留白最大的标签位置。"""
    ax.apply_aspect()
    renderer = ax.figure.canvas.get_renderer()
    panel = ax.bbox.padded(-2)
    font_size = min(7.0, 27 / np.sqrt(len(layout)))
    pixels_per_point = renderer.points_to_pixels(1)
    occupied = []
    for row in layout.itertuples(index=False):
        x, y = ax.transData.transform((row.x, row.y))
        half = diameters[row.module] / 2 * pixels_per_point
        occupied.append(Bbox.from_extents(x - half, y - half, x + half, y + half))

    for row in layout.sort_values("is_hub", ascending=False, kind="stable").itertuples(index=False):
        gap = diameters[row.module] / 2 + 1.5
        top, bottom = ((0, gap), "center", "bottom"), ((0, -gap), "center", "top")
        right, left = ((gap, 0), "left", "center"), ((-gap, 0), "right", "center")
        if abs(row.x) > 0.25 * ax.get_xlim()[1]:
            candidates = (right, top, bottom, left) if row.x > 0 else (left, top, bottom, right)
        else:
            candidates = (top, bottom, right, left)

        label = ax.annotate(
            fill(row.module, width=20, break_on_hyphens=False),
            (row.x, row.y), xytext=(0, gap), textcoords="offset points",
            fontsize=font_size, color=idop.TEXT_COLOR,
            path_effects=[withStroke(linewidth=1.8, foreground="white")],
            clip_on=True, zorder=4,
        )
        placements = []
        for offset, ha, va in candidates:
            label.set_position(offset)
            label.set_horizontalalignment(ha)
            label.set_verticalalignment(va)
            box = label.get_window_extent(renderer).padded(1)
            inside = Bbox.intersection(box, panel)
            clipped = box.width * box.height - (inside.width * inside.height if inside else 0)
            overlaps = [Bbox.intersection(box, obstacle) for obstacle in occupied]
            collision = sum(item.width * item.height for item in overlaps if item is not None)
            placements.append((clipped * 5 + collision, offset, ha, va, box))
            if clipped < 1e-6 and collision < 1e-6:
                break
        _, offset, ha, va, box = min(placements, key=lambda placement: placement[0])
        label.set_position(offset)
        label.set_horizontalalignment(ha)
        label.set_verticalalignment(va)
        occupied.append(box)


def build_multilayer_figure(
    inter_cluster: Mapping[str, pd.DataFrame],
    intra_cluster: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    conditions: Sequence[str] | None = None,
    modules: Sequence[str] | None = None,
    title: str = "Two-layer IDOP network",
    max_edges_per_panel: int | None = None,
    curvature_scale: float = DEFAULT_EDGE_CURVATURE_SCALE,
) -> Figure:
    """生成模块间网络与模块内子网的条件对照图。"""
    available = list(dict.fromkeys([*inter_cluster, *intra_cluster]))
    columns = list(conditions) if conditions is not None else available
    if not columns or len(columns) != len(set(columns)) or set(columns) - set(available):
        raise ValueError(f"条件须互不重复且来自输入。可用条件：{', '.join(available)}")
    if max_edges_per_panel is not None and max_edges_per_panel < 1:
        raise ValueError("每格最多边数须为正整数。")
    if not np.isfinite(curvature_scale) or curvature_scale < 0:
        raise ValueError("边曲率倍率须为非负有限数。")

    module_names = set()
    built_modules = set()
    panel_matrix = {}
    for condition in columns:
        inter = inter_cluster.get(condition)
        if inter is not None:
            panel_matrix[(None, condition)] = inter
            module_names.update(str(node).strip() for node in inter.index)
        for module, frame in intra_cluster.get(condition, {}).items():
            module_names.add(module)
            built_modules.add(module)
            panel_matrix[(module, condition)] = frame

    rows = list(modules) if modules is not None else idop.sort_modules(built_modules)
    if len(rows) != len(set(rows)) or set(rows) - module_names:
        raise ValueError(f"模块须互不重复且来自输入。可用模块：{', '.join(idop.sort_modules(module_names))}")

    panel_size = 1.8
    row_keys = [None, *rows]
    width = 0.96 + panel_size * len(columns)
    height = 1.15 + panel_size * len(row_keys)
    fonts = {font.name for font in fontManager.ttflist}
    cjk = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if name in fonts), None)
    with mpl.rc_context({"font.family": ["DejaVu Sans"] + ([cjk] if cjk else [])}):
        figure = Figure(figsize=(width, height), facecolor="white")
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(
            len(row_keys) + 1, len(columns) + 1,
            width_ratios=[0.78, *([panel_size] * len(columns))],
            height_ratios=[0.37, *([panel_size] * len(row_keys))],
            left=0.12 / width, right=1 - 0.08 / width,
            bottom=0.37 / height, top=1 - 0.40 / height,
            wspace=0, hspace=0,
        )
        figure.text(0.20 / width, 1 - 0.18 / height, title,
                    va="center", fontsize=14, weight="bold", color=LABEL_COLOR)

        for col_number, condition in enumerate(columns, start=1):
            heading = figure.add_subplot(grid[0, col_number])
            heading.set_facecolor(HEADER_COLOR)
            heading.set_xticks([])
            heading.set_yticks([])
            for spine in heading.spines.values():
                spine.set_color(PANEL_BORDER)
                spine.set_linewidth(0.55)
            heading.text(0.5, 0.5, condition, ha="center", va="center",
                         fontsize=9, weight="bold", color=LABEL_COLOR, wrap=True)

        for row_number, module in enumerate(row_keys, start=1):
            heading = figure.add_subplot(grid[row_number, 0])
            heading.set_axis_off()
            label = module if module is not None else "Inter-\nmodule"
            heading.text(0.94, 0.5, label, ha="right", va="center",
                         fontsize=9, weight="bold", color=LABEL_COLOR)
            panels = []
            for condition in columns:
                frame = panel_matrix.get((module, condition))
                # 每行使用同一布局种子，降低跨条件比较时的无关旋转和抖动。
                key = "inter-module" if module is None else module
                panels.append(_prepare_panel(frame, key, max_edges_per_panel) if frame is not None else None)
            present = [panel for panel in panels if panel is not None]
            bound = max(
                (panel[0][["x", "y"]].abs().to_numpy().max() for panel in present),
                default=0.0,
            )
            radius = max(4.0, 1.42 * bound)
            pooled = pd.concat([panel[2] for panel in present], ignore_index=True) if present else pd.DataFrame()
            width_levels = idop._edge_width_levels(pooled)
            for col_number, panel in enumerate(panels, start=1):
                axis = figure.add_subplot(grid[row_number, col_number])
                _draw_panel(axis, panel, radius, width_levels, curvature_scale)

        handles = [
            Line2D([], [], color=idop.POSITIVE_EDGE_COLOR, lw=1.8, label="Positive weight"),
            Line2D([], [], color=idop.NEGATIVE_EDGE_COLOR, lw=1.8, label="Negative weight"),
        ]
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04 / height),
                      ncol=2, fontsize=8, frameon=False)
    return figure


def plot_multilayer(
    inter_cluster: Mapping[str, pd.DataFrame],
    intra_cluster: Mapping[str, Mapping[str, pd.DataFrame]],
    output_path: str | Path,
    *,
    conditions: Sequence[str] | None = None,
    modules: Sequence[str] | None = None,
    title: str = "Two-layer IDOP network",
    max_edges_per_panel: int | None = None,
    curvature_scale: float = DEFAULT_EDGE_CURVATURE_SCALE,
) -> Path:
    """绘制并导出模块间网络与模块内子网的条件对照图。"""
    path = Path(output_path)
    if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("输出文件扩展名须为 .png、.svg 或 .pdf。")
    figure = build_multilayer_figure(
        inter_cluster, intra_cluster, conditions=conditions, modules=modules,
        title=title, max_edges_per_panel=max_edges_per_panel, curvature_scale=curvature_scale,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"pdf.fonttype": 42}):
        figure.savefig(path, dpi=240, facecolor="white")
    return path


def create_selector_window():
    """打开导出 ZIP，选择模块子网并导出对照图。"""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("IDOP 多层网络")
    root.minsize(460, 345)
    container = ttk.Frame(root, padding=16)
    container.pack(fill="both", expand=True)
    status = tk.StringVar(value="请先打开 NetRecon 的 Multi-Layer 导出 ZIP。")
    file_label = tk.StringVar()
    inter, intra = {}, {}
    input_path = None

    def open_export():
        nonlocal inter, intra, input_path
        filename = filedialog.askopenfilename(
            parent=root, title="打开 NetRecon 多层导出",
            filetypes=[("NetRecon ZIP", "*.zip")],
        )
        if not filename:
            return
        try:
            new_inter, new_intra = read_netrecon_export(filename)
        except (OSError, BadZipFile, ValueError) as exc:
            messagebox.showerror("无法打开导出", str(exc), parent=root)
            return
        inter, intra = new_inter, new_intra
        input_path = Path(filename)
        file_label.set(input_path.name)
        module_list.delete(0, tk.END)
        modules = idop.sort_modules({module for networks in intra.values() for module in networks})
        for module in modules:
            module_list.insert(tk.END, module)
        draw_button.state(["!disabled"])
        status.set(
            f"读取到 {len(modules)} 个已构网模块，点击模块名可多选。"
            if modules else "已读取模块间网络。该 ZIP 没有模块内构网结果。"
        )

    def save_figure():
        selected = [module_list.get(index) for index in module_list.curselection()]
        if not selected and not inter:
            status.set("该 ZIP 只有模块内网络，请至少选择一个模块。")
            return
        filename = filedialog.asksaveasfilename(
            parent=root, title="保存 IDOP 多层网络图",
            initialdir=input_path.parent, initialfile=f"{input_path.stem}_figure.svg",
            defaultextension=".svg",
            filetypes=[("SVG 矢量图", "*.svg"), ("PNG 图片", "*.png"), ("PDF 图", "*.pdf")],
        )
        if not filename:
            return
        status.set("正在绘图…")
        root.update_idletasks()
        try:
            path = plot_multilayer(inter, intra, filename, modules=selected)
        except (OSError, ValueError) as exc:
            status.set("导出失败。")
            messagebox.showerror("导出失败", str(exc), parent=root)
        else:
            status.set(f"已导出 {path.name}")

    file_bar = ttk.Frame(container)
    file_bar.pack(fill="x", pady=(0, 12))
    ttk.Button(file_bar, text="打开 NetRecon ZIP", command=open_export).pack(side="left")
    ttk.Label(file_bar, textvariable=file_label, padding=(8, 0)).pack(side="left", fill="x")
    ttk.Label(container, text="选择要显示的模块内网络").pack(anchor="w")
    choices = ttk.Frame(container)
    choices.pack(fill="both", expand=True, pady=(5, 8))
    scroll = ttk.Scrollbar(choices, orient="vertical")
    scroll.pack(side="right", fill="y")
    module_list = tk.Listbox(
        choices, selectmode=tk.MULTIPLE, exportselection=False,
        height=8, yscrollcommand=scroll.set,
    )
    module_list.pack(side="left", fill="both", expand=True)
    scroll.configure(command=module_list.yview)
    actions = ttk.Frame(container)
    actions.pack(fill="x")
    ttk.Button(actions, text="全选", command=lambda: module_list.selection_set(0, tk.END)).pack(side="left")
    ttk.Button(actions, text="清空", command=lambda: module_list.selection_clear(0, tk.END)).pack(side="left", padx=5)
    draw_button = ttk.Button(actions, text="绘制所选模块", command=save_figure, state="disabled")
    draw_button.pack(side="right")
    ttk.Label(container, text="不选模块时，仅绘制模块间网络。").pack(anchor="w", pady=(9, 0))
    ttk.Label(container, textvariable=status, wraplength=430).pack(anchor="w", pady=(5, 0))
    return root


def main():
    if len(sys.argv) == 1:
        create_selector_window().mainloop()
        return

    parser = argparse.ArgumentParser(description="绘制 IDOP 多层网总图。")
    parser.add_argument("input", type=Path, help="NetRecon 多层导出 ZIP 或解压目录。不传时打开选模块窗口")
    parser.add_argument("--out", type=Path, default=Path("idop_multilayer.png"))
    parser.add_argument("--conditions", nargs="+", help="选择并排序条件列")
    parser.add_argument("--modules", nargs="+", help="选择并排序模块行")
    parser.add_argument("--title", default="Two-layer IDOP network")
    parser.add_argument("--max-edges-per-panel", type=int, help="限制每格展示的边数")
    parser.add_argument("--curvature-scale", type=float, default=DEFAULT_EDGE_CURVATURE_SCALE, help="边曲率倍率")
    args = parser.parse_args()

    try:
        inter, intra = read_netrecon_export(args.input)
        path = plot_multilayer(
            inter, intra, args.out, conditions=args.conditions, modules=args.modules,
            title=args.title, max_edges_per_panel=args.max_edges_per_panel, curvature_scale=args.curvature_scale,
        )
    except (OSError, BadZipFile, ValueError) as exc:
        parser.error(str(exc))
    print(path.resolve())


if __name__ == "__main__":
    main()
