# GLMY barcode

利用 `src/network_data.py` 生成的有向加权边表，计算并绘制 GLMY **regular path homology** 的持续条形码。

## 使用

直接在编辑器打开 `src/plot_barcode.py` 并点击 **Run** 即可：默认调用同目录 `src/network_data.py` 的 `main()` 生成边表，保存到 `output/network.csv`，再生成 **300 DPI** 的 `output/barcode.png` 和区间表 `output/barcode.csv`，不弹出图窗。也可先运行 `src/network_data.py`，再运行 `src/plot_barcode.py`；绘图时会按当前模拟参数重新生成一次边表。默认路径按脚本位置定位，不依赖编辑器的工作目录。修改模拟参数时，编辑 `src/network_data.py` 的 `main()` 即可。

也可不带参数运行；如需额外显示图窗，可添加 `--show`：

Python 数据脚本可以由 `main()` 返回包含 `source,target,weight` 列的 DataFrame；也兼容 `main()` 调用一次 `generate_network_data(...)` 后只保存 CSV、不返回值的写法。后一种情况直接接收本次生成的边表及其节点元数据，不读取旧 CSV。若在生成后还会替换或筛选边表，建议显式 `return edgelist` 以明确最终输入。

```bash
python src/plot_barcode.py
python src/plot_barcode.py --show
```

在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python -m src.plot_barcode output/network.csv --n-nodes 12 --output output/barcode.png
```

输出 `output/barcode.png` 和区间表 `output/barcode.csv`。

在 Python 中：

```python
from src.network_data import generate_network_data
from src.plot_barcode import plot_barcode, compute_barcode

edgelist = generate_network_data(n=12, p=0.2, seed=42)
barcode, fig, axes = plot_barcode(
    edgelist,
    max_dim=2,                    # 默认计算 H_0、H_1、H_2
    save_path="output/barcode.png",
    show=True,
)
print(barcode)
barcode.to_csv("output/barcode.csv", index=False)

# 只计算，不绘图：
barcode = compute_barcode(edgelist, max_dim=2)

# CSV 无法记录孤立点，因此需提供完整节点列表：
barcode, fig, axes = plot_barcode("output/network.csv", nodes=range(12))
```

`plot_barcode` 返回 `(barcode, fig, axes)`。`barcode` 是包含 `dimension`、`birth`、`death` 三列的 DataFrame，每行为半开区间 `[birth, death)`；`death=inf` 表示在最终图中仍然存在。黑色空心圆表示出生，黑色实心方块表示死亡，黑色箭头右侧标注 +∞ 表示无限延伸。β₀ 条形码为红色，y 轴刻度为黑色；β₁ 条形码和 y 轴刻度为绿色。两者的区间序号均作为 y 轴刻度标签，从最下面的 #1 向上编号。

子图从上到下按维度递减排列，纵轴用数学公式渲染为具体的 β₁、β₀ 等，不显示子图标题；横轴标签为 `filtration`。条形码按区间表的倒序从下向上绘制。图宽为 4.5 英寸，每个子图绘图区宽高比固定为 4:2，相邻子图无留白。`axes` 始终为一维数组，按子图从上到下排列。

## 数学约定

默认计算并绘制 β₂、β₁、β₀ 三个子图，β₂ 条形码和序号为蓝色。若某个维度没有非零长度持续区间，子图显示 `No nonzero intervals`。命令行可用 `--max-dim 1` 恢复只画 β₁ 和 β₀。

- 过滤：`G_t = (V, {(u,v): |weight(u,v)| <= t})`，`t >= 0`；所有节点在 `t=0` 出现。
- 保留有向边；不做无向化，不添加最短路或传递闭包边。
- 系数域固定为 **F₂**，计算普通（非约化）同调；不代表实数域下的结果。
- 权重只决定边的出现时间；`sign` 列不参与计算，也不是带符号系数的同调。
- 使用 regular 路径：相邻节点不得相同，允许 `u,v,u` 等非相邻重复节点；边界中的非 regular 项取零。
- 每个阈值构造 `Ω_p = {c ∈ A_p : ∂c ∈ A_(p-1)}`，扩充与过滤相容的基，然后约化边界矩阵，得到持续区间。不能直接把全部有向路径当作一个边界封闭的链复形。
- 计算到 `H_k` 时，会构造到 `Ω_(k+1)`，以包含最高请求维度的死亡事件。同一阈值出现的边一起处理，不返回零长度区间。

定义依据：[Chowdhury & Mémoli, Persistent Path Homology of Directed Networks](https://research.math.osu.edu/networks/pubs/pph-arxiv.pdf)，第 4–5 节。

## 数据与规模

输入为 DataFrame 或 CSV 路径，必须有 `source,target,weight` 列。重复有向边按最小绝对权重合并；自环、缺失值和无穷权重会报错。`nodes` 提供额外节点，与所有边端点取并集。

生成器的 DataFrame 通过 `attrs['nodes']` 保存孤立点；函数直接接收该 DataFrame 时会自动读取。CSV 不保留此信息，需显式传入 `nodes`，命令行可用 `--n-nodes` 指定编号为 `0,...,n-1` 的节点。

这是面向小型稀疏网络的精确参考实现。路径数量随维度和网络密度快速增长，默认 `max_paths=100000` 限制构造的 allowed 路径总数；超限会报错，不会静默截断。该限制不是运行时间或内存保证。

## Homology representative 可视化

直接打开 `src/plot_homology_representative.py` 点击 **Run**，或者：

```bash
python src/plot_homology_representative.py
```

默认读取同目录 `src/network_data.py`，为 β₁、β₂ 的每条非零长度持续区间各绘制一个代表元。所有小图使用相同的圆形节点布局，保留孤立点；绿色表示 β₁，蓝色表示 β₂。每个面板编号与条形码在该维度从下向上的 `#1, #2, ...` 一致，标题给出对应的 `[birth, death)`。代表元图按编号从左到右、从上到下排列，超过 5 个自动换行，不省略区间。

默认在 `output` 输出：

- `homology_representative_beta1.png`、`homology_representative_beta2.png`：300 DPI 代表元图。
- `homology_representative_intervals.csv`：代表元编号、出生/死亡值、路径项数和支撑边数。
- `homology_representative_paths.csv`：每个代表元的完整路径链，每项系数为 1，运算在 F₂ 上。
- `homology_representative_edges.csv`：各代表元的支撑边及原始正负权重、绝对进入时间。
- `homology_representative_nodes.csv`：全部节点及其共用绘图坐标。
- `homology_representative_barcode.csv`：本次计算的完整条形码；同时更新 `barcode.png`、`barcode.csv`，确保与代表元使用同一份实际生成的数据。

这里的面板数是**全部非零长度持续区间数**，包括有限区间，不是最终过滤阈值下的 Betti 数。代表元在对应区间内有效；不同面板可能属于不同过滤阶段。β₂ 的小图展示 2-path 链所涉及的有向边的并集，仅凭边图不能唯一确定二维链，需查看 `paths.csv` 中的具体路径项。算法不保证最短代表元，代表元也不唯一。

有限区间使用对应约化死亡列表示的循环链；无限区间使用零列的基变换循环链。关于约化矩阵与代表循环的对应关系，可参见 [PHAT — Persistent Homology Algorithms Toolbox](https://doi.org/10.1016/j.jsc.2016.03.008)。

Python 调用：

```python
from src.network_data import generate_network_data
from src.plot_homology_representative import plot_homology_representative

edgelist = generate_network_data(n=12, p=0.3, seed=42)
representatives, figures = plot_homology_representative(
    edgelist, output_dir="output", max_columns=5,
)
for rep in representatives:
    print(rep.dimension, rep.number, rep.birth, rep.death, rep.paths)
```

支持 CSV 路径输入、`nodes=range(n)`、`show_node_labels=False`。命令行可用 `--no-labels` 隐藏节点名称，`--max-columns 4` 调整每行面板数，`--vector` 额外导出可编辑的 SVG/PDF。`--color-by-sign`（Python 参数 `color_by_sign=True`）可将支撑边改为正红、负蓝、零灰，标题仍按同调维度着色；默认继续使用 β₁ 绿、β₂ 蓝。

## 随滤值恢复网络的过程图

直接打开 `src/plot_filtration_process.py` 点击 **Run**，不需要先运行其他脚本。它会调用同目录 `network_data.py`，使用同一份实际生成的数据计算 H₀–H₂ 条形码，并在上方标出阶段 `[1]`、`[2]` 等，在下方画出对应的网络和 `(β₀, β₁, β₂)`。

默认使用全部有向边，按 `|weight| <= t` 依次恢复。所有节点（包括孤立点）在所有阶段使用相同圆形布局。默认选择 `t=0`、两个边出现时间的分位数、最后一条边的出现时间；权重重复时可能少于 4 个不同阶段，零权重边会在 `t=0` 出现。没有加任何最短路或传递闭包边。上方条形码旁的括号是该维度的持续区间数，下方的 β 三元组是各阈值的实际 Betti 数。

下方全部网络阶段默认横向排成一行，使用代表元图的黑色实心节点、外侧节点编号和浅灰色方框，方向箭头位于边的中间。若需要手动换行，可用 `--columns 2` 指定每行的阶段数。

在脚本顶部可修改点击 Run 时使用的默认配置：

```python
N_STAGES = 4
FILTRATION_VALUES = None  # 或设成 [0.0, 0.3, 0.6, 1.0]
EDGE_SIGN = "all"        # 可选 "positive" 或 "negative"
```

也可命令行指定滤值或只画正边子图：

```bash
python src/plot_filtration_process.py
python src/plot_filtration_process.py --thresholds 0 0.3 0.6 1.0
python src/plot_filtration_process.py --edge-sign positive
python src/plot_filtration_process.py --color-by-sign
```

正/负边选择使用 `weight` 的实际正负号，仍然保留原网络全部节点，并重新计算所选子图的同调。若想观察很短的 β₂ 区间，可把该区间出生值加入 `FILTRATION_VALUES`；等分位阶段不保证覆盖每个同调事件。

默认生成 `output/filtration_process.png`（300 DPI），以及同名前缀的 `_snapshots.csv`、`_barcode.csv`、`_edges.csv`、`_nodes.csv`，分别记录阶段摘要、完整条形码、各阶段实际存在的边、共用节点布局。`--vector` 可额外导出 SVG/PDF。

`--color-by-sign`（Python 参数 `color_by_sign=True`）将网络正边画成红色、负边画成蓝色、零权重边画成灰色，仅改变绘图，不改变同调结果；条形码颜色仍按维度区分。`_edges.csv` 同时记录原始带符号的 `weight`、对应的 `sign` 和绝对值 `entry_time`。重复有向边采用最早进入的那条记录的权重及符号；绝对权重相同则取输入中第一条。正/负筛选发生在合并重复边之前。

Python 调用返回 `(barcode, snapshots, figure)`：

```python
from src.network_data import generate_network_data
from src.plot_filtration_process import plot_filtration_process

edgelist = generate_network_data(n=16, p=0.4, seed=42)
barcode, snapshots, figure = plot_filtration_process(
    edgelist,
    filtration_values=[0.0, 0.3, 0.6, 1.0],
    save_path="output/filtration_process.png",
)
```

## 对比正负边的两种展示方式

直接打开 `src/plot_signed_filtration_process.py` 点击 **Run**，即可一次输出三张恢复过程图及每组 β₁、β₂ 代表元，不需要先运行其他脚本：

```text
output/sign_comparison/
  01_all_edges_colored/filtration_process.png
  02_separate_signs/positive/filtration_process.png
  02_separate_signs/negative/filtration_process.png
```

第一种方式保留全部边，正边红色、负边蓝色，barcode 与原来的全边计算一致。第二种方式分别保留 `weight > 0` 或 `weight < 0` 的边，各自重新计算 H₀–H₂，绘制对应的 barcode 与网络恢复过程；正子图用红边，负子图用蓝边。零权重边只出现在全边图中。三张图保留全部节点，使用相同圆形布局和同一组滤值，便于逐阶段比较。

上述三个子目录还分别输出 `homology_representative_beta1.png`、`homology_representative_beta2.png`，以及同名前缀的 `_barcode.csv`、`_intervals.csv`、`_paths.csv`、`_edges.csv` 和 `_nodes.csv`。每个代表元由该组实际选中的边表计算，使用与恢复图相同的完整节点和坐标，并按正负号着色。编号 `#1, #2, ...` 对应该组 barcode 在该维度倒序的持续区间；不是简单从全边代表元中删除异号边。

没有非零长度区间的维度仍保存图片，明确显示 `No nonzero intervals`。β₂ 图片只展示路径链的支撑边，完整二维链见 `_paths.csv`。命令行 `--max-columns` 控制代表元每行最多几个面板，`--columns` 仍只控制恢复过程图。

脚本仅调用数据生成器一次，即使设置 `seed=None`，三张图也使用同一份边表。默认阶段由三组图的边进入时间共同确定，并覆盖三个最终网络；重复边可能导致正/负子图中某条边晚于全边图出现。可在新脚本顶部修改 `N_STAGES` 或 `FILTRATION_VALUES`。指定滤值时严格使用给定阶段。

每个结果子目录保存 **300 DPI PNG**、对应 CSV 表和 `network.csv`；根目录保存原始边表 `input_edges.csv`、完整节点表 `input_nodes.csv`、筛选数量及代表元数量统计 `selection_summary.csv` 和并排比较所需的阶段表 `comparison_snapshots.csv`。恢复图和代表元共用同一次生成的数据。重新运行会更新这些结果。这里计算的仍是 F₂ 上的普通 GLMY 同调；正负号用于显示或选择子图，不作为边界算子的带符号系数。

```bash
python src/plot_signed_filtration_process.py
python src/plot_signed_filtration_process.py --thresholds 0 0.3 0.6 1.0
python src/plot_signed_filtration_process.py --vector
```

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖有向三环、传递三角形、菱形路径组合、双向边、有限死亡区间、孤立点、相同阈值和无效输入；并用独立的稠密 F₂ 线性代数实现，在随机小图上核对 H₀–H₂ 所有阈值对的持续映射秩。
