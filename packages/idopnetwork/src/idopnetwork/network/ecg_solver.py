"""Numerical kernel from user-supplied idopECG(4).zip, src/idopECG.py.

Author: Yu Wang. The numerical implementation is preserved; plotting/CLI imports
are excluded and output_path=None disables disk writes for concurrent web sessions.
"""
import math
from itertools import permutations
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from scipy.integrate import cumulative_trapezoid
from sklearn.linear_model import Lasso

def get_smoothed_data(
    data: pd.DataFrame,
    wavelet: str = "rbio3.9",
    level: int = 2,
    alpha: float = 0.5,
    mode: str = "soft",
) -> pd.DataFrame:
    """
    Smooth ECG data with the stationary wavelet transform (SWT).

    Note
    ----
    逐导联处理。只有细节系数 D1..D_level 参与阈值处理，更低频带全部留在近似
    系数里，因此 level 越大能压掉的抖动频段越靠下。1000 样本的 swt_max_level
    是 3，默认 level=2 覆盖 12.5-50 Hz。
    sigma 取最细尺度 (level 1) 细节系数的 MAD/0.6745，所有层共用同一个阈值；
    逐层各自估计会把该层的 QRS 能量误当噪声，反而削峰。
    阈值 = alpha * sigma * sqrt(2 log N)。

    Parameters
    ----------
    data : (n_timepoints, n_channels) | pd.DataFrame
        columns 可含 time，其余每列一个导联。
    wavelet : str
        小波基。
    level : int
        SWT 分解层数，实际取 max(1, min(level, swt_max_level(n_timepoints)))。
    alpha : float
        阈值系数，越大越平滑。
    mode : str
        pywt.threshold 的阈值方式，"soft"（默认）或 "garrote"。

    Returns
    -------
    data_smoothed : (n_timepoints, n_channels) | pd.DataFrame
        与输入同列、同索引的平滑结果。
    """
    data_smoothed = data.copy()
    lead_columns = [column for column in data.columns if column != "time"]
    level = max(1, min(int(level), pywt.swt_max_level(len(data_smoothed))))

    for lead in lead_columns:
        signal = data[lead].to_numpy(dtype=float)
        coefficients = pywt.swt(signal, wavelet=wavelet, level=level)

        # coefficients[-1] 是最细尺度 (level 1) 的细节系数
        detail = coefficients[-1][1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
        threshold = alpha * sigma * np.sqrt(2.0 * np.log(signal.size))

        # 常量导联的 sigma = 0，阈值会出现 0/0，直接跳过
        if sigma > 0:
            coefficients = [
                (approx, pywt.threshold(layer, threshold, mode=mode))
                for approx, layer in coefficients
            ]
        fitted = pywt.iswt(coefficients, wavelet=wavelet)
        # 防御: 残留的非有限值置 0, 避免污染下游
        data_smoothed[lead] = np.nan_to_num(
            fitted, nan=0.0, posinf=0.0, neginf=0.0)

    return data_smoothed


def get_support_sets(
    data: pd.DataFrame,
    output_path: Path | None = None,
    alpha_ratio: float = 0.1,
    window_size: int | None = None,
    stride: int | None = None,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """
    Select each lead's support set by sliding-window Lasso voting.

    Note
    ----
    对每个 target 导联用其余 11 个导联做 Lasso，在长度 window_size、步长 stride
    的滑窗上重复，统计每条有向边被选中的窗口频率，保留频率严格大于 threshold
    的边。每个窗口内 x 与 y 各自 z-score，故无需截距；alpha 取
    alpha_ratio * alpha_max，其中 alpha_max 是标准化后 x 与 y 的最大相关系数，
    使不同窗口的正则强度可比。time 列不参与选边，自环不参与选择。

    Parameters
    ----------
    data : (n_timepoints, n_channels) | pd.DataFrame
        平滑后的 ECG 数据，每列一个导联，可含 time 列。
    output_path : Path | None
        输出目录，默认项目下的 output/support_set。
    alpha_ratio : float
        每个窗口内 Lasso alpha 与 alpha_max 的比例，默认 0.1。
    window_size : int | None
        窗口样本数，默认 max(4, len(data) // 5)。
    stride : int | None
        滑动步长，默认 max(1, window_size // 4)。
    threshold : float
        入选频率阈值，默认 0.9，频率严格大于才保留。

    Returns
    -------
    support_sets : pd.DataFrame
        columns = ["target", "source"]，source 形如 "{aVR,aVL,V6}"。
        同时保存 support_set.csv 与 edge_frequency.csv（行 target、列 source）。
    """
    if not 0 < alpha_ratio <= 1:
        raise ValueError("alpha_ratio must be in (0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    lead_columns = [column for column in data.columns if column != "time"]
    if len(lead_columns) < 2 or not data.columns.is_unique:
        raise ValueError("Data must contain at least two uniquely named ECG leads.")
    values = data[lead_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ECG data must contain only finite values.")
    if window_size is None:
        window_size = max(4, len(data) // 5)
    if not isinstance(window_size, (int, np.integer)) or not 2 <= window_size <= len(data):
        raise ValueError("window_size must be an integer between 2 and len(data).")
    if stride is None:
        stride = max(1, window_size // 4)
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer.")

    edge_counts = {
        target: {source: 0 for source in lead_columns if source != target}
        for target in lead_columns
    }
    n_windows = 0

    for start in range(0, len(data) - window_size + 1, stride):
        window = values[start:start + window_size]
        n_windows += 1

        # 每个窗口独立 z-score, x 与 y 同时标准化
        mean = window.mean(axis=0)
        scale = window.std(axis=0)
        scale[scale == 0] = 1.0
        window = (window - mean) / scale

        for target_index, target in enumerate(lead_columns):
            source_indices = [
                index for index in range(len(lead_columns)) if index != target_index
            ]
            x = window[:, source_indices]
            y = window[:, target_index]

            # 自适应正则: 标准化后 x.T @ y / n 即相关系数, 取最大者为 alpha_max
            alpha_max = np.max(np.abs(x.T @ y)) / len(y)
            if alpha_max == 0.0:
                # 极端情况: 该窗口内所有自变量与 y 完全无关
                continue

            model = Lasso(
                alpha=alpha_ratio * alpha_max,
                fit_intercept=False,
                max_iter=10000,
                tol=1e-5,
            )
            model.fit(x, y)
            for local_index, coefficient in enumerate(model.coef_):
                if coefficient != 0.0:
                    edge_counts[target][lead_columns[source_indices[local_index]]] += 1

    if n_windows == 0:
        raise RuntimeError("no sliding window generated")

    frequency = pd.DataFrame(
        [
            [
                edge_counts[target][source] / n_windows if source != target else np.nan
                for source in lead_columns
            ]
            for target in lead_columns
        ],
        index=lead_columns,
        columns=lead_columns,
    )

    support_sets = pd.DataFrame(
        [
            {
                "target": target,
                "source": "{" + ",".join([
                    source for source in lead_columns
                    if source != target
                    and edge_counts[target][source] / n_windows > threshold
                ]) + "}",
            }
            for target in lead_columns
        ],
        columns=["target", "source"],
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        support_sets.to_csv(
            output_path / "support_set.csv", index=False, encoding="utf-8")
        frequency.to_csv(
            output_path / "edge_frequency.csv", index_label="target", encoding="utf-8")
    support_sets.attrs["edge_frequency"] = frequency
    return support_sets


def _source_orderings(sources: list[str], limit: int = 120) -> list[tuple]:
    """来源集的所有排列；超出 limit 时用固定种子抽样，保证可复现。"""
    if math.factorial(len(sources)) <= limit:
        return list(permutations(sources))
    rng = np.random.default_rng(0)
    seen, orderings = set(), []
    while len(orderings) < limit:
        index = tuple(rng.permutation(len(sources)))
        if index not in seen:
            seen.add(index)
            orderings.append(tuple(sources[i] for i in index))
    return orderings


def _edge_slope_sign(rate_curve: np.ndarray, source_signal: np.ndarray) -> float:
    """该边变化率贡献对来源电平的回归斜率，其符号定义为边的方向。"""
    centered = source_signal - source_signal.mean()
    energy = float(np.dot(centered, centered))
    return float(np.dot(rate_curve, centered) / energy) if energy > 0 else 0.0


def _sources_by_target(support_sets: pd.DataFrame, leads: list[str]) -> dict:
    """把 support_sets 的 "{aVR,aVL}" 解析成 {target: [source, ...]}。"""
    if not {"target", "source"}.issubset(support_sets.columns):
        raise ValueError("support_sets must contain target and source columns.")
    if support_sets["target"].duplicated().any() or set(support_sets["target"]) != set(leads):
        raise ValueError("support_sets must contain each ECG target exactly once.")
    sources_by_target = {}
    for target, source_text in support_sets[["target", "source"]].itertuples(
            index=False, name=None):
        if (not isinstance(source_text, str)
                or not source_text.strip().startswith("{")
                or not source_text.strip().endswith("}")):
            raise ValueError(f"Invalid source set for {target}: expected {{I,II}} or {{}}.")
        inner = source_text.strip()[1:-1].strip()
        sources = [source.strip() for source in inner.split(",")] if inner else []
        if len(sources) != len(set(sources)) or any(
                source not in leads or source == target for source in sources):
            raise ValueError(f"Invalid, duplicate, or self source for {target}.")
        sources_by_target[target] = sources
    return sources_by_target


def _legendre_basis(signal: np.ndarray, order: int, time: np.ndarray) -> tuple:
    """
    来源信号的勒让德变化率基及其积分。

    z_s 把信号取值范围映射到 [-1, 1]；基不含 P0，并按时间均值中心化后再积分，
    使常数变化率不进入交互项、累计贡献不随记录长度线性发散。
    """
    lower, upper = signal.min(), signal.max()
    if upper > lower:
        z = 2.0 * (signal - lower) / (upper - lower) - 1.0
    else:
        z = np.zeros(len(signal))
    raw = np.polynomial.legendre.legvander(z, order)[:, 1:]
    raw_integrals = cumulative_trapezoid(raw, time, axis=0, initial=0)
    centers = raw_integrals[-1] / (time[-1] - time[0])
    rates = raw - centers
    if upper == lower:
        rates[:] = 0.0
    integrals = cumulative_trapezoid(rates, time, axis=0, initial=0)
    return rates, integrals, centers, float(lower), float(upper)


def get_basis(
    data: pd.DataFrame,
    support_sets: pd.DataFrame,
    sampling_rate: float = 100.0,
    fourier_order: int | None = None,
    fourier_period: float | None = None,
    fourier_max_frequency: float = 10.0,
    legendre_order: int = 3,
) -> dict:
    """
    Define the basis expansion of Q_j(t) and Q_jj'(x_j'), without fitting.

    Note
    ----
    自身项 Q_j(t) 用傅里叶基：常数 + sum_k [a_k cos(w_k t) + b_k sin(w_k t)]，
    w_k = 2*pi*k/P。带宽 = fourier_order / P，而 P 默认取整段记录长度，所以阶数
    必须随记录长度缩放；默认由 fourier_max_frequency=10 Hz 推出
    fourier_order = ceil(10 * P)。9.99 s 记录配 20 阶只有 2 Hz，连 QRS 都表达
    不了，这一点是此前 R^2 偏低的主因。
    交互项 Q_jj'(x_j') 用来源信号的勒让德多项式 P_1..P_legendre_order。

    两种基都返回变化率基与其从 t0 起的累积积分。因为不对信号做数值求导，后续
    拟合一律在积分域进行：x_j(t) - x_j(t0) = int Q_j + sum int Q_jj'。
    本函数只定义基、不做任何拟合，因此可以对同一份基反复以不同设置求解。

    Parameters
    ----------
    data : pd.DataFrame
        平滑后的 ECG 数据，每列一个导联，可含以秒为单位、严格递增的 time 列。
    support_sets : pd.DataFrame
        get_support_sets 的返回值，用于确定需要构造哪些来源的基。
    sampling_rate : float
        无 time 列时的采样率，默认 100 Hz。
    fourier_order : int | None
        自身项谐波数，默认由 fourier_max_frequency * fourier_period 推出。
    fourier_period : float | None
        傅里叶基周期（秒），默认记录首尾时间差。
    fourier_max_frequency : float
        自身项最高频率（Hz），默认 10。
    legendre_order : int
        交互项最高阶数，默认 3。

    Returns
    -------
    basis : dict
        time: (n_timepoints,) 秒；
        self: rates/integrals (n_timepoints, 2*fourier_order+1) 与 labels；
        sources: {导联: rates/integrals (n_timepoints, legendre_order)、
            centers、lower、upper}；
        settings: 本次基展开的参数摘要。
    """
    leads = [column for column in data.columns if column != "time"]
    if not leads or not data.columns.is_unique or len(data) < 3:
        raise ValueError("Data must contain unique ECG leads and at least three samples.")
    if not np.isfinite(data[leads].to_numpy(dtype=float)).all():
        raise ValueError("ECG data must contain only finite values.")
    if fourier_order is not None and (
            isinstance(fourier_order, (bool, np.bool_))
            or not isinstance(fourier_order, (int, np.integer)) or fourier_order < 0):
        raise ValueError("fourier_order must be a nonnegative integer, or None.")
    if (isinstance(legendre_order, (bool, np.bool_))
            or not isinstance(legendre_order, (int, np.integer)) or legendre_order < 1):
        raise ValueError("legendre_order must be an integer >= 1.")

    if "time" in data.columns:
        time = data["time"].to_numpy(dtype=float)
    else:
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be finite and positive.")
        time = np.arange(len(data)) / sampling_rate
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be finite and strictly increasing.")

    elapsed = time - time[0]
    duration = elapsed[-1]
    if fourier_period is None:
        fourier_period = duration
    if not np.isfinite(fourier_period) or fourier_period <= 0:
        raise ValueError("fourier_period must be finite and positive.")
    if fourier_order is None:
        if not np.isfinite(fourier_max_frequency) or fourier_max_frequency <= 0:
            raise ValueError("fourier_max_frequency must be finite and positive.")
        fourier_order = int(np.ceil(fourier_max_frequency * fourier_period))
    if fourier_order and fourier_order / fourier_period >= 0.5 / np.max(np.diff(time)):
        raise ValueError("Highest Fourier frequency must be below the sampling limit.")

    rates = [np.ones(len(time))]
    integrals = [elapsed]
    labels = [("constant", 0)]
    for order in range(1, fourier_order + 1):
        omega = 2.0 * np.pi * order / fourier_period
        phase = omega * elapsed
        rates.extend([np.cos(phase), np.sin(phase)])
        integrals.extend([np.sin(phase) / omega, (1.0 - np.cos(phase)) / omega])
        labels.extend([("cos", order), ("sin", order)])

    sources_by_target = _sources_by_target(support_sets, leads)
    selected = {source for sources in sources_by_target.values() for source in sources}
    sources = {}
    for source in leads:
        if source not in selected:
            continue
        source_rates, source_integrals, centers, lower, upper = _legendre_basis(
            data[source].to_numpy(dtype=float), legendre_order, time)
        sources[source] = {
            "rates": source_rates, "integrals": source_integrals,
            "centers": centers, "lower": lower, "upper": upper,
        }

    return {
        "time": time,
        "self": {
            "rates": np.column_stack(rates),
            "integrals": np.column_stack(integrals),
            "labels": labels,
        },
        "sources": sources,
        "settings": {
            "sampling_rate": sampling_rate if "time" not in data.columns else np.nan,
            "time_origin": float(time[0]), "time_end": float(time[-1]),
            "n_samples": len(time),
            "fourier_order": fourier_order, "fourier_period": fourier_period,
            "fourier_bandwidth_hz": fourier_order / fourier_period,
            "legendre_order": legendre_order,
        },
    }


def _fit_integral_basis(
    design: np.ndarray,
    response: np.ndarray,
    tolerance: float,
    ridge_alpha: float,
    n_samples: int,
) -> tuple:
    """按列 RMS 归一化后的（可选岭）最小二乘；返回原尺系数、有效列与归一化设计。"""
    scales = np.sqrt(np.mean(design ** 2, axis=0))
    active = scales > tolerance
    scaled = design[:, active] / scales[active]
    coefficients = np.zeros(design.shape[1])
    if scaled.shape[1]:
        if ridge_alpha:
            fit_design = np.vstack([
                scaled, np.sqrt(n_samples * ridge_alpha) * np.eye(scaled.shape[1]),
            ])
            fit_response = np.concatenate([response, np.zeros(scaled.shape[1])])
        else:
            fit_design, fit_response = scaled, response
        solution = np.linalg.lstsq(fit_design, fit_response, rcond=None)[0]
        coefficients[active] = solution / scales[active]
    return coefficients, active, scaled


def _orthogonalize_on_prefix(
    integrals: np.ndarray,
    rates: np.ndarray,
    prefix_rates: list,
    prefix_integrals: list,
    tolerance: float,
) -> tuple:
    """把一块基对既有前缀的积分空间正交化；前缀为空时原样返回。"""
    if not prefix_integrals:
        return rates, integrals
    block_integrals = np.column_stack(prefix_integrals)
    block_rates = np.column_stack(prefix_rates)
    scales = np.sqrt(np.mean(block_integrals ** 2, axis=0))
    active = scales > tolerance
    if not active.any():
        return rates, integrals
    scaled = block_integrals[:, active] / scales[active]
    projection = np.zeros((block_integrals.shape[1], integrals.shape[1]))
    projection[active] = (
        np.linalg.lstsq(scaled, integrals, rcond=None)[0] / scales[active, None])
    return (rates - block_rates @ projection,
            integrals - block_integrals @ projection)


def get_effects(
    data: pd.DataFrame,
    support_sets: pd.DataFrame,
    output_path: Path | None = None,
    sampling_rate: float = 100.0,
    fourier_order: int | None = None,
    fourier_period: float | None = None,
    fourier_max_frequency: float = 10.0,
    legendre_order: int = 3,
    ridge_alpha: float = 1e-6,
) -> dict[str, pd.DataFrame]:
    """
    Fit MODEs and split each response into independent and cross effects.

    Note
    ----
    模型（对每个 target 导联 j，S_j 由选边器给出）：

        dx_j/dt = Q_j(t) + sum_{j' in S_j} Q_jj'(x_j')

    基由 get_basis 定义，拟合在积分域完成：
        x_j(t) - x_j(t0) = int Q_j + sum int Q_jj'
    因此全程不做数值求导，初值固定为输入首个样本。

    分摊约定：交互项优先。交互项先在完整响应上按来源排列依次正交化并取平均
    (Shapley)，每条边只解释前面各边剩余的部分；自身项再对全部交互基正交化后
    拟合残差。这样做是因为交互基约有 99.4% 的能量落在自身项的时间模板空间内，
    若让两者联合求解，分摊会退化成近零空间里的任意分配（实测出现负份额与百倍
    于信号的单边幅值）。依次正交化 + Shapley 保证单边幅值有界、且分摊与来源
    顺序无关；各排列的总拟合相同，所以 R^2 与联合拟合完全一致。

    边权重 weight = <累计效应, 响应> / ||响应||^2，恒正，同一目标的各边之和等于
    该导联交互项的解释份额。边符号 sign 由变化率贡献对来源电平的回归斜率决定。
    两者都只由效应曲线导出，且都是条件量：只在相同的支撑集、带宽、分摊约定下
    有意义，不可跨设置比较。

    Parameters
    ----------
    data : pd.DataFrame
        平滑后的 ECG 数据，可含 time 列。
    support_sets : pd.DataFrame
        get_support_sets 的返回值。
    output_path : Path | None
        输出目录，默认项目下的 output/effect。
    sampling_rate : float
        无 time 列时的采样率，默认 100 Hz。
    fourier_order, fourier_period, fourier_max_frequency, legendre_order
        透传给 get_basis，决定自身项带宽与交互项阶数。
    ridge_alpha : float
        两阶段归一化设计矩阵上的岭惩罚，默认 1e-6；0 为纯最小二乘。

    Returns
    -------
    effects : dict[str, pd.DataFrame]
        coefficients / self_effects / interaction_effects / reconstruction /
        diagnostics / network / orthogonalization / settings，同名保存为 CSV。
    """
    basis = get_basis(
        data,
        support_sets,
        sampling_rate=sampling_rate,
        fourier_order=fourier_order,
        fourier_period=fourier_period,
        fourier_max_frequency=fourier_max_frequency,
        legendre_order=legendre_order,
    )
    time = basis["time"]
    self_rates = basis["self"]["rates"]
    self_integrals = basis["self"]["integrals"]
    self_labels = basis["self"]["labels"]
    source_basis = basis["sources"]

    leads = [column for column in data.columns if column != "time"]
    sources_by_target = _sources_by_target(support_sets, leads)
    n_samples = len(time)
    duration = float(time[-1] - time[0])

    if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("ridge_alpha must be finite and nonnegative.")
    # 正交投影不加岭惩罚，否则投影残差不再与自身空间正交
    tolerance = np.finfo(float).eps * max(1.0, duration) * 100

    # 各来源基对自身空间正交化：既是分摊起点，也记录交互基与自身空间的重叠度
    self_scales = np.sqrt(np.mean(self_integrals ** 2, axis=0))
    self_active = self_scales > tolerance
    self_scaled = self_integrals[:, self_active] / self_scales[self_active]
    bases = {}
    projection_rows = []
    for source, block in source_basis.items():
        integrals, rates = block["integrals"], block["rates"]
        projection = np.zeros((self_integrals.shape[1], integrals.shape[1]))
        if self_active.any():
            projection[self_active] = (
                np.linalg.lstsq(self_scaled, integrals, rcond=None)[0]
                / self_scales[self_active, None])
        orthogonal_integrals = integrals - self_integrals @ projection
        orthogonal_rates = rates - self_rates @ projection
        residual_norm = np.sqrt(np.mean(orthogonal_integrals ** 2, axis=0))
        raw_norm = np.sqrt(np.mean(integrals ** 2, axis=0))
        inactive = residual_norm <= np.maximum(tolerance, raw_norm * 1e-12)
        orthogonal_integrals[:, inactive] = 0.0
        orthogonal_rates[:, inactive] = 0.0
        bases[source] = (
            orthogonal_rates, orthogonal_integrals,
            block["centers"], block["lower"], block["upper"],
        )
        for degree in range(integrals.shape[1]):
            for basis_index, (name, order) in enumerate(self_labels):
                projection_rows.append({
                    "source": source, "legendre_order": degree + 1,
                    "fourier_basis": name, "fourier_order": order,
                    "projection_coefficient": projection[basis_index, degree],
                    "active": bool(not inactive[degree]),
                })

    coefficient_rows, self_frames, interaction_frames = [], [], []
    reconstruction_frames, diagnostic_rows, network_rows = [], [], []

    for target in leads:
        sources = sources_by_target[target]
        signal = data[target].to_numpy(dtype=float)
        response = signal - signal[0]
        response_energy = float(np.dot(response, response))

        # ---- 交互项优先：依次正交化 + Shapley 平均 ----
        orderings = _source_orderings(sources)
        n_orderings = len(orderings)
        edge_curves = {source: np.zeros(n_samples) for source in sources}
        edge_rates = {source: np.zeros(n_samples) for source in sources}
        for ordering in orderings:
            prefix_rates, prefix_integrals = [], []
            remaining = response.copy()
            for source in ordering:
                block_rates, block_integrals = _orthogonalize_on_prefix(
                    source_basis[source]["integrals"],
                    source_basis[source]["rates"],
                    prefix_rates,
                    prefix_integrals,
                    tolerance,
                )
                beta = np.linalg.lstsq(block_integrals, remaining, rcond=None)[0]
                curve = block_integrals @ beta
                edge_curves[source] += curve / n_orderings
                edge_rates[source] += (block_rates @ beta) / n_orderings
                remaining = remaining - curve
                prefix_rates.append(block_rates)
                prefix_integrals.append(block_integrals)

        interaction_sum = (
            sum(edge_curves.values()) if sources else np.zeros(n_samples))
        interaction_rate = (
            sum(edge_rates.values()) if sources else np.zeros(n_samples))

        # ---- 自身项：对全部交互基正交化后拟合残差 ----
        if sources:
            self_fit_rates, self_fit_integrals = _orthogonalize_on_prefix(
                self_integrals,
                self_rates,
                [source_basis[source]["rates"] for source in sources],
                [source_basis[source]["integrals"] for source in sources],
                tolerance,
            )
        else:
            self_fit_rates, self_fit_integrals = self_rates, self_integrals
        own_coefficients, own_active, own_scaled = _fit_integral_basis(
            self_fit_integrals,
            response - interaction_sum,
            tolerance,
            ridge_alpha,
            n_samples,
        )
        self_rate = self_fit_rates @ own_coefficients
        self_cumulative = self_fit_integrals @ own_coefficients
        self_trajectory = signal[0] + self_cumulative

        cross_design = (
            np.column_stack([
                source_basis[source]["integrals"] for source in sources])
            if sources else np.empty((n_samples, 0)))
        cross_scales = np.sqrt(np.mean(cross_design ** 2, axis=0))
        cross_active = cross_scales > tolerance
        cross_scaled = cross_design[:, cross_active] / cross_scales[cross_active]
        scaled = np.column_stack([own_scaled, cross_scaled])
        orthogonality_error = (
            np.max(np.abs(own_scaled.T @ cross_scaled)) / n_samples
            if cross_scaled.shape[1] else 0.0
        )

        self_frames.append(pd.DataFrame({
            "time": time, "target": target, "rate": self_rate,
            "cumulative_effect": self_cumulative,
            "self_trajectory": self_trajectory,
        }))
        for index, (name, order) in enumerate(self_labels):
            coefficient_rows.append({
                "target": target, "source": target, "effect_type": "self",
                "basis": name, "order": order,
                "coefficient": own_coefficients[index],
                "basis_center": 0.0, "source_min": np.nan, "source_max": np.nan,
                "active": bool(own_active[index]),
            })

        contribution_norm = float(np.linalg.norm(self_cumulative))
        n_active_parameters = int(own_active.sum())
        for source in sources:
            rates, integrals, centers, lower, upper = bases[source]
            # 系数是 Shapley 平均后累计效应在该来源基上的最小二乘投影, 用于描述形状
            beta = np.linalg.lstsq(integrals, edge_curves[source], rcond=None)[0]
            active_source = np.sqrt(np.mean(integrals ** 2, axis=0)) > tolerance
            n_active_parameters += int(active_source.sum())
            rate, cumulative = edge_rates[source], edge_curves[source]
            contribution_norm += float(np.linalg.norm(cumulative))
            interaction_frames.append(pd.DataFrame({
                "time": time, "target": target, "source": source,
                "rate": rate, "cumulative_effect": cumulative,
            }))
            network_rows.append({
                "source": source, "target": target,
                "weight": (float(np.dot(cumulative, response)) / response_energy
                           if response_energy > 0 else 0.0),
                "sign": "+" if _edge_slope_sign(
                    rate, data[source].to_numpy(dtype=float)) >= 0 else "-",
                "n_orderings": n_orderings,
                "direction_basis": "selector_target_source",
            })
            for degree in range(integrals.shape[1]):
                coefficient_rows.append({
                    "target": target, "source": source,
                    "effect_type": "interaction",
                    "basis": "legendre", "order": degree + 1,
                    "coefficient": beta[degree],
                    "basis_center": centers[degree],
                    "source_min": lower, "source_max": upper,
                    "active": bool(active_source[degree]),
                })

        reconstructed = self_trajectory + interaction_sum
        residual = signal - reconstructed
        reconstruction_frames.append(pd.DataFrame({
            "time": time, "target": target, "smoothed": signal,
            "self_trajectory": self_trajectory, "interaction_sum": interaction_sum,
            "reconstructed": reconstructed, "residual": residual,
            "fitted_rate": self_rate + interaction_rate,
        }))

        singular_values = np.linalg.svd(scaled, compute_uv=False)
        rank = np.linalg.matrix_rank(scaled)
        condition = (singular_values[0] / singular_values[-1]
                     if rank == scaled.shape[1] else np.inf)
        total_variation = float(np.sum((signal - signal.mean()) ** 2))
        fit_norm = float(np.linalg.norm(reconstructed - signal[0]))
        diagnostic_rows.append({
            "target": target, "n_sources": len(sources),
            "n_parameters": int(self_integrals.shape[1]) + legendre_order * len(sources),
            "n_active_parameters": n_active_parameters, "rank": rank,
            "rank_deficient": bool(rank < scaled.shape[1]),
            "condition_number": condition,
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r_squared": (1.0 - np.sum(residual ** 2) / total_variation
                          if total_variation > 0 else np.nan),
            "self_energy_share": (float(np.dot(self_cumulative, response))
                                 / response_energy if response_energy > 0 else np.nan),
            "edge_energy_share": (float(np.dot(interaction_sum, response))
                                 / response_energy if response_energy > 0 else np.nan),
            "cancellation_ratio": (contribution_norm / fit_norm
                                   if fit_norm > 0 else np.nan),
            "self_cross_orthogonality_error": orthogonality_error,
            "n_orderings": n_orderings,
        })

    settings = dict(basis["settings"])
    settings.update({
        "method": "interaction_first_sequential_orthogonal_shapley",
        "allocation": "interaction_first",
        "self_argument": "time",
        "interaction_argument": "source_signal",
        "ridge_alpha": ridge_alpha,
        "interaction_centering": "time_mean",
        "initial_value": "first_smoothed_sample",
        "orthogonalization_space": "integrated_fourier_basis",
    })

    effects = {
        "coefficients": pd.DataFrame(coefficient_rows),
        "self_effects": pd.concat(self_frames, ignore_index=True),
        "interaction_effects": (
            pd.concat(interaction_frames, ignore_index=True) if interaction_frames
            else pd.DataFrame(
                columns=["time", "target", "source", "rate", "cumulative_effect"])),
        "reconstruction": pd.concat(reconstruction_frames, ignore_index=True),
        "diagnostics": pd.DataFrame(diagnostic_rows),
        "network": pd.DataFrame(network_rows, columns=[
            "source", "target", "weight", "sign", "n_orderings", "direction_basis"]),
        "orthogonalization": pd.DataFrame(projection_rows, columns=[
            "source", "legendre_order", "fourier_basis", "fourier_order",
            "projection_coefficient", "active"]),
        "settings": pd.DataFrame([settings]),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, table in effects.items():
            table.to_csv(output_path / f"{name}.csv", index=False, encoding="utf-8")
    return effects


def get_edge_list(
    effects: dict[str, pd.DataFrame],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Export the weighted directed edge list consumed by the downstream network stage.

    Note
    ----
    把 get_effects 的 network 表整理成 source / target / weight / sign 四列。
    weight 是该边累计效应占目标导联响应能量的比例，已对所有来源排列取平均，
    因此与顺序无关；同一目标各边权重之和等于该导联交互项的解释份额。
    sign 由该边变化率贡献对来源电平的回归斜率决定，正斜率为 positive、
    负斜率为 negative。方向沿用选边器的 target/source 角色，属于建模约定，
    不是数据推断（同期回归无法辨识方向）。
    两者都是条件量：只在相同的支撑集、带宽与分摊约定下有意义，不可跨设置比较。

    Parameters
    ----------
    effects : dict[str, pd.DataFrame]
        get_effects 的返回值，需要其中的 network 表。
    output_path : Path | None
        输出目录，默认项目下的 output/network。

    Returns
    -------
    edge_list : pd.DataFrame
        columns = ["source", "target", "weight", "sign"]，按 weight 降序；
        同时保存 edge_list.csv（无索引）。
    """
    if "network" not in effects:
        raise ValueError("effects must contain the network table.")
    network = effects["network"]
    required = {"source", "target", "weight", "sign"}
    if not required.issubset(network.columns):
        raise ValueError(f"network must contain columns: {sorted(required)}")
    if network.empty:
        raise ValueError("network is empty; nothing to export.")
    weights = network["weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("network weights must be finite.")
    signs = network["sign"].to_numpy().astype(str)
    if not set(signs).issubset({"+", "-"}):
        raise ValueError('network sign must be "+" or "-".')

    edge_list = pd.DataFrame({
        "source": network["source"].to_numpy(),
        "target": network["target"].to_numpy(),
        "weight": weights,
        "sign": np.where(signs == "-", "negative", "positive"),
    }).sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        edge_list.to_csv(output_path / "edge_list.csv", index=False, encoding="utf-8")
    return edge_list
