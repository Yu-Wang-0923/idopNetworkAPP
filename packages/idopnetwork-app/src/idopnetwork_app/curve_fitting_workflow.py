"""Upload-to-results workflow for the Curve Fitting page.

支持两种数据类型：

``quasi_dynamic``（默认）
    静态数据 → 列变换 → 按行和排序成拟动态序列 → 幂律拟合 ``y = a·x^b``。

``dynamic``
    时间序列（如 12 导联 ECG）→ 逐通道 SWT 平稳小波软阈值去噪。

两条路线导出**同一套三表契约**（FunClu / NetRecon 共用），dynamic 模式下含义重定义为：

    quasi_dynamic/   观测数据（原始波形，按时间索引）
    curve_sample/    SWT 去噪后的波形（新求解器的设计与响应输入）
    curve_params/    每个通道的 SWT 参数（wavelet / level / sigma / threshold）

因此 FunClu 与 NetRecon 的 ZIP 读取逻辑无需改动。
"""

import io
import zipfile

import numpy as np
import pandas as pd

from idopnetwork.curve_fitting import power_fitting, preprocess
from idopnetwork.curve_fitting.dynamic import (
    DEFAULT_ALPHA as SWT_DEFAULT_ALPHA,
    DEFAULT_LEVEL as SWT_DEFAULT_LEVEL,
    DEFAULT_WAVELET as SWT_DEFAULT_WAVELET,
    swt_denoise,
)

DATA_MODE_QUASI = "quasi_dynamic"
DATA_MODE_DYNAMIC = "dynamic"

DATA_MODE_LABELS = {
    DATA_MODE_QUASI: "Quasi-dynamic（静态数据 → 幂律拟合）",
    DATA_MODE_DYNAMIC: "Dynamic（时间序列 → SWT 去噪）",
}


def fit_uploaded_csv(
    content: bytes, *, data_mode: str = DATA_MODE_QUASI,
    first_transform: str = "Z_min_add1",
    second_transform: str = "Log10_1p", n_samples: int = 30,
    trim_percent: float = 1.0,
    swt_wavelet: str = SWT_DEFAULT_WAVELET,
    swt_level: int = SWT_DEFAULT_LEVEL,
    swt_alpha: float = SWT_DEFAULT_ALPHA,
) -> dict[str, pd.DataFrame]:
    """按所选数据类型拟合一个上传的 CSV。

    Returns
    -------
    dict
        键固定为 ``quasi_dynamic`` / ``curve_params`` / ``curve_sample``，
        但 dynamic 模式下含义见模块 docstring。
    """
    original = pd.read_csv(io.BytesIO(content), index_col=0)
    if original.empty or original.columns.has_duplicates:
        raise ValueError("CSV 必须包含数据，且特征名称不能重复。")

    if str(data_mode) == DATA_MODE_DYNAMIC:
        numeric = original.apply(pd.to_numeric, errors="coerce")
        if numeric.to_numpy(dtype=float).size == 0 or not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).any():
            raise ValueError("动态模式要求各列都是数值型时间序列。")
        fitted, params = swt_denoise(
            numeric,
            wavelet=str(swt_wavelet),
            level=int(swt_level),
            alpha=float(swt_alpha),
        )
        return {
            "quasi_dynamic": numeric,
            "curve_sample": fitted,
            "curve_params": params,
        }

    if not 0 <= trim_percent < 100:
        raise ValueError("去除比例必须在 0（含）到 100（不含）之间。")
    transformed = preprocess(preprocess(original, first_transform), second_transform)
    sums = transformed.sum(axis=1).to_numpy()
    order = np.argsort(sums, kind="stable")
    quasi = transformed.iloc[order].copy()
    quasi.index = pd.Index(np.log1p(sums[order]), name="quasi time")
    quasi = quasi.loc[quasi.index > 0]
    quasi = quasi.iloc[int(trim_percent / 100 * len(quasi)):]
    params, samples = power_fitting(quasi, n_samples=n_samples)
    return {"quasi_dynamic": quasi, "curve_params": params, "curve_sample": samples}


def build_fitting_export(results: dict[str, dict[str, pd.DataFrame]]) -> bytes:
    """Keep the three-table ZIP contract consumed by FunClu and NetRecon."""
    buffer = io.BytesIO()
    used = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, tables in results.items():
            folder = name.replace("\\", "_").replace("/", "_").strip() or "export"
            if folder in used:
                raise ValueError("文件名规范化后重复，请重命名文件后重试。")
            used.add(folder)
            for table in ("quasi_dynamic", "curve_params", "curve_sample"):
                archive.writestr(f"{folder}/{table}.csv", tables[table].to_csv(index=True))
    return buffer.getvalue()
