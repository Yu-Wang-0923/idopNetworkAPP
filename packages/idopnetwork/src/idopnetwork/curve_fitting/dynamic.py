"""动态数据（时间序列）拟合：SWT 平稳小波软阈值去噪。

移植自新版 ``idopECG.py`` 的 ``data_fit``：逐通道做 Stationary Wavelet
Transform (SWT)，用最细层细节系数的 MAD 估计噪声 ``sigma = MAD / 0.6745``，
阈值取 ``alpha · sigma · sqrt(2 log N)``（soft 阈值），再逆变换还原。

与静态数据的幂律拟合（``fitting.power_fitting``）并列，供「曲线拟合」页的
「数据类型」选项在准动态 / 动态两条路线之间切换。

依赖
----
需要 **PyWavelets**（导入时才加载，未安装时给出明确报错）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WAVELET = "rbio3.9"
DEFAULT_LEVEL = 2
DEFAULT_ALPHA = 0.5


def swt_denoise(
    data: pd.DataFrame,
    *,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对每个通道做 SWT 软阈值去噪。

    Parameters
    ----------
    data:
        ``(n_timepoints, n_channels)`` 的时间序列（index 为时间 / 采样点）。
    wavelet:
        小波基，默认 ``"rbio3.9"``（ECG 常用）。
    level:
        SWT 分解层数上限；实际取 ``min(level, pywt.swt_max_level(N))``。
    alpha:
        阈值系数（``alpha · sigma · sqrt(2 log N)``）。

    Returns
    -------
    fitted, params
        ``fitted`` 与输入同形状、同索引的去噪结果；
        ``params`` 为每个通道的 ``wavelet / level / sigma / threshold`` 诊断表。
    """
    try:
        import pywt
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "动态数据拟合（SWT 去噪）需要 PyWavelets，请安装：pip install PyWavelets"
        ) from exc

    if data.empty:
        raise ValueError("输入数据为空，无法做 SWT 去噪。")

    n_timepoints = data.shape[0]
    effective_level = min(int(level), pywt.swt_max_level(n_timepoints))
    fitted = data.copy()
    records: list[dict[str, object]] = []

    for column in data.columns:
        signal = data[column].to_numpy(dtype=float)

        if effective_level < 1:
            # 序列过短（如单点），SWT 无法分解：原样返回
            fitted[column] = signal
            records.append({
                "feature": column, "wavelet": wavelet, "level": 0,
                "sigma": 0.0, "threshold": 0.0, "denoised": False,
            })
            continue

        coefficients = pywt.swt(signal, wavelet=wavelet, level=effective_level)

        # 用最细层细节系数做稳健噪声估计
        detail = coefficients[-1][1]
        sigma = float(np.median(np.abs(detail - np.median(detail))) / 0.6745)
        threshold = float(alpha * sigma * np.sqrt(2.0 * np.log(n_timepoints)))

        if sigma > 0.0:
            coefficients_fit = [
                (approx, pywt.threshold(detail_layer, threshold, mode="soft"))
                for approx, detail_layer in coefficients
            ]
            denoised = True
        else:
            # 常量通道（如全 0）：阈值化会出现 0/0 = NaN，直接跳过
            coefficients_fit = list(coefficients)
            denoised = False

        restored = pywt.iswt(coefficients_fit, wavelet=wavelet)
        fitted[column] = np.nan_to_num(
            restored, nan=0.0, posinf=0.0, neginf=0.0
        )

        records.append({
            "feature": column, "wavelet": wavelet, "level": int(effective_level),
            "sigma": sigma, "threshold": threshold, "denoised": denoised,
        })

    params = pd.DataFrame(records).set_index("feature")
    return fitted, params
