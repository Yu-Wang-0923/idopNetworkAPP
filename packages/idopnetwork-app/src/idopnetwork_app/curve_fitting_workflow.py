"""Upload-to-results workflow matching the supplied FunClu v4 example."""
import io
import zipfile

import numpy as np
import pandas as pd

from idopnetwork.curve_fitting import power_fitting, preprocess


def fit_uploaded_csv(content: bytes) -> dict[str, pd.DataFrame]:
    original = pd.read_csv(io.BytesIO(content), index_col=0)
    if original.empty or original.columns.has_duplicates:
        raise ValueError("CSV 必须包含数据，且特征名称不能重复。")
    transformed = preprocess(preprocess(original, "Z_min_add1"), "Log10_1p")
    sums = transformed.sum(axis=1).to_numpy()
    order = np.argsort(sums, kind="stable")
    quasi = transformed.iloc[order].copy()
    quasi.index = pd.Index(np.log1p(sums[order]), name="quasi time")
    quasi = quasi.loc[quasi.index > 0]
    quasi = quasi.iloc[int(0.01 * len(quasi)):]
    params, samples = power_fitting(quasi, n_samples=30)
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
