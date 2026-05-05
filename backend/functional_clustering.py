"""Functional Clustering 后端模块。

本文件目前仅实现 :class:`FunClu` 的**数据准备阶段**：
- 从多 condition 的 ``list[pd.DataFrame]`` 中抽取共同特征列；
- 将每个 condition 转换为 ``(n_features, n_times_i)`` 的 ``torch.Tensor``；
- 保留每个 condition 的时间索引（``np.ndarray``）。

EM 拟合、KMeans 初始化、E/M 步等后续逻辑会在后续步骤逐步补齐。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class FunClu:
    """多 condition 函数聚类（构建中）。

    当前阶段：仅完成数据准备（``_prepare_data``）。

    Args:
        device: 张量驻留设备；默认 ``torch.device('cpu')``。
        dtype: 张量精度；默认 ``torch.float64``。
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.dtype: torch.dtype = dtype

        self.common_cols: Optional[List[str]] = None
        self.n_features: int = 0
        self.n_conditions: int = 0
        self.times_list: Optional[List[np.ndarray]] = None
        self.n_times_conditions: Optional[List[int]] = None

    def _prepare_data(
        self,
        data: List[pd.DataFrame],
    ) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """从多 condition 的 DataFrame 列表抽取张量与时间向量。

        各 condition 的时间长度可以不同，不做对齐；仅按列名取**交集**，
        确保每个 condition 在相同的特征列上参与聚类。

        Args:
            data: 长度为 ``n_conditions`` 的列表，第 i 个元素为
                ``(n_times_i, n_features_i)`` 的 ``pd.DataFrame``，
                行索引解释为时间/伪时间，列为特征。

        Returns:
            ``(X_list, times_list)``：

            - ``X_list``：长度为 ``n_conditions`` 的列表；第 i 项为形如
              ``(n_features, n_times_i)`` 的 ``torch.Tensor``，
              其中 ``n_features`` 等于所有 condition 的列名交集大小；
            - ``times_list``：长度为 ``n_conditions`` 的列表；第 i 项为
              ``(n_times_i,)`` 的 ``np.ndarray``（``float64``）。

        Raises:
            ValueError: 若 ``data`` 为空，或所有 condition 的列交集为空。
        """
        if not data:
            raise ValueError("data 不能为空：至少需要 1 个 condition 的 DataFrame")

        common_cols: List[str] = list(data[0].columns)
        for d in data[1:]:
            cols_d = set(d.columns)
            common_cols = [c for c in common_cols if c in cols_d]
        if len(common_cols) == 0:
            raise ValueError("所有 condition 的 DataFrame 不存在共同列；无法聚类")

        self.common_cols = common_cols
        self.n_features = len(common_cols)
        self.n_conditions = len(data)

        X_list: List[torch.Tensor] = []
        times_list: List[np.ndarray] = []
        for df in data:
            sub = df[common_cols]
            vals = sub.to_numpy(dtype=np.float64, copy=True).T
            idx = np.asarray(sub.index, dtype=np.float64)
            X_list.append(torch.from_numpy(vals).to(self.device, self.dtype))
            times_list.append(idx)

        self.times_list = times_list
        self.n_times_conditions = [len(t) for t in times_list]

        return X_list, times_list

    def __repr__(self) -> str:
        return (
            f"FunClu(n_conditions={self.n_conditions}, "
            f"n_features={self.n_features}, "
            f"n_times_conditions={self.n_times_conditions})"
        )
