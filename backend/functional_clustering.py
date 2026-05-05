"""Functional Clustering 后端模块。

当前进度：
- ``_prepare_data``：多 condition DataFrame → ``(n_features, n_times_i)`` 张量列表（已完成）。
- ``_initialize``：基于 KMeans / MiniBatchKMeans 给出 EM 的 4 件套初值
  ``(labels, weights, mu_params, cov_params)``，外加 ``centers_kl``（KMeans 子中心，
  调试/可视化用）与 ``backend``（实际使用的 KMeans 后端名）。

EM 主循环（``_e_step / _m_step / fit``）会在后续步骤补齐。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from backend.curve_fitting import fit_power_loglinear


class FunClu:
    """多 condition 函数聚类（构建中）。

    Args:
        n_components: 簇数 K，默认 3。
        max_iter: EM 主循环最大迭代次数，默认 50（本步未启用）。
        tol: EM 收敛阈值（log-likelihood 增量），默认 1e-4（本步未启用）。
        device: 张量驻留设备，默认 ``torch.device('cpu')``。
        dtype: 张量精度，默认 ``torch.float64``。
        kmeans_minibatch_threshold: 当 ``n_features`` ≥ 该阈值且
            ``use_minibatch_kmeans is None`` 时自动启用 ``MiniBatchKMeans``，默认 8000。
        minibatch_batch_size: ``MiniBatchKMeans`` 的 batch 大小，默认 4096。
        minibatch_max_iter: ``MiniBatchKMeans`` 的最大迭代次数，默认 100。
        use_minibatch_kmeans: ``None`` 自动；``True`` 强制 MiniBatch；``False`` 强制全量。
        random_state: KMeans 随机种子，默认 42。
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 50,
        tol: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        *,
        kmeans_minibatch_threshold: int = 8_000,
        minibatch_batch_size: int = 4_096,
        minibatch_max_iter: int = 100,
        use_minibatch_kmeans: Optional[bool] = None,
        random_state: int = 42,
    ) -> None:
        self.n_components: int = int(n_components)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.dtype: torch.dtype = dtype
        self.kmeans_minibatch_threshold: int = int(kmeans_minibatch_threshold)
        self.minibatch_batch_size: int = int(minibatch_batch_size)
        self.minibatch_max_iter: int = int(minibatch_max_iter)
        self.use_minibatch_kmeans: Optional[bool] = use_minibatch_kmeans
        self.random_state: int = int(random_state)

        # ── 数据维度（_prepare_data 写入） ────────────────────────────────────
        self.common_cols: Optional[List[str]] = None
        self.n_features: int = 0
        self.n_conditions: int = 0
        self.times_list: Optional[List[np.ndarray]] = None
        self.n_times_conditions: Optional[List[int]] = None

        # ── 模型参数（_initialize / EM 写入） ─────────────────────────────────
        self.params_mu: Optional[torch.Tensor] = None    # (K, L, 2) [a, b]
        self.params_cov: Optional[torch.Tensor] = None   # (K, L, 2) [phi, gamma]
        self.weights: Optional[torch.Tensor] = None      # (K,)

        # ── 结果（EM 写入） ───────────────────────────────────────────────────
        self.labels: Optional[torch.Tensor] = None
        self.log_likelihood: Optional[float] = None
        self.neg_log_likelihood: Optional[float] = None
        self.bic: Optional[float] = None
        self.n_params: Optional[int] = None
        self.converged: bool = False
        self._kmeans_init_backend: str = ""

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: data preparation
    # ────────────────────────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────────────
    # Step 2: KMeans-based initialization
    # ────────────────────────────────────────────────────────────────────────
    def _initialize(
        self,
        X_list: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """基于 KMeans / MiniBatchKMeans 给出 EM 4 件套初值。

        流程：

        1. 沿时间轴拼接所有 condition：得到 ``(N, sum(n_t_i))`` 的输入矩阵。
        2. 选择 KMeans 后端（按 ``use_minibatch_kmeans`` 与
           ``kmeans_minibatch_threshold`` 自动切换）并跑一次。
        3. 把簇中心按 ``n_t_i`` 切回每个 ``(k, i)`` 的"子中心"。
        4. 对每个 ``(k, i)`` 用双对数线性回归拟合 ``y = a · t^b`` →
           ``params_mu[k, i] = (a, b)``。
        5. 估计 SAD1 的 ``(phi, gamma)``：对该簇所有特征求残差均值序列
           ``r̄ = mean(X_ki - μ_ki)``；``phi`` 取一阶自相关并夹紧到
           ``[-0.99, 0.99]``；``gamma = sqrt(var(r̄) · (1 - phi²) + 1e-6)``。
           **空簇**或 ``n_t_i == 1`` 等退化情况按文档兜底（``phi=0, gamma=1``）。

        本方法**会写入** ``self.params_mu / params_cov / weights / labels /
        _kmeans_init_backend``，便于 EM 步直接使用；同时把上述结果与
        ``centers_kl / backend`` 一起作为字典返回。

        Args:
            X_list: ``_prepare_data`` 已经准备好的张量列表，长度 = ``n_conditions``，
                第 i 项形如 ``(n_features, n_times_i)``。

        Returns:
            字典，含以下键：

            - ``labels``：``torch.LongTensor``，形如 ``(n_features,)``。
            - ``weights``：``torch.Tensor``，形如 ``(K,)``，权重之和归一化为 1。
            - ``mu_params``：``torch.Tensor``，形如 ``(K, L, 2)``，最后一维为 ``[a, b]``。
            - ``cov_params``：``torch.Tensor``，形如 ``(K, L, 2)``，最后一维为
              ``[phi, gamma]``。
            - ``centers_kl``：``List[List[np.ndarray]]``，``centers_kl[k][i]`` 为
              该 ``(k, i)`` 的 KMeans 子中心，形如 ``(n_t_i,)``。
            - ``backend``：实际使用的 KMeans 后端名（``"KMeans"`` 或 ``"MiniBatchKMeans"``）。

        Raises:
            RuntimeError: 若尚未调用过 ``_prepare_data``（``n_conditions == 0``）。
        """
        if self.n_conditions == 0 or self.times_list is None or self.n_times_conditions is None:
            raise RuntimeError(
                "_initialize 调用前必须先调用 _prepare_data 设置 n_conditions / times_list"
            )
        if len(X_list) != self.n_conditions:
            raise ValueError(
                f"X_list 长度 {len(X_list)} 与 n_conditions {self.n_conditions} 不一致"
            )

        K = self.n_components
        L = self.n_conditions
        n_t_per: List[int] = list(self.n_times_conditions)

        # 1) 拼接：(N, sum(n_t_i))
        X_concat = torch.cat(X_list, dim=1)
        N = int(X_concat.shape[0])
        X_np = X_concat.detach().cpu().numpy()
        if K > N:
            raise ValueError(
                f"n_components={K} 超过特征数 N={N}；请减小 n_components"
            )

        # 2) KMeans 后端选择
        use_mb = self.use_minibatch_kmeans
        if use_mb is None:
            use_mb = N >= self.kmeans_minibatch_threshold

        if use_mb:
            bs = max(256, min(self.minibatch_batch_size, N))
            km = MiniBatchKMeans(
                n_clusters=K,
                init="k-means++",
                batch_size=bs,
                n_init=3,
                max_iter=self.minibatch_max_iter,
                random_state=self.random_state,
            ).fit(X_np)
            backend = "MiniBatchKMeans"
        else:
            km = KMeans(
                n_clusters=K,
                init="k-means++",
                n_init=10,
                random_state=self.random_state,
            ).fit(X_np)
            backend = "KMeans"

        labels_np: np.ndarray = km.labels_.astype(np.int64, copy=False)
        centers_concat: np.ndarray = km.cluster_centers_.astype(np.float64, copy=False)

        # 簇大小 / 权重
        sizes = np.bincount(labels_np, minlength=K)
        weights_np = sizes.astype(np.float64) / max(N, 1)
        if weights_np.sum() > 0:
            weights_np = weights_np / weights_np.sum()

        # 3) 切回 (k, i) 子中心
        centers_kl: List[List[np.ndarray]] = [
            [np.empty((0,), dtype=np.float64) for _ in range(L)] for _ in range(K)
        ]
        offset = 0
        for i in range(L):
            n_t = n_t_per[i]
            for k in range(K):
                centers_kl[k][i] = centers_concat[k, offset : offset + n_t].copy()
            offset += n_t

        # 4) 拟合 (a, b)
        params_mu_np = np.zeros((K, L, 2), dtype=np.float64)
        for k in range(K):
            for i in range(L):
                t = np.asarray(self.times_list[i], dtype=np.float64)
                y_center = centers_kl[k][i]
                a, b = fit_power_loglinear(
                    t,
                    y_center,
                    clip_a=(1e-8, np.inf),
                    clip_b=(0.01, 10.0),
                )
                params_mu_np[k, i, 0] = a
                params_mu_np[k, i, 1] = b

        # 5) 估计 (phi, gamma)
        params_cov_np = np.zeros((K, L, 2), dtype=np.float64)
        for k in range(K):
            mask_k = labels_np == k
            n_k = int(mask_k.sum())
            for i in range(L):
                n_t = n_t_per[i]
                if n_k == 0:
                    params_cov_np[k, i] = (0.0, 1.0)
                    continue

                X_i_np = X_list[i].detach().cpu().numpy()
                X_ki = X_i_np[mask_k, :]  # (n_k, n_t)

                a = float(params_mu_np[k, i, 0])
                b = float(params_mu_np[k, i, 1])
                t = np.asarray(self.times_list[i], dtype=np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    mu_curve = a * np.power(t, b)
                R = X_ki - mu_curve[None, :]                # (n_k, n_t)
                R = np.where(np.isfinite(R), R, 0.0)
                r_bar = R.mean(axis=0)                       # (n_t,)

                if n_t > 1 and np.std(r_bar) > 0:
                    with np.errstate(invalid="ignore"):
                        cc = np.corrcoef(r_bar[:-1], r_bar[1:])[0, 1]
                    if not np.isfinite(cc):
                        cc = 0.0
                    phi = float(np.clip(cc, -0.99, 0.99))
                    gamma_sq = max(float(np.var(r_bar)) * (1.0 - phi * phi), 1e-6)
                    gamma = float(np.sqrt(gamma_sq))
                elif n_t > 0:
                    phi = 0.0
                    gamma = float(np.sqrt(max(float(np.var(r_bar)), 1e-6)))
                else:
                    phi, gamma = 0.0, 1.0

                params_cov_np[k, i, 0] = phi
                params_cov_np[k, i, 1] = gamma

        # 6) numpy → torch & 写入 self
        labels_t = torch.from_numpy(labels_np).to(self.device).long()
        weights_t = torch.from_numpy(weights_np.astype(np.float64)).to(self.device, self.dtype)
        params_mu_t = torch.from_numpy(params_mu_np).to(self.device, self.dtype)
        params_cov_t = torch.from_numpy(params_cov_np).to(self.device, self.dtype)

        self.labels = labels_t
        self.weights = weights_t
        self.params_mu = params_mu_t
        self.params_cov = params_cov_t
        self._kmeans_init_backend = backend

        return {
            "labels": labels_t,
            "weights": weights_t,
            "mu_params": params_mu_t,
            "cov_params": params_cov_t,
            "centers_kl": centers_kl,
            "backend": backend,
        }

    def __repr__(self) -> str:
        return (
            f"FunClu(n_components={self.n_components}, "
            f"n_conditions={self.n_conditions}, "
            f"n_features={self.n_features}, "
            f"backend={self._kmeans_init_backend!r})"
        )
