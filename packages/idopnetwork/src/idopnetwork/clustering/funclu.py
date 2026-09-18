"""Application-compatible interface backed by the FunClu v4 solver."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

from idopnetwork.curve_fitting import power_fitting
from ._v4 import FunClu as FunCluV4, select_best_k


class FunClu:
    """Retain application tensor/export interfaces while using v4 for all fitting.

    ``params_mu`` remains (a, b) for existing consumers; the native standardized
    (c, beta) parameters are available on ``kernel_.params_mu_``.
    """
    def __init__(self, n_components=3, max_iter=1000, tol=1e-4,
                 device=None, dtype=torch.float64, *, random_state=42,
                 use_minibatch_kmeans=None, kmeans_minibatch_threshold=8000,
                 minibatch_batch_size=4096, minibatch_max_iter=100, **options):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self.random_state = int(random_state)
        self.use_minibatch_kmeans = use_minibatch_kmeans
        self.options = options
        self.common_cols = None
        self.n_features = self.n_conditions = 0
        self.times_list = self.n_times_conditions = None
        self.params_mu = self.params_cov = self.weights = self.labels = None

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


    def fit(self, data, *, parameter_data=None, verbose=False):
        self._prepare_data(data)
        if parameter_data is not None and len(parameter_data) != len(data):
            raise ValueError("参数表数量必须与 condition 数量一致")
        samples, parameters = {}, {}
        for i, df in enumerate(data):
            t = np.asarray(df.index, dtype=float)
            if len(t) < 2 or not np.isfinite(t).all() or (t <= 0).any() or np.unique(t).size < 2:
                raise ValueError("聚类要求至少两个不同的有限正数时间点，请重新运行拟合器")
            if df.columns.has_duplicates:
                raise ValueError("特征名称不能重复")
            samples[str(i)] = df
            parameters[str(i)] = (power_fitting(df)[0] if parameter_data is None
                                  else parameter_data[i])
        options = dict(self.options)
        options.setdefault("compute_device", str(self.device))
        options.setdefault("use_mini_batch", True if self.use_minibatch_kmeans is None
                           else self.use_minibatch_kmeans)
        options.setdefault("progress", verbose)
        kernel = FunCluV4(n_components=self.n_components, max_iter=self.max_iter,
                          tol=self.tol, random_state=self.random_state, **options)
        kernel._prepare_inputs(samples, parameters)
        if not 1 <= self.n_components <= kernel.n_features_:
            raise ValueError(f"K 必须介于 1 和有效共同特征数 {kernel.n_features_} 之间")
        kernel.fit(samples, parameters)
        self.kernel_ = kernel
        self.common_cols = list(kernel.feature_names_)
        self.n_features = kernel.n_features_
        self.excluded_features_ = kernel.excluded_features_
        mu = kernel.params_mu_.copy()
        for i, t in enumerate(self.times_list):
            log_t = np.log(t)
            b = mu[:, i, 1] / max(log_t.std(), 1e-8)
            mu[:, i, 0] *= np.exp(-b * log_t.mean())
            mu[:, i, 1] = b
        self.params_mu = torch.as_tensor(mu, dtype=self.dtype, device=self.device)
        self.params_cov = torch.as_tensor(kernel.params_cov_, dtype=self.dtype, device=self.device)
        self.weights = torch.as_tensor(kernel.weights_, dtype=self.dtype, device=self.device)
        self.labels = torch.as_tensor(kernel.labels_, dtype=torch.long, device=self.device)
        self.log_likelihood = kernel.log_likelihood_
        self.neg_log_likelihood = -self.log_likelihood
        self.n_params = self.n_components - 1 + 2 * self.n_components * self.n_conditions + 2 * self.n_conditions
        self.bic = float(kernel.bic())
        self.converged = kernel.converged_
        self.n_iter_run = kernel.n_iter_
        self.loglik_history = kernel.log_likelihood_history_.tolist()
        return self

    def predict(self, data=None):
        if not hasattr(self, "kernel_"):
            raise RuntimeError("请先调用 fit")
        if data is None:
            return self.labels.clone()
        # Use the training time basis and covariance without changing fitted state.
        if len(data) != self.n_conditions:
            raise ValueError("预测 condition 数量必须与训练一致")
        import copy
        kernel = copy.copy(self.kernel_)
        features = list(data[0].columns)
        for i, df in enumerate(data):
            if not np.array_equal(np.asarray(df.index, dtype=float), self.times_list[i]):
                raise ValueError("预测时间点必须与训练一致")
            features = [c for c in features if c in df.columns]
        if not features:
            raise ValueError("预测数据不存在共同特征")
        kernel.sample_sets_ = {str(i): df.loc[:, features] for i, df in enumerate(data)}
        if any(not np.isfinite(df.to_numpy(float)).all() for df in kernel.sample_sets_.values()):
            raise ValueError("预测数据必须为有限数值")
        kernel.n_features_ = len(features)
        kernel._prepare_solver_data()
        convert = lambda a: torch.as_tensor(a, dtype=kernel.dtype_, device=kernel.device_)
        resp, _ = kernel._e_step(convert(kernel.weights_), convert(kernel.params_mu_),
                                 convert(kernel.shared_params_cov_))
        return resp.argmax(dim=1).to(self.device)

    def get_cluster_curves(self, condition_idx=0):
        if not hasattr(self, "kernel_"):
            raise RuntimeError("请先调用 fit")
        frame = self.kernel_.mean_curves_[self.kernel_.condition_names_[condition_idx]]
        return frame.index.to_numpy(float), frame.to_numpy(float).T

    def get_params(self) -> Dict[str, Optional[np.ndarray]]:
        """以 numpy 字典导出关键模型参数，便于持久化或表格展示。

        Returns:
            字典：``mu_params (K,L,2)``、``cov_params (K,L,2)``、``weights (K,)``、
            ``labels (N,)``；尚未填充的字段为 ``None``。
        """
        return {
            "mu_params": (
                self.params_mu.detach().cpu().numpy()
                if self.params_mu is not None
                else None
            ),
            "cov_params": (
                self.params_cov.detach().cpu().numpy()
                if self.params_cov is not None
                else None
            ),
            "weights": (
                self.weights.detach().cpu().numpy()
                if self.weights is not None
                else None
            ),
            "labels": (
                self.labels.detach().cpu().numpy()
                if self.labels is not None
                else None
            ),
        }



def _standardized_curve_matrix(X_list: List[torch.Tensor]) -> np.ndarray:
    """Return a standardized feature-by-time matrix for distance-based scores."""
    if not X_list:
        return np.empty((0, 0), dtype=np.float64)

    X_concat = torch.cat(X_list, dim=1).detach().cpu().numpy().astype(np.float64)
    mean = np.nanmean(X_concat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    X_concat = np.where(np.isfinite(X_concat), X_concat, mean)

    scale = np.nanstd(X_concat, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (X_concat - mean) / scale


def _compute_silhouette_index(
    X_list: List[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> float:
    """Compute Silhouette Index (SI) from final hard cluster labels."""
    if labels is None:
        return float("nan")

    y = labels.detach().cpu().numpy().astype(int)
    n_samples = int(y.size)
    n_labels = int(np.unique(y).size)
    if n_samples < 3 or n_labels < 2 or n_labels >= n_samples:
        return float("nan")

    try:
        return float(silhouette_score(_standardized_curve_matrix(X_list), y))
    except Exception:
        return float("nan")


def _cluster_size_stats(
    labels: Optional[torch.Tensor],
    n_components: int,
    min_cluster_size: int,
) -> Dict[str, Any]:
    """Summarize hard-label cluster sizes for K-selection diagnostics."""
    if labels is None:
        return {
            "min_cluster_size": float("nan"),
            "max_cluster_size": float("nan"),
            "small_cluster_count": 0,
            "empty_cluster_count": 0,
            "passes_min_cluster_size": False,
        }

    y = labels.detach().cpu().numpy().astype(int)
    counts = np.bincount(y, minlength=int(n_components))
    return {
        "min_cluster_size": int(counts.min()) if counts.size else 0,
        "max_cluster_size": int(counts.max()) if counts.size else 0,
        "small_cluster_count": int((counts < int(min_cluster_size)).sum()),
        "empty_cluster_count": int((counts == 0).sum()),
        "passes_min_cluster_size": bool(np.all(counts >= int(min_cluster_size))),
    }


def _finite_values(values: List[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _aggregate_values(values: List[Any], method: str) -> float:
    arr = _finite_values(values)
    if arr.size == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(arr))
    return float(np.median(arr))


def _mean_values(values: List[Any]) -> float:
    arr = _finite_values(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _median_values(values: List[Any]) -> float:
    arr = _finite_values(values)
    return float(np.median(arr)) if arr.size else float("nan")


def _median_int_values(values: List[Any]) -> int:
    arr = _finite_values(values)
    return int(np.median(arr)) if arr.size else 0


def _std_values(values: List[Any]) -> float:
    arr = _finite_values(values)
    return float(np.std(arr, ddof=0)) if arr.size else float("nan")


def compute_bic_scores(
    data: List[pd.DataFrame],
    k_min: int = 2,
    k_max: int = 10,
    step: int = 1,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    n_random_starts: int = 1,
    aggregation: str = "median",
    min_cluster_size: int = 1,
    parameter_data: Optional[List[pd.DataFrame]] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Scan a range of K values and return BIC scores.

    For each K in [k_min, k_max] with given step, instantiates a
    :class:`FunClu`, calls :meth:`~FunClu.fit`, and collects BIC, log-likelihood,
    convergence status, etc.  Single-K failures are recorded as NaN rows rather
    than aborting the scan.

    Args:
        data: One DataFrame per condition (same format as :meth:`FunClu.fit`).
        k_min: Smallest K to try (must be ≥ 2).
        k_max: Largest K to try (capped to the number of features in *data*).
        step: Stride between consecutive K values.
        max_iter: Passed to :class:`FunClu`.
        tol: Passed to :class:`FunClu`.
        random_state: Base seed. When ``n_random_starts > 1``, seeds are
            ``random_state, random_state + 1, ...``.
        n_random_starts: Number of seeds to run for each K.
        aggregation: ``"median"`` or ``"mean"`` for the primary ``BIC`` / ``SI``.
        min_cluster_size: Minimum hard-label members required per cluster.
        verbose: Passed to :meth:`FunClu.fit`.
        progress_callback: Called after each K as ``callback(idx_1based, total)``.

    Returns:
        DataFrame with columns ``K``, ``BIC``, ``SI``, ``log_likelihood``, ``NLL``,
        aggregate diagnostics, seed counts, and cluster-size constraint flags.
    """
    if k_min < 2:
        raise ValueError(f"k_min must be >= 2, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if n_random_starts < 1:
        raise ValueError(f"n_random_starts must be >= 1, got {n_random_starts}")
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")

    n_features = min((d.shape[1] for d in data), default=0)
    if n_features < 2:
        raise ValueError(f"Data has fewer than 2 features (got {n_features})")
    capped_k_max = min(k_max, n_features)

    ks = list(range(k_min, capped_k_max + 1, step))
    if not ks:
        raise ValueError(
            f"No K values to scan: k_min={k_min}, "
            f"k_max={k_max}, n_features={n_features}"
        )

    rows: List[Dict[str, Any]] = []
    total = len(ks)
    seeds = [int(random_state) + offset for offset in range(int(n_random_starts))]

    for idx, k in enumerate(ks):
        seed_rows: List[Dict[str, Any]] = []
        for seed in seeds:
            seed_row: Dict[str, Any] = {
                "seed": seed,
                "BIC": float("nan"),
                "SI": float("nan"),
                "log_likelihood": float("nan"),
                "NLL": float("nan"),
                "n_params": 0,
                "converged": False,
                "n_iter_run": 0,
                "n_features": 0,
                "n_conditions": 0,
                "min_cluster_size": float("nan"),
                "max_cluster_size": float("nan"),
                "small_cluster_count": 0,
                "empty_cluster_count": 0,
                "passes_min_cluster_size": False,
                "fit_success": False,
                "error": "",
            }
            try:
                model = FunClu(
                    n_components=k, max_iter=max_iter, tol=tol,
                    random_state=seed,
                )
                model.fit(data, parameter_data=parameter_data, verbose=verbose)
                X_list = [torch.as_tensor(d.loc[:, model.common_cols].to_numpy(float).T) for d in data]
                size_stats = _cluster_size_stats(
                    model.labels,
                    n_components=k,
                    min_cluster_size=min_cluster_size,
                )
                seed_row.update({
                    "BIC": model.bic,
                    "SI": _compute_silhouette_index(X_list, model.labels),
                    "log_likelihood": model.log_likelihood,
                    "NLL": model.neg_log_likelihood,
                    "n_params": model.n_params,
                    "converged": model.converged,
                    "n_iter_run": model.n_iter_run,
                    "n_features": model.n_features,
                    "n_conditions": model.n_conditions,
                    "fit_success": bool(np.isfinite(model.bic)),
                    **size_stats,
                })
            except Exception as error:
                seed_row["error"] = str(error)
            seed_rows.append(seed_row)

        fit_successes = int(sum(bool(r["fit_success"]) for r in seed_rows))
        row: Dict[str, Any] = {
            "K": k,
            "error": "; ".join(r["error"] for r in seed_rows if r["error"]),
            "BIC": _aggregate_values([r["BIC"] for r in seed_rows], aggregation),
            "BIC_mean": _mean_values([r["BIC"] for r in seed_rows]),
            "BIC_median": _median_values([r["BIC"] for r in seed_rows]),
            "BIC_sd": _std_values([r["BIC"] for r in seed_rows]),
            "SI": _aggregate_values([r["SI"] for r in seed_rows], aggregation),
            "SI_mean": _mean_values([r["SI"] for r in seed_rows]),
            "SI_median": _median_values([r["SI"] for r in seed_rows]),
            "log_likelihood": _aggregate_values(
                [r["log_likelihood"] for r in seed_rows], aggregation
            ),
            "NLL": _aggregate_values([r["NLL"] for r in seed_rows], aggregation),
            "n_params": _median_int_values([r["n_params"] for r in seed_rows]),
            "converged": bool(
                _mean_values([float(r["converged"]) for r in seed_rows]) >= 0.5
            ),
            "converged_rate": _mean_values(
                [float(r["converged"]) for r in seed_rows]
            ),
            "n_iter_run": _aggregate_values(
                [r["n_iter_run"] for r in seed_rows], aggregation
            ),
            "n_features": _median_int_values([r["n_features"] for r in seed_rows]),
            "n_conditions": _median_int_values([r["n_conditions"] for r in seed_rows]),
            "min_cluster_size": _aggregate_values(
                [r["min_cluster_size"] for r in seed_rows], aggregation
            ),
            "min_cluster_size_min": (
                float(np.min(_finite_values([r["min_cluster_size"] for r in seed_rows])))
                if _finite_values([r["min_cluster_size"] for r in seed_rows]).size
                else float("nan")
            ),
            "max_cluster_size": _aggregate_values(
                [r["max_cluster_size"] for r in seed_rows], aggregation
            ),
            "small_cluster_count": _aggregate_values(
                [r["small_cluster_count"] for r in seed_rows], aggregation
            ),
            "small_cluster_count_max": (
                float(np.max(_finite_values([r["small_cluster_count"] for r in seed_rows])))
                if _finite_values([r["small_cluster_count"] for r in seed_rows]).size
                else float("nan")
            ),
            "empty_cluster_count": _aggregate_values(
                [r["empty_cluster_count"] for r in seed_rows], aggregation
            ),
            "size_ok_rate": _mean_values(
                [float(r["passes_min_cluster_size"]) for r in seed_rows]
            ),
            "passes_min_cluster_size": bool(
                _mean_values([float(r["passes_min_cluster_size"]) for r in seed_rows])
                >= 0.5
            ),
            "fit_successes": fit_successes,
            "random_starts": int(n_random_starts),
            "aggregation": aggregation,
            "seed_start": int(random_state),
            "seed_end": int(random_state) + int(n_random_starts) - 1,
            "min_cluster_size_threshold": int(min_cluster_size),
        }
        rows.append(row)

        if progress_callback is not None:
            progress_callback(idx + 1, total)

    return pd.DataFrame(rows)
