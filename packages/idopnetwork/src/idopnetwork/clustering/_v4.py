"""Numerical FunClu v4 kernel, ported from the supplied funclu_v4(3).py."""
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from idopnetwork.curve_fitting import power_fitting

class FunClu:
    """Finite-mixture clustering of fitted quasi-dynamic curves."""

    def __init__(
        self,
        n_components: int,
        max_iter: int = 1000,
        inner_max_iter: int = 1,
        tol: float = 1e-4, # 外部迭代收敛阈值
        inner_tol: float = 1e-3, # 内部迭代收敛阈值
        beta_bounds: tuple[float, float] = (-6.0, 6.0),
        beta_xtol: float = 1e-4, # beta参数收敛阈值
        phi_bound: float = 0.90, # phi参数边界
        phi_ridge: float = 0.30, # phi参数正则化参数
        tightness_weight: float = 0.0, # 松松聚类
        n_init: int = 10,
        use_mini_batch: bool = True, # True False
        balance_conditions: bool = True,
        compute_device: str = "auto",
        random_state: int = 0,
        progress: bool = True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.inner_max_iter = inner_max_iter
        self.tol = tol
        self.inner_tol = inner_tol
        self.beta_bounds = beta_bounds
        self.beta_xtol = beta_xtol
        self.phi_bound = phi_bound
        self.phi_ridge = phi_ridge
        self.tightness_weight = tightness_weight
        self.n_init = n_init
        self.use_mini_batch = use_mini_batch
        self.balance_conditions = balance_conditions
        self.compute_device = compute_device
        self.random_state = random_state
        self.progress = progress

    def _prepare_inputs(self, sample_sets, parameter_sets):
        self.condition_names_ = list(sample_sets)
        first = self.condition_names_[0]
        original_features = pd.Index(sample_sets[first].columns)
        common_features = original_features

        for condition in self.condition_names_:
            common_features = common_features.intersection(
                sample_sets[condition].columns,
                sort=False,
            )
            common_features = common_features.intersection(
                parameter_sets[condition].index,
                sort=False,
            )

        valid = np.ones(len(common_features), dtype=bool)
        for condition in self.condition_names_:
            curves = sample_sets[condition].loc[:, common_features].to_numpy(float)
            params = parameter_sets[condition].loc[common_features, ["c", "beta"]].to_numpy(float)
            valid &= np.isfinite(curves).all(axis=0)
            valid &= np.isfinite(params).all(axis=1) & (params[:, 0] > 0)

        self.feature_names_ = common_features[valid]
        self.excluded_features_ = original_features.difference(self.feature_names_, sort=False)
        self.sample_sets_ = {
            condition: sample_sets[condition].loc[:, self.feature_names_].copy()
            for condition in self.condition_names_
        }
        self.parameter_sets_ = {
            condition: parameter_sets[condition].loc[self.feature_names_].copy()
            for condition in self.condition_names_
        }
        self.n_features_ = len(self.feature_names_)
        self.n_conditions_ = len(self.condition_names_)

    def _combine_sample_sets(self):
        self.curves_ = np.concatenate(
            [
                self.sample_sets_[condition].to_numpy(float).T
                for condition in self.condition_names_
            ],
            axis=1,
        )

    def _build_initialization_data(self):
        blocks = []
        for condition in self.condition_names_:
            params = self.parameter_sets_[condition]
            block = np.column_stack((
                np.log(params["c"].to_numpy(float)),
                params["beta"].to_numpy(float),
            ))
            center = np.median(block, axis=0)
            q25, q75 = np.quantile(block, [0.25, 0.75], axis=0)
            scale = (q75 - q25) / 1.349
            scale = np.where(scale > 1e-12, scale, block.std(axis=0))
            scale = np.where(scale > 1e-12, scale, 1.0)
            blocks.append((block - center) / scale)
        self.initialization_data_ = np.concatenate(blocks, axis=1)

    def _initialize_clusters(self):
        method = MiniBatchKMeans if self.use_mini_batch else KMeans
        self.clustering_model_ = method(
            n_clusters=self.n_components,
            init="k-means++",
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.initial_labels_ = self.clustering_model_.fit_predict(
            self.initialization_data_
        )
        self.initial_cluster_sizes_ = np.bincount(
            self.initial_labels_,
            minlength=self.n_components,
        )

    def _initialize_weights(self):
        self.initial_weights_ = (
            self.initial_cluster_sizes_
            / self.initial_cluster_sizes_.sum()
        )

    def _initialize_mean_curves(self):
        membership = np.eye(self.n_components)[self.initial_labels_]
        columns = [f"cluster_{k}" for k in range(self.n_components)]
        self.initial_mean_curves_ = {}
        for condition, samples in self.sample_sets_.items():
            means = (
                samples.to_numpy(float)
                @ membership
                / np.maximum(self.initial_cluster_sizes_, 1)
            )
            # Identical curves can leave KMeans clusters empty. Keep finite
            # initial means so zero-weight components do not poison the E-step.
            empty = self.initial_cluster_sizes_ == 0
            if empty.any():
                means[:, empty] = samples.to_numpy(float).mean(axis=1, keepdims=True)
            self.initial_mean_curves_[condition] = pd.DataFrame(
                means,
                index=samples.index,
                columns=columns,
            )

    def _initialize_mean_parameters(self):
        self.initial_params_mu_ = np.empty(
            (self.n_components, self.n_conditions_, 2)
        )
        for ell, curves in enumerate(self.initial_mean_curves_.values()):
            params, _ = power_fitting(curves, n_samples=len(curves))
            self.initial_params_mu_[:, ell] = params[["c", "beta"]].to_numpy(float)

    def _initialize_covariance_parameters(self):
        self.initial_params_cov_ = np.tile(
            [0.01, 0.95],
            (self.n_components, self.n_conditions_, 1),
        )

    def _initialize_responsibilities(self):
        self.initial_responsibilities_ = np.eye(self.n_components)[
            self.initial_labels_
        ]

    def _initialize(self):
        self._combine_sample_sets()
        self._build_initialization_data()
        self._initialize_clusters()
        self._initialize_weights()
        self._initialize_mean_curves()
        self._initialize_mean_parameters()
        self._initialize_covariance_parameters()
        self._initialize_responsibilities()

    def _set_device(self):
        if self.compute_device == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            name = self.compute_device
        self.device_ = torch.device(name)
        self.dtype_ = torch.float32 if self.device_.type == "cuda" else torch.float64

    def _prepare_solver_data(self):
        self._set_device()
        dtype, device = self.dtype_, self.device_
        self._prepared = []
        curve_blocks, statistic_blocks, self._curve_slices = [], [], []
        start = 0

        for condition in self.condition_names_:
            samples = self.sample_sets_[condition]
            time = torch.as_tensor(
                np.array(samples.index, dtype=float, copy=True),
                dtype=dtype,
                device=device,
            )
            curves = torch.as_tensor(
                np.array(samples.to_numpy(float).T, dtype=float, order="C", copy=True),
                dtype=dtype,
                device=device,
            )
            log_time = torch.log(time)
            center = log_time.mean()
            scale = torch.clamp(log_time.std(unbiased=False), min=1e-8)
            z = (log_time - center) / scale
            sum_y2 = torch.sum(curves.square(), dim=1)
            sum_y2_previous = torch.sum(curves[:, :-1].square(), dim=1)
            sum_cross = torch.sum(curves[:, 1:] * curves[:, :-1], dim=1)
            n_time = curves.shape[1]

            self._prepared.append({
                "curves": curves,
                "z": z,
                "sum_y2": sum_y2,
                "sum_y2_previous": sum_y2_previous,
                "sum_cross": sum_cross,
                "n_time": n_time,
            })
            curve_blocks.append(curves)
            statistic_blocks.extend([
                sum_y2[:, None],
                sum_y2_previous[:, None],
                sum_cross[:, None],
            ])
            self._curve_slices.append(slice(start, start + n_time))
            start += n_time

        self._combined_curves = torch.cat(curve_blocks, dim=1).contiguous()
        self._combined_statistics = torch.cat(statistic_blocks, dim=1).contiguous()
        counts = np.asarray([x["n_time"] for x in self._prepared], dtype=float)
        condition_weights = counts.mean() / counts if self.balance_conditions else np.ones_like(counts)
        self.condition_weights_ = condition_weights
        self._condition_weights = torch.as_tensor(condition_weights, dtype=dtype, device=device)
        self._log_two_pi = math.log(2.0 * math.pi)

    @staticmethod
    def _evaluate_power(z, parameters):
        exponent = torch.clamp(parameters[:, 1, None] * z[None, :], -60.0, 60.0)
        return parameters[:, 0, None] * torch.exp(exponent)

    def _fit_power_sad(self, curves, z, phi, initial_parameters):
        transformed_curves = curves.clone()
        transformed_curves[:, 1:] -= phi * curves[:, :-1]
        curve_norm = (
            torch.sum(transformed_curves.square(), dim=1)
            + self.tightness_weight * torch.sum(curves.square(), dim=1)
        )

        def profile(beta):
            basis = torch.exp(torch.clamp(beta[:, None] * z[None, :], -60.0, 60.0))
            transformed_basis = basis.clone()
            transformed_basis[:, 1:] -= phi * basis[:, :-1]
            numerator = torch.sum(transformed_basis * transformed_curves, dim=1)
            denominator = torch.sum(transformed_basis.square(), dim=1).clamp_min(1e-12)
            numerator += self.tightness_weight * torch.sum(basis * curves, dim=1)
            denominator += self.tightness_weight * torch.sum(basis.square(), dim=1)
            c = torch.clamp(numerator / denominator, min=0.0)
            loss = (curve_norm - 2.0 * c * numerator + c.square() * denominator).clamp_min(0.0)
            return loss, c

        lower, upper = self.beta_bounds
        golden = 0.5 * (math.sqrt(5.0) - 1.0)
        lo = torch.full((len(curves),), lower, dtype=self.dtype_, device=self.device_)
        hi = torch.full_like(lo, upper)
        beta_1, beta_2 = hi - golden * (hi - lo), lo + golden * (hi - lo)
        loss_1, loss_2 = profile(beta_1)[0], profile(beta_2)[0]
        n_search = max(1, math.ceil(math.log(self.beta_xtol / (upper - lower)) / math.log(golden)))

        for _ in range(n_search):
            move_right = loss_1 > loss_2
            next_lo = torch.where(move_right, beta_1, lo)
            next_hi = torch.where(move_right, hi, beta_2)
            candidate = torch.where(
                move_right,
                next_lo + golden * (next_hi - next_lo),
                next_hi - golden * (next_hi - next_lo),
            )
            candidate_loss = profile(candidate)[0]
            beta_1, loss_1, beta_2, loss_2 = (
                torch.where(move_right, beta_2, candidate),
                torch.where(move_right, loss_2, candidate_loss),
                torch.where(move_right, candidate, beta_1),
                torch.where(move_right, candidate_loss, loss_1),
            )
            lo, hi = next_lo, next_hi

        best_beta = torch.where(loss_1 <= loss_2, beta_1, beta_2)
        best_loss, best_c = profile(best_beta)
        candidates = (
            initial_parameters[:, 1].clamp(lower, upper),
            torch.zeros_like(best_beta),
            torch.full_like(best_beta, lower),
            torch.full_like(best_beta, upper),
        )
        for candidate_beta in candidates:
            candidate_loss, candidate_c = profile(candidate_beta)
            better = candidate_loss < best_loss
            best_loss = torch.where(better, candidate_loss, best_loss)
            best_beta = torch.where(better, candidate_beta, best_beta)
            best_c = torch.where(better, candidate_c, best_c)

        zero_curve = torch.max(torch.abs(curves), dim=1).values <= 1e-12
        best_c = torch.where(zero_curve, torch.zeros_like(best_c), best_c)
        best_beta = torch.where(zero_curve, torch.zeros_like(best_beta), best_beta)
        return torch.stack((best_c, best_beta), dim=1)

    def _e_step(self, weights, params_mu, params_cov):
        feature_constant = torch.zeros(self.n_features_, dtype=self.dtype_, device=self.device_)
        cluster_constant = torch.zeros(self.n_components, dtype=self.dtype_, device=self.device_)
        coefficient_blocks = []

        for ell, data in enumerate(self._prepared):
            phi = params_cov[ell, 0]
            gamma = params_cov[ell, 1].clamp_min(1e-8)
            condition_weight = self._condition_weights[ell]
            mean_curves = self._evaluate_power(data["z"], params_mu[:, ell])
            transformed_mean = mean_curves.clone()
            transformed_mean[:, 1:] -= phi * mean_curves[:, :-1]
            coefficient = transformed_mean.clone()
            coefficient[:, :-1] -= phi * transformed_mean[:, 1:]
            coefficient += self.tightness_weight * mean_curves
            inverse_variance = condition_weight / gamma.square()
            coefficient_blocks.append(coefficient * inverse_variance)

            feature_quadratic = (
                data["sum_y2"] - 2.0 * phi * data["sum_cross"]
                + phi.square() * data["sum_y2_previous"]
                + self.tightness_weight * data["sum_y2"]
            )
            feature_constant += (
                -0.5 * inverse_variance * feature_quadratic
                - condition_weight * (
                    (1.0 + self.tightness_weight)
                    * data["n_time"]
                    * (0.5 * self._log_two_pi + torch.log(gamma))
                )
            )
            cluster_constant += -0.5 * inverse_variance * (
                torch.sum(transformed_mean.square(), dim=1)
                + self.tightness_weight * torch.sum(mean_curves.square(), dim=1)
            )

        coefficients = torch.cat(coefficient_blocks, dim=1).contiguous()
        log_joint = (
            torch.log(weights.clamp_min(1e-30))[None]
            + feature_constant[:, None]
            + cluster_constant[None]
            + self._combined_curves @ coefficients.T
        )
        normalizer = torch.logsumexp(log_joint, dim=1, keepdim=True)
        responsibilities = torch.exp(log_joint - normalizer)
        responsibilities /= responsibilities.sum(dim=1, keepdim=True).clamp_min(1e-30)
        log_likelihood = float(normalizer.to(torch.float64).sum().item())
        return responsibilities, log_likelihood

    def _m_step(self, responsibilities, params_mu, params_cov):
        cluster_sizes = responsibilities.sum(dim=0)
        active = cluster_sizes > 1e-8
        weights = (cluster_sizes / self.n_features_).clamp_min(1e-12)
        weights /= weights.sum()
        new_mu, new_cov = params_mu.clone(), params_cov.clone()
        responsibility_transpose = responsibilities.T.contiguous()
        weighted_sums = responsibility_transpose @ self._combined_curves
        weighted_statistics = responsibility_transpose @ self._combined_statistics

        for ell, data in enumerate(self._prepared):
            weighted_sum = weighted_sums[:, self._curve_slices[ell]]
            offset = 3 * ell
            weighted_y2 = weighted_statistics[:, offset]
            weighted_y2_previous = weighted_statistics[:, offset + 1]
            weighted_cross = weighted_statistics[:, offset + 2]
            mean_curves = weighted_sum / cluster_sizes[:, None].clamp_min(1e-12)
            mean_curves = torch.where(active[:, None], mean_curves, torch.zeros_like(mean_curves))
            mean_parameters = new_mu[:, ell].clone()
            phi = new_cov[ell, 0].clone()
            gamma = new_cov[ell, 1].clamp_min(1e-8).clone()

            for _ in range(self.inner_max_iter):
                previous_parameters = mean_parameters.clone()
                previous_phi, previous_gamma = phi.clone(), gamma.clone()
                if bool(active.any().item()):
                    mean_parameters[active] = self._fit_power_sad(
                        mean_curves[active],
                        data["z"],
                        phi,
                        mean_parameters[active],
                    )

                fitted = self._evaluate_power(data["z"], mean_parameters)
                residual_ss = (
                    weighted_y2 - 2.0 * torch.sum(weighted_sum * fitted, dim=1)
                    + cluster_sizes * torch.sum(fitted.square(), dim=1)
                )
                residual_ss_previous = (
                    weighted_y2_previous
                    - 2.0 * torch.sum(weighted_sum[:, :-1] * fitted[:, :-1], dim=1)
                    + cluster_sizes * torch.sum(fitted[:, :-1].square(), dim=1)
                )
                residual_cross = (
                    weighted_cross
                    - torch.sum(weighted_sum[:, 1:] * fitted[:, :-1], dim=1)
                    - torch.sum(weighted_sum[:, :-1] * fitted[:, 1:], dim=1)
                    + cluster_sizes * torch.sum(fitted[:, 1:] * fitted[:, :-1], dim=1)
                )
                denominator = torch.sum(residual_ss_previous)
                phi = torch.where(
                    denominator > 1e-12,
                    torch.clamp(
                        torch.sum(residual_cross)
                        / ((1.0 + self.phi_ridge) * denominator).clamp_min(1e-12),
                        -self.phi_bound,
                        self.phi_bound,
                    ),
                    torch.zeros_like(phi),
                )
                innovation_ss = (
                    residual_ss
                    - 2.0 * phi * residual_cross
                    + phi.square() * residual_ss_previous
                    + self.tightness_weight * residual_ss
                )
                gamma = torch.sqrt(
                    torch.clamp(
                        torch.sum(innovation_ss)
                        / (
                            (1.0 + self.tightness_weight)
                            * self.n_features_
                            * data["n_time"]
                        ),
                        min=1e-12,
                    )
                )

                parameter_change = torch.max(torch.abs(mean_parameters - previous_parameters))
                covariance_change = torch.maximum(
                    torch.abs(phi - previous_phi),
                    torch.abs(gamma - previous_gamma),
                )
                scale = 1.0 + torch.maximum(
                    torch.max(torch.abs(previous_parameters)),
                    torch.maximum(torch.abs(previous_phi), torch.abs(previous_gamma)),
                )
                if float(torch.maximum(parameter_change, covariance_change).item()) <= self.inner_tol * float(scale.item()):
                    break

            new_mu[:, ell] = mean_parameters
            new_cov[ell] = torch.stack((phi, gamma))

        return weights, new_mu, new_cov, cluster_sizes

    def _solve(self):
        weights = torch.as_tensor(self.initial_weights_, dtype=self.dtype_, device=self.device_).clone()
        params_mu = torch.as_tensor(self.initial_params_mu_, dtype=self.dtype_, device=self.device_).clone()
        params_cov = torch.as_tensor(self.initial_params_cov_[0], dtype=self.dtype_, device=self.device_).clone()
        responsibilities, log_likelihood = self._e_step(weights, params_mu, params_cov)
        history, converged, n_iter = [log_likelihood], False, 0

        if self.progress:
            print(f"[funclu] EM start | log_likelihood={log_likelihood:,.6f} | device={self.device_}")

        for iteration in range(1, self.max_iter + 1):
            new_weights, new_mu, new_cov, _ = self._m_step(responsibilities, params_mu, params_cov)
            new_responsibilities, new_log_likelihood = self._e_step(new_weights, new_mu, new_cov)
            change = new_log_likelihood - log_likelihood
            threshold = max(self.tol, 1e-5 if self.device_.type == "cuda" else self.tol) * (1.0 + abs(log_likelihood))

            if self.progress:
                print(
                    f"[funclu] iter={iteration:03d} | "
                    f"log_likelihood={new_log_likelihood:,.6f} | change={change:+.6e}"
                )
            if change < -threshold:
                break
            if change >= 0.0:
                weights, params_mu, params_cov = new_weights, new_mu, new_cov
                responsibilities, log_likelihood = new_responsibilities, new_log_likelihood
                history.append(log_likelihood)
                n_iter = iteration
            if abs(change) <= threshold:
                converged = True
                break

        self._weights = weights
        self._params_mu = params_mu
        self._params_cov = params_cov
        self._responsibilities = responsibilities
        self.log_likelihood_ = log_likelihood
        self.log_likelihood_history_ = np.asarray(history)
        self.n_iter_ = n_iter
        self.converged_ = converged

    def _finalize(self):
        self.weights_ = self._weights.detach().cpu().numpy()
        self.params_mu_ = self._params_mu.detach().cpu().numpy()
        self.shared_params_cov_ = self._params_cov.detach().cpu().numpy()
        self.params_cov_ = np.broadcast_to(
            self.shared_params_cov_[None],
            (self.n_components, self.n_conditions_, 2),
        ).copy()
        self.responsibilities_ = self._responsibilities.detach().cpu().numpy()
        self.labels_ = np.argmax(self.responsibilities_, axis=1)
        self.cluster_sizes_ = np.bincount(self.labels_, minlength=self.n_components)
        self.effective_cluster_sizes_ = self.responsibilities_.sum(axis=0)
        self.mean_curves_ = {}

        for ell, condition in enumerate(self.condition_names_):
            z = self._prepared[ell]["z"].detach().cpu().numpy()
            c = self.params_mu_[:, ell, 0]
            beta = self.params_mu_[:, ell, 1]
            values = (c[:, None] * np.exp(np.clip(beta[:, None] * z[None], -60.0, 60.0))).T
            self.mean_curves_[condition] = pd.DataFrame(
                values,
                index=self.sample_sets_[condition].index,
                columns=[f"cluster_{k}" for k in range(self.n_components)],
            )

        self.assignments_ = pd.DataFrame({
            "feature": self.feature_names_,
            "cluster": self.labels_,
            "probability": self.responsibilities_.max(axis=1),
        }).set_index("feature")
        self.compute_device_ = str(self.device_)
        if self.device_.type == "cuda":
            torch.cuda.synchronize()
        for name in (
            "_weights",
            "_params_mu",
            "_params_cov",
            "_responsibilities",
            "_prepared",
            "_combined_curves",
            "_combined_statistics",
            "_condition_weights",
        ):
            delattr(self, name)

    def fit(self, sample_sets: dict[str, pd.DataFrame], parameter_sets: dict[str, pd.DataFrame]):
        self._prepare_inputs(sample_sets, parameter_sets)
        if not 1 <= self.n_components <= self.n_features_:
            raise ValueError("n_components must not exceed the valid common feature count")
        for samples in self.sample_sets_.values():
            time = samples.index.to_numpy(float)
            if len(time) < 2 or not np.isfinite(time).all() or (time <= 0).any() or np.unique(time).size < 2:
                raise ValueError("Samples require at least two distinct finite positive times")
        self._initialize()
        self._prepare_solver_data()
        self._solve()
        self._finalize()
        if not np.isfinite(self.log_likelihood_) or not np.isfinite(self.responsibilities_).all():
            raise ValueError("FunClu produced non-finite results; check data magnitudes and parameters")
        return self

    def fit_predict(self, sample_sets, parameter_sets):
        return self.fit(sample_sets, parameter_sets).predict()

    def predict(self):
        return self.labels_.copy()

    def predict_proba(self):
        return self.responsibilities_.copy()

    def score(self):
        return self.log_likelihood_

    def bic(self):
        n_parameters = (
            self.n_components - 1
            + 2 * self.n_components * self.n_conditions_
            + 2 * self.n_conditions_
        )
        return -2.0 * self.log_likelihood_ + n_parameters * np.log(self.n_features_)

    def result(self):
        return {
            "labels": self.labels_,
            "responsibilities": self.responsibilities_,
            "weights": self.weights_,
            "params_mu": self.params_mu_,
            "params_cov": self.params_cov_,
            "shared_params_cov": self.shared_params_cov_,
            "condition_names": self.condition_names_,
            "feature_names": self.feature_names_,
            "excluded_features": self.excluded_features_,
            "cluster_sizes": self.cluster_sizes_,
            "effective_cluster_sizes": self.effective_cluster_sizes_,
            "mean_curves": self.mean_curves_,
            "log_likelihood": self.log_likelihood_,
            "log_likelihood_history": self.log_likelihood_history_,
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "compute_device": self.compute_device_,
            "condition_weights": self.condition_weights_,
        }

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assignments = self.assignments_.copy()
        assignments["cluster"] += 1
        assignments.to_csv(output_dir / "assignments.csv")

        pd.DataFrame(
            self.responsibilities_,
            index=self.feature_names_,
            columns=[f"M{k + 1}" for k in range(self.n_components)],
        ).rename_axis("feature").to_csv(output_dir / "responsibilities.csv")

        pd.DataFrame({
            "cluster": np.arange(1, self.n_components + 1),
            "weight": self.weights_,
            "cluster_size": self.cluster_sizes_,
            "effective_cluster_size": self.effective_cluster_sizes_,
        }).to_csv(output_dir / "clusters.csv", index=False)

        mean_parameters = []
        for k in range(self.n_components):
            for ell, condition in enumerate(self.condition_names_):
                mean_parameters.append({
                    "cluster": k + 1,
                    "condition": condition,
                    "c": self.params_mu_[k, ell, 0],
                    "beta": self.params_mu_[k, ell, 1],
                })
        pd.DataFrame(mean_parameters).to_csv(
            output_dir / "mean_parameters.csv",
            index=False,
        )

        pd.DataFrame({
            "condition": self.condition_names_,
            "phi": self.shared_params_cov_[:, 0],
            "gamma": self.shared_params_cov_[:, 1],
        }).to_csv(output_dir / "sad_parameters.csv", index=False)

        pd.DataFrame({
            "iteration": np.arange(len(self.log_likelihood_history_)),
            "log_likelihood": self.log_likelihood_history_,
        }).to_csv(output_dir / "log_likelihood.csv", index=False)

        pd.DataFrame({
            "condition_code": [f"C{ell + 1}" for ell in range(self.n_conditions_)],
            "condition": self.condition_names_,
        }).to_csv(output_dir / "conditions.csv", index=False)

        pd.DataFrame([{
            "n_components": self.n_components,
            "n_features": self.n_features_,
            "n_conditions": self.n_conditions_,
            "log_likelihood": self.log_likelihood_,
            "bic": self.bic(),
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "compute_device": self.compute_device_,
            "phi_bound": self.phi_bound,
            "phi_ridge": self.phi_ridge,
            "tightness_weight": self.tightness_weight,
        }]).to_csv(output_dir / "model_summary.csv", index=False)

        result_dir = output_dir / "res"
        result_dir.mkdir(parents=True, exist_ok=True)
        for k in range(self.n_components):
            folder = result_dir / f"M{k + 1}"
            folder.mkdir(parents=True, exist_ok=True)
            members = self.feature_names_[self.labels_ == k]
            for ell, condition in enumerate(self.condition_names_):
                curves = self.sample_sets_[condition].loc[:, members].copy()
                curves["mean"] = self.mean_curves_[condition].iloc[:, k]
                curves.to_csv(folder / f"C{ell + 1}_M{k + 1}.csv")

        return output_dir


def select_best_k(
    sample_sets: dict[str, pd.DataFrame],
    parameter_sets: dict[str, pd.DataFrame],
    k_values: list[int] | tuple[int, ...] | None = None,
    k_min: int = 2,
    k_max: int = 500,
    k_step: int = 10,
    max_iter: int = 1000,
    min_cluster_size: int = 10,
    n_init: int = 10,
    compute_device: str = "auto",
    random_state: int = 42,
    progress: bool = True,
    **funclu_options,
):
    """Fit every supplied K with identical precision and select the lowest BIC."""

    n_features = min(samples.shape[1] for samples in sample_sets.values())
    k_max = min(k_max, n_features // max(min_cluster_size, 1))
    if k_values is None:
        candidates = list(range(k_min, k_max + 1, max(k_step, 1)))
        if candidates and candidates[-1] != k_max:
            candidates.append(k_max)
    else:
        candidates = sorted({
            int(k) for k in k_values
            if k_min <= int(k) <= k_max
        })

    records = []
    options = dict(funclu_options)
    for key in ("n_components", "max_iter", "n_init", "compute_device", "random_state", "progress"):
        options.pop(key, None)

    def fit_candidate(k, index):
        if progress:
            print(f"[funclu] BIC {index}/{len(candidates)}: fitting K={k} ...", flush=True)
        started = perf_counter()
        try:
            model = FunClu(
                n_components=k,
                max_iter=max_iter,
                n_init=n_init,
                compute_device=compute_device,
                random_state=random_state,
                progress=False,
                **options,
            ).fit(sample_sets, parameter_sets)
            weights = model.cluster_sizes_ / model.cluster_sizes_.sum()
            positive_weights = weights[weights > 0]
            entropy = -np.sum(positive_weights * np.log(positive_weights)) / np.log(k)
            record = {
                "k": k,
                "bic": model.bic(),
                "log_likelihood": model.log_likelihood_,
                "n_iter": model.n_iter_,
                "converged": model.converged_,
                "min_cluster_size": int(model.cluster_sizes_.min()),
                "normalized_entropy": float(entropy),
                "eligible": bool(model.cluster_sizes_.min() >= min_cluster_size),
                "seconds": perf_counter() - started,
                "error": "",
            }
        except Exception as error:
            model = None
            record = {
                "k": k,
                "bic": np.inf,
                "log_likelihood": np.nan,
                "n_iter": 0,
                "converged": False,
                "min_cluster_size": 0,
                "normalized_entropy": np.nan,
                "eligible": False,
                "seconds": perf_counter() - started,
                "error": str(error),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        records.append(record)
        if progress:
            if model is None:
                print(f"[funclu] BIC K={k} failed: {record['error']}", flush=True)
            else:
                print(
                    f"[funclu] BIC K={k} | BIC={record['bic']:,.4f} | "
                    f"iter={record['n_iter']} | converged={record['converged']} | "
                    f"min_size={record['min_cluster_size']} | {record['seconds']:.1f}s",
                    flush=True,
                )
        return model, record

    best_model, best_record, best_key = None, None, None
    for index, k in enumerate(candidates, start=1):
        model, record = fit_candidate(k, index)
        if model is None:
            continue
        quality = (
            0 if record["eligible"] and record["converged"]
            else 1 if record["eligible"]
            else 2
        )
        key = (quality, record["bic"])
        if best_key is None or key < best_key:
            del best_model
            best_model, best_record, best_key = model, record, key
        else:
            del model

    bic_table = pd.DataFrame(records)
    bic_table["selected"] = False
    if best_record is not None:
        bic_table.loc[bic_table["k"] == best_record["k"], "selected"] = True
        if progress:
            print(
                f"[funclu] BIC selected K={best_record['k']} | "
                f"BIC={best_record['bic']:,.4f} | converged={best_record['converged']}",
                flush=True,
            )

    return {
        "best_k": None if best_record is None else best_record["k"],
        "best_model": best_model,
        "bic_table": bic_table,
        "candidates": candidates,
    }

