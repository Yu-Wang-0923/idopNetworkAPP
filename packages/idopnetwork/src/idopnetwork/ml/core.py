"""Machine-learning helpers for Hub identification.

The NetAnal Machine Learning page accepts both common spreadsheet layouts:

    rows = samples/plants, columns = variables/genes
    rows = variables/genes, columns = samples/observations

Preprocessing converts either layout into the internal convention where
variables are rows and samples are columns.

For every target variable, the module fits a sparse linear model using all
other variables as predictors. A positive standardized coefficient is treated
as a promoting association (source promotes target), while a negative
coefficient is treated as an inhibiting association.
"""

from __future__ import annotations

from itertools import combinations, permutations
import io
import zipfile
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
import warnings

try:  # Optional at runtime; required only when classifier="xgboost".
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - depends on optional local environment.
    XGBClassifier = None  # type: ignore[assignment]


def _read_zip_csv(zf: zipfile.ZipFile, name: str, *, index_col: int | None = None) -> pd.DataFrame:
    with zf.open(name) as fh:
        return pd.read_csv(io.BytesIO(fh.read()), index_col=index_col)


def load_funclu_k_export(zip_bytes: bytes) -> dict[str, Any]:
    """Read the ``funclu_k_export.zip`` package produced by FunClu-K.

    The returned object keeps the same layer structure used by Multi-Layer
    NetRecon:

    - ``cluster_centers/<condition>/...`` maps to ``inter_cluster`` data.
    - ``cluster_members/<condition>/<cluster>_...`` maps to ``intra_cluster`` data.
    """
    labels_df: pd.DataFrame | None = None
    cluster_sizes_df: pd.DataFrame | None = None
    centers: dict[str, pd.DataFrame] = {}
    center_responses: dict[str, pd.DataFrame] = {}
    members: dict[str, dict[str, pd.DataFrame]] = {}
    member_responses: dict[str, dict[str, pd.DataFrame]] = {}
    members_list: list[str] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members_list = sorted(n for n in zf.namelist() if not n.endswith("/"))
        for name in members_list:
            parts = name.split("/")
            if name == "labels.csv":
                labels_df = _read_zip_csv(zf, name)
            elif name == "cluster_sizes.csv":
                cluster_sizes_df = _read_zip_csv(zf, name)
            elif (
                len(parts) == 3
                and parts[0] == "cluster_centers"
                and parts[2] == "cluster_center_curve_sample.csv"
            ):
                centers[parts[1]] = _read_zip_csv(zf, name, index_col=0)
            elif (
                len(parts) == 3
                and parts[0] == "cluster_centers"
                and parts[2] == "cluster_center_quasi_dynamic.csv"
            ):
                center_responses[parts[1]] = _read_zip_csv(zf, name, index_col=0)
            elif (
                len(parts) == 3
                and parts[0] == "cluster_members"
                and parts[2].endswith("_curve_sample.csv")
            ):
                cluster_name = parts[2].removesuffix("_curve_sample.csv")
                members.setdefault(parts[1], {})[cluster_name] = _read_zip_csv(
                    zf,
                    name,
                    index_col=0,
                )
            elif (
                len(parts) == 3
                and parts[0] == "cluster_members"
                and parts[2].endswith("_quasi_dynamic.csv")
            ):
                cluster_name = parts[2].removesuffix("_quasi_dynamic.csv")
                member_responses.setdefault(parts[1], {})[cluster_name] = _read_zip_csv(
                    zf,
                    name,
                    index_col=0,
                )

    if labels_df is None:
        raise ValueError("FunClu-K ZIP is missing labels.csv.")
    if cluster_sizes_df is None:
        cluster_sizes_df = pd.DataFrame()
    if not centers:
        raise ValueError("FunClu-K ZIP is missing cluster center curve_sample files.")
    if not center_responses:
        raise ValueError("FunClu-K ZIP is missing cluster center quasi_dynamic files.")
    if not members:
        raise ValueError("FunClu-K ZIP is missing cluster member curve_sample files.")
    if not member_responses:
        raise ValueError("FunClu-K ZIP is missing cluster member quasi_dynamic files.")

    return {
        "labels": labels_df,
        "cluster_sizes": cluster_sizes_df,
        "centers": dict(sorted(centers.items())),
        "center_responses": dict(sorted(center_responses.items())),
        "members": {
            cond: dict(sorted(cluster_map.items()))
            for cond, cluster_map in sorted(members.items())
        },
        "member_responses": {
            cond: dict(sorted(cluster_map.items()))
            for cond, cluster_map in sorted(member_responses.items())
        },
        "zip_members": members_list,
    }


def funclu_k_export_summary(funclu_export: dict[str, Any]) -> pd.DataFrame:
    """Summarize ML-ready datasets inside a FunClu-K export."""
    rows: list[dict[str, Any]] = []
    for condition, df in funclu_export["center_responses"].items():
        rows.append(
            {
                "layer": "inter_cluster",
                "condition": condition,
                "cluster": "",
                "data_source": "quasi_dynamic",
                "variables": int(df.shape[1]),
                "samples": int(df.shape[0]),
            }
        )
    for condition, cluster_map in funclu_export["member_responses"].items():
        for cluster, df in cluster_map.items():
            rows.append(
                {
                    "layer": "intra_cluster",
                    "condition": condition,
                    "cluster": cluster,
                    "data_source": "quasi_dynamic",
                    "variables": int(df.shape[1]),
                    "samples": int(df.shape[0]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["layer", "condition", "cluster", "data_source", "variables", "samples"],
    )


def module_feature_map_from_labels(labels_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return ``cluster -> feature names`` from a FunClu ``labels.csv`` table."""
    if labels_df.empty:
        raise ValueError("FunClu labels table is empty.")

    if "feature" in labels_df.columns:
        feature_col = "feature"
    else:
        feature_col = str(labels_df.columns[0])

    if "cluster" in labels_df.columns:
        cluster_col = "cluster"
    elif "cluster_id" in labels_df.columns:
        cluster_col = "cluster_id"
    else:
        raise ValueError("FunClu labels table must contain a cluster or cluster_id column.")

    clean = labels_df[[feature_col, cluster_col]].copy()
    clean[feature_col] = clean[feature_col].astype(str).str.strip()
    clean[cluster_col] = clean[cluster_col].astype(str).str.strip()
    clean = clean[(clean[feature_col] != "") & (clean[cluster_col] != "")]

    cluster_map: dict[str, list[str]] = {}
    for cluster, group in clean.groupby(cluster_col, sort=False):
        label = str(cluster)
        if cluster_col == "cluster_id" and not label.upper().startswith("M"):
            label = f"M{label}"
        features = []
        seen: set[str] = set()
        for feature in group[feature_col].tolist():
            feature = str(feature)
            if feature not in seen:
                features.append(feature)
                seen.add(feature)
        cluster_map[label] = features

    if not cluster_map:
        raise ValueError("No cluster-feature mapping could be read from FunClu labels.")
    return dict(sorted(cluster_map.items(), key=lambda item: _module_sort_key(item[0])))


def _module_sort_key(label: str) -> tuple[int, str]:
    text = str(label)
    if len(text) > 1 and text[0].upper() == "M" and text[1:].isdigit():
        return (int(text[1:]), text)
    return (10**9, text)


def _prepare_condition_frame(
    raw_df: pd.DataFrame,
    *,
    first_column_as_sample_id: bool,
    condition: str,
) -> tuple[pd.DataFrame, list[str]]:
    if raw_df.empty:
        raise ValueError(f"Condition {condition!r} CSV is empty.")

    if first_column_as_sample_id and raw_df.shape[1] >= 2:
        sample_ids = _make_unique_labels(raw_df.iloc[:, 0].tolist())
        feature_part = raw_df.iloc[:, 1:].copy()
    else:
        sample_ids = [f"{condition}_S{i}" for i in range(1, raw_df.shape[0] + 1)]
        feature_part = raw_df.copy()

    feature_part.columns = [str(col).strip() for col in feature_part.columns]
    numeric = feature_part.apply(pd.to_numeric, errors="coerce")
    numeric.index = sample_ids
    return numeric, sample_ids


def prepare_module_classification_dataset(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    *,
    first_column_as_sample_id: bool = True,
    max_missing_fraction: float = 0.5,
) -> dict[str, Any]:
    """Merge condition CSV tables into a sample-by-feature classification dataset."""
    if len(condition_tables) < 2:
        raise ValueError("At least two condition CSV files are required for classification.")

    cluster_map = module_feature_map_from_labels(labels_df)
    mapped_features: list[str] = []
    seen_features: set[str] = set()
    for cluster in sorted(cluster_map, key=_module_sort_key):
        for feature in cluster_map[cluster]:
            if feature not in seen_features:
                mapped_features.append(feature)
                seen_features.add(feature)

    condition_frames: dict[str, pd.DataFrame] = {}
    sample_rows: list[dict[str, Any]] = []
    for condition, raw_df in condition_tables.items():
        frame, sample_ids = _prepare_condition_frame(
            raw_df,
            first_column_as_sample_id=first_column_as_sample_id,
            condition=condition,
        )
        condition_frames[str(condition)] = frame
        sample_rows.append(
            {
                "condition": str(condition),
                "samples": int(frame.shape[0]),
                "numeric_features": int(frame.shape[1]),
            }
        )

    feature_sets = [set(frame.columns) for frame in condition_frames.values()]
    common_feature_set = set.intersection(*feature_sets)
    usable_features = [feature for feature in mapped_features if feature in common_feature_set]
    if not usable_features:
        raise ValueError(
            "No FunClu-labeled features were found in every uploaded condition CSV."
        )

    x_parts: list[pd.DataFrame] = []
    y_values: list[str] = []
    sample_info_rows: list[dict[str, Any]] = []
    for condition, frame in condition_frames.items():
        part = frame.loc[:, usable_features].copy()
        x_parts.append(part)
        y_values.extend([condition] * len(part))
        for sample_id in part.index:
            sample_info_rows.append({"sample_id": str(sample_id), "condition": condition})

    x = pd.concat(x_parts, axis=0)
    x.index = _make_unique_labels([str(idx) for idx in x.index])
    y = pd.Series(y_values, index=x.index, name="condition")

    missing_fraction = x.isna().mean(axis=0)
    keep_missing = missing_fraction <= float(max_missing_fraction)
    x = x.loc[:, keep_missing].copy()

    std = x.std(axis=0, ddof=0, skipna=True).fillna(0.0)
    keep_nonconstant = std > 1e-12
    x = x.loc[:, keep_nonconstant].copy()
    if x.empty:
        raise ValueError("No usable features remain after missing/constant filtering.")

    filtered_cluster_map: dict[str, list[str]] = {}
    for cluster, features in cluster_map.items():
        filtered_cluster_map[cluster] = [feature for feature in features if feature in x.columns]

    feature_summary = pd.DataFrame(
        [
            {
                "module": cluster,
                "funclu_features": int(len(features)),
                "usable_features": int(len(filtered_cluster_map[cluster])),
            }
            for cluster, features in cluster_map.items()
        ]
    )
    sample_summary = pd.DataFrame(sample_rows)
    sample_info = pd.DataFrame(sample_info_rows)

    diagnostics = {
        "conditions": int(len(condition_tables)),
        "samples": int(len(y)),
        "funclu_features": int(len(mapped_features)),
        "features_common_to_all_conditions": int(len(usable_features)),
        "features_used": int(x.shape[1]),
        "features_dropped_missing": int((~keep_missing).sum()),
        "features_dropped_constant": int((~keep_nonconstant).sum()),
        "max_missing_fraction": float(max_missing_fraction),
    }
    return {
        "x": x,
        "y": y,
        "cluster_map": filtered_cluster_map,
        "sample_summary": sample_summary,
        "sample_info": sample_info,
        "feature_summary": feature_summary,
        "diagnostics": diagnostics,
    }


def _transform_edge_node_matrix(
    x: pd.DataFrame,
    *,
    transform: str,
) -> pd.DataFrame:
    """Return a numeric node matrix suitable for sample-level edge construction."""
    filled = x.apply(pd.to_numeric, errors="coerce").copy()
    medians = filled.median(axis=0, skipna=True).fillna(0.0)
    filled = filled.fillna(medians).fillna(0.0)

    transform = str(transform)
    if transform == "raw":
        return filled
    if transform == "centered":
        return filled - filled.mean(axis=0)
    if transform == "zscore":
        std = filled.std(axis=0, ddof=0).replace(0.0, 1.0)
        return (filled - filled.mean(axis=0)) / std
    raise ValueError("transform must be one of: 'zscore', 'centered', 'raw'.")


def _sample_edge_values(
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    method = str(method)
    if method == "product":
        return source_values * target_values
    if method == "signed_difference":
        return source_values - target_values
    if method == "absolute_difference":
        return np.abs(source_values - target_values)
    if method == "similarity":
        return 1.0 / (1.0 + np.abs(source_values - target_values))
    raise ValueError(
        "method must be one of: 'product', 'signed_difference', "
        "'absolute_difference', 'similarity'."
    )


def generate_sample_level_intra_edge_table(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    *,
    modules: list[str] | None = None,
    first_column_as_sample_id: bool = True,
    max_missing_fraction: float = 0.5,
    transform: str = "zscore",
    method: str = "product",
    direction: str = "undirected",
    include_self_edges: bool = False,
    max_edges_per_module: int | None = 500,
) -> dict[str, Any]:
    """Generate sample-level intra-module edge features for Node+Edge ML.

    The generated table is intentionally sample-level and does not reuse
    condition-level NetRecon edge weights. For each selected module, candidate
    edges are constructed from module member node values for every sample.
    """
    dataset = prepare_module_classification_dataset(
        condition_tables,
        labels_df,
        first_column_as_sample_id=first_column_as_sample_id,
        max_missing_fraction=max_missing_fraction,
    )
    x: pd.DataFrame = dataset["x"]
    cluster_map: dict[str, list[str]] = dataset["cluster_map"]
    sample_info = dataset["sample_info"].copy()
    sample_info["ml_index"] = x.index.astype(str)
    sample_info["sample_id"] = sample_info["sample_id"].astype(str)
    sample_info["condition"] = sample_info["condition"].astype(str)

    available_modules = [
        module for module in sorted(cluster_map, key=_module_sort_key)
        if len(cluster_map.get(module, [])) > 0
    ]
    if modules is None:
        selected_modules = available_modules
    else:
        requested = [str(module) for module in modules]
        selected_modules = [
            module for module in requested
            if module in cluster_map and len(cluster_map.get(module, [])) > 0
        ]
    if not selected_modules:
        raise ValueError("No selected FunClu modules have usable node features.")

    direction = str(direction)
    if direction not in {"undirected", "directed"}:
        raise ValueError("direction must be 'undirected' or 'directed'.")

    max_edges = (
        int(max_edges_per_module)
        if max_edges_per_module is not None and int(max_edges_per_module) > 0
        else None
    )
    node_x = _transform_edge_node_matrix(x, transform=str(transform))
    sample_lookup = sample_info.set_index("ml_index").loc[node_x.index]

    edge_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for module in selected_modules:
        features = [feature for feature in cluster_map[module] if feature in node_x.columns]
        if len(features) < 2 and not include_self_edges:
            summary_rows.append(
                {
                    "module": module,
                    "usable_features": int(len(features)),
                    "candidate_edges": 0,
                    "retained_edges": 0,
                    "rows": 0,
                    "status": "skipped: less than 2 usable features",
                }
            )
            continue

        if direction == "directed":
            candidate_pairs = list(permutations(features, 2))
            if include_self_edges:
                candidate_pairs.extend((feature, feature) for feature in features)
        else:
            candidate_pairs = list(combinations(features, 2))
            if include_self_edges:
                candidate_pairs.extend((feature, feature) for feature in features)

        if not candidate_pairs:
            summary_rows.append(
                {
                    "module": module,
                    "usable_features": int(len(features)),
                    "candidate_edges": 0,
                    "retained_edges": 0,
                    "rows": 0,
                    "status": "skipped: no candidate edges",
                }
            )
            continue

        if max_edges is not None and len(candidate_pairs) > max_edges:
            scored_pairs: list[tuple[float, str, str]] = []
            for source, target in candidate_pairs:
                values = _sample_edge_values(
                    node_x[source].to_numpy(dtype=float, copy=False),
                    node_x[target].to_numpy(dtype=float, copy=False),
                    method=str(method),
                )
                score = float(np.nanvar(values))
                scored_pairs.append((score, str(source), str(target)))
            scored_pairs.sort(key=lambda item: item[0], reverse=True)
            retained_pairs = [(source, target) for _, source, target in scored_pairs[:max_edges]]
        else:
            retained_pairs = [(str(source), str(target)) for source, target in candidate_pairs]

        module_frames: list[pd.DataFrame] = []
        for source, target in retained_pairs:
            values = _sample_edge_values(
                node_x[source].to_numpy(dtype=float, copy=False),
                node_x[target].to_numpy(dtype=float, copy=False),
                method=str(method),
            )
            module_frames.append(
                pd.DataFrame(
                    {
                        "condition": sample_lookup["condition"].to_numpy(),
                        "sample_id": sample_lookup["sample_id"].to_numpy(),
                        "module": module,
                        "from": source,
                        "to": target,
                        "weight": values,
                    }
                )
            )

        module_edge_table = pd.concat(module_frames, axis=0, ignore_index=True)
        edge_rows.append(module_edge_table)
        summary_rows.append(
            {
                "module": module,
                "usable_features": int(len(features)),
                "candidate_edges": int(len(candidate_pairs)),
                "retained_edges": int(len(retained_pairs)),
                "rows": int(module_edge_table.shape[0]),
                "status": "ok",
            }
        )

    if not edge_rows:
        raise ValueError("No sample-level edge rows were generated.")

    edge_table = pd.concat(edge_rows, axis=0, ignore_index=True)
    edge_table["weight"] = pd.to_numeric(edge_table["weight"], errors="coerce").fillna(0.0)
    summary = pd.DataFrame(summary_rows)
    return {
        "edge_table": edge_table,
        "summary": summary,
        "dataset": {
            "sample_summary": dataset["sample_summary"],
            "feature_summary": dataset["feature_summary"],
            "diagnostics": dataset["diagnostics"],
        },
        "context": {
            "modules": selected_modules,
            "transform": str(transform),
            "method": str(method),
            "direction": direction,
            "include_self_edges": bool(include_self_edges),
            "max_edges_per_module": int(max_edges) if max_edges is not None else 0,
            "rows": int(edge_table.shape[0]),
            "edge_features": int(
                edge_table.loc[:, ["module", "from", "to"]].drop_duplicates().shape[0]
            ),
            "samples": int(sample_lookup.shape[0]),
        },
    }


def _classification_pipeline(classifier: str, *, random_state: int) -> Pipeline:
    classifier = str(classifier)
    if classifier == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            random_state=int(random_state),
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )
    elif classifier == "linear_svm":
        svm = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=int(random_state),
        )
        estimator = CalibratedClassifierCV(
            estimator=svm,
            method="sigmoid",
            cv=2,
        )
    elif classifier == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is not installed in this Python environment. "
                "Install it with `pip install xgboost` to use classifier='xgboost'."
            )
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=int(random_state),
            n_jobs=-1,
        )
    else:
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=int(random_state),
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def _model_y_for_classifier(
    y: pd.Series,
    *,
    classifier: str,
) -> tuple[pd.Series, LabelEncoder | None]:
    """Encode labels only for estimators that require numeric class ids."""
    if str(classifier) != "xgboost":
        return y, None

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str).to_numpy())
    return pd.Series(encoded, index=y.index, name=y.name), encoder


def _cross_validate_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    classifier: str,
    cv_folds: int,
    random_state: int,
    task: str,
) -> dict[str, float]:
    y_model, _ = _model_y_for_classifier(y, classifier=classifier)
    class_counts = y_model.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Classification requires at least two classes.")
    n_splits = min(int(cv_folds), int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Each class needs at least two samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    if task == "one_vs_rest":
        scoring["roc_auc"] = "roc_auc"
    elif len(class_counts) > 2:
        scoring["roc_auc"] = "roc_auc_ovr_weighted"

    model = _classification_pipeline(classifier, random_state=random_state)
    try:
        cv_result = cross_validate(
            model,
            x,
            y_model,
            cv=cv,
            scoring=scoring,
            error_score=np.nan,
        )
    except Exception:
        scoring.pop("roc_auc", None)
        cv_result = cross_validate(
            model,
            x,
            y_model,
            cv=cv,
            scoring=scoring,
            error_score=np.nan,
        )

    out: dict[str, float] = {"cv_folds_used": float(n_splits)}
    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc"]:
        values = cv_result.get(f"test_{metric}")
        if values is None:
            out[f"{metric}_mean"] = np.nan
            out[f"{metric}_std"] = np.nan
        else:
            values = np.asarray(values, dtype=float)
            out[f"{metric}_mean"] = float(np.nanmean(values))
            out[f"{metric}_std"] = float(np.nanstd(values, ddof=0))

    primary_metric = (
        "roc_auc"
        if pd.notna(out.get("roc_auc_mean", np.nan))
        else "balanced_accuracy"
    )
    out["primary_score_mean"] = float(out[f"{primary_metric}_mean"])
    out["primary_score_std"] = float(out[f"{primary_metric}_std"])
    out["primary_metric"] = primary_metric  # type: ignore[assignment]
    return out


def _module_feature_sets(
    x: pd.DataFrame,
    cluster_map: dict[str, list[str]],
    *,
    include_all: bool = True,
) -> dict[str, list[str]]:
    module_features = {
        module: [feature for feature in features if feature in x.columns]
        for module, features in cluster_map.items()
    }
    if include_all:
        module_features["All"] = list(x.columns)
    return module_features


def _score_fitted_classifier(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: str,
) -> dict[str, float | str]:
    y_pred = model.predict(x_test)
    out: dict[str, float | str] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": np.nan,
    }

    try:
        proba = model.predict_proba(x_test)
        if task == "one_vs_rest":
            if len(np.unique(y_test)) >= 2 and proba.shape[1] >= 2:
                out["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
        elif proba.shape[1] >= 2:
            classes = list(model.named_steps["classifier"].classes_)
            out["roc_auc"] = float(
                roc_auc_score(
                    y_test,
                    proba,
                    labels=classes,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except Exception:
        out["roc_auc"] = np.nan

    primary_metric = "roc_auc" if pd.notna(out["roc_auc"]) else "balanced_accuracy"
    out["primary_metric"] = primary_metric
    out["primary_score"] = float(out[primary_metric])
    return out


def run_module_stability_validation(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    *,
    first_column_as_sample_id: bool = True,
    max_missing_fraction: float = 0.5,
    task: str = "one_vs_rest",
    positive_label: str | None = None,
    classifier: str = "logistic_regression",
    cv_folds: int = 5,
    n_repeats: int = 20,
    include_all: bool = True,
    random_state: int = 123,
) -> dict[str, Any]:
    """Repeated CV stability of module ranks for condition classification."""
    dataset = prepare_module_classification_dataset(
        condition_tables,
        labels_df,
        first_column_as_sample_id=first_column_as_sample_id,
        max_missing_fraction=max_missing_fraction,
    )
    x: pd.DataFrame = dataset["x"]
    y: pd.Series = dataset["y"].astype(str)
    cluster_map: dict[str, list[str]] = dataset["cluster_map"]

    task = str(task)
    if task not in {"one_vs_rest", "multiclass"}:
        raise ValueError("task must be 'one_vs_rest' or 'multiclass'.")
    if task == "one_vs_rest":
        if positive_label is None:
            raise ValueError("positive_label is required for one-vs-rest stability.")
        positive_label = str(positive_label)
        if positive_label not in set(y):
            raise ValueError(f"Positive label {positive_label!r} was not found.")
        y_model = (y == positive_label).astype(int)
        task_label = f"{positive_label} vs Other"
    else:
        y_model = y
        task_label = "multiclass"

    y_model, _ = _model_y_for_classifier(y_model, classifier=classifier)
    class_counts = y_model.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Stability validation requires at least two classes.")
    n_splits = min(int(cv_folds), int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Each class needs at least two samples for repeated CV.")
    repeats = max(1, int(n_repeats))

    module_features = _module_feature_sets(
        x,
        cluster_map,
        include_all=bool(include_all),
    )
    split_rows: list[dict[str, Any]] = []
    split_index = 0
    for repeat_idx in range(repeats):
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(random_state) + repeat_idx,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y_model), start=1):
            split_index += 1
            fold_rows: list[dict[str, Any]] = []
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y_model.iloc[train_idx]
            y_test = y_model.iloc[test_idx]

            for module in sorted(module_features, key=lambda m: (m == "All", _module_sort_key(m))):
                features = module_features[module]
                base_row: dict[str, Any] = {
                    "split_id": split_index,
                    "repeat": repeat_idx + 1,
                    "fold": fold_idx,
                    "module": module,
                    "task": task_label,
                    "classifier": classifier,
                    "n_features": int(len(features)),
                    "train_samples": int(len(y_train)),
                    "test_samples": int(len(y_test)),
                }
                if not features:
                    base_row.update(
                        {
                            "status": "no usable features",
                            "primary_metric": "",
                            "primary_score": np.nan,
                            "accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "f1_macro": np.nan,
                            "roc_auc": np.nan,
                        }
                    )
                else:
                    try:
                        model = _classification_pipeline(
                            classifier,
                            random_state=int(random_state) + repeat_idx,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=ConvergenceWarning)
                            model.fit(x_train.loc[:, features], y_train)
                        scores = _score_fitted_classifier(
                            model,
                            x_test.loc[:, features],
                            y_test,
                            task=task,
                        )
                    except Exception as exc:
                        scores = {
                            "status": f"failed: {exc}",
                            "primary_metric": "",
                            "primary_score": np.nan,
                            "accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "f1_macro": np.nan,
                            "roc_auc": np.nan,
                        }
                    else:
                        scores["status"] = "ok"
                    base_row.update(scores)
                fold_rows.append(base_row)

            fold_df = pd.DataFrame(fold_rows)
            ok_mask = fold_df["status"].eq("ok") & pd.notna(fold_df["primary_score"])
            if ok_mask.any():
                ranked = fold_df.loc[ok_mask].sort_values(
                    ["primary_score", "n_features"],
                    ascending=[False, False],
                )
                rank_map = {
                    str(module): rank
                    for rank, module in enumerate(ranked["module"].astype(str), start=1)
                }
                fold_df["split_rank"] = fold_df["module"].astype(str).map(rank_map)
            else:
                fold_df["split_rank"] = np.nan
            split_rows.extend(fold_df.to_dict("records"))

    split_scores = pd.DataFrame(split_rows)
    summary_rows: list[dict[str, Any]] = []
    for module, group in split_scores.groupby("module", sort=False):
        ok = group[group["status"].eq("ok") & pd.notna(group["primary_score"])].copy()
        values = ok["primary_score"].astype(float).to_numpy()
        ranks = ok["split_rank"].astype(float).to_numpy()
        if values.size == 0:
            summary_rows.append(
                {
                    "module": module,
                    "status": "no successful splits",
                    "primary_metric": "",
                    "primary_score_mean": np.nan,
                    "primary_score_std": np.nan,
                    "primary_score_p025": np.nan,
                    "primary_score_p975": np.nan,
                    "mean_rank": np.nan,
                    "rank_std": np.nan,
                    "rank_1_frequency": np.nan,
                    "top_2_frequency": np.nan,
                    "top_3_frequency": np.nan,
                    "successful_splits": 0,
                    "total_splits": int(group["split_id"].nunique()),
                    "n_features": int(group["n_features"].max()) if "n_features" in group else 0,
                }
            )
            continue

        primary_metric = (
            ok["primary_metric"].mode().iloc[0]
            if not ok["primary_metric"].dropna().empty
            else "primary_score"
        )
        summary_rows.append(
            {
                "module": module,
                "status": "ok",
                "primary_metric": str(primary_metric),
                "primary_score_mean": float(np.mean(values)),
                "primary_score_std": float(np.std(values, ddof=0)),
                "primary_score_p025": float(np.quantile(values, 0.025)),
                "primary_score_p975": float(np.quantile(values, 0.975)),
                "mean_rank": float(np.nanmean(ranks)),
                "rank_std": float(np.nanstd(ranks, ddof=0)),
                "rank_1_frequency": float(np.mean(ranks == 1)),
                "top_2_frequency": float(np.mean(ranks <= 2)),
                "top_3_frequency": float(np.mean(ranks <= 3)),
                "successful_splits": int(len(ok)),
                "total_splits": int(group["split_id"].nunique()),
                "n_features": int(group["n_features"].max()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["rank_1_frequency", "primary_score_mean", "mean_rank"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    summary_df.insert(0, "stability_rank", np.arange(1, len(summary_df) + 1))

    return {
        "summary": summary_df,
        "split_scores": split_scores,
        "dataset": {
            "sample_summary": dataset["sample_summary"],
            "feature_summary": dataset["feature_summary"],
            "diagnostics": dataset["diagnostics"],
        },
        "task": task,
        "positive_label": positive_label,
        "cv_folds_used": int(n_splits),
        "n_repeats": int(repeats),
    }


def run_module_classification_validation(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    *,
    first_column_as_sample_id: bool = True,
    max_missing_fraction: float = 0.5,
    task: str = "one_vs_rest",
    positive_label: str | None = None,
    classifier: str = "logistic_regression",
    cv_folds: int = 5,
    random_state: int = 123,
) -> dict[str, Any]:
    """Score each FunClu module as a condition classifier.

    ``task='one_vs_rest'`` is the main topology-Hub validation mode: it asks
    whether the topology Hub module for one condition is also the strongest
    classifier for that condition.
    """
    dataset = prepare_module_classification_dataset(
        condition_tables,
        labels_df,
        first_column_as_sample_id=first_column_as_sample_id,
        max_missing_fraction=max_missing_fraction,
    )
    x: pd.DataFrame = dataset["x"]
    y: pd.Series = dataset["y"]
    cluster_map: dict[str, list[str]] = dataset["cluster_map"]

    task = str(task)
    if task not in {"one_vs_rest", "multiclass"}:
        raise ValueError("task must be 'one_vs_rest' or 'multiclass'.")

    if task == "one_vs_rest":
        if positive_label is None:
            raise ValueError("positive_label is required for one-vs-rest classification.")
        positive_label = str(positive_label)
        if positive_label not in set(y.astype(str)):
            raise ValueError(f"Positive label {positive_label!r} was not found.")
        y_model = (y.astype(str) == positive_label).astype(int)
        task_label = f"{positive_label} vs Other"
    else:
        y_model = y.astype(str)
        task_label = "multiclass"

    module_features = {
        module: [feature for feature in features if feature in x.columns]
        for module, features in cluster_map.items()
    }
    module_features["All"] = list(x.columns)

    rows: list[dict[str, Any]] = []
    for module in sorted(module_features, key=lambda m: (m == "All", _module_sort_key(m))):
        features = module_features[module]
        base_row: dict[str, Any] = {
            "module": module,
            "task": task_label,
            "classifier": classifier,
            "n_features": int(len(features)),
            "n_samples": int(len(y_model)),
            "n_classes": int(y_model.nunique()),
        }
        if len(features) == 0:
            base_row.update(
                {
                    "status": "no usable features",
                    "primary_metric": "",
                    "primary_score_mean": np.nan,
                    "primary_score_std": np.nan,
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "balanced_accuracy_mean": np.nan,
                    "balanced_accuracy_std": np.nan,
                    "f1_macro_mean": np.nan,
                    "f1_macro_std": np.nan,
                    "roc_auc_mean": np.nan,
                    "roc_auc_std": np.nan,
                    "cv_folds_used": np.nan,
                }
            )
        else:
            try:
                scores = _cross_validate_classifier(
                    x.loc[:, features],
                    y_model,
                    classifier=classifier,
                    cv_folds=int(cv_folds),
                    random_state=int(random_state),
                    task=task,
                )
            except Exception as exc:
                scores = {
                    "status": f"failed: {exc}",
                    "primary_metric": "",
                    "primary_score_mean": np.nan,
                    "primary_score_std": np.nan,
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "balanced_accuracy_mean": np.nan,
                    "balanced_accuracy_std": np.nan,
                    "f1_macro_mean": np.nan,
                    "f1_macro_std": np.nan,
                    "roc_auc_mean": np.nan,
                    "roc_auc_std": np.nan,
                    "cv_folds_used": np.nan,
                }
            else:
                scores["status"] = "ok"
            base_row.update(scores)
        rows.append(base_row)

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values(
        ["primary_score_mean", "n_features"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    scores_df.insert(0, "module_rank", np.arange(1, len(scores_df) + 1))

    return {
        "scores": scores_df,
        "dataset": {
            "sample_summary": dataset["sample_summary"],
            "feature_summary": dataset["feature_summary"],
            "diagnostics": dataset["diagnostics"],
        },
        "task": task,
        "positive_label": positive_label,
    }


def run_module_single_feature_validation(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    *,
    modules: list[str],
    first_column_as_sample_id: bool = True,
    max_missing_fraction: float = 0.5,
    task: str = "one_vs_rest",
    positive_label: str | None = None,
    classifier: str = "logistic_regression",
    cv_folds: int = 5,
    random_state: int = 123,
    max_features_per_module: int | None = None,
) -> dict[str, Any]:
    """Score each single feature inside selected modules as a classifier.

    This follows the Feishu validation figure pattern: compare a module's
    whole-feature score against the scores obtained by using each feature in
    that module alone.
    """
    selected_modules = []
    seen_modules: set[str] = set()
    for module in modules:
        module = str(module)
        if module and module not in seen_modules:
            selected_modules.append(module)
            seen_modules.add(module)
    if not selected_modules:
        raise ValueError("At least one module is required.")

    dataset = prepare_module_classification_dataset(
        condition_tables,
        labels_df,
        first_column_as_sample_id=first_column_as_sample_id,
        max_missing_fraction=max_missing_fraction,
    )
    x: pd.DataFrame = dataset["x"]
    y: pd.Series = dataset["y"].astype(str)
    cluster_map: dict[str, list[str]] = dataset["cluster_map"]

    task = str(task)
    if task not in {"one_vs_rest", "multiclass"}:
        raise ValueError("task must be 'one_vs_rest' or 'multiclass'.")

    if task == "one_vs_rest":
        if positive_label is None:
            raise ValueError("positive_label is required for one-vs-rest validation.")
        positive_label = str(positive_label)
        if positive_label not in set(y):
            raise ValueError(f"Positive label {positive_label!r} was not found.")
        y_model = (y == positive_label).astype(int)
        task_label = f"{positive_label} vs Other"
    else:
        y_model = y
        task_label = "multiclass"

    max_features = (
        int(max_features_per_module)
        if max_features_per_module is not None and int(max_features_per_module) > 0
        else None
    )

    module_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    selected_features: dict[str, list[str]] = {}
    for module in selected_modules:
        if module not in cluster_map:
            raise ValueError(f"Module {module!r} was not found in FunClu labels.")

        features = [feature for feature in cluster_map[module] if feature in x.columns]
        if max_features is not None and len(features) > max_features:
            variances = x.loc[:, features].var(axis=0, skipna=True).sort_values(
                ascending=False
            )
            features = [str(feature) for feature in variances.head(max_features).index]
        selected_features[module] = [str(feature) for feature in features]

        module_base: dict[str, Any] = {
            "module": module,
            "task": task_label,
            "classifier": classifier,
            "n_features": int(len(features)),
            "n_samples": int(len(y_model)),
            "n_classes": int(pd.Series(y_model).nunique()),
        }
        if not features:
            module_scores = {
                "status": "no usable features",
                "primary_metric": "",
                "primary_score_mean": np.nan,
                "primary_score_std": np.nan,
                "accuracy_mean": np.nan,
                "accuracy_std": np.nan,
                "balanced_accuracy_mean": np.nan,
                "balanced_accuracy_std": np.nan,
                "f1_macro_mean": np.nan,
                "f1_macro_std": np.nan,
                "roc_auc_mean": np.nan,
                "roc_auc_std": np.nan,
                "cv_folds_used": np.nan,
            }
        else:
            try:
                module_scores = _cross_validate_classifier(
                    x.loc[:, features],
                    y_model,
                    classifier=classifier,
                    cv_folds=int(cv_folds),
                    random_state=int(random_state),
                    task=task,
                )
            except Exception as exc:
                module_scores = {
                    "status": f"failed: {exc}",
                    "primary_metric": "",
                    "primary_score_mean": np.nan,
                    "primary_score_std": np.nan,
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "balanced_accuracy_mean": np.nan,
                    "balanced_accuracy_std": np.nan,
                    "f1_macro_mean": np.nan,
                    "f1_macro_std": np.nan,
                    "roc_auc_mean": np.nan,
                    "roc_auc_std": np.nan,
                    "cv_folds_used": np.nan,
                }
            else:
                module_scores["status"] = "ok"
        module_base.update(module_scores)
        module_rows.append(module_base)

        for feature in features:
            feature_base: dict[str, Any] = {
                "module": module,
                "feature": str(feature),
                "task": task_label,
                "classifier": classifier,
                "n_features": 1,
                "n_samples": int(len(y_model)),
                "n_classes": int(pd.Series(y_model).nunique()),
                "module_primary_score_mean": module_base.get(
                    "primary_score_mean",
                    np.nan,
                ),
                "module_primary_score_std": module_base.get(
                    "primary_score_std",
                    np.nan,
                ),
                "module_primary_metric": module_base.get("primary_metric", ""),
                "module_n_features": int(len(features)),
            }
            try:
                feature_scores = _cross_validate_classifier(
                    x.loc[:, [feature]],
                    y_model,
                    classifier=classifier,
                    cv_folds=int(cv_folds),
                    random_state=int(random_state),
                    task=task,
                )
            except Exception as exc:
                feature_scores = {
                    "status": f"failed: {exc}",
                    "primary_metric": "",
                    "primary_score_mean": np.nan,
                    "primary_score_std": np.nan,
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "balanced_accuracy_mean": np.nan,
                    "balanced_accuracy_std": np.nan,
                    "f1_macro_mean": np.nan,
                    "f1_macro_std": np.nan,
                    "roc_auc_mean": np.nan,
                    "roc_auc_std": np.nan,
                    "cv_folds_used": np.nan,
                }
            else:
                feature_scores["status"] = "ok"
            feature_base.update(feature_scores)
            feature_rows.append(feature_base)

    module_scores_df = pd.DataFrame(module_rows)
    if not module_scores_df.empty:
        module_scores_df = module_scores_df.sort_values(
            ["primary_score_mean", "n_features"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        module_scores_df.insert(
            0,
            "module_single_feature_rank",
            np.arange(1, len(module_scores_df) + 1),
        )

    feature_scores_df = pd.DataFrame(feature_rows)
    if not feature_scores_df.empty:
        feature_scores_df = feature_scores_df.sort_values(
            ["module", "primary_score_mean", "feature"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)
        feature_scores_df.insert(
            0,
            "feature_rank_within_module",
            feature_scores_df.groupby("module").cumcount() + 1,
        )

    return {
        "feature_scores": feature_scores_df,
        "module_scores": module_scores_df,
        "dataset": {
            "sample_summary": dataset["sample_summary"],
            "feature_summary": dataset["feature_summary"],
            "diagnostics": dataset["diagnostics"],
        },
        "selected_features": selected_features,
        "context": {
            "modules": selected_modules,
            "task": task,
            "task_label": task_label,
            "positive_label": positive_label if positive_label is not None else "",
            "classifier": classifier,
            "cv_folds": int(cv_folds),
            "max_features_per_module": max_features,
        },
    }


def run_intra_module_feature_importance(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    *,
    module: str,
    first_column_as_sample_id: bool = True,
    max_missing_fraction: float = 0.5,
    positive_label: str | None = None,
    classifier: str = "logistic_regression",
    cv_folds: int = 5,
    random_state: int = 123,
    n_repeats: int = 20,
) -> dict[str, Any]:
    """Rank features inside one FunClu module for one-vs-rest validation.

    This is the intra-cluster counterpart of module-level classification:
    after a topology/ML workflow points to a candidate Hub module, the model
    asks which features inside that module carry the strongest condition signal.
    """
    if positive_label is None:
        raise ValueError("positive_label is required for intra-module validation.")

    dataset = prepare_module_classification_dataset(
        condition_tables,
        labels_df,
        first_column_as_sample_id=first_column_as_sample_id,
        max_missing_fraction=max_missing_fraction,
    )
    x: pd.DataFrame = dataset["x"]
    y: pd.Series = dataset["y"].astype(str)
    cluster_map: dict[str, list[str]] = dataset["cluster_map"]

    module = str(module)
    positive_label = str(positive_label)
    if positive_label not in set(y):
        raise ValueError(f"Positive label {positive_label!r} was not found.")
    if module not in cluster_map:
        raise ValueError(f"Module {module!r} was not found in FunClu labels.")

    features = [feature for feature in cluster_map[module] if feature in x.columns]
    if not features:
        raise ValueError(f"Module {module!r} has no usable features after filtering.")

    y_model = (y == positive_label).astype(int)
    x_module = x.loc[:, features].copy()
    module_scores = _cross_validate_classifier(
        x_module,
        y_model,
        classifier=classifier,
        cv_folds=int(cv_folds),
        random_state=int(random_state),
        task="one_vs_rest",
    )

    model = _classification_pipeline(classifier, random_state=int(random_state))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_module, y_model)

    repeats = max(1, int(n_repeats))
    try:
        perm = permutation_importance(
            model,
            x_module,
            y_model,
            scoring="roc_auc",
            n_repeats=repeats,
            random_state=int(random_state),
            n_jobs=1,
        )
        perm_mean = perm.importances_mean
        perm_std = perm.importances_std
    except Exception:
        perm_mean = np.full(len(features), np.nan)
        perm_std = np.full(len(features), np.nan)

    estimator = model.named_steps["classifier"]
    coefficients = np.full(len(features), np.nan, dtype=float)
    embedded_importance = np.full(len(features), np.nan, dtype=float)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 2 and coef.shape[0] >= 1:
            coefficients = coef[0]
    if hasattr(estimator, "feature_importances_"):
        embedded_importance = np.asarray(estimator.feature_importances_, dtype=float)

    pos_mask = y == positive_label
    positive_means = x_module.loc[pos_mask].mean(axis=0, skipna=True)
    other_means = x_module.loc[~pos_mask].mean(axis=0, skipna=True)
    missing_fraction = x_module.isna().mean(axis=0)

    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(features):
        coefficient = float(coefficients[idx])
        mean_positive = float(positive_means.loc[feature])
        mean_other = float(other_means.loc[feature])
        mean_difference = mean_positive - mean_other
        if pd.notna(coefficient):
            if coefficient > 0:
                direction = f"higher -> {positive_label}"
            elif coefficient < 0:
                direction = f"higher -> other"
            else:
                direction = "neutral"
        elif mean_difference > 0:
            direction = f"higher in {positive_label}"
        elif mean_difference < 0:
            direction = "higher in other"
        else:
            direction = "neutral"

        rows.append(
            {
                "module": module,
                "feature": str(feature),
                "positive_condition": positive_label,
                "classifier": classifier,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient) if pd.notna(coefficient) else np.nan,
                "embedded_importance": float(embedded_importance[idx]),
                "permutation_importance_mean": float(perm_mean[idx]),
                "permutation_importance_std": float(perm_std[idx]),
                "mean_positive": mean_positive,
                "mean_other": mean_other,
                "mean_difference": mean_difference,
                "abs_mean_difference": abs(mean_difference),
                "missing_fraction": float(missing_fraction.loc[feature]),
                "direction": direction,
            }
        )

    importance_df = pd.DataFrame(rows)
    importance_df = importance_df.sort_values(
        [
            "permutation_importance_mean",
            "abs_coefficient",
            "embedded_importance",
            "abs_mean_difference",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    importance_df.insert(0, "feature_rank", np.arange(1, len(importance_df) + 1))

    module_scores.update(
        {
            "module": module,
            "positive_label": positive_label,
            "classifier": classifier,
            "n_features": int(len(features)),
            "n_samples": int(len(y_model)),
        }
    )

    return {
        "importance": importance_df,
        "module_score": module_scores,
        "dataset": {
            "sample_summary": dataset["sample_summary"],
            "feature_summary": dataset["feature_summary"],
            "diagnostics": dataset["diagnostics"],
        },
        "selected_features": features,
    }


def run_intra_node_edge_feature_validation(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    module: str,
    sample_col: str,
    source_col: str,
    target_col: str,
    weight_col: str,
    condition_col: str | None = None,
    first_column_as_sample_id: bool = True,
    max_missing_fraction: float = 0.5,
    task: str = "one_vs_rest",
    positive_label: str | None = None,
    classifier: str = "logistic_regression",
    cv_folds: int = 5,
    random_state: int = 123,
    max_edges: int | None = 5000,
) -> dict[str, Any]:
    """Compare intra-module node features, edge features, and their combination.

    ``edge_table`` must be sample-level or instance-level long data. Required
    columns are sample id, source, target, and weight. Each edge is pivoted to a
    column named ``source->target`` and aligned to the condition CSV samples.
    """
    dataset = prepare_module_classification_dataset(
        condition_tables,
        labels_df,
        first_column_as_sample_id=first_column_as_sample_id,
        max_missing_fraction=max_missing_fraction,
    )
    x: pd.DataFrame = dataset["x"]
    y: pd.Series = dataset["y"].astype(str)
    cluster_map: dict[str, list[str]] = dataset["cluster_map"]

    module = str(module)
    if module not in cluster_map:
        raise ValueError(f"Module {module!r} was not found in FunClu labels.")
    node_features = [feature for feature in cluster_map[module] if feature in x.columns]
    if not node_features:
        raise ValueError(f"Module {module!r} has no usable node features.")

    required_cols = [sample_col, source_col, target_col, weight_col]
    missing_cols = [col for col in required_cols if col not in edge_table.columns]
    if missing_cols:
        raise ValueError(f"Edge table is missing required columns: {missing_cols}")

    task = str(task)
    if task not in {"one_vs_rest", "multiclass"}:
        raise ValueError("task must be 'one_vs_rest' or 'multiclass'.")
    if task == "one_vs_rest":
        if positive_label is None:
            raise ValueError("positive_label is required for one-vs-rest validation.")
        positive_label = str(positive_label)
        if positive_label not in set(y):
            raise ValueError(f"Positive label {positive_label!r} was not found.")
        y_model = (y == positive_label).astype(int)
        task_label = f"{positive_label} vs Other"
    else:
        y_model = y
        task_label = "multiclass"

    sample_info = dataset["sample_info"].copy()
    if len(sample_info) != len(x):
        raise ValueError("Sample metadata does not align with the ML matrix.")
    sample_info["ml_index"] = x.index.astype(str)
    sample_info["sample_id"] = sample_info["sample_id"].astype(str)
    sample_info["condition"] = sample_info["condition"].astype(str)

    use_condition_alignment = bool(condition_col) and condition_col in edge_table.columns
    if use_condition_alignment:
        sample_info["_edge_align_key"] = (
            sample_info["condition"] + "\x1f" + sample_info["sample_id"]
        )
        if sample_info["_edge_align_key"].duplicated().any():
            duplicated = sample_info.loc[
                sample_info["_edge_align_key"].duplicated(),
                ["condition", "sample_id"],
            ].head(5).to_dict("records")
            raise ValueError(
                "Condition + sample_id must be unique to align sample-level "
                f"edge features. Duplicated examples: {duplicated}"
            )
    elif sample_info["sample_id"].duplicated().any():
        duplicated = sample_info.loc[
            sample_info["sample_id"].duplicated(),
            "sample_id",
        ].head(5).tolist()
        raise ValueError(
            "Sample IDs must be unique to align sample-level edge features. "
            f"Duplicated examples: {duplicated}. If sample IDs repeat across "
            "conditions, include a condition column in the edge table and select it."
        )
    else:
        sample_info["_edge_align_key"] = sample_info["sample_id"]

    module_node_set = set(str(feature) for feature in node_features)
    edges = edge_table.loc[:, required_cols].copy()
    if use_condition_alignment:
        edges[condition_col] = edge_table[condition_col].astype(str)
    edges[sample_col] = edges[sample_col].astype(str)
    edges[source_col] = edges[source_col].astype(str)
    edges[target_col] = edges[target_col].astype(str)
    edges[weight_col] = pd.to_numeric(edges[weight_col], errors="coerce")
    edges = edges.dropna(subset=[sample_col, source_col, target_col, weight_col])
    edges = edges[
        edges[source_col].isin(module_node_set)
        & edges[target_col].isin(module_node_set)
    ].copy()
    if edges.empty:
        raise ValueError(
            f"No sample-level edges remained for module {module!r}. Make sure "
            "source/target names match FunClu feature names."
        )
    edges["edge_feature"] = edges[source_col] + "->" + edges[target_col]

    if use_condition_alignment:
        edges["_edge_align_key"] = (
            edges[condition_col].astype(str) + "\x1f" + edges[sample_col].astype(str)
        )
    else:
        edges["_edge_align_key"] = edges[sample_col].astype(str)

    valid_sample_keys = set(sample_info["_edge_align_key"])
    edges = edges[edges["_edge_align_key"].isin(valid_sample_keys)].copy()
    if edges.empty:
        raise ValueError(
            "No edge rows matched the uploaded condition sample IDs. Check the "
            "sample_id column, and if selected, the condition column in the edge table."
        )

    edge_wide = edges.pivot_table(
        index="_edge_align_key",
        columns="edge_feature",
        values=weight_col,
        aggfunc="mean",
        fill_value=0.0,
    )
    edge_wide.columns = [str(col) for col in edge_wide.columns]

    if max_edges is not None and int(max_edges) > 0 and edge_wide.shape[1] > int(max_edges):
        variances = edge_wide.var(axis=0, ddof=0).sort_values(ascending=False)
        edge_wide = edge_wide.loc[:, variances.head(int(max_edges)).index].copy()

    sample_order = sample_info.set_index("_edge_align_key").loc[:, "ml_index"]
    edge_x = edge_wide.reindex(sample_order.index).fillna(0.0)
    edge_x.index = sample_order.to_numpy()
    edge_x = edge_x.reindex(index=x.index).fillna(0.0)
    if edge_x.empty or edge_x.shape[1] == 0:
        raise ValueError("No usable edge features remained after alignment.")

    node_x = x.loc[:, node_features].copy()
    combined_x = pd.concat(
        [
            node_x.add_prefix("node:"),
            edge_x.add_prefix("edge:"),
        ],
        axis=1,
    )

    feature_sets = {
        "Node only": node_x,
        "Edge only": edge_x,
        "Node + Edge": combined_x,
    }
    rows: list[dict[str, Any]] = []
    for feature_set, matrix in feature_sets.items():
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "module": module,
            "task": task_label,
            "classifier": classifier,
            "n_features": int(matrix.shape[1]),
            "n_samples": int(matrix.shape[0]),
        }
        try:
            scores = _cross_validate_classifier(
                matrix,
                y_model,
                classifier=classifier,
                cv_folds=int(cv_folds),
                random_state=int(random_state),
                task=task,
            )
        except Exception as exc:
            scores = {
                "status": f"failed: {exc}",
                "primary_metric": "",
                "primary_score_mean": np.nan,
                "primary_score_std": np.nan,
                "accuracy_mean": np.nan,
                "accuracy_std": np.nan,
                "balanced_accuracy_mean": np.nan,
                "balanced_accuracy_std": np.nan,
                "f1_macro_mean": np.nan,
                "f1_macro_std": np.nan,
                "roc_auc_mean": np.nan,
                "roc_auc_std": np.nan,
                "cv_folds_used": np.nan,
            }
        else:
            scores["status"] = "ok"
        row.update(scores)
        rows.append(row)

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values(
        ["primary_score_mean", "n_features"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    scores_df.insert(0, "validation_rank", np.arange(1, len(scores_df) + 1))

    edge_summary = (
        edges.groupby("edge_feature", sort=False)
        .agg(
            observed_samples=(sample_col, "nunique"),
            mean_weight=(weight_col, "mean"),
            std_weight=(weight_col, "std"),
            nonzero_edges=(weight_col, lambda values: int((values != 0).sum())),
        )
        .reset_index()
    )
    edge_summary["abs_mean_weight"] = edge_summary["mean_weight"].abs()
    edge_summary = edge_summary.sort_values(
        ["observed_samples", "abs_mean_weight"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    return {
        "scores": scores_df,
        "edge_summary": edge_summary,
        "node_features": node_features,
        "edge_features": list(edge_x.columns),
        "dataset": {
            "sample_summary": dataset["sample_summary"],
            "feature_summary": dataset["feature_summary"],
            "diagnostics": dataset["diagnostics"],
        },
        "context": {
            "module": module,
            "task": task,
            "task_label": task_label,
            "positive_label": positive_label if positive_label is not None else "",
            "classifier": classifier,
            "cv_folds": int(cv_folds),
            "sample_col": sample_col,
            "source_col": source_col,
            "target_col": target_col,
            "weight_col": weight_col,
            "condition_col": condition_col if condition_col is not None else "",
            "matched_edge_rows": int(len(edges)),
            "matched_edge_features": int(edge_x.shape[1]),
            "matched_samples": int(edge_x.shape[0]),
            "max_edges": int(max_edges) if max_edges is not None else 0,
            "alignment_mode": (
                "condition + sample_id" if use_condition_alignment else "sample_id"
            ),
        },
    }


def predict_unknown_condition_samples(
    condition_tables: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    unknown_table: pd.DataFrame,
    *,
    module: str = "All",
    first_column_as_sample_id: bool = True,
    unknown_first_column_as_sample_id: bool | None = None,
    max_missing_fraction: float = 0.5,
    task: str = "multiclass",
    positive_label: str | None = None,
    classifier: str = "logistic_regression",
    random_state: int = 123,
) -> dict[str, Any]:
    """Train a condition classifier and predict labels for unknown samples."""
    if unknown_first_column_as_sample_id is None:
        unknown_first_column_as_sample_id = first_column_as_sample_id

    dataset = prepare_module_classification_dataset(
        condition_tables,
        labels_df,
        first_column_as_sample_id=first_column_as_sample_id,
        max_missing_fraction=max_missing_fraction,
    )
    x: pd.DataFrame = dataset["x"]
    y: pd.Series = dataset["y"].astype(str)
    cluster_map: dict[str, list[str]] = dataset["cluster_map"]

    module = str(module)
    if module == "All":
        features = list(x.columns)
    else:
        if module not in cluster_map:
            raise ValueError(f"Module {module!r} was not found in FunClu labels.")
        features = [feature for feature in cluster_map[module] if feature in x.columns]
    if not features:
        raise ValueError(f"Prediction feature set {module!r} has no usable features.")

    task = str(task)
    if task not in {"one_vs_rest", "multiclass"}:
        raise ValueError("task must be 'one_vs_rest' or 'multiclass'.")
    if task == "one_vs_rest":
        if positive_label is None:
            raise ValueError("positive_label is required for one-vs-rest prediction.")
        positive_label = str(positive_label)
        if positive_label not in set(y):
            raise ValueError(f"Positive label {positive_label!r} was not found.")
        y_model = (y == positive_label).astype(int)
        class_labels = ["Other", positive_label]
    else:
        y_model = y
        class_labels = sorted(y.unique().tolist())

    unknown_numeric, sample_ids = _prepare_condition_frame(
        unknown_table,
        first_column_as_sample_id=bool(unknown_first_column_as_sample_id),
        condition="unknown",
    )
    if unknown_numeric.empty:
        raise ValueError("Unknown sample CSV is empty after reading numeric columns.")
    unknown_x = unknown_numeric.reindex(columns=features)

    model = _classification_pipeline(classifier, random_state=int(random_state))
    y_fit, y_encoder = _model_y_for_classifier(y_model, classifier=classifier)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x.loc[:, features], y_fit)

    pred = model.predict(unknown_x)
    proba = model.predict_proba(unknown_x)
    estimator = model.named_steps["classifier"]
    model_classes = list(estimator.classes_)

    rows: list[dict[str, Any]] = []
    for row_idx, sample_id in enumerate(sample_ids):
        if task == "one_vs_rest":
            positive_class_index = model_classes.index(1) if 1 in model_classes else -1
            positive_prob = (
                float(proba[row_idx, positive_class_index])
                if positive_class_index >= 0
                else np.nan
            )
            predicted_label = positive_label if int(pred[row_idx]) == 1 else "Other"
            confidence = (
                max(positive_prob, 1.0 - positive_prob)
                if pd.notna(positive_prob)
                else np.nan
            )
            row = {
                "sample_id": str(sample_id),
                "predicted_label": predicted_label,
                "confidence": float(confidence) if pd.notna(confidence) else np.nan,
                f"prob_{positive_label}": positive_prob,
                "prob_Other": 1.0 - positive_prob if pd.notna(positive_prob) else np.nan,
            }
        else:
            if y_encoder is not None:
                decoded_classes = [
                    str(value)
                    for value in y_encoder.inverse_transform(
                        np.asarray(model_classes, dtype=int)
                    )
                ]
                predicted_label = str(
                    y_encoder.inverse_transform(
                        np.asarray([int(pred[row_idx])], dtype=int)
                    )[0]
                )
            else:
                decoded_classes = [str(cls) for cls in model_classes]
                predicted_label = str(pred[row_idx])
            class_prob = {
                f"prob_{decoded_classes[col_idx]}": float(proba[row_idx, col_idx])
                for col_idx in range(len(decoded_classes))
            }
            predicted_prob_key = f"prob_{predicted_label}"
            row = {
                "sample_id": str(sample_id),
                "predicted_label": predicted_label,
                "confidence": class_prob.get(predicted_prob_key, np.nan),
            }
            row.update(class_prob)
        rows.append(row)

    prediction_df = pd.DataFrame(rows)
    missing_features = [feature for feature in features if feature not in unknown_numeric.columns]
    extra_features = [feature for feature in unknown_numeric.columns if feature not in features]
    context = {
        "module": module,
        "classifier": classifier,
        "task": task,
        "positive_label": positive_label if positive_label is not None else "",
        "n_training_samples": int(len(y_model)),
        "n_unknown_samples": int(len(prediction_df)),
        "n_features_used": int(len(features)),
        "classes": class_labels,
        "missing_features_in_unknown": missing_features,
        "extra_features_ignored": extra_features,
    }
    return {
        "predictions": prediction_df,
        "context": context,
        "dataset": {
            "sample_summary": dataset["sample_summary"],
            "feature_summary": dataset["feature_summary"],
            "diagnostics": dataset["diagnostics"],
        },
    }


def topology_hubs_from_adjacencies(
    topology_adjs: dict[str, pd.DataFrame],
    *,
    rank_metric: str = "out_degree",
    edge_threshold: float = 0.0,
) -> pd.DataFrame:
    """Return the top inter-cluster Hub for each NetRecon condition adjacency."""
    rows: list[dict[str, Any]] = []
    for key, adj in topology_adjs.items():
        if not str(key).startswith("inter_cluster/"):
            continue
        condition = str(key).split("/", 1)[1]
        hub_table = hub_table_from_adjacency(
            adj,
            edge_threshold=float(edge_threshold),
            rank_metric=str(rank_metric),
        )
        if hub_table.empty:
            continue
        top = hub_table.iloc[0]
        rows.append(
            {
                "condition": condition,
                "topology_key": key,
                "topology_hub": str(top["variable"]),
                "rank_metric": rank_metric,
                "rank_value": float(top[rank_metric]) if rank_metric in top else np.nan,
                "out_degree": int(top["out_degree"]) if "out_degree" in top else np.nan,
                "out_strength": float(top["out_strength"]) if "out_strength" in top else np.nan,
            }
        )
    return pd.DataFrame(rows)


def get_funclu_ml_matrix(
    funclu_export: dict[str, Any],
    *,
    layer: str,
    condition: str,
    cluster: str = "",
    data_source: str = "quasi_dynamic",
) -> pd.DataFrame:
    """Return an internal variable-by-sample matrix from a FunClu-K export."""
    layer = str(layer)
    data_source = str(data_source)
    if data_source not in {"quasi_dynamic", "curve_sample"}:
        raise ValueError("data_source must be 'quasi_dynamic' or 'curve_sample'.")

    if layer == "inter_cluster":
        source_map = (
            funclu_export["center_responses"]
            if data_source == "quasi_dynamic"
            else funclu_export["centers"]
        )
        if condition not in source_map:
            raise ValueError(f"Condition {condition!r} not found in inter-cluster data.")
        df = source_map[condition]
    elif layer == "intra_cluster":
        source_map = (
            funclu_export["member_responses"]
            if data_source == "quasi_dynamic"
            else funclu_export["members"]
        )
        if condition not in source_map or cluster not in source_map[condition]:
            raise ValueError(
                f"Condition/cluster {condition!r}/{cluster!r} not found in intra-cluster data."
            )
        df = source_map[condition][cluster]
    else:
        raise ValueError("layer must be 'inter_cluster' or 'intra_cluster'.")

    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric.index = _make_unique_labels(list(numeric.index))
    numeric.columns = _make_unique_labels(list(numeric.columns))
    matrix = numeric.T
    matrix.index.name = "variable"
    matrix = matrix.dropna(axis=1, how="all")
    if matrix.isna().any().any():
        row_medians = matrix.median(axis=1)
        matrix = matrix.T.fillna(row_medians).T
    keep_variable = matrix.std(axis=1, ddof=0) > 1e-12
    matrix = matrix.loc[keep_variable].copy()
    return matrix


def load_idopnetwork_adjacencies(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Read adjacency matrices from a NetRecon export ZIP."""
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("adjacency_matrix.csv"):
                label = name[: -len("/adjacency_matrix.csv")] if "/" in name else "single_layer"
                out[label] = _read_zip_csv(zf, name, index_col=0)
    if not out:
        raise ValueError("No adjacency_matrix.csv files were found in the NetRecon ZIP.")
    return out


def matching_topology_key(*, layer: str, condition: str, cluster: str = "") -> str:
    """Return the NetRecon adjacency key matching a FunClu layer selection."""
    if layer == "inter_cluster":
        return f"inter_cluster/{condition}"
    if layer == "intra_cluster":
        return f"intra_cluster/{condition}/{cluster}"
    return "single_layer"


def hub_table_from_adjacency(
    adj_df: pd.DataFrame,
    *,
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
    rank_metric: str = "out_degree",
) -> pd.DataFrame:
    """Compute an IDOP-style topology Hub table from an adjacency matrix."""
    adj = adj_df.copy()
    adj.index = adj.index.astype(str)
    adj.columns = adj.columns.astype(str)
    if set(adj.index) != set(adj.columns):
        raise ValueError("Adjacency matrix index and columns must contain the same nodes.")
    adj = adj.loc[list(adj.index), list(adj.index)].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    values = adj.to_numpy(dtype=float, copy=True)
    if not include_self_edges:
        np.fill_diagonal(values, 0.0)
    edge_mat = pd.DataFrame(values, index=adj.index, columns=adj.columns)
    active = edge_mat.abs() > float(edge_threshold)

    hub = pd.DataFrame(
        {
            "variable": list(adj.index),
            "out_degree": active.sum(axis=1).astype(int).reindex(adj.index).values,
            "in_degree": active.sum(axis=0).astype(int).reindex(adj.index).values,
            "out_strength": edge_mat.abs().sum(axis=1).reindex(adj.index).values,
            "in_strength": edge_mat.abs().sum(axis=0).reindex(adj.index).values,
            "promoting_out_strength": edge_mat.clip(lower=0.0).sum(axis=1).reindex(adj.index).values,
            "inhibiting_out_strength": (-edge_mat.clip(upper=0.0)).sum(axis=1).reindex(adj.index).values,
            "node_weight": np.diag(adj.to_numpy(dtype=float, copy=False)),
        }
    )
    hub["total_degree"] = hub["out_degree"] + hub["in_degree"]
    hub["total_strength"] = hub["out_strength"] + hub["in_strength"]

    if rank_metric not in hub.columns:
        rank_metric = "out_degree"
    sort_cols = []
    for col in [rank_metric, "out_degree", "out_strength"]:
        if col in hub.columns and col not in sort_cols:
            sort_cols.append(col)
    hub = hub.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    hub.insert(0, "rank", np.arange(1, len(hub) + 1))
    return hub


def compare_hub_tables(
    ml_hub_df: pd.DataFrame,
    topology_hub_df: pd.DataFrame,
    *,
    rank_metric: str = "out_degree",
    topology_top_n: int = 3,
    ml_top_n: int = 20,
) -> dict[str, Any]:
    """Compare ML Hub ranking against topology Hub ranking on common nodes."""
    def _unique_cols(cols: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for col in cols:
            if col not in seen:
                out.append(col)
                seen.add(col)
        return out

    metric = str(rank_metric)
    if metric not in ml_hub_df.columns:
        metric = "out_degree"
    if metric not in topology_hub_df.columns:
        metric = "out_degree"

    ml_ranked = ml_hub_df.copy()
    topology_ranked = topology_hub_df.copy()
    ml_ranked["variable"] = ml_ranked["variable"].astype(str)
    topology_ranked["variable"] = topology_ranked["variable"].astype(str)

    ml_sort_cols = _unique_cols(
        [col for col in [metric, "out_degree", "out_strength"] if col in ml_ranked.columns]
    )
    topology_sort_cols = _unique_cols(
        [col for col in [metric, "out_degree", "out_strength"] if col in topology_ranked.columns]
    )
    ml_ranked = ml_ranked.sort_values(ml_sort_cols, ascending=False).reset_index(drop=True)
    topology_ranked = topology_ranked.sort_values(
        topology_sort_cols,
        ascending=False,
    ).reset_index(drop=True)
    ml_ranked["ml_rank"] = np.arange(1, len(ml_ranked) + 1)
    topology_ranked["topology_rank"] = np.arange(1, len(topology_ranked) + 1)

    ml_cols = _unique_cols(["variable", "ml_rank"] + [
        col for col in [metric, "out_degree", "out_strength", "hub_score", "target_r2"] if col in ml_ranked.columns
    ])
    topology_cols = _unique_cols(["variable", "topology_rank"] + [
        col for col in [metric, "out_degree", "out_strength"] if col in topology_ranked.columns
    ])
    detail = ml_ranked[ml_cols].merge(
        topology_ranked[topology_cols],
        on="variable",
        how="inner",
        suffixes=("_ml", "_topology"),
    )

    topology_top = set(topology_ranked.head(int(topology_top_n))["variable"])
    ml_top = set(ml_ranked.head(int(ml_top_n))["variable"])
    overlap = sorted(topology_top & ml_top)
    detail["in_topology_top_n"] = detail["variable"].isin(topology_top)
    detail["in_ml_top_n"] = detail["variable"].isin(ml_top)
    detail["rank_delta_ml_minus_topology"] = detail["ml_rank"] - detail["topology_rank"]

    ml_metric_col = f"{metric}_ml" if f"{metric}_ml" in detail.columns else metric
    topology_metric_col = (
        f"{metric}_topology" if f"{metric}_topology" in detail.columns else metric
    )
    if (
        len(detail) >= 2
        and ml_metric_col in detail.columns
        and topology_metric_col in detail.columns
        and detail[ml_metric_col].nunique(dropna=True) > 1
        and detail[topology_metric_col].nunique(dropna=True) > 1
    ):
        spearman = detail[ml_metric_col].corr(detail[topology_metric_col], method="spearman")
    else:
        spearman = np.nan

    return {
        "detail": detail.sort_values("topology_rank").reset_index(drop=True),
        "summary": {
            "rank_metric": metric,
            "common_nodes": int(len(detail)),
            "topology_top_n": int(topology_top_n),
            "ml_top_n": int(ml_top_n),
            "overlap_count": int(len(overlap)),
            "overlap_nodes": ", ".join(overlap),
            "overlap_rate_vs_topology_top_n": (
                float(len(overlap) / max(1, int(topology_top_n)))
            ),
            "spearman_metric_correlation": (
                float(spearman) if pd.notna(spearman) else np.nan
            ),
        },
    }


def _make_unique_labels(labels: list[Any]) -> list[str]:
    """Return stable, non-empty labels while preserving the original order."""
    counts: dict[str, int] = {}
    out: list[str] = []

    for i, label in enumerate(labels, start=1):
        base = str(label).strip()
        if not base or base.lower() == "nan":
            base = f"Var_{i}"

        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            out.append(base)
        else:
            out.append(f"{base}_{counts[base]}")

    return out


def prepare_variable_sample_matrix(
    raw_df: pd.DataFrame,
    *,
    input_orientation: str = "samples_by_variables",
    first_column_as_names: bool = True,
    max_missing_fraction: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean an uploaded table into a variable-by-sample matrix.

    Parameters
    ----------
    raw_df:
        DataFrame read directly from the uploaded CSV.
    input_orientation:
        ``"samples_by_variables"`` means CSV rows are samples/plants and CSV
        columns are variables/genes. ``"variables_by_samples"`` means CSV rows
        are variables/genes and CSV columns are samples.
    first_column_as_names:
        When true, the first CSV column is used as row names and excluded from
        numeric modeling. In ``samples_by_variables`` mode these are sample IDs;
        in ``variables_by_samples`` mode these are variable/gene names.
    max_missing_fraction:
        Variables with a larger fraction of missing sample values are dropped.

    Returns
    -------
    tuple
        ``(matrix, diagnostics)`` where ``matrix`` has variables on rows and
        samples on columns.
    """
    if raw_df.empty:
        raise ValueError("The uploaded CSV is empty.")

    orientation = str(input_orientation).lower()
    if orientation not in {"samples_by_variables", "variables_by_samples"}:
        raise ValueError(
            "input_orientation must be 'samples_by_variables' or 'variables_by_samples'."
        )

    if first_column_as_names:
        if raw_df.shape[1] < 2:
            raise ValueError("At least one row-name column and one numeric column are required.")
        row_labels = _make_unique_labels(raw_df.iloc[:, 0].tolist())
        numeric_part = raw_df.iloc[:, 1:].copy()
    else:
        row_labels = [
            f"Row_{i}" for i in range(1, raw_df.shape[0] + 1)
        ]
        numeric_part = raw_df.copy()

    raw_numeric = numeric_part.apply(pd.to_numeric, errors="coerce")
    raw_numeric.index = row_labels

    if orientation == "variables_by_samples":
        variable_labels = (
            row_labels
            if first_column_as_names
            else [f"Var_{i}" for i in range(1, raw_numeric.shape[0] + 1)]
        )
        numeric = raw_numeric.copy()
        numeric.index = variable_labels
    else:
        variable_labels = _make_unique_labels(list(raw_numeric.columns))
        sample_labels = (
            row_labels
            if first_column_as_names
            else [f"Sample_{i}" for i in range(1, raw_numeric.shape[0] + 1)]
        )
        raw_numeric.index = sample_labels
        raw_numeric.columns = variable_labels
        numeric = raw_numeric.T

    raw_variables, raw_samples = numeric.shape
    raw_missing_values = int(numeric.isna().sum().sum())

    all_missing_samples = numeric.columns[numeric.isna().all(axis=0)].tolist()
    if all_missing_samples:
        numeric = numeric.drop(columns=all_missing_samples)

    if numeric.shape[1] < 3:
        raise ValueError("At least 3 usable samples are required for ML Hub identification.")

    missing_fraction = numeric.isna().mean(axis=1)
    keep_missing = missing_fraction <= float(max_missing_fraction)
    dropped_missing = int((~keep_missing).sum())
    numeric = numeric.loc[keep_missing].copy()

    if numeric.empty:
        raise ValueError("No variables remain after missing-value filtering.")

    missing_to_impute = int(numeric.isna().sum().sum())
    row_medians = numeric.median(axis=1)
    numeric = numeric.T.fillna(row_medians).T

    std = numeric.std(axis=1, ddof=0)
    keep_variable = std > 1e-12
    dropped_constant = int((~keep_variable).sum())
    numeric = numeric.loc[keep_variable].copy()

    if numeric.shape[0] < 2:
        raise ValueError("At least 2 non-constant variables are required.")

    sample_labels = _make_unique_labels(list(numeric.columns))
    numeric.columns = sample_labels

    diagnostics = {
        "raw_variables": int(raw_variables),
        "raw_samples": int(raw_samples),
        "variables": int(numeric.shape[0]),
        "samples": int(numeric.shape[1]),
        "raw_missing_values": int(raw_missing_values),
        "missing_values_imputed": int(missing_to_impute),
        "dropped_variables_missing": dropped_missing,
        "dropped_variables_constant": dropped_constant,
        "dropped_samples_all_missing": int(len(all_missing_samples)),
        "max_missing_fraction": float(max_missing_fraction),
        "input_orientation": orientation,
    }
    return numeric, diagnostics


def _select_variables(
    matrix: pd.DataFrame,
    *,
    max_variables: int | None,
    strategy: str,
) -> tuple[pd.DataFrame, list[str]]:
    if max_variables is None or int(max_variables) <= 0 or matrix.shape[0] <= int(max_variables):
        return matrix.copy(), []

    max_variables = int(max_variables)
    strategy = str(strategy).lower()

    if strategy == "first":
        selected = list(matrix.index[:max_variables])
    elif strategy == "mean_abs":
        scores = matrix.abs().mean(axis=1).sort_values(ascending=False)
        selected = scores.head(max_variables).index.tolist()
    else:
        scores = matrix.var(axis=1, ddof=0).sort_values(ascending=False)
        selected = scores.head(max_variables).index.tolist()

    dropped = [x for x in matrix.index.tolist() if x not in set(selected)]
    return matrix.loc[selected].copy(), dropped


def _standardize_rows(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, ddof=0, keepdims=True)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    return (values - means) / stds


def _positive_scale(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    max_value = float(values.max()) if len(values) else 0.0
    if max_value <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return values / max_value


def _summarize_hubs(
    edges_df: pd.DataFrame,
    variables: list[str],
    model_scores: pd.DataFrame,
    *,
    random_state: int = 123,
) -> pd.DataFrame:
    base = pd.DataFrame({"variable": variables})

    if edges_df.empty:
        for col in [
            "out_strength",
            "in_strength",
            "total_strength",
            "out_degree",
            "in_degree",
            "promoting_out_strength",
            "inhibiting_out_strength",
            "pagerank",
            "betweenness",
            "hub_score",
        ]:
            base[col] = 0.0
        base["total_degree"] = 0
        if not model_scores.empty:
            target_r2 = model_scores.set_index("target")["r2"].reindex(variables).fillna(0.0)
            base["target_r2"] = target_r2.clip(lower=0.0).values
        else:
            base["target_r2"] = 0.0
        base["role"] = "No edges"
        base.insert(0, "rank", np.arange(1, len(base) + 1))
        return base

    out_strength = edges_df.groupby("source")["abs_weight"].sum()
    in_strength = edges_df.groupby("target")["abs_weight"].sum()
    out_degree = edges_df.groupby("source")["target"].nunique()
    in_degree = edges_df.groupby("target")["source"].nunique()

    promoting = edges_df[edges_df["coefficient"] > 0]
    inhibiting = edges_df[edges_df["coefficient"] < 0]
    promoting_out = promoting.groupby("source")["abs_weight"].sum()
    inhibiting_out = inhibiting.groupby("source")["abs_weight"].sum()

    hub = base.set_index("variable")
    hub["out_strength"] = out_strength.reindex(variables).fillna(0.0)
    hub["in_strength"] = in_strength.reindex(variables).fillna(0.0)
    hub["total_strength"] = hub["out_strength"] + hub["in_strength"]
    hub["out_degree"] = out_degree.reindex(variables).fillna(0).astype(int)
    hub["in_degree"] = in_degree.reindex(variables).fillna(0).astype(int)
    hub["total_degree"] = hub["out_degree"] + hub["in_degree"]
    hub["promoting_out_strength"] = promoting_out.reindex(variables).fillna(0.0)
    hub["inhibiting_out_strength"] = inhibiting_out.reindex(variables).fillna(0.0)

    target_r2 = model_scores.set_index("target")["r2"].reindex(variables).fillna(0.0)
    hub["target_r2"] = target_r2.clip(lower=0.0)

    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    for row in edges_df.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            weight=float(row.abs_weight),
            distance=1.0 / (float(row.abs_weight) + 1e-12),
        )

    if graph.number_of_edges() > 0:
        pagerank = nx.pagerank(graph, weight="weight")
        if graph.number_of_nodes() > 450:
            k = min(120, graph.number_of_nodes())
            betweenness = nx.betweenness_centrality(
                graph,
                k=k,
                weight="distance",
                seed=random_state,
                normalized=True,
            )
        else:
            betweenness = nx.betweenness_centrality(
                graph,
                weight="distance",
                normalized=True,
            )
    else:
        pagerank = {node: 0.0 for node in variables}
        betweenness = {node: 0.0 for node in variables}

    hub["pagerank"] = pd.Series(pagerank).reindex(variables).fillna(0.0)
    hub["betweenness"] = pd.Series(betweenness).reindex(variables).fillna(0.0)

    hub["hub_score"] = (
        0.45 * _positive_scale(hub["out_strength"])
        + 0.25 * _positive_scale(hub["in_strength"])
        + 0.20 * _positive_scale(hub["pagerank"])
        + 0.10 * _positive_scale(hub["betweenness"])
    )

    def _role(row: pd.Series) -> str:
        if row["out_strength"] >= 1.25 * max(row["in_strength"], 1e-12):
            return "Driver Hub"
        if row["in_strength"] >= 1.25 * max(row["out_strength"], 1e-12):
            return "Receiver Hub"
        if row["total_strength"] > 0:
            return "Connector Hub"
        return "Peripheral"

    hub["role"] = hub.apply(_role, axis=1)
    hub = hub.reset_index()
    hub = hub.sort_values(
        ["hub_score", "total_strength", "out_strength"],
        ascending=False,
    ).reset_index(drop=True)
    hub.insert(0, "rank", np.arange(1, len(hub) + 1))
    return hub


def infer_signed_hub_network(
    matrix: pd.DataFrame,
    *,
    max_variables: int | None = 250,
    variable_selection: str = "variance",
    l1_ratio: float = 0.75,
    cv_folds: int = 5,
    coefficient_threshold: float = 0.03,
    max_edges: int | None = 5000,
    random_state: int = 123,
    max_iter: int = 20000,
) -> dict[str, Any]:
    """Infer a signed directed Hub network with target-wise ElasticNet models."""
    if matrix.shape[0] < 2:
        raise ValueError("At least 2 variables are required.")
    if matrix.shape[1] < 3:
        raise ValueError("At least 3 samples are required.")

    work_matrix, dropped_variables = _select_variables(
        matrix,
        max_variables=max_variables,
        strategy=variable_selection,
    )

    variables = work_matrix.index.astype(str).tolist()
    n_variables = len(variables)
    n_samples = int(work_matrix.shape[1])
    z_values = _standardize_rows(work_matrix)
    sample_by_variable = z_values.T

    l1_ratio = float(np.clip(l1_ratio, 0.01, 1.0))
    coefficient_threshold = max(0.0, float(coefficient_threshold))
    cv = min(max(2, int(cv_folds)), n_samples)

    edge_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for target_idx, target in enumerate(variables):
        feature_indices = [i for i in range(n_variables) if i != target_idx]
        source_names = [variables[i] for i in feature_indices]
        x = sample_by_variable[:, feature_indices]
        y = sample_by_variable[:, target_idx]

        if n_samples >= 4:
            model = ElasticNetCV(
                l1_ratio=l1_ratio,
                cv=cv,
                fit_intercept=False,
                max_iter=int(max_iter),
                random_state=int(random_state),
                selection="cyclic",
            )
        else:
            model = ElasticNet(
                alpha=0.05,
                l1_ratio=l1_ratio,
                fit_intercept=False,
                max_iter=int(max_iter),
                random_state=int(random_state),
                selection="cyclic",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x, y)

        prediction = model.predict(x)
        r2 = float(r2_score(y, prediction))
        rmse = float(np.sqrt(mean_squared_error(y, prediction)))
        alpha = float(getattr(model, "alpha_", getattr(model, "alpha", np.nan)))
        coefficients = np.asarray(model.coef_, dtype=float)

        active = np.where(np.abs(coefficients) >= coefficient_threshold)[0]
        for local_idx in active:
            coef = float(coefficients[local_idx])
            if abs(coef) <= 1e-12:
                continue
            source = source_names[int(local_idx)]
            edge_rows.append(
                {
                    "source": source,
                    "target": target,
                    "coefficient": coef,
                    "abs_weight": abs(coef),
                    "effect": "promote" if coef > 0 else "inhibit",
                    "target_r2": r2,
                    "target_rmse": rmse,
                    "alpha": alpha,
                }
            )

        model_rows.append(
            {
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "alpha": alpha,
                "selected_predictors": int(len(active)),
            }
        )

    edges_df = pd.DataFrame(edge_rows)
    if not edges_df.empty:
        edges_df = edges_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
        if max_edges is not None and int(max_edges) > 0 and len(edges_df) > int(max_edges):
            edges_df = edges_df.head(int(max_edges)).copy()

    model_scores = pd.DataFrame(model_rows).sort_values("r2", ascending=False).reset_index(drop=True)
    hub_scores = _summarize_hubs(
        edges_df,
        variables,
        model_scores,
        random_state=int(random_state),
    )
    adjacency = signed_edges_to_adjacency(edges_df, variables)

    possible_edges = max(1, n_variables * (n_variables - 1))
    metadata = {
        "variables_used": int(n_variables),
        "samples_used": int(n_samples),
        "variables_dropped_by_selection": int(len(dropped_variables)),
        "dropped_variable_names": dropped_variables,
        "edge_count": int(len(edges_df)),
        "network_density": float(len(edges_df) / possible_edges),
        "l1_ratio": float(l1_ratio),
        "cv_folds": int(cv),
        "coefficient_threshold": float(coefficient_threshold),
        "max_edges": None if max_edges is None else int(max_edges),
        "variable_selection": str(variable_selection),
    }

    return {
        "edges": edges_df,
        "hub_scores": hub_scores,
        "model_scores": model_scores,
        "adjacency": adjacency,
        "selected_matrix": work_matrix,
        "metadata": metadata,
    }


def signed_edges_to_adjacency(
    edges_df: pd.DataFrame,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Convert ``source -> target`` edge rows into an adjacency matrix."""
    if variables is None:
        if edges_df.empty:
            variables = []
        else:
            variables = sorted(set(edges_df["source"]).union(set(edges_df["target"])))

    adj = pd.DataFrame(0.0, index=list(variables), columns=list(variables))
    if edges_df.empty:
        return adj

    for row in edges_df.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        if source in adj.index and target in adj.columns:
            adj.loc[source, target] = float(row.coefficient)
    return adj
