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

import io
import zipfile
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


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


def _cross_validate_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    classifier: str,
    cv_folds: int,
    random_state: int,
    task: str,
) -> dict[str, float]:
    class_counts = y.value_counts()
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
            y,
            cv=cv,
            scoring=scoring,
            error_score=np.nan,
        )
    except Exception:
        scoring.pop("roc_auc", None)
        cv_result = cross_validate(
            model,
            x,
            y,
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
