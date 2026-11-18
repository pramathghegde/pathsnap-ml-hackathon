"""
Training and cross-validation benchmarking pipeline for PathSnap candidate classifiers.

Responsibilities:
1. Multi-model candidate classifier benchmarking across five distinct model families.
2. Leakage-free GroupKFold cross-validation grouped strictly by trip_id.
3. Precomputing identical validation splits across all five model families.
4. Profiling training time and predict_proba() inference latency per fold.
5. Robust metric collection and handling of unseen validation classes.
6. Model selection via Macro F1, followed by fresh refitting on full permitted training data.
7. Serializing the final production model and experiment metadata sidecars.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from src.models.evaluate import (
    UnseenClassError,
    aggregate_cv_metrics,
    evaluate_model,
)
from src.models.model_factory import (
    SUPPORTED_MODELS,
    get_model,
    get_models,
)


def _compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hash of a pandas DataFrame."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


def _sanitize_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize model hyperparameters safely for JSON export."""
    sanitized = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized


def validate_training_inputs(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    feature_cols: Optional[List[str]] = None,
    n_splits: int = 5,
) -> List[str]:
    """
    Validate input data integrity for GroupKFold training.

    Args:
        df: Input pandas DataFrame containing features, target, and groups.
        target_col: Name of the target column.
        group_col: Name of the group column (e.g. 'trip_id').
        feature_cols: Optional list of feature column names. If None, inferred automatically.
        n_splits: Number of GroupKFold splits.

    Returns:
        List of resolved feature column names.

    Raises:
        ValueError: If validation checks fail (missing columns, missing values,
                    insufficient groups, insufficient classes).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    # Target checks
    if df[target_col].isna().any():
        n_na = int(df[target_col].isna().sum())
        raise ValueError(f"Target column '{target_col}' contains {n_na} missing/NaN values.")

    unique_classes = df[target_col].unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Target column '{target_col}' must contain at least 2 distinct classes; "
            f"found {len(unique_classes)}: {unique_classes}."
        )

    # Group checks
    if df[group_col].isna().any():
        n_na = int(df[group_col].isna().sum())
        raise ValueError(f"Group column '{group_col}' contains {n_na} missing/NaN values.")

    unique_groups = df[group_col].nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"Number of unique groups in '{group_col}' ({unique_groups}) is less than "
            f"n_splits ({n_splits}). GroupKFold requires at least {n_splits} groups."
        )

    # Resolve and validate feature columns
    if feature_cols is None:
        metadata_cols = {group_col, target_col, "point_idx", "candidate_arc_id", "timestamp"}
        resolved_features = [col for col in df.columns if col not in metadata_cols]
    else:
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(
                f"Specified feature columns missing from DataFrame: {missing_features}"
            )
        resolved_features = list(feature_cols)

    if not resolved_features:
        raise ValueError("No feature columns identified for model training.")

    cols_with_na = [
        col for col in resolved_features if df[col].isna().any()
    ]
    if cols_with_na:
        raise ValueError(f"Feature columns contain NaN values: {cols_with_na}")

    return resolved_features


def run_training_pipeline(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = "trip_id",
    feature_cols: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    class_weight: Optional[Any] = None,
    refit_final: bool = True,
    save_artifacts: bool = False,
    artifacts_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    selected_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute group-aware multi-model cross-validation and optional final model refit.

    Workflow:
    1. Validates inputs without assuming feature schema or label encodings.
    2. Precomputes identical GroupKFold splits across all 5 model families.
    3. Fits each model on each training fold, timing fit and inference execution.
    4. Evaluates fold performance with evaluate_model(); gracefully logs unseen validation
       classes without dropping folds from aggregated reports.
    5. Aggregates metrics (mean and std) using aggregate_cv_metrics().
    6. Selects top model via mean Macro F1 and checks for validation limitations.
    7. Refits fresh estimator on complete dataset (if refit_final is True).
    8. Persists final model (.joblib), metadata (.json), and summaries (.csv, .json).

    Args:
        df: Input training DataFrame.
        target_col: Target label column name.
        group_col: Group identifier column (defaults to 'trip_id').
        feature_cols: List of feature names, or None to infer automatically.
        n_splits: Number of GroupKFold partitions (default: 5).
        random_state: Seed for reproducible random states.
        class_weight: Class weighting configuration (e.g. 'balanced' or None).
        refit_final: Whether to refit the best model on the complete training set.
        save_artifacts: Whether to save artifacts to disk.
        artifacts_dir: Destination directory for model artifacts.
        results_dir: Destination directory for benchmark summaries.
        overwrite: Whether to overwrite existing artifact files.
        selected_models: Optional list of model names to run; defaults to all 5.

    Returns:
        Dictionary containing:
        - "models": Dict of per-model fold breakdowns and aggregated metrics.
        - "summary_table": pandas DataFrame of model comparison benchmarks.
        - "best_model_name": Name of the top model selected by Macro F1.
        - "best_cv_metrics": Aggregated CV metrics of the selected model.
        - "selection_warning": Note if the selected model had invalid/incomplete folds.
        - "final_model": Fitted final estimator instance (if refit_final is True).
        - "final_metadata": Metadata dictionary for the final fitted model.
    """
    # 1. Validate inputs and resolve feature columns
    resolved_features = validate_training_inputs(
        df=df,
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
        n_splits=n_splits,
    )

    X = df[resolved_features]
    y = df[target_col].to_numpy()
    groups = df[group_col].to_numpy()

    # 2. Precompute GroupKFold splits once to guarantee identical partitions across models
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups=groups))

    # Explicitly verify strict group separation across all folds
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise ValueError(
                f"Group leakage detected in fold {fold_idx + 1}: overlapping groups {overlap}"
            )

    # 3. Resolve model families to benchmark
    all_model_names = list(SUPPORTED_MODELS)
    if selected_models is not None:
        invalid_names = [m for m in selected_models if m not in SUPPORTED_MODELS]
        if invalid_names:
            raise ValueError(
                f"Unknown models requested: {invalid_names}. Supported: {list(SUPPORTED_MODELS)}"
            )
        target_model_names = [m for m in selected_models]
    else:
        target_model_names = all_model_names

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    # 4. Train and cross-evaluate each model on identical precomputed folds
    for model_name in target_model_names:
        fold_metrics: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Instantiate a fresh model instance per fold
            model = get_model(
                name=model_name,
                random_state=random_state,
                class_weight=class_weight,
            )

            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            # Measure training execution time
            t_train_start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time_sec = time.perf_counter() - t_train_start

            # Evaluate on validation split
            try:
                eval_metrics = evaluate_model(model, X_val, y_val)
                eval_metrics["log_loss_valid"] = True
                eval_metrics["log_loss_note"] = None
            except UnseenClassError as e:
                # Validation fold has classes absent from training fold:
                # Capture inference timing and compute non-probabilistic metrics (accuracy, macro F1)
                t_infer_start = time.perf_counter()
                y_proba = model.predict_proba(X_val)
                inference_time_sec = time.perf_counter() - t_infer_start

                classes = (
                    model.classes_
                    if hasattr(model, "classes_")
                    else model.named_steps["classifier"].classes_
                )
                y_pred = classes[np.argmax(y_proba, axis=1)]
                acc = float(accuracy_score(y_val, y_pred))
                f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
                n_val = len(y_val)
                latency_ms = (
                    (inference_time_sec / n_val) * 1000.0 if n_val > 0 else 0.0
                )

                eval_metrics = {
                    "accuracy": acc,
                    "macro_f1": f1,
                    "log_loss": float(np.nan),
                    "log_loss_valid": False,
                    "log_loss_note": (
                        f"Fold {fold_idx + 1} validation contains unseen classes: "
                        f"{sorted(list(e.unseen_classes))}. Log loss is undefined."
                    ),
                    "inference_time_sec": float(inference_time_sec),
                    "inference_latency_ms_per_sample": float(latency_ms),
                }

            eval_metrics["fold"] = fold_idx + 1
            eval_metrics["train_time_sec"] = float(train_time_sec)
            fold_metrics.append(eval_metrics)

        # 5. Aggregate metrics across folds
        aggregated = aggregate_cv_metrics(fold_metrics)
        all_results[model_name] = {
            "aggregate": aggregated,
            "folds": fold_metrics,
        }

        # Prepare summary row
        row = {"model_name": model_name}
        for k, v in aggregated.items():
            if not isinstance(v, list):
                row[k] = v
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 6. Model Selection via mean Macro F1
    # Sort models by macro_f1_mean descending
    ranked_models = sorted(
        all_results.keys(),
        key=lambda m: (
            all_results[m]["aggregate"].get("macro_f1_mean", -1.0)
            if not np.isnan(all_results[m]["aggregate"].get("macro_f1_mean", -1.0))
            else -1.0
        ),
        reverse=True,
    )
    best_model_name = ranked_models[0]
    best_aggregate = all_results[best_model_name]["aggregate"]

    selection_warning = None
    if best_aggregate.get("log_loss_invalid_folds", 0) > 0:
        selection_warning = (
            f"Selected model '{best_model_name}' has "
            f"{best_aggregate['log_loss_invalid_folds']}/{n_splits} folds with invalid log loss."
        )

    # 7. Final Model Refit (Permitted Full Training Set Only)
    final_model = None
    final_metadata = None

    if refit_final:
        final_model = get_model(
            name=best_model_name,
            random_state=random_state,
            class_weight=class_weight,
        )
        t_refit_start = time.perf_counter()
        final_model.fit(X, y)
        refit_time_sec = time.perf_counter() - t_refit_start

        classes = (
            final_model.classes_
            if hasattr(final_model, "classes_")
            else final_model.named_steps["classifier"].classes_
        )

        final_metadata = {
            "model_name": best_model_name,
            "hyperparameters": _sanitize_hyperparameters(final_model.get_params()),
            "feature_names": resolved_features,
            "target_classes": [str(c) for c in classes],
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_size": {"n_samples": int(len(df)), "n_features": len(resolved_features)},
            "dataset_hash": _compute_dataset_hash(df),
            "scikit_learn_version": sklearn.__version__,
            "cv_metrics": best_aggregate,
            "refit_time_sec": float(refit_time_sec),
            "selection_warning": selection_warning,
        }

    # 8. Persist Artifacts (if requested)
    if save_artifacts:
        target_artifacts_dir = Path(artifacts_dir or "artifacts/models")
        target_results_dir = Path(results_dir or "experiments/results")

        target_artifacts_dir.mkdir(parents=True, exist_ok=True)
        target_results_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_results_dir / "benchmark_summary.csv"
        json_path = target_results_dir / "benchmark_summary.json"

        if not overwrite:
            for p in (csv_path, json_path):
                if p.exists():
                    raise FileExistsError(
                        f"Artifact '{p}' already exists. Pass overwrite=True to replace."
                    )

        summary_df.to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "benchmark_summary": summary_rows,
                    "best_model": best_model_name,
                    "selection_warning": selection_warning,
                    "detailed_results": all_results,
                },
                f,
                indent=2,
            )

        if final_model is not None and final_metadata is not None:
            slug = best_model_name.lower().replace(" ", "_")
            model_file = target_artifacts_dir / f"{slug}_refitted.joblib"
            meta_file = target_artifacts_dir / f"{slug}_metadata.json"

            if not overwrite:
                for p in (model_file, meta_file):
                    if p.exists():
                        raise FileExistsError(
                            f"Artifact '{p}' already exists. Pass overwrite=True to replace."
                        )

            joblib.dump(final_model, model_file)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(final_metadata, f, indent=2)

    return {
        "models": all_results,
        "summary_table": summary_df,
        "best_model_name": best_model_name,
        "best_cv_metrics": best_aggregate,
        "selection_warning": selection_warning,
        "final_model": final_model,
        "final_metadata": final_metadata,
    }
