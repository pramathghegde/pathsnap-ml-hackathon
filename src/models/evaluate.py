"""
Model evaluation module for PathSnap candidate classification.

Provides:
- evaluate_model: Times probability prediction, derives discrete class predictions,
  validates probability constraints, checks for unseen validation classes, and
  computes standard classification metrics (Accuracy, Macro F1, Log Loss, Latency).
- aggregate_cv_metrics: Aggregates fold-level metrics across GroupKFold splits,
  computing mean and standard deviation while transparently tracking invalid folds.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


class UnseenClassError(ValueError):
    """Raised when validation data contains classes absent from trained model classes."""

    def __init__(self, unseen_classes: Set[Any], model_classes: Set[Any]) -> None:
        self.unseen_classes = set(unseen_classes)
        self.model_classes = set(model_classes)
        super().__init__(
            f"Validation target contains classes absent from trained model classes. "
            f"Unseen classes: {sorted(list(self.unseen_classes))}. "
            f"Model classes: {sorted(list(self.model_classes))}."
        )


def _validate_probabilities(
    y_proba: np.ndarray,
    n_expected_samples: int,
    n_expected_classes: int,
    atol: float = 1e-3,
) -> None:
    """
    Validate that predicted probabilities satisfy mathematical constraints.

    Args:
        y_proba: Predicted probability matrix.
        n_expected_samples: Expected number of rows (samples).
        n_expected_classes: Expected number of columns (classes).
        atol: Absolute tolerance for row sum check.

    Raises:
        ValueError: If probability output violates shape, finiteness, non-negativity,
                    or row-sum unity constraints.
    """
    if not isinstance(y_proba, np.ndarray):
        y_proba = np.asarray(y_proba)

    if y_proba.ndim != 2:
        raise ValueError(
            f"Predicted probability array must be 2-dimensional; got shape {y_proba.shape}."
        )

    if y_proba.shape[0] != n_expected_samples:
        raise ValueError(
            f"Probability sample count ({y_proba.shape[0]}) does not match "
            f"expected sample count ({n_expected_samples})."
        )

    if y_proba.shape[1] != n_expected_classes:
        raise ValueError(
            f"Probability class count ({y_proba.shape[1]}) does not match "
            f"number of classes in model ({n_expected_classes})."
        )

    if not np.all(np.isfinite(y_proba)):
        raise ValueError("Predicted probabilities contain NaN, null, or infinite values.")

    if np.any(y_proba < 0.0):
        min_val = float(y_proba.min())
        raise ValueError(
            f"Predicted probabilities contain negative values (minimum found: {min_val})."
        )

    row_sums = y_proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(
            f"Predicted probabilities do not sum to 1.0 within tolerance {atol}. "
            f"Row sum min={float(row_sums.min()):.4f}, max={float(row_sums.max()):.4f}."
        )


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> Dict[str, Any]:
    """
    Evaluate a fitted candidate classification model or pipeline.

    Measures probability inference latency (`predict_proba`) required for downstream
    HMM emission scoring, verifies output probability validity, detects missing
    validation classes, and computes Accuracy, Macro F1, Log Loss, and Latency.

    Note: Latency measured here reflects pointwise classifier candidate scoring only,
    distinct from end-to-end trajectory reconstruction latency.

    Args:
        model: Fitted scikit-learn classifier or Pipeline exposing `classes_` and `predict_proba`.
        X_test: Test feature matrix (numpy array or pandas DataFrame).
        y_test: True test labels (numpy array, pandas Series, or list).

    Returns:
        Dictionary containing:
        - "accuracy": Top-1 accuracy score.
        - "macro_f1": Macro-averaged F1 score.
        - "log_loss": Multi-class / binary cross-entropy loss.
        - "inference_time_sec": Total elapsed time for predict_proba() in seconds.
        - "inference_latency_ms_per_sample": Average latency per sample in milliseconds.

    Raises:
        AttributeError: If model lacks `classes_` or `predict_proba`.
        UnseenClassError: If validation labels contain classes absent from training.
        ValueError: If probability matrix violates mathematical consistency.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model {type(model).__name__} does not implement 'predict_proba()'."
        )

    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model, "named_steps") and hasattr(
        model.named_steps.get("classifier", None), "classes_"
    ):
        classes = model.named_steps["classifier"].classes_
    else:
        raise AttributeError("The provided model does not expose a 'classes_' attribute.")

    y_test_arr = np.asarray(y_test)
    n_samples = len(y_test_arr)

    # 1. Measure inference latency exclusively around predict_proba()
    start_time = time.perf_counter()
    y_proba = model.predict_proba(X_test)
    inference_time_sec = time.perf_counter() - start_time

    # 2. Validate probability output consistency
    _validate_probabilities(
        y_proba=y_proba,
        n_expected_samples=n_samples,
        n_expected_classes=len(classes),
    )

    # 3. Derive predicted class labels from model.classes_ and argmax
    y_pred = classes[np.argmax(y_proba, axis=1)]

    # 4. Check for unseen validation classes before computing metrics
    unseen_classes = set(y_test_arr) - set(classes)
    if unseen_classes:
        raise UnseenClassError(
            unseen_classes=unseen_classes,
            model_classes=set(classes),
        )

    # 5. Compute evaluation metrics
    accuracy = float(accuracy_score(y_test_arr, y_pred))
    macro_f1 = float(f1_score(y_test_arr, y_pred, average="macro", zero_division=0))
    loss = float(log_loss(y_test_arr, y_proba, labels=classes))

    latency_ms_per_sample = (
        (inference_time_sec / n_samples) * 1000.0 if n_samples > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "log_loss": loss,
        "inference_time_sec": float(inference_time_sec),
        "inference_latency_ms_per_sample": float(latency_ms_per_sample),
    }


def aggregate_cv_metrics(
    fold_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate cross-validation fold metrics into summary statistics.

    Computes mean and standard deviation for:
    - accuracy
    - macro_f1
    - log_loss
    - train_time_sec
    - inference_time_sec
    - inference_latency_ms_per_sample

    Explicitly handles invalid log-loss folds without silently discarding them.
    If any fold has an invalid or missing log loss, `log_loss_mean` is reported as NaN,
    and detailed diagnostic fields (`log_loss_valid_folds`, `log_loss_invalid_folds`,
    `log_loss_status`, `log_loss_invalid_reasons`) are populated.

    Args:
        fold_metrics: List of metric dictionaries, one per cross-validation fold.

    Returns:
        Dictionary of aggregated mean and standard deviation statistics.
    """
    if not fold_metrics:
        raise ValueError("Cannot aggregate empty fold_metrics list.")

    total_folds = len(fold_metrics)
    summary: Dict[str, Any] = {
        "total_folds": total_folds,
    }

    # Standard metrics where values are expected to be numeric across all folds
    standard_keys = [
        "accuracy",
        "macro_f1",
        "train_time_sec",
        "inference_time_sec",
        "inference_latency_ms_per_sample",
    ]

    for key in standard_keys:
        values = [
            f[key]
            for f in fold_metrics
            if key in f and f[key] is not None and not np.isnan(f[key])
        ]
        if len(values) == total_folds:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=0))
        elif values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=0))
            summary[f"{key}_valid_folds"] = len(values)
        else:
            summary[f"{key}_mean"] = float(np.nan)
            summary[f"{key}_std"] = float(np.nan)

    # Rigorous log_loss handling
    valid_log_losses = []
    invalid_reasons = []

    for i, fold in enumerate(fold_metrics):
        is_valid = fold.get("log_loss_valid", True)
        val = fold.get("log_loss", None)

        if not is_valid or val is None or np.isnan(val):
            reason = fold.get(
                "log_loss_note",
                f"Fold {fold.get('fold', i + 1)} has invalid/missing log_loss.",
            )
            invalid_reasons.append(reason)
        else:
            valid_log_losses.append(float(val))

    n_valid = len(valid_log_losses)
    n_invalid = len(invalid_reasons)

    summary["log_loss_valid_folds"] = n_valid
    summary["log_loss_total_folds"] = total_folds
    summary["log_loss_invalid_folds"] = n_invalid

    if n_invalid == 0:
        summary["log_loss_mean"] = float(np.mean(valid_log_losses))
        summary["log_loss_std"] = float(np.std(valid_log_losses, ddof=0))
        summary["log_loss_status"] = "ALL_FOLDS_VALID"
        summary["log_loss_invalid_reasons"] = []
    else:
        # Do not silently exclude invalid folds; mark log_loss_mean as NaN
        summary["log_loss_mean"] = float(np.nan)
        summary["log_loss_std"] = float(np.nan)
        summary["log_loss_status"] = (
            f"INVALID_FOLDS_DETECTED: {n_invalid}/{total_folds} folds were invalid"
        )
        summary["log_loss_invalid_reasons"] = invalid_reasons
        # Also provide valid fold average for diagnostic inspection
        if valid_log_losses:
            summary["log_loss_valid_mean"] = float(np.mean(valid_log_losses))
            summary["log_loss_valid_std"] = float(np.std(valid_log_losses, ddof=0))
        else:
            summary["log_loss_valid_mean"] = float(np.nan)
            summary["log_loss_valid_std"] = float(np.nan)

    return summary


def compute_observation_ranking_metrics(
    df: pd.DataFrame,
    proba_col: str = "pred_proba",
    target_col: str = "is_match",
    group_cols: Sequence[str] = ("trip_id", "observation_id"),
    tie_breaker_col: Optional[str] = "perp_dist",
    ascending_tie_breaker: bool = True,
    total_observations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute observation-level candidate ranking metrics and diagnostics.

    Evaluates:
    1. Eligible Top-1 Accuracy: Accuracy on observations containing at least one true positive candidate.
    2. Overall Top-1 Accuracy: End-to-end accuracy across all observations, treating candidate misses as failures.
    3. Candidate Generation Recall: Proportion of evaluated observations that have at least one true positive candidate.
    4. Ranking Diagnostics: Breakdown of total, eligible, missed, and correctly matched observations.

    Args:
        df: DataFrame containing candidate rows with predictions, labels, and observation groupings.
        proba_col: Column name containing predicted positive class probability P(target=1).
        target_col: Column name containing binary ground truth label (1 if matching arc, 0 otherwise).
        group_cols: Sequence of column names uniquely identifying each GPS observation.
        tie_breaker_col: Optional column for deterministic tie-breaking (e.g. 'perp_dist').
        ascending_tie_breaker: Whether smaller values in tie_breaker_col are preferred (True for distance/rank).
        total_observations: Optional total observation count (e.g., if observations with 0 candidate rows were omitted).
                            If None, derived directly as unique observation groups in df.

    Returns:
        Dictionary containing ranking metrics and observation diagnostics:
        - "num_observations_total": Total evaluated observations.
        - "num_observations_eligible": Observations where true arc was in candidate set.
        - "num_candidate_misses": Observations where true arc was absent from candidate set.
        - "num_correct_top1": Eligible observations where Top-1 candidate matches ground truth.
        - "eligible_top1_accuracy": num_correct_top1 / num_observations_eligible.
        - "overall_top1_accuracy": num_correct_top1 / num_observations_total.
        - "candidate_generation_recall": num_observations_eligible / num_observations_total.

    Raises:
        ValueError: If df is empty, missing required columns, contains NaN/infinite probabilities,
                    or contains invalid non-binary targets.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a pandas DataFrame; got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty; cannot evaluate candidate ranking.")

    required_cols = list(group_cols) + [proba_col, target_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from DataFrame: {missing_cols}")

    # Validate probabilities
    probas = df[proba_col].to_numpy()
    if not np.all(np.isfinite(probas)):
        raise ValueError(f"Probability column '{proba_col}' contains NaN, null, or infinite values.")

    if np.any((probas < -1e-5) | (probas > 1.0 + 1e-4)):
        raise ValueError(
            f"Probability column '{proba_col}' contains values outside probability range [0, 1] "
            f"(min={float(np.min(probas)):.4f}, max={float(np.max(probas)):.4f})."
        )

    # Validate target labels
    targets = df[target_col].to_numpy()
    if not np.all(np.isfinite(targets)):
        raise ValueError(f"Target column '{target_col}' contains NaN or infinite values.")

    unique_targets = set(np.unique(targets))
    if not unique_targets.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Target column '{target_col}' must contain binary values in {{0, 1}}; found {unique_targets}."
        )

    # Prepare DataFrame for deterministic ranking
    keep_cols = list(group_cols) + [proba_col, target_col]
    sort_cols = list(group_cols) + [proba_col]
    ascending_list = [True] * len(group_cols) + [False]

    if tie_breaker_col is not None and tie_breaker_col in df.columns:
        keep_cols.append(tie_breaker_col)
        sort_cols.append(tie_breaker_col)
        ascending_list.append(ascending_tie_breaker)

    work_df = df[keep_cols].copy()
    # Stable sort by group, descending probability, then tie breaker
    work_df.sort_values(by=sort_cols, ascending=ascending_list, kind="mergesort", inplace=True)

    # Select Top-1 candidate per observation
    top1_df = work_df.drop_duplicates(subset=list(group_cols), keep="first")

    # Determine observation eligibility: observation has at least one candidate with target == 1
    obs_eligibility = df.groupby(list(group_cols), sort=False)[target_col].max().rename("_is_eligible")
    top1_merged = top1_df.merge(obs_eligibility, on=list(group_cols), how="left")

    n_obs_in_df = len(top1_merged)
    n_total_obs = (
        total_observations
        if (total_observations is not None and total_observations >= n_obs_in_df)
        else n_obs_in_df
    )

    is_eligible = (top1_merged["_is_eligible"] == 1).to_numpy()
    n_eligible_obs = int(np.sum(is_eligible))
    n_misses = int(n_total_obs - n_eligible_obs)

    # Correct Top-1 prediction: Top-1 candidate has target_col == 1
    top1_targets = top1_merged[target_col].to_numpy().astype(int)
    n_correct_top1 = int(np.sum((top1_targets == 1) & is_eligible))

    # Metric calculations with zero-division safety
    eligible_top1_acc = float(n_correct_top1 / n_eligible_obs) if n_eligible_obs > 0 else 0.0
    overall_top1_acc = float(n_correct_top1 / n_total_obs) if n_total_obs > 0 else 0.0
    cand_recall = float(n_eligible_obs / n_total_obs) if n_total_obs > 0 else 0.0

    return {
        "num_observations_total": n_total_obs,
        "num_observations_eligible": n_eligible_obs,
        "num_candidate_misses": n_misses,
        "num_correct_top1": n_correct_top1,
        "eligible_top1_accuracy": eligible_top1_acc,
        "overall_top1_accuracy": overall_top1_acc,
        "candidate_generation_recall": cand_recall,
    }


def evaluate_candidate_ranking(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "is_match",
    group_cols: Sequence[str] = ("trip_id", "observation_id"),
    tie_breaker_col: Optional[str] = "perp_dist",
    ascending_tie_breaker: bool = True,
    total_observations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate candidate-level classification and observation-level ranking for a fitted model.

    Args:
        model: Fitted estimator or Pipeline implementing predict_proba().
        df: Evaluation DataFrame containing feature columns, target, and group identifiers.
        feature_cols: List of numerical feature column names to feed to model.
        target_col: Binary target column name.
        group_cols: Column names identifying unique GPS observations.
        tie_breaker_col: Optional tie-breaking column name.
        ascending_tie_breaker: Whether smaller tie-breaker value is preferred.
        total_observations: Optional total observation count.

    Returns:
        Consolidated dictionary of candidate metrics, ranking metrics, and inference latency.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Model {type(model).__name__} does not implement 'predict_proba()'.")

    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model, "named_steps") and hasattr(
        model.named_steps.get("classifier", None), "classes_"
    ):
        classes = model.named_steps["classifier"].classes_
    else:
        raise AttributeError("The provided model does not expose a 'classes_' attribute.")

    # 1. Profile inference latency
    X_eval = df[feature_cols]
    t_start = time.perf_counter()
    y_proba = model.predict_proba(X_eval)
    infer_time_sec = time.perf_counter() - t_start

    # Validate probability matrix
    _validate_probabilities(
        y_proba=y_proba,
        n_expected_samples=len(df),
        n_expected_classes=len(classes),
    )

    # 2. Extract probability of positive class (label 1)
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    elif 1.0 in classes:
        pos_idx = int(np.where(classes == 1.0)[0][0])
    elif len(classes) == 2:
        pos_idx = 1
    else:
        pos_idx = 0

    scores = y_proba[:, pos_idx]
    y_true = df[target_col].to_numpy().astype(int)
    y_pred = (scores >= 0.5).astype(int)

    # 3. Candidate-level metrics
    cand_acc = float(accuracy_score(y_true, y_pred))
    cand_prec = float(precision_score(y_true, y_pred, zero_division=0))
    cand_rec = float(recall_score(y_true, y_pred, zero_division=0))
    cand_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        cand_log_loss = float(log_loss(y_true, y_proba, labels=classes))
    except Exception:
        cand_log_loss = float(np.nan)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 4. Observation-level ranking metrics
    scored_df = df[list(group_cols) + [target_col]].copy()
    scored_df["_pred_score"] = scores
    if tie_breaker_col is not None and tie_breaker_col in df.columns:
        scored_df[tie_breaker_col] = df[tie_breaker_col]

    ranking_metrics = compute_observation_ranking_metrics(
        df=scored_df,
        proba_col="_pred_score",
        target_col=target_col,
        group_cols=group_cols,
        tie_breaker_col=tie_breaker_col,
        ascending_tie_breaker=ascending_tie_breaker,
        total_observations=total_observations,
    )

    # 5. Latency calculations
    n_candidates = len(df)
    n_observations = ranking_metrics["num_observations_total"]
    lat_cand = float((infer_time_sec / n_candidates) * 1000.0) if n_candidates > 0 else 0.0
    lat_obs = float((infer_time_sec / n_observations) * 1000.0) if n_observations > 0 else 0.0

    result = {
        # Observation-level ranking metrics
        **ranking_metrics,
        # Candidate-level classification metrics
        "candidate_accuracy": cand_acc,
        "candidate_precision": cand_prec,
        "candidate_recall": cand_rec,
        "candidate_macro_f1": cand_macro_f1,
        "candidate_log_loss": cand_log_loss,
        "num_candidate_rows": n_candidates,
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        # Latency profiling
        "inference_time_sec": float(infer_time_sec),
        "inference_latency_ms_per_candidate": float(lat_cand),
        "inference_latency_ms_per_observation": float(lat_obs),
    }

    return result


def aggregate_ranking_cv_metrics(
    fold_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate cross-validation fold ranking metrics into mean and standard deviation summary statistics.
    """
    if not fold_metrics:
        raise ValueError("Cannot aggregate empty fold_metrics list.")

    total_folds = len(fold_metrics)
    summary: Dict[str, Any] = {"total_folds": total_folds}

    numeric_keys = [
        "eligible_top1_accuracy",
        "overall_top1_accuracy",
        "candidate_generation_recall",
        "candidate_macro_f1",
        "candidate_accuracy",
        "candidate_precision",
        "candidate_recall",
        "candidate_log_loss",
        "train_time_sec",
        "inference_time_sec",
        "inference_latency_ms_per_candidate",
        "inference_latency_ms_per_observation",
    ]

    for key in numeric_keys:
        values = [
            f[key]
            for f in fold_metrics
            if key in f and f[key] is not None and not np.isnan(f[key])
        ]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=0))
        else:
            summary[f"{key}_mean"] = float(np.nan)
            summary[f"{key}_std"] = float(np.nan)

    sum_keys = [
        "num_observations_total",
        "num_observations_eligible",
        "num_candidate_misses",
        "num_correct_top1",
        "num_candidate_rows",
    ]
    for key in sum_keys:
        values = [f[key] for f in fold_metrics if key in f and f[key] is not None]
        if values:
            summary[f"{key}_sum"] = int(np.sum(values))

    return summary

