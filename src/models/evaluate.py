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
from typing import Any, Dict, List, Optional, Set
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


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
