"""
Unit tests for src/models/evaluate.py using isolated synthetic data.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.models.evaluate import (
    evaluate_model,
    aggregate_cv_metrics,
    UnseenClassError,
    _validate_probabilities,
)


class MockProbClassifier:
    """Mock classifier for testing probability validation edge cases."""

    def __init__(self, probas: np.ndarray, classes: np.ndarray) -> None:
        self.probas = probas
        self.classes_ = classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.probas


def test_evaluate_model_normal():
    """Test normal evaluation on synthetic 3-class classification data."""
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_classes=3,
        random_state=42,
    )
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X[:40], y[:40])

    metrics = evaluate_model(clf, X[40:], y[40:])

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "log_loss" in metrics
    assert "inference_time_sec" in metrics
    assert "inference_latency_ms_per_sample" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["inference_time_sec"] >= 0.0
    assert metrics["inference_latency_ms_per_sample"] >= 0.0


def test_evaluate_model_unseen_classes():
    """Test that UnseenClassError is raised when validation target contains unseen classes."""
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=3,
        n_redundant=1,
        n_classes=3,
        random_state=42,
    )
    # Fit only on classes 0 and 1
    train_mask = y != 2
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X[train_mask], y[train_mask])

    # Test set contains class 2
    X_test = X[y == 2]
    y_test = y[y == 2]

    with pytest.raises(UnseenClassError) as exc_info:
        evaluate_model(clf, X_test, y_test)

    assert 2 in exc_info.value.unseen_classes
    assert set(clf.classes_) == exc_info.value.model_classes


def test_validate_probabilities_dimensions():
    """Test that invalid probability shapes raise ValueError."""
    # 1D array
    with pytest.raises(ValueError, match="2-dimensional"):
        _validate_probabilities(np.array([0.5, 0.5]), n_expected_samples=1, n_expected_classes=2)

    # Sample count mismatch
    with pytest.raises(ValueError, match="expected sample count"):
        _validate_probabilities(
            np.array([[0.5, 0.5]]), n_expected_samples=2, n_expected_classes=2
        )

    # Class count mismatch
    with pytest.raises(ValueError, match="number of classes"):
        _validate_probabilities(
            np.array([[0.5, 0.5]]), n_expected_samples=1, n_expected_classes=3
        )


def test_validate_probabilities_values():
    """Test that NaN, negative, or un-normalized probabilities raise ValueError."""
    # NaN
    with pytest.raises(ValueError, match="NaN, null, or infinite"):
        _validate_probabilities(
            np.array([[np.nan, 0.5]]), n_expected_samples=1, n_expected_classes=2
        )

    # Negative values
    with pytest.raises(ValueError, match="negative values"):
        _validate_probabilities(
            np.array([[-0.2, 1.2]]), n_expected_samples=1, n_expected_classes=2
        )

    # Does not sum to 1.0
    with pytest.raises(ValueError, match="do not sum to 1.0"):
        _validate_probabilities(
            np.array([[0.3, 0.3]]), n_expected_samples=1, n_expected_classes=2
        )


def test_evaluate_model_catches_invalid_probabilities():
    """Test that evaluate_model rejects models returning invalid probabilities."""
    X = np.array([[1.0, 2.0]])
    y = np.array([0])
    classes = np.array([0, 1])

    # Model returning row sums that don't equal 1
    bad_clf = MockProbClassifier(probas=np.array([[0.2, 0.2]]), classes=classes)
    with pytest.raises(ValueError, match="do not sum to 1.0"):
        evaluate_model(bad_clf, X, y)


def test_aggregate_cv_metrics_all_valid():
    """Test aggregation when all folds have valid metrics."""
    fold_metrics = [
        {
            "fold": 1,
            "accuracy": 0.80,
            "macro_f1": 0.78,
            "log_loss": 0.40,
            "log_loss_valid": True,
            "train_time_sec": 0.10,
            "inference_time_sec": 0.01,
            "inference_latency_ms_per_sample": 0.1,
        },
        {
            "fold": 2,
            "accuracy": 0.90,
            "macro_f1": 0.88,
            "log_loss": 0.30,
            "log_loss_valid": True,
            "train_time_sec": 0.12,
            "inference_time_sec": 0.012,
            "inference_latency_ms_per_sample": 0.12,
        },
    ]

    summary = aggregate_cv_metrics(fold_metrics)

    assert summary["total_folds"] == 2
    assert summary["accuracy_mean"] == pytest.approx(0.85)
    assert summary["accuracy_std"] == pytest.approx(0.05)
    assert summary["log_loss_mean"] == pytest.approx(0.35)
    assert summary["log_loss_status"] == "ALL_FOLDS_VALID"
    assert summary["log_loss_invalid_folds"] == 0
    assert summary["log_loss_valid_folds"] == 2


def test_aggregate_cv_metrics_with_invalid_log_loss():
    """Test aggregation when a fold has an invalid log_loss value."""
    fold_metrics = [
        {
            "fold": 1,
            "accuracy": 0.80,
            "macro_f1": 0.78,
            "log_loss": 0.40,
            "log_loss_valid": True,
            "train_time_sec": 0.10,
            "inference_time_sec": 0.01,
            "inference_latency_ms_per_sample": 0.1,
        },
        {
            "fold": 2,
            "accuracy": 0.70,
            "macro_f1": 0.68,
            "log_loss": np.nan,
            "log_loss_valid": False,
            "log_loss_note": "Validation fold contains unseen classes: [3]",
            "train_time_sec": 0.11,
            "inference_time_sec": 0.011,
            "inference_latency_ms_per_sample": 0.11,
        },
    ]

    summary = aggregate_cv_metrics(fold_metrics)

    assert summary["total_folds"] == 2
    assert summary["accuracy_mean"] == pytest.approx(0.75)
    # Log loss mean must be NaN to avoid misleading aggregate results
    assert np.isnan(summary["log_loss_mean"])
    assert summary["log_loss_invalid_folds"] == 1
    assert summary["log_loss_valid_folds"] == 1
    assert "INVALID_FOLDS_DETECTED" in summary["log_loss_status"]
    assert len(summary["log_loss_invalid_reasons"]) == 1
    assert "unseen classes" in summary["log_loss_invalid_reasons"][0]
    # Check that valid mean is preserved for reference
    assert summary["log_loss_valid_mean"] == pytest.approx(0.40)
