"""
Unit tests for observation-level candidate ranking evaluation in src/models/evaluate.py.

Covers:
- Correct Top-1 selection
- Incorrect Top-1 selection
- Multiple observations across multiple trips
- Candidate-generation misses (all candidates negative)
- Deterministic probability tie-breaking
- Invalid and missing probability values (NaN, Inf, out of bounds)
- Invalid non-binary targets
- Empty candidate DataFrame / empty groups
- Correct eligible and overall metric denominators
- Integration with evaluate_candidate_ranking() using a fitted model
- Cross-validation aggregation via aggregate_ranking_cv_metrics()
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import (
    aggregate_ranking_cv_metrics,
    compute_observation_ranking_metrics,
    evaluate_candidate_ranking,
)


def test_correct_top1_selection():
    """Verify that when the highest probability candidate is positive, Top-1 is correct."""
    df = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "observation_id": [0, 0, 0],
            "candidate_arc_id": [101, 102, 103],
            "pred_proba": [0.15, 0.75, 0.10],
            "is_match": [0, 1, 0],
            "perp_dist": [10.0, 2.5, 15.0],
        }
    )

    metrics = compute_observation_ranking_metrics(df)
    assert metrics["num_observations_total"] == 1
    assert metrics["num_observations_eligible"] == 1
    assert metrics["num_candidate_misses"] == 0
    assert metrics["num_correct_top1"] == 1
    assert metrics["eligible_top1_accuracy"] == pytest.approx(1.0)
    assert metrics["overall_top1_accuracy"] == pytest.approx(1.0)
    assert metrics["candidate_generation_recall"] == pytest.approx(1.0)


def test_incorrect_top1_selection():
    """Verify that when the highest probability candidate is negative, Top-1 is counted as incorrect."""
    df = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "observation_id": [0, 0, 0],
            "candidate_arc_id": [101, 102, 103],
            "pred_proba": [0.85, 0.10, 0.05],  # Wrong candidate 101 has highest score
            "is_match": [0, 1, 0],             # True candidate is 102
            "perp_dist": [8.0, 3.0, 12.0],
        }
    )

    metrics = compute_observation_ranking_metrics(df)
    assert metrics["num_observations_total"] == 1
    assert metrics["num_observations_eligible"] == 1
    assert metrics["num_candidate_misses"] == 0
    assert metrics["num_correct_top1"] == 0
    assert metrics["eligible_top1_accuracy"] == pytest.approx(0.0)
    assert metrics["overall_top1_accuracy"] == pytest.approx(0.0)
    assert metrics["candidate_generation_recall"] == pytest.approx(1.0)


def test_candidate_generation_miss():
    """Verify that observations without any positive candidate are classified as misses and fail overall Top-1."""
    df = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "observation_id": [0, 0],
            "candidate_arc_id": [201, 202],
            "pred_proba": [0.90, 0.10],
            "is_match": [0, 0],  # Candidate miss: no candidate corresponds to reference route
            "perp_dist": [45.0, 50.0],
        }
    )

    metrics = compute_observation_ranking_metrics(df)
    assert metrics["num_observations_total"] == 1
    assert metrics["num_observations_eligible"] == 0
    assert metrics["num_candidate_misses"] == 1
    assert metrics["num_correct_top1"] == 0
    # Eligible accuracy is 0.0 when eligible count is 0
    assert metrics["eligible_top1_accuracy"] == pytest.approx(0.0)
    assert metrics["overall_top1_accuracy"] == pytest.approx(0.0)
    assert metrics["candidate_generation_recall"] == pytest.approx(0.0)


def test_multiple_observations_and_trips_with_misses():
    """Verify ranking evaluation across multiple trips, multiple observations, and candidate misses."""
    records = [
        # Trip 1, Obs 0: Eligible, Correct Top-1
        {"trip_id": "T1", "observation_id": 0, "pred_proba": 0.8, "is_match": 1, "perp_dist": 2.0},
        {"trip_id": "T1", "observation_id": 0, "pred_proba": 0.2, "is_match": 0, "perp_dist": 15.0},
        # Trip 1, Obs 1: Eligible, Incorrect Top-1
        {"trip_id": "T1", "observation_id": 1, "pred_proba": 0.9, "is_match": 0, "perp_dist": 1.0},
        {"trip_id": "T1", "observation_id": 1, "pred_proba": 0.1, "is_match": 1, "perp_dist": 12.0},
        # Trip 2, Obs 0: Candidate Miss (no match)
        {"trip_id": "T2", "observation_id": 0, "pred_proba": 0.7, "is_match": 0, "perp_dist": 40.0},
        {"trip_id": "T2", "observation_id": 0, "pred_proba": 0.3, "is_match": 0, "perp_dist": 48.0},
        # Trip 2, Obs 1: Eligible, Correct Top-1
        {"trip_id": "T2", "observation_id": 1, "pred_proba": 0.65, "is_match": 1, "perp_dist": 4.0},
        {"trip_id": "T2", "observation_id": 1, "pred_proba": 0.35, "is_match": 0, "perp_dist": 20.0},
    ]
    df = pd.DataFrame(records)

    metrics = compute_observation_ranking_metrics(df)
    # Total observations = 4 (T1-0, T1-1, T2-0, T2-1)
    assert metrics["num_observations_total"] == 4
    # Eligible observations = 3 (T1-0, T1-1, T2-1)
    assert metrics["num_observations_eligible"] == 3
    # Candidate misses = 1 (T2-0)
    assert metrics["num_candidate_misses"] == 1
    # Correct Top-1 = 2 (T1-0 and T2-1)
    assert metrics["num_correct_top1"] == 2

    # Verify mathematical relationship: overall = eligible * recall
    assert metrics["eligible_top1_accuracy"] == pytest.approx(2 / 3)
    assert metrics["overall_top1_accuracy"] == pytest.approx(2 / 4)
    assert metrics["candidate_generation_recall"] == pytest.approx(3 / 4)
    assert metrics["overall_top1_accuracy"] == pytest.approx(
        metrics["eligible_top1_accuracy"] * metrics["candidate_generation_recall"]
    )


def test_deterministic_probability_ties():
    """Verify deterministic tie-breaking using perpendicular distance when probabilities tie."""
    df = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "observation_id": [0, 0, 0],
            "candidate_arc_id": [1, 2, 3],
            "pred_proba": [0.50, 0.50, 0.10],  # Candidate 1 and 2 tie on probability
            "is_match": [1, 0, 0],
            "perp_dist": [3.5, 8.0, 20.0],     # Candidate 1 has smaller distance (preferred)
        }
    )

    # Smaller perp_dist should win tie
    metrics = compute_observation_ranking_metrics(df, tie_breaker_col="perp_dist", ascending_tie_breaker=True)
    assert metrics["num_correct_top1"] == 1
    assert metrics["eligible_top1_accuracy"] == pytest.approx(1.0)

    # Reversed tie-breaker preference (larger perp_dist wins) -> candidate 2 chosen (wrong)
    metrics_rev = compute_observation_ranking_metrics(df, tie_breaker_col="perp_dist", ascending_tie_breaker=False)
    assert metrics_rev["num_correct_top1"] == 0
    assert metrics_rev["eligible_top1_accuracy"] == pytest.approx(0.0)


def test_invalid_probability_values():
    """Verify that NaN, infinite, or out-of-range probabilities raise ValueError."""
    # NaN in probability
    df_nan = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "observation_id": [0],
            "pred_proba": [np.nan],
            "is_match": [1],
        }
    )
    with pytest.raises(ValueError, match="NaN, null, or infinite"):
        compute_observation_ranking_metrics(df_nan)

    # Inf in probability
    df_inf = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "observation_id": [0],
            "pred_proba": [np.inf],
            "is_match": [1],
        }
    )
    with pytest.raises(ValueError, match="NaN, null, or infinite"):
        compute_observation_ranking_metrics(df_inf)

    # Probability < 0
    df_neg = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "observation_id": [0],
            "pred_proba": [-0.2],
            "is_match": [1],
        }
    )
    with pytest.raises(ValueError, match="outside probability range"):
        compute_observation_ranking_metrics(df_neg)

    # Probability > 1
    df_high = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "observation_id": [0],
            "pred_proba": [1.5],
            "is_match": [1],
        }
    )
    with pytest.raises(ValueError, match="outside probability range"):
        compute_observation_ranking_metrics(df_high)


def test_invalid_target_values():
    """Verify non-binary target labels raise ValueError."""
    df_bad_target = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "observation_id": [0],
            "pred_proba": [0.5],
            "is_match": [2],  # Non-binary class
        }
    )
    with pytest.raises(ValueError, match="must contain binary values"):
        compute_observation_ranking_metrics(df_bad_target)


def test_empty_dataframe():
    """Verify empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="DataFrame is empty"):
        compute_observation_ranking_metrics(pd.DataFrame())


def test_missing_required_columns():
    """Verify missing required column raises ValueError."""
    df = pd.DataFrame({"trip_id": ["T1"], "observation_id": [0]})
    with pytest.raises(ValueError, match="Required columns missing"):
        compute_observation_ranking_metrics(df)


def test_total_observations_override():
    """Verify that passing total_observations accounts for completely omitted observations."""
    # Suppose 1 observation was completely dropped upstream (0 candidate rows)
    df = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "observation_id": [0, 0],
            "pred_proba": [0.8, 0.2],
            "is_match": [1, 0],
        }
    )
    # Total observations is 2, but only 1 is in df
    metrics = compute_observation_ranking_metrics(df, total_observations=2)
    assert metrics["num_observations_total"] == 2
    assert metrics["num_observations_eligible"] == 1
    assert metrics["num_candidate_misses"] == 1
    assert metrics["num_correct_top1"] == 1
    assert metrics["eligible_top1_accuracy"] == pytest.approx(1.0)
    assert metrics["overall_top1_accuracy"] == pytest.approx(0.5)
    assert metrics["candidate_generation_recall"] == pytest.approx(0.5)


def test_evaluate_candidate_ranking_with_fitted_model():
    """Verify evaluate_candidate_ranking() integrates model inference with ranking metrics."""
    # Synthetic dataset with 6 observations across 2 trips
    rng = np.random.RandomState(42)
    records = []
    for trip_idx in [1, 2]:
        for obs_idx in range(3):
            is_miss = (trip_idx == 2 and obs_idx == 2)
            for cand_idx in range(4):
                is_m = 1 if (not is_miss and cand_idx == 1) else 0
                records.append(
                    {
                        "trip_id": f"T{trip_idx}",
                        "observation_id": obs_idx,
                        "candidate_arc_id": cand_idx,
                        "feat_0": rng.randn(),
                        "feat_1": rng.randn(),
                        "perp_dist": float(cand_idx * 5.0),
                        "is_match": is_m,
                    }
                )
    df = pd.DataFrame(records)

    # Train simple logistic regression
    clf = LogisticRegression(random_state=42)
    clf.fit(df[["feat_0", "feat_1"]], df["is_match"])

    res = evaluate_candidate_ranking(
        model=clf,
        df=df,
        feature_cols=["feat_0", "feat_1"],
        target_col="is_match",
        group_cols=("trip_id", "observation_id"),
        tie_breaker_col="perp_dist",
    )

    # Verify presence of all expected metric keys
    expected_keys = [
        "num_observations_total",
        "num_observations_eligible",
        "num_candidate_misses",
        "num_correct_top1",
        "eligible_top1_accuracy",
        "overall_top1_accuracy",
        "candidate_generation_recall",
        "candidate_accuracy",
        "candidate_precision",
        "candidate_recall",
        "candidate_macro_f1",
        "candidate_log_loss",
        "num_candidate_rows",
        "confusion_matrix",
        "inference_time_sec",
        "inference_latency_ms_per_candidate",
        "inference_latency_ms_per_observation",
    ]
    for key in expected_keys:
        assert key in res

    assert res["num_observations_total"] == 6
    assert res["num_observations_eligible"] == 5
    assert res["num_candidate_misses"] == 1
    assert res["num_candidate_rows"] == 24
    assert 0.0 <= res["eligible_top1_accuracy"] <= 1.0
    assert 0.0 <= res["overall_top1_accuracy"] <= 1.0
    assert res["candidate_generation_recall"] == pytest.approx(5 / 6)
    assert res["inference_latency_ms_per_candidate"] >= 0.0
    assert res["inference_latency_ms_per_observation"] >= 0.0


def test_aggregate_ranking_cv_metrics():
    """Verify CV aggregation computes correct mean and std for ranking metrics."""
    fold_metrics = [
        {
            "eligible_top1_accuracy": 0.80,
            "overall_top1_accuracy": 0.72,
            "candidate_generation_recall": 0.90,
            "candidate_macro_f1": 0.85,
            "candidate_accuracy": 0.94,
            "candidate_precision": 0.75,
            "candidate_recall": 0.78,
            "candidate_log_loss": 0.14,
            "train_time_sec": 10.0,
            "inference_time_sec": 0.5,
            "inference_latency_ms_per_candidate": 0.002,
            "inference_latency_ms_per_observation": 0.015,
            "num_observations_total": 100,
            "num_observations_eligible": 90,
            "num_candidate_misses": 10,
            "num_correct_top1": 72,
            "num_candidate_rows": 800,
        },
        {
            "eligible_top1_accuracy": 0.84,
            "overall_top1_accuracy": 0.756,
            "candidate_generation_recall": 0.90,
            "candidate_macro_f1": 0.87,
            "candidate_accuracy": 0.95,
            "candidate_precision": 0.79,
            "candidate_recall": 0.80,
            "candidate_log_loss": 0.12,
            "train_time_sec": 12.0,
            "inference_time_sec": 0.6,
            "inference_latency_ms_per_candidate": 0.0024,
            "inference_latency_ms_per_observation": 0.018,
            "num_observations_total": 100,
            "num_observations_eligible": 90,
            "num_candidate_misses": 10,
            "num_correct_top1": 76,
            "num_candidate_rows": 820,
        },
    ]

    agg = aggregate_ranking_cv_metrics(fold_metrics)
    assert agg["total_folds"] == 2
    assert agg["eligible_top1_accuracy_mean"] == pytest.approx(0.82)
    assert agg["eligible_top1_accuracy_std"] == pytest.approx(0.02)
    assert agg["overall_top1_accuracy_mean"] == pytest.approx(0.738)
    assert agg["candidate_macro_f1_mean"] == pytest.approx(0.86)
    assert agg["num_observations_total_sum"] == 200
    assert agg["num_observations_eligible_sum"] == 180
    assert agg["num_candidate_misses_sum"] == 20
    assert agg["num_correct_top1_sum"] == 148
    assert agg["num_candidate_rows_sum"] == 1620
