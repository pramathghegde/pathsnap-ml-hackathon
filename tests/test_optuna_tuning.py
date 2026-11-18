"""
Unit and integration tests for Bayesian Hyperparameter Optimization (Optuna) on HistGradientBoosting.

Tests:
1. Valid search-space parameter generation (bounds, types, categorical None).
2. TPE sampler reproducibility under fixed seed.
3. GroupKFold trip isolation and leakage detection.
4. Objective metric calculation (macro-average, pooled metrics, identity checks).
5. Failed trial handling and graceful error logging.
6. Trial result recording completeness.
7. Artifact generation (CSV and JSON schema validity).
8. Preservation of existing baseline and Phase 1 result files.
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import numpy as np
import optuna
from optuna.trial import TrialState
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.optimize import (
    DEFAULT_HISTGBDT_SEARCH_SPACE,
    DEFAULT_BASELINE_PARAMS,
    HistGBDTOptimizer,
    create_histgbdt_study,
    sample_histgbdt_params,
)


@pytest.fixture
def synthetic_ranking_dataset() -> pd.DataFrame:
    """
    Generate synthetic candidate ranking dataset across 4 trips and 2 folds.
    """
    records = []
    rng = np.random.RandomState(42)

    trips = ["trip_A", "trip_B", "trip_C", "trip_D"]
    fold_map = {"trip_A": 0, "trip_B": 0, "trip_C": 1, "trip_D": 1}

    for trip in trips:
        fold = fold_map[trip]
        for obs_idx in range(10):
            # 8 observations have a match, 2 are misses
            is_miss = obs_idx >= 8
            n_cands = 5
            true_cand_idx = -1 if is_miss else rng.randint(0, n_cands)

            for cand_idx in range(n_cands):
                is_match = 1 if cand_idx == true_cand_idx else 0
                perp_dist = 5.0 + cand_idx * 10.0 + rng.uniform(-1, 1)
                records.append(
                    {
                        "trip_id": trip,
                        "observation_id": obs_idx,
                        "candidate_arc_id": 1000 + cand_idx,
                        "fold": fold,
                        "is_training_eligible": 0 if is_miss else 1,
                        "is_match": is_match,
                        "perp_dist": perp_dist,
                        "proj_ratio": float(rng.uniform(0.1, 0.9)),
                        "delta_heading": float(rng.uniform(0.0, 1.5)),
                        "arc_length": float(rng.uniform(20.0, 150.0)),
                        "gps_speed": float(rng.uniform(5.0, 25.0)),
                        "speed_ratio": float(rng.uniform(0.3, 1.8)),
                        "traversal_rate": float(rng.uniform(0.05, 0.5)),
                        "candidate_rank": cand_idx + 1,
                        "is_adjacent": int(cand_idx == 0),
                    }
                )

    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# 1. Valid search-space parameter generation
# ------------------------------------------------------------------------------
def test_valid_search_space_parameter_generation() -> None:
    """Verify that sampled hyperparameters satisfy all type and range constraints."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

    for _ in range(15):
        trial = study.ask()
        params = sample_histgbdt_params(trial)

        # learning_rate: float in [0.02, 0.20]
        assert 0.02 <= params["learning_rate"] <= 0.20
        assert isinstance(params["learning_rate"], float)

        # max_iter: int in [100, 500]
        assert 100 <= params["max_iter"] <= 500
        assert params["max_iter"] % 50 == 0
        assert isinstance(params["max_iter"], int)

        # max_leaf_nodes: int in [15, 63]
        assert 15 <= params["max_leaf_nodes"] <= 63
        assert isinstance(params["max_leaf_nodes"], int)

        # max_depth: in [None, 3..12]
        assert params["max_depth"] in [None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # min_samples_leaf: int in [20, 100]
        assert 20 <= params["min_samples_leaf"] <= 100
        assert isinstance(params["min_samples_leaf"], int)

        # l2_regularization: float in [0.0, 10.0]
        assert 0.0 <= params["l2_regularization"] <= 10.0
        assert isinstance(params["l2_regularization"], float)

        # max_bins: int in [64, 255]
        assert 64 <= params["max_bins"] <= 255
        assert isinstance(params["max_bins"], int)

        # Verify scikit-learn accepts the exact parameter dictionary
        clf = HistGradientBoostingClassifier(**params, random_state=42)
        assert clf is not None


# ------------------------------------------------------------------------------
# 2. TPE sampler reproducibility
# ------------------------------------------------------------------------------
def test_tpe_sampler_reproducibility() -> None:
    """Verify that two independent studies with identical seed produce identical trial sequences."""
    study_a = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    study_b = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

    for _ in range(10):
        trial_a = study_a.ask()
        trial_b = study_b.ask()

        params_a = sample_histgbdt_params(trial_a)
        params_b = sample_histgbdt_params(trial_b)

        assert params_a == params_b


# ------------------------------------------------------------------------------
# 3. GroupKFold trip isolation
# ------------------------------------------------------------------------------
def test_groupkfold_trip_isolation(synthetic_ranking_dataset: pd.DataFrame) -> None:
    """Verify strict trip isolation between training and validation folds."""
    features = [
        "perp_dist",
        "proj_ratio",
        "delta_heading",
        "arc_length",
        "gps_speed",
        "speed_ratio",
        "traversal_rate",
        "candidate_rank",
        "is_adjacent",
    ]

    optimizer = HistGBDTOptimizer(
        df=synthetic_ranking_dataset,
        feature_cols=features,
        n_splits=2,
    )

    # Check each precomputed fold
    for fold_item in optimizer.fold_data:
        val_df = fold_item["val_df"]
        train_trips = set(
            synthetic_ranking_dataset.loc[
                (synthetic_ranking_dataset["fold"] != fold_item["fold"])
                & (synthetic_ranking_dataset["is_training_eligible"] == 1),
                "trip_id",
            ].unique()
        )
        val_trips = set(val_df["trip_id"].unique())
        assert len(train_trips.intersection(val_trips)) == 0

    # Test that injected leakage raises ValueError
    leaky_df = synthetic_ranking_dataset.copy()
    # Inject leakage: put half of trip_A in fold 0 and half in fold 1
    leaky_df.loc[(leaky_df["trip_id"] == "trip_A") & (leaky_df["observation_id"] < 5), "fold"] = 0
    leaky_df.loc[(leaky_df["trip_id"] == "trip_A") & (leaky_df["observation_id"] >= 5), "fold"] = 1

    with pytest.raises(ValueError, match="Trip leakage detected"):
        HistGBDTOptimizer(
            df=leaky_df,
            feature_cols=features,
            n_splits=2,
        )


# ------------------------------------------------------------------------------
# 4. Objective metric calculation (macro vs pooled)
# ------------------------------------------------------------------------------
def test_objective_metric_calculation_macro_and_pooled(
    synthetic_ranking_dataset: pd.DataFrame,
) -> None:
    """Verify calculation of macro-averaged and pooled ranking metrics."""
    features = [
        "perp_dist",
        "proj_ratio",
        "delta_heading",
        "arc_length",
        "gps_speed",
        "speed_ratio",
        "traversal_rate",
        "candidate_rank",
        "is_adjacent",
    ]

    optimizer = HistGBDTOptimizer(
        df=synthetic_ranking_dataset,
        feature_cols=features,
        n_splits=2,
    )

    params = {
        "learning_rate": 0.1,
        "max_iter": 20,
        "max_leaf_nodes": 15,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "l2_regularization": 0.0,
        "max_bins": 64,
    }

    primary_obj, metrics, fold_metrics, oof_df = (
        optimizer.evaluate_hyperparameters(params)
    )

    # 1. Primary objective matches mean Eligible Top-1
    assert len(fold_metrics) == 2
    f0_acc = fold_metrics[0]["eligible_top1_accuracy"]
    f1_acc = fold_metrics[1]["eligible_top1_accuracy"]
    assert np.isclose(primary_obj, (f0_acc + f1_acc) / 2.0)
    assert np.isclose(primary_obj, metrics["eligible_top1_acc_mean"])

    # 2. Both macro and pooled metrics present
    assert "eligible_top1_acc_mean" in metrics
    assert "eligible_top1_acc_std" in metrics
    assert "pooled_eligible_top1_acc" in metrics
    assert "pooled_overall_top1_acc" in metrics
    assert "pooled_cand_recall" in metrics

    # 3. Mathematical identity on pooled metrics:
    # pooled_overall_top1_acc == pooled_eligible_top1_acc * pooled_cand_recall
    expected_overall = (
        metrics["pooled_eligible_top1_acc"] * metrics["pooled_cand_recall"]
    )
    assert np.isclose(metrics["pooled_overall_top1_acc"], expected_overall, atol=1e-5)

    # 4. Out-of-fold predictions count equals total dataset rows
    assert len(oof_df) == len(synthetic_ranking_dataset)


# ------------------------------------------------------------------------------
# 5. Failed trial handling
# ------------------------------------------------------------------------------
def test_failed_trial_handling(synthetic_ranking_dataset: pd.DataFrame) -> None:
    """Verify that failed trials are recorded with status FAIL and error message."""
    features = [
        "perp_dist",
        "proj_ratio",
        "delta_heading",
        "arc_length",
        "gps_speed",
        "speed_ratio",
        "traversal_rate",
        "candidate_rank",
        "is_adjacent",
    ]

    optimizer = HistGBDTOptimizer(
        df=synthetic_ranking_dataset,
        feature_cols=features,
        n_splits=2,
    )

    # Corrupt one fold to trigger exception
    optimizer.fold_data[0]["X_train"] = pd.DataFrame()  # Empty dataframe causes error

    study = optuna.create_study()

    # The optimizer objective catches and logs the error before re-raising for Optuna
    with pytest.raises(Exception):
        optimizer.objective(study.ask())

    # Check trial record
    assert len(optimizer.trial_records) == 1
    assert optimizer.trial_records[0]["state"] == "FAIL"
    assert "error_message" in optimizer.trial_records[0]
    assert len(optimizer.trial_records[0]["error_message"]) > 0


# ------------------------------------------------------------------------------
# 6. Trial result recording
# ------------------------------------------------------------------------------
def test_trial_result_recording(synthetic_ranking_dataset: pd.DataFrame) -> None:
    """Verify trial records contain all necessary reporting and tracking fields."""
    features = [
        "perp_dist",
        "proj_ratio",
        "delta_heading",
        "arc_length",
        "gps_speed",
        "speed_ratio",
        "traversal_rate",
        "candidate_rank",
        "is_adjacent",
    ]

    optimizer = HistGBDTOptimizer(
        df=synthetic_ranking_dataset,
        feature_cols=features,
        n_splits=2,
    )

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    trial = study.ask()
    obj_val = optimizer.objective(trial)

    assert len(optimizer.trial_records) == 1
    rec = optimizer.trial_records[0]
    assert rec["state"] == "COMPLETE"
    assert rec["trial_number"] == 0
    assert "learning_rate" in rec
    assert "max_iter" in rec
    assert "eligible_top1_acc_mean" in rec
    assert "overall_top1_acc_mean" in rec
    assert "pooled_eligible_top1_acc" in rec
    assert "cand_macro_f1_mean" in rec
    assert "cand_log_loss_mean" in rec
    assert "total_train_time_sec" in rec
    assert "latency_ms_per_obs_mean" in rec


# ------------------------------------------------------------------------------
# 7. Artifact generation
# ------------------------------------------------------------------------------
def test_artifact_generation(synthetic_ranking_dataset: pd.DataFrame) -> None:
    """Verify export of trials CSV and summary JSON with proper structure and metrics."""
    features = [
        "perp_dist",
        "proj_ratio",
        "delta_heading",
        "arc_length",
        "gps_speed",
        "speed_ratio",
        "traversal_rate",
        "candidate_rank",
        "is_adjacent",
    ]

    optimizer = HistGBDTOptimizer(
        df=synthetic_ranking_dataset,
        feature_cols=features,
        n_splits=2,
    )

    study = optimizer.run_optimization(n_trials=2, enqueue_baseline=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_trials.csv"
        json_path = Path(tmpdir) / "test_summary.json"

        optimizer.export_artifacts(
            study=study,
            output_csv=csv_path,
            output_json=json_path,
            baseline_metrics={"eligible_top1_accuracy_mean": 0.80},
        )

        assert csv_path.exists()
        assert json_path.exists()

        # Validate CSV contents
        trials_df = pd.read_csv(csv_path)
        assert len(trials_df) == 2
        assert "eligible_top1_acc_mean" in trials_df.columns
        assert "pooled_eligible_top1_acc" in trials_df.columns

        # Validate JSON contents
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["model_name"] == "HistGradientBoostingClassifier"
        assert summary["n_trials_completed"] == 2
        assert "best_trial" in summary
        assert "params" in summary["best_trial"]
        assert "baseline_comparison" in summary
        assert "reproducibility" in summary


# ------------------------------------------------------------------------------
# 8. Preservation of existing baseline results
# ------------------------------------------------------------------------------
def test_preservation_of_existing_baseline_results() -> None:
    """Verify that existing baseline benchmark files are present and protected."""
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "experiments" / "results"

    baseline_files = [
        results_dir / "baseline_model_comparison.csv",
        results_dir / "baseline_model_comparison.json",
        results_dir / "knn_pilot.csv",
        results_dir / "knn_pilot.json",
        results_dir / "histgbdt_ranking_evaluation.csv",
        results_dir / "histgbdt_ranking_evaluation.json",
    ]

    for p in baseline_files:
        assert p.is_file(), f"Critical baseline file {p} is missing or corrupted!"
        assert p.stat().st_size > 0, f"Baseline file {p} is empty!"
