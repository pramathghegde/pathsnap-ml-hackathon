"""
Unit tests for src/models/train.py using isolated synthetic data.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    run_training_pipeline,
    validate_training_inputs,
)
from src.models.model_factory import SUPPORTED_MODELS


def _generate_synthetic_df(
    n_samples: int = 100,
    n_trips: int = 10,
    n_features: int = 4,
    n_classes: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Helper to generate a mock dataset with trip_id groupings and target classes."""
    rng = np.random.RandomState(random_state)
    features = {f"feat_{i}": rng.randn(n_samples) for i in range(n_features)}
    trip_ids = [f"trip_{i % n_trips:03d}" for i in range(n_samples)]
    targets = rng.choice(range(n_classes), size=n_samples)

    df = pd.DataFrame(features)
    df["trip_id"] = trip_ids
    df["label"] = targets
    return df


def test_validate_training_inputs_errors():
    """Verify input validation catches all invalid inputs."""
    df = _generate_synthetic_df()

    # Non-dataframe
    with pytest.raises(ValueError, match="Expected pandas DataFrame"):
        validate_training_inputs("not_a_df", target_col="label", group_col="trip_id")

    # Empty dataframe
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_training_inputs(pd.DataFrame(), target_col="label", group_col="trip_id")

    # Missing target col
    with pytest.raises(ValueError, match="Target column 'missing' not found"):
        validate_training_inputs(df, target_col="missing", group_col="trip_id")

    # Missing group col
    with pytest.raises(ValueError, match="Group column 'missing' not found"):
        validate_training_inputs(df, target_col="label", group_col="missing")

    # Target containing NaNs
    df_na_target = df.copy()
    df_na_target.loc[0, "label"] = np.nan
    with pytest.raises(ValueError, match="Target column 'label' contains .* missing"):
        validate_training_inputs(df_na_target, target_col="label", group_col="trip_id")

    # Target with < 2 unique classes
    df_single_class = df.copy()
    df_single_class["label"] = 1
    with pytest.raises(ValueError, match="must contain at least 2 distinct classes"):
        validate_training_inputs(df_single_class, target_col="label", group_col="trip_id")

    # Group column containing NaNs
    df_na_group = df.copy()
    df_na_group.loc[0, "trip_id"] = np.nan
    with pytest.raises(ValueError, match="Group column 'trip_id' contains .* missing"):
        validate_training_inputs(df_na_group, target_col="label", group_col="trip_id")

    # Fewer unique groups than n_splits
    with pytest.raises(ValueError, match="is less than n_splits"):
        validate_training_inputs(df, target_col="label", group_col="trip_id", n_splits=15)

    # Specified feature column missing
    with pytest.raises(ValueError, match="Specified feature columns missing"):
        validate_training_inputs(
            df, target_col="label", group_col="trip_id", feature_cols=["feat_0", "nonexistent"]
        )

    # Feature column containing NaNs
    df_na_feat = df.copy()
    df_na_feat.loc[0, "feat_0"] = np.nan
    with pytest.raises(ValueError, match="Feature columns contain NaN"):
        validate_training_inputs(df_na_feat, target_col="label", group_col="trip_id")


def test_group_kfold_separation_and_identical_splits():
    """Verify GroupKFold ensures zero trip leakage across splits."""
    df = _generate_synthetic_df(n_samples=80, n_trips=8)
    res = run_training_pipeline(
        df=df,
        target_col="label",
        group_col="trip_id",
        n_splits=4,
        selected_models=["Logistic Regression", "Decision Tree"],
        refit_final=False,
        save_artifacts=False,
    )

    # Check both models evaluated on identical 4 folds
    lr_folds = res["models"]["Logistic Regression"]["folds"]
    dt_folds = res["models"]["Decision Tree"]["folds"]
    assert len(lr_folds) == 4
    assert len(dt_folds) == 4

    for f_lr, f_dt in zip(lr_folds, dt_folds):
        assert f_lr["fold"] == f_dt["fold"]


def test_all_five_models_complete_cv():
    """Verify that all five distinct model families complete GroupKFold cross-validation."""
    df = _generate_synthetic_df(n_samples=50, n_trips=5, n_classes=2, random_state=123)

    results = run_training_pipeline(
        df=df,
        target_col="label",
        group_col="trip_id",
        n_splits=3,
        refit_final=True,
        save_artifacts=False,
    )

    # Verify all 5 models are present in results
    for model_name in SUPPORTED_MODELS:
        assert model_name in results["models"]
        agg = results["models"][model_name]["aggregate"]
        assert "accuracy_mean" in agg
        assert "macro_f1_mean" in agg
        assert "log_loss_mean" in agg
        assert "train_time_sec_mean" in agg
        assert "inference_time_sec_mean" in agg
        assert "inference_latency_ms_per_sample_mean" in agg

    # Verify best model selection
    assert results["best_model_name"] in SUPPORTED_MODELS
    assert results["final_model"] is not None
    assert results["final_metadata"]["model_name"] == results["best_model_name"]


def test_final_refit_and_artifact_serialization(tmp_path: Path):
    """Verify model refit on full data, serialization, and metadata preservation in tmp_path."""
    df = _generate_synthetic_df(n_samples=60, n_trips=6, n_classes=2, random_state=99)

    art_dir = tmp_path / "artifacts"
    res_dir = tmp_path / "experiments"

    res = run_training_pipeline(
        df=df,
        target_col="label",
        group_col="trip_id",
        n_splits=3,
        selected_models=["Decision Tree"],
        refit_final=True,
        save_artifacts=True,
        artifacts_dir=art_dir,
        results_dir=res_dir,
        overwrite=False,
    )

    best_slug = res["best_model_name"].lower().replace(" ", "_")
    model_path = art_dir / f"{best_slug}_refitted.joblib"
    meta_path = art_dir / f"{best_slug}_metadata.json"
    csv_path = res_dir / "benchmark_summary.csv"
    json_path = res_dir / "benchmark_summary.json"

    # Verify files created in tmp_path
    assert model_path.exists()
    assert meta_path.exists()
    assert csv_path.exists()
    assert json_path.exists()

    # Verify model reloads cleanly
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict_proba")

    # Verify metadata contents
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_name"] == res["best_model_name"]
    assert "dataset_hash" in meta
    assert meta["dataset_size"]["n_samples"] == len(df)
    assert "cv_metrics" in meta

    # Verify overwrite protection
    with pytest.raises(FileExistsError):
        run_training_pipeline(
            df=df,
            target_col="label",
            group_col="trip_id",
            n_splits=3,
            selected_models=["Decision Tree"],
            refit_final=True,
            save_artifacts=True,
            artifacts_dir=art_dir,
            results_dir=res_dir,
            overwrite=False,
        )


def test_unseen_validation_classes_handled():
    """Verify that unseen validation classes are captured without crashing CV or dropping folds."""
    # Construct a dataset where trip 'trip_isolated' uniquely possesses class 2
    n_samples = 40
    df = pd.DataFrame(
        {
            "feat_0": np.random.randn(n_samples),
            "feat_1": np.random.randn(n_samples),
            "trip_id": ["trip_1"] * 15 + ["trip_2"] * 15 + ["trip_isolated"] * 10,
            # Classes 0 and 1 for trip_1 and trip_2; class 2 only for trip_isolated
            "label": [0] * 8 + [1] * 7 + [0] * 7 + [1] * 8 + [2] * 10,
        }
    )

    # 3 splits: fold with trip_isolated in val will encounter class 2, which was unseen in training
    res = run_training_pipeline(
        df=df,
        target_col="label",
        group_col="trip_id",
        n_splits=3,
        selected_models=["Logistic Regression"],
        refit_final=False,
        save_artifacts=False,
    )

    agg = res["models"]["Logistic Regression"]["aggregate"]
    assert agg["log_loss_invalid_folds"] >= 1
    assert np.isnan(agg["log_loss_mean"])
    assert "INVALID_FOLDS_DETECTED" in agg["log_loss_status"]
    # Accuracy and macro F1 should still be computed
    assert not np.isnan(agg["accuracy_mean"])
    assert not np.isnan(agg["macro_f1_mean"])


def test_metadata_columns_excluded_from_automatic_feature_inference():
    """
    Verify that all metadata and audit columns are excluded from automatic feature inference:
    - observation_id
    - has_positive_candidate
    - is_training_eligible
    - fold
    - point_idx, candidate_arc_id, timestamp, trip_id, target
    And verify that the nine intended predictive features are selected.
    """
    n_samples = 60
    intended_features = [
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

    data = {f: np.random.randn(n_samples) for f in intended_features}
    # Add metadata and audit columns
    data["trip_id"] = [f"trip_{i % 6:03d}" for i in range(n_samples)]
    data["observation_id"] = np.arange(n_samples)
    data["point_idx"] = np.arange(n_samples)
    data["candidate_arc_id"] = np.random.randint(1, 1000, size=n_samples)
    data["timestamp"] = np.linspace(1.0, 60.0, n_samples)
    data["has_positive_candidate"] = np.random.choice([0, 1], size=n_samples)
    data["is_training_eligible"] = data["has_positive_candidate"]
    data["fold"] = np.random.choice(range(5), size=n_samples)
    data["is_match"] = np.random.choice([0, 1], size=n_samples)

    df = pd.DataFrame(data)

    resolved = validate_training_inputs(
        df=df,
        target_col="is_match",
        group_col="trip_id",
        feature_cols=None,
    )

    # 1. Verify specific metadata columns are strictly excluded
    assert "observation_id" not in resolved
    assert "has_positive_candidate" not in resolved
    assert "is_training_eligible" not in resolved
    assert "fold" not in resolved
    assert "point_idx" not in resolved
    assert "candidate_arc_id" not in resolved
    assert "timestamp" not in resolved
    assert "trip_id" not in resolved
    assert "is_match" not in resolved

    # 2. Verify all nine intended predictive features are present and matched
    assert sorted(resolved) == sorted(intended_features)


def test_explicit_feature_cols_selection():
    """Verify that explicitly supplied feature_cols continues to work and takes precedence."""
    n_samples = 50
    df = pd.DataFrame(
        {
            "trip_id": [f"trip_{i % 5:03d}" for i in range(n_samples)],
            "is_match": np.random.choice([0, 1], size=n_samples),
            "observation_id": np.arange(n_samples),
            "fold": np.random.choice(range(5), size=n_samples),
            "perp_dist": np.random.randn(n_samples),
            "delta_heading": np.random.randn(n_samples),
            "unused_col": np.random.randn(n_samples),
        }
    )

    explicit_features = ["perp_dist", "delta_heading"]
    resolved = validate_training_inputs(
        df=df,
        target_col="is_match",
        group_col="trip_id",
        feature_cols=explicit_features,
    )

    assert resolved == explicit_features


def test_training_pipeline_auto_feature_inference_end_to_end():
    """Verify end-to-end training pipeline preserves metadata exclusion during full refit."""
    n_samples = 60
    df = pd.DataFrame(
        {
            "trip_id": [f"trip_{i % 6:03d}" for i in range(n_samples)],
            "is_match": np.random.choice([0, 1], size=n_samples),
            "observation_id": np.arange(n_samples),
            "has_positive_candidate": np.ones(n_samples, dtype=int),
            "is_training_eligible": np.ones(n_samples, dtype=int),
            "fold": np.random.choice(range(5), size=n_samples),
            "perp_dist": np.random.randn(n_samples),
            "proj_ratio": np.random.uniform(0, 1, size=n_samples),
            "delta_heading": np.random.uniform(0, np.pi, size=n_samples),
        }
    )

    res = run_training_pipeline(
        df=df,
        target_col="is_match",
        group_col="trip_id",
        feature_cols=None,  # Rely on automatic feature inference
        n_splits=3,
        selected_models=["Decision Tree"],
        refit_final=True,
        save_artifacts=False,
    )

    meta_features = res["final_metadata"]["feature_names"]
    assert "observation_id" not in meta_features
    assert "has_positive_candidate" not in meta_features
    assert "is_training_eligible" not in meta_features
    assert "fold" not in meta_features
    assert sorted(meta_features) == ["delta_heading", "perp_dist", "proj_ratio"]
