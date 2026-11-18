"""
Unit and integration tests for PathSnap inference module (PathSnapPredictor).

Tests:
1. Model artifact loading from disk.
2. Metadata validation and schema verification.
3. Feature ordering tolerance (shuffled columns re-aligned correctly).
4. Missing feature detection and error reporting.
5. Probability generation within mathematical bounds [0, 1].
6. Correct positive-class probability extraction using model.classes_.
7. Identifier preservation (trip_id, observation_id, candidate_arc_id).
8. Invalid input handling (empty DataFrame, NaN in features, incorrect types).
9. Prediction reproducibility across repeated calls.
10. Candidate ranking and Top-1 selection logic.
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.predict import (
    DEFAULT_PREDICTIVE_FEATURES,
    PathSnapPredictor,
    load_predictor,
)


@pytest.fixture
def mock_model_artifacts():
    """
    Train a dummy HistGradientBoostingClassifier and serialize with metadata.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_path = tmp_path / "test_model.joblib"
        metadata_path = tmp_path / "test_model_metadata.json"

        features = list(DEFAULT_PREDICTIVE_FEATURES)
        rng = np.random.RandomState(42)

        # Create dummy training data
        N = 200
        X_data = rng.randn(N, len(features))
        y_data = rng.choice([0, 1], size=N, p=[0.7, 0.3])

        clf = HistGradientBoostingClassifier(max_iter=10, random_state=42)
        clf.fit(X_data, y_data)

        # Save model
        joblib.dump(clf, model_path)

        # Save metadata
        metadata = {
            "model_name": "HistGradientBoostingClassifier",
            "version": "1.0.0-test",
            "feature_names": features,
            "classes": [int(c) for c in clf.classes_],
            "random_seed": 42,
            "training_dataset": "synthetic_test_dataset",
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        yield model_path, metadata_path, features


@pytest.fixture
def sample_candidate_df():
    """
    Create sample candidate observation DataFrame with features and identifiers.
    """
    features = list(DEFAULT_PREDICTIVE_FEATURES)
    rng = np.random.RandomState(123)

    records = []
    for obs in range(5):
        for cand in range(4):
            rec = {
                "trip_id": "test_trip_1",
                "observation_id": obs,
                "candidate_arc_id": 100 + cand,
            }
            for f in features:
                rec[f] = float(rng.uniform(0.1, 10.0))
            records.append(rec)

    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# 1. Model artifact loading
# ------------------------------------------------------------------------------
def test_predictor_model_loading(mock_model_artifacts):
    """Verify clean initialization of PathSnapPredictor from disk."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    assert predictor.model is not None
    assert predictor.version == "1.0.0-test"
    assert predictor.model_name == "HistGradientBoostingClassifier"
    assert predictor.pos_class_index == 1

    # Inferred metadata path
    predictor_inferred = PathSnapPredictor(model_path=model_path)
    assert predictor_inferred.metadata_path == metadata_path


def test_predictor_missing_files():
    """Verify FileNotFoundError on non-existent model or metadata paths."""
    with pytest.raises(FileNotFoundError):
        PathSnapPredictor(model_path="non_existent_model.joblib")


# ------------------------------------------------------------------------------
# 2. Metadata validation
# ------------------------------------------------------------------------------
def test_metadata_validation(mock_model_artifacts):
    """Verify validation of mandatory metadata keys."""
    model_path, metadata_path, _ = mock_model_artifacts

    # Write corrupt metadata missing 'feature_names'
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_meta_path = Path(tmpdir) / "bad_metadata.json"
        with open(bad_meta_path, "w", encoding="utf-8") as f:
            json.dump({"model_name": "TestModel"}, f)

        with pytest.raises(ValueError, match="missing mandatory keys"):
            PathSnapPredictor(model_path=model_path, metadata_path=bad_meta_path)


# ------------------------------------------------------------------------------
# 3. Feature ordering tolerance
# ------------------------------------------------------------------------------
def test_feature_ordering_tolerance(mock_model_artifacts, sample_candidate_df):
    """Verify predictor generates identical predictions regardless of column order."""
    model_path, metadata_path, features = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    # 1. Normal ordering
    probs_normal = predictor.predict_proba(sample_candidate_df)

    # 2. Reverse feature column ordering in input DataFrame
    shuffled_features = list(reversed(features))
    id_cols = ["trip_id", "observation_id", "candidate_arc_id"]
    df_shuffled = sample_candidate_df[id_cols + shuffled_features].copy()

    probs_shuffled = predictor.predict_proba(df_shuffled)

    np.testing.assert_allclose(probs_normal, probs_shuffled, atol=1e-7)


# ------------------------------------------------------------------------------
# 4. Missing feature detection
# ------------------------------------------------------------------------------
def test_missing_feature_detection(mock_model_artifacts, sample_candidate_df):
    """Verify ValueError is raised if any mandatory predictive feature is missing."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    # Drop a feature
    df_missing = sample_candidate_df.drop(columns=["perp_dist"])

    with pytest.raises(ValueError, match="missing required features.*perp_dist"):
        predictor.predict_proba(df_missing)


# ------------------------------------------------------------------------------
# 5. Probability generation within bounds
# ------------------------------------------------------------------------------
def test_probability_generation_bounds(mock_model_artifacts, sample_candidate_df):
    """Verify predicted probabilities are within [0.0, 1.0] and contain no NaNs."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    probas = predictor.predict_proba(sample_candidate_df)

    assert isinstance(probas, np.ndarray)
    assert len(probas) == len(sample_candidate_df)
    assert np.all(np.isfinite(probas))
    assert np.all((probas >= 0.0) & (probas <= 1.0))


# ------------------------------------------------------------------------------
# 6. Correct positive-class probability extraction
# ------------------------------------------------------------------------------
def test_positive_class_index_resolution(mock_model_artifacts):
    """Verify that predictor correctly identifies positive class 1."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    assert predictor.pos_class_index == 1
    assert predictor.model.classes_[predictor.pos_class_index] == 1


# ------------------------------------------------------------------------------
# 7. Identifier preservation
# ------------------------------------------------------------------------------
def test_identifier_preservation(mock_model_artifacts, sample_candidate_df):
    """Verify identifier columns are intact in predict_candidates output."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    scored_df = predictor.predict_candidates(sample_candidate_df)

    for col in ["trip_id", "observation_id", "candidate_arc_id"]:
        assert col in scored_df.columns
        pd.testing.assert_series_equal(
            scored_df[col], sample_candidate_df[col]
        )

    assert "match_probability" in scored_df.columns
    assert len(scored_df) == len(sample_candidate_df)


# ------------------------------------------------------------------------------
# 8. Invalid input handling
# ------------------------------------------------------------------------------
def test_invalid_input_handling(mock_model_artifacts, sample_candidate_df):
    """Verify handling of empty DataFrame, NaNs, and incorrect types."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    # Empty DataFrame
    with pytest.raises(ValueError, match="Input candidate DataFrame is empty"):
        predictor.predict_proba(pd.DataFrame())

    # Incorrect type
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        predictor.predict_proba([1, 2, 3])

    # NaN in features
    df_nan = sample_candidate_df.copy()
    df_nan.loc[0, "perp_dist"] = np.nan
    with pytest.raises(ValueError, match="contain missing/NaN values"):
        predictor.predict_proba(df_nan)


# ------------------------------------------------------------------------------
# 9. Prediction reproducibility
# ------------------------------------------------------------------------------
def test_prediction_reproducibility(mock_model_artifacts, sample_candidate_df):
    """Verify that repeated predictions on same input yield identical results."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    p1 = predictor.predict_proba(sample_candidate_df)
    p2 = predictor.predict_proba(sample_candidate_df)

    np.testing.assert_array_equal(p1, p2)


# ------------------------------------------------------------------------------
# 10. Candidate ranking and Top-1 selection
# ------------------------------------------------------------------------------
def test_candidate_ranking_and_top1_selection(mock_model_artifacts, sample_candidate_df):
    """Verify candidate ranking and Top-1 extraction per observation."""
    model_path, metadata_path, _ = mock_model_artifacts
    predictor = PathSnapPredictor(model_path=model_path, metadata_path=metadata_path)

    # Rank candidates
    ranked = predictor.rank_candidates(sample_candidate_df)
    assert "model_rank" in ranked.columns
    assert "match_probability" in ranked.columns

    # Check ranks 1..4 present for each observation
    obs_groups = ranked.groupby(["trip_id", "observation_id"])
    for _, grp in obs_groups:
        assert list(grp["model_rank"]) == [1, 2, 3, 4]
        # Check sorted descending by probability
        probs = grp["match_probability"].values
        assert np.all(probs[:-1] >= probs[1:])

    # Top-1 selection
    top1 = predictor.select_top1_candidates(sample_candidate_df)
    assert len(top1) == sample_candidate_df["observation_id"].nunique()
    assert "match_probability" in top1.columns


# ------------------------------------------------------------------------------
# 11. Production model artifact integration
# ------------------------------------------------------------------------------
def test_production_model_artifact_load_and_predict(sample_candidate_df):
    """Verify that the production artifact in artifacts/models loads and predicts cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    prod_model_path = repo_root / "artifacts" / "models" / "pathsnap_histgbdt.joblib"
    prod_meta_path = repo_root / "artifacts" / "models" / "pathsnap_histgbdt_metadata.json"

    if not prod_model_path.exists():
        pytest.skip("Production model artifact not found.")

    predictor = load_predictor(model_dir=prod_model_path.parent)
    assert predictor.model_name == "HistGradientBoostingClassifier"
    assert predictor.pos_class_index == 1
    assert len(predictor.feature_names) == 9

    # Score sample candidate DataFrame
    probs = predictor.predict_proba(sample_candidate_df)
    assert len(probs) == len(sample_candidate_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Top-1 selection
    top1 = predictor.select_top1_candidates(sample_candidate_df)
    assert len(top1) == sample_candidate_df["observation_id"].nunique()

