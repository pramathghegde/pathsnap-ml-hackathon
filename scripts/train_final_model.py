"""
Train and serialize the final PathSnap HistGradientBoosting candidate ranking model.

Uses:
- Complete training dataset (data/processed/train_features.csv)
- Training filter: is_training_eligible == 1 (1,426,177 eligible rows)
- Exactly the 9 predictive features from configs/feature_schema.json
- Validated Trial 6 Bayesian hyperparameters
- Serialization to artifacts/models/pathsnap_histgbdt.joblib & metadata JSON
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.predict import load_predictor


DATA_PATH = REPO_ROOT / "data" / "processed" / "train_features.csv"
SCHEMA_PATH = REPO_ROOT / "configs" / "feature_schema.json"
VALIDATION_DETAILS_PATH = (
    REPO_ROOT / "experiments" / "results" / "trial6_validation_details.json"
)
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "models"
MODEL_PATH = ARTIFACTS_DIR / "pathsnap_histgbdt.joblib"
METADATA_PATH = ARTIFACTS_DIR / "pathsnap_histgbdt_metadata.json"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main(overwrite: bool = False) -> None:
    # 1. Overwrite check
    if not overwrite:
        for p in (MODEL_PATH, METADATA_PATH):
            if p.exists():
                raise FileExistsError(
                    f"Artifact '{p}' already exists. Pass overwrite=True to replace."
                )

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Schema not found at: {SCHEMA_PATH}")

    # 2. Load schema
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema["features"]
    target_col = schema.get("target", "is_match")
    filter_col = schema.get("training_filter_column", "is_training_eligible")

    # 3. Load validated Trial 6 hyperparameters and CV metrics
    with open(VALIDATION_DETAILS_PATH, "r", encoding="utf-8") as f:
        val_details = json.load(f)

    hyperparameters = val_details["params"]
    cv_metrics = val_details["metrics"]
    baseline_comp = val_details["baseline_comparison"]

    print("=" * 80)
    print("PATHSNAP: FINAL MODEL TRAINING (HISTGRADIENTBOOSTING)")
    print("=" * 80)
    print(f"Data Source:       {DATA_PATH}")
    print(f"Filter Column:     '{filter_col} == 1' (Strictly eligible rows)")
    print(f"Predictive Features ({len(feature_cols)}): {feature_cols}")
    print(f"Selected Config:   Trial 6 (Tuned Bayesian Hyperparameters)")
    for k, v in hyperparameters.items():
        print(f"  • {k:20s}: {v}")
    print("=" * 80)

    # 4. Load dataset
    print(f"\n[1/4] Loading training dataset...")
    t0 = time.time()
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "trip_id": str,
            "observation_id": int,
            "candidate_arc_id": int,
            "fold": int,
            target_col: int,
            filter_col: int,
        },
    )
    t_load = time.time() - t0
    data_hash = compute_file_sha256(DATA_PATH)
    print(f"  Loaded {len(df):,} total rows in {t_load:.2f}s (SHA-256: {data_hash[:16]}...)")

    # 5. Filter strictly to eligible training observations
    print(f"\n[2/4] Applying training filter '{filter_col} == 1'...")
    train_mask = df[filter_col] == 1
    df_train = df.loc[train_mask]
    n_train_total = len(df_train)
    n_pos = int((df_train[target_col] == 1).sum())
    n_neg = int((df_train[target_col] == 0).sum())
    print(f"  Eligible training rows: {n_train_total:,} (Pos: {n_pos:,}, Neg: {n_neg:,}, Ratio: {n_neg/n_pos:.2f}:1)")

    X_train = df_train[feature_cols]
    y_train = df_train[target_col].to_numpy().astype(int)

    # Assert strictly no metadata/target in X_train
    forbidden_cols = ["trip_id", "observation_id", "candidate_arc_id", target_col, "fold"]
    for c in forbidden_cols:
        assert c not in X_train.columns, f"Forbidden column '{c}' leaked into feature matrix!"

    # 6. Fit final model on complete permitted training data
    print(f"\n[3/4] Fitting final HistGradientBoostingClassifier on all {n_train_total:,} rows...")
    model = HistGradientBoostingClassifier(
        **hyperparameters,
        random_state=42,
    )

    t_fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_duration = time.perf_counter() - t_fit_start
    print(f"  Final model fitted in {fit_duration:.2f}s (n_iter_ = {model.n_iter_})")
    print(f"  Classes discovered: {model.classes_}")

    # 7. Construct and serialize artifacts
    print(f"\n[4/4] Serializing model and metadata to {ARTIFACTS_DIR}...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Joblib model
    joblib.dump(model, MODEL_PATH, compress=3)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved model -> {MODEL_PATH} ({model_size_mb:.2f} MB)")

    # Metadata sidecar
    metadata: Dict[str, Any] = {
        "model_name": "HistGradientBoostingClassifier",
        "version": "1.0.0",
        "description": "Production candidate emission scoring model for PathSnap map matching.",
        "feature_names": list(feature_cols),
        "classes": [int(c) for c in model.classes_],
        "positive_class": 1,
        "selected_hyperparameters": hyperparameters,
        "best_optuna_trial": 6,
        "training_dataset": "data/processed/train_features.csv",
        "training_dataset_sha256": data_hash,
        "training_filter": f"{filter_col} == 1",
        "num_training_samples": n_train_total,
        "num_positive_samples": n_pos,
        "num_negative_samples": n_neg,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": float(fit_duration),
        "random_seed": 42,
        "sklearn_version": sklearn.__version__,
        "cross_validation_metrics": {
            "eligible_top1_accuracy_mean": cv_metrics["eligible_top1_acc_mean"],
            "eligible_top1_accuracy_std": cv_metrics["eligible_top1_acc_std"],
            "overall_top1_accuracy_mean": cv_metrics["overall_top1_acc_mean"],
            "overall_top1_accuracy_std": cv_metrics["overall_top1_acc_std"],
            "candidate_generation_recall_mean": cv_metrics["cand_recall_mean"],
            "candidate_macro_f1_mean": cv_metrics["cand_macro_f1_mean"],
            "candidate_log_loss_mean": cv_metrics["cand_log_loss_mean"],
            "pooled_eligible_top1_accuracy": cv_metrics["pooled_eligible_top1_acc"],
            "pooled_overall_top1_accuracy": cv_metrics["pooled_overall_top1_acc"],
            "inference_latency_ms_per_observation": cv_metrics["latency_ms_per_obs_mean"],
        },
        "baseline_comparison": baseline_comp,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata -> {METADATA_PATH}")

    # 8. Test load with PathSnapPredictor
    print(f"\nVerifying inference readiness with load_predictor()...")
    predictor = load_predictor(model_dir=ARTIFACTS_DIR)
    sample_eval = df.iloc[:10][feature_cols]
    sample_probs = predictor.predict_proba(sample_eval)
    print(f"  Verified: Predictor loaded successfully. Sample probabilities: {sample_probs[:3]}")
    print("\n" + "=" * 80)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main(overwrite=True)
