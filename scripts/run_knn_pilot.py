"""
PathSnap Controlled KNN Feasibility Pilot Runner.

Evaluates K-Nearest Neighbors on a deterministic, observation-level sub-sample
of the training data to measure computational scaling and classification viability
on candidate-level trajectory data.

Protocol:
1. Load data/processed/train_features.csv only.
2. Require and filter rows using is_training_eligible == 1.
3. Select 12 training trips deterministically using random seed 42.
4. Within each selected trip, sample 10% of unique observation_id values using seed 42
   (RandomState initialized once before the trip loop).
5. Retain ALL candidate rows belonging to every sampled observation.
6. Use the exact nine predictive features from configs/feature_schema.json.
7. Execute 3-fold GroupKFold cross-validation via run_training_pipeline() with:
   - selected_models=["KNN"]
   - n_splits=3
   - random_state=42
   - class_weight=None
   - refit_final=False
   - save_artifacts=False
8. Print dataset statistics, fold-level metrics, and aggregate metrics.
9. Persist results to experiments/results/knn_pilot.json and knn_pilot.csv
   with strict overwrite protection.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train import run_training_pipeline


def run_knn_pilot() -> None:
    data_path = REPO_ROOT / "data" / "processed" / "train_features.csv"
    schema_path = REPO_ROOT / "configs" / "feature_schema.json"
    output_json = REPO_ROOT / "experiments" / "results" / "knn_pilot.json"
    output_csv = REPO_ROOT / "experiments" / "results" / "knn_pilot.csv"

    # 10. Enforce strict overwrite protection
    if output_json.exists():
        raise FileExistsError(f"Output file already exists: {output_json}")
    if output_csv.exists():
        raise FileExistsError(f"Output file already exists: {output_csv}")

    if not data_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")

    # Load feature schema
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema["features"]
    target_col = schema.get("target", "is_match")
    group_col = schema.get("group_column", "trip_id")
    filter_col = schema.get("training_filter_column", "is_training_eligible")

    print("=" * 80)
    print("PATHSNAP: CONTROLLED KNN COMPUTATIONAL FEASIBILITY PILOT")
    print("=" * 80)
    print("Note: This is a computational feasibility pilot, not a directly")
    print("comparable full-dataset benchmark.")
    print("-" * 80)

    # 1. Load train_features.csv only
    print(f"Loading {data_path}...")
    t_start = time.time()
    df_raw = pd.read_csv(data_path, dtype={group_col: str})

    # 2. Require is_training_eligible; raise ValueError if missing
    if filter_col not in df_raw.columns:
        raise ValueError(
            f"Required training filter column '{filter_col}' not found in {data_path}. "
            "Cannot proceed without eligible observation filtering."
        )

    df_eligible = df_raw[df_raw[filter_col] == 1].copy().reset_index(drop=True)

    # 3. Select 12 training trips deterministically using random seed 42
    unique_trips = sorted(df_eligible[group_col].unique())
    trip_rng = np.random.RandomState(42)
    selected_trips = sorted(trip_rng.choice(unique_trips, size=12, replace=False))

    # 4 & 5. Sample 10% of unique observation_id per trip and retain ALL candidate rows
    # Initialize observation sampling RNG once before the trip loop
    obs_rng = np.random.RandomState(42)
    sampled_dfs = []

    for tid in selected_trips:
        trip_df = df_eligible[df_eligible[group_col] == tid]
        obs_ids = np.unique(trip_df["observation_id"])
        n_sample_obs = max(1, int(round(len(obs_ids) * 0.10)))
        sampled_obs = obs_rng.choice(obs_ids, size=n_sample_obs, replace=False)
        sampled_trip_df = trip_df[trip_df["observation_id"].isin(sampled_obs)]
        sampled_dfs.append(sampled_trip_df)

    pilot_df = pd.concat(sampled_dfs, ignore_index=True)

    n_selected_trips = int(pilot_df[group_col].nunique())
    n_sampled_observations = int(pilot_df.groupby([group_col, "observation_id"]).ngroups)
    n_candidate_rows = int(len(pilot_df))

    # 8. Print dataset statistics
    print(f"Number of selected trips:       {n_selected_trips}")
    print(f"Number of sampled observations: {n_sampled_observations:,}")
    print(f"Number of candidate rows:       {n_candidate_rows:,}")
    print(f"Predictive features (9):        {feature_cols}")
    print(f"Selected Trip IDs:              {selected_trips}")
    print("-" * 80)

    # 7. Execute 3-fold GroupKFold via run_training_pipeline()
    pipeline_res = run_training_pipeline(
        df=pilot_df,
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
        n_splits=3,
        random_state=42,
        class_weight=None,
        refit_final=False,
        save_artifacts=False,
        selected_models=["KNN"],
    )

    wall_clock_time = time.time() - t_start
    knn_results = pipeline_res["models"]["KNN"]
    folds = knn_results["folds"]
    agg = knn_results["aggregate"]

    # 8. Print fold-level metrics
    print("\n--- KNN Fold-Level Evaluation (3-Fold GroupKFold) ---")
    header = (
        f"{'Fold':^6} | {'Train (s)':^10} | {'Infer (s)':^10} | "
        f"{'Latency (ms)':^12} | {'Accuracy':^10} | {'Macro F1':^10} | {'Log Loss':^10}"
    )
    print(header)
    print("-" * len(header))
    for f in folds:
        fold_id = f.get("fold", "-")
        t_tr = f.get("train_time_sec", 0.0)
        t_inf = f.get("inference_time_sec", 0.0)
        lat = f.get("inference_latency_ms_per_sample", 0.0)
        acc = f.get("accuracy", 0.0)
        f1 = f.get("macro_f1", 0.0)
        ll = f.get("log_loss", np.nan)
        print(f"{fold_id:^6} | {t_tr:10.4f} | {t_inf:10.4f} | {lat:12.6f} | {acc:10.6f} | {f1:10.6f} | {ll:10.6f}")

    # 8. Print aggregate metrics
    print("\n--- Aggregate Metrics (Mean ± Std) ---")
    print(f"  • Macro F1:          {agg['macro_f1_mean']:.6f} ± {agg['macro_f1_std']:.6f}")
    print(f"  • Accuracy:          {agg['accuracy_mean']:.6f} ± {agg['accuracy_std']:.6f}")
    print(f"  • Log Loss:          {agg['log_loss_mean']:.6f} ± {agg['log_loss_std']:.6f}")
    print(f"  • Train Time / Fold: {agg['train_time_sec_mean']:.4f}s ± {agg['train_time_sec_std']:.4f}s")
    print(f"  • Infer Time / Fold: {agg['inference_time_sec_mean']:.4f}s ± {agg['inference_time_sec_std']:.4f}s")
    print(f"  • Inference Latency: {agg['inference_latency_ms_per_sample_mean']:.6f} ms/sample")
    print(f"Total Wall-Clock Execution Time: {wall_clock_time:.2f}s")
    print("=" * 80)

    # 9. Save results to CSV and JSON
    summary_row = {
        "model_name": "KNN",
        "pilot_scale": "12_trips_10pct_obs",
        "num_trips": n_selected_trips,
        "num_observations": n_sampled_observations,
        "num_candidate_rows": n_candidate_rows,
        "n_splits": 3,
        "macro_f1_mean": agg["macro_f1_mean"],
        "macro_f1_std": agg["macro_f1_std"],
        "accuracy_mean": agg["accuracy_mean"],
        "accuracy_std": agg["accuracy_std"],
        "log_loss_mean": agg["log_loss_mean"],
        "log_loss_std": agg["log_loss_std"],
        "train_time_sec_mean": agg["train_time_sec_mean"],
        "train_time_sec_std": agg["train_time_sec_std"],
        "inference_time_sec_mean": agg["inference_time_sec_mean"],
        "inference_time_sec_std": agg["inference_time_sec_std"],
        "inference_latency_ms_per_sample_mean": agg["inference_latency_ms_per_sample_mean"],
        "wall_clock_time_sec": round(wall_clock_time, 2),
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary_row]).to_csv(output_csv, index=False)
    print(f"Saved pilot summary CSV -> {output_csv}")

    json_payload = {
        "pilot_metadata": {
            "description": "PathSnap KNN Computational Feasibility Pilot",
            "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_selected_trips": n_selected_trips,
            "selected_trips": selected_trips,
            "num_sampled_observations": n_sampled_observations,
            "num_candidate_rows": n_candidate_rows,
            "sampling_rate_per_trip": 0.10,
            "random_seed": 42,
            "n_splits": 3,
            "features": feature_cols,
            "target": target_col,
            "wall_clock_time_sec": round(wall_clock_time, 2),
            "feasibility_note": "Pilot on ~14.6k rows to measure computational scaling; not directly comparable to 1.42M row benchmarks.",
        },
        "aggregate_metrics": agg,
        "fold_metrics": folds,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved pilot detailed JSON -> {output_json}")


if __name__ == "__main__":
    run_knn_pilot()
