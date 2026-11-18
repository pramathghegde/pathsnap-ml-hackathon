"""
PathSnap HistGradientBoosting 5-Fold Candidate Ranking Benchmark.

Executes:
1. Leakage-free 5-fold cross-validation on all 80 training trips using pre-assigned folds.
2. Fits HistGradientBoosting exclusively on eligible observations (is_training_eligible == 1).
3. Evaluates out-of-fold predictions on the full validation population (including candidate misses).
4. Computes:
   - Eligible Top-1 Accuracy
   - Overall Top-1 Accuracy
   - Candidate Generation Recall
   - Candidate-level Accuracy, Precision, Recall, Macro F1, Log Loss, and Confusion Matrix
   - Prediction latency profiling (per candidate and per observation)
5. Persists results to experiments/results/histgbdt_ranking_evaluation.csv and .json
   without overwriting existing baseline benchmark files.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.evaluate import (
    aggregate_ranking_cv_metrics,
    evaluate_candidate_ranking,
)
from src.models.model_factory import get_model


def run_histgbdt_ranking_benchmark(
    data_path: Path,
    schema_path: Path,
    output_csv: Path,
    output_json: Path,
    overwrite: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute 5-fold out-of-fold candidate ranking evaluation for HistGradientBoosting.
    """
    # Overwrite protection
    if not overwrite:
        for p in (output_csv, output_json):
            if p.exists():
                raise FileExistsError(
                    f"Output file '{p}' already exists. Pass overwrite=True to replace."
                )

    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")

    # Load schema
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema["features"]
    target_col = schema.get("target", "is_match")
    group_col = schema.get("group_column", "trip_id")
    filter_col = schema.get("training_filter_column", "is_training_eligible")
    fold_col = schema.get("fold_column", "fold")
    n_splits = schema.get("n_splits", 5)

    print("=" * 80)
    print("PATHSNAP: HISTGRADIENTBOOSTING CANDIDATE RANKING BENCHMARK")
    print("=" * 80)
    print(f"Data Source:       {data_path}")
    print(f"Feature Schema:    {schema_path}")
    print(f"Target Column:     '{target_col}'")
    print(f"Group Column:      '{group_col}' (5 Folds)")
    print(f"Training Filter:   '{filter_col} == 1' (Strictly eligible rows)")
    print(f"Features ({len(feature_cols)}):     {feature_cols}")
    print("=" * 80)

    # Load dataset
    print(f"\n[1/3] Loading dataset from {data_path}...")
    t_load_start = time.time()
    df = pd.read_csv(
        data_path,
        dtype={
            group_col: str,
            "observation_id": int,
            "candidate_arc_id": int,
            fold_col: int,
            target_col: int,
            filter_col: int,
        },
    )
    t_load = time.time() - t_load_start
    print(f"  Loaded {len(df):,} candidate rows across {df[group_col].nunique()} trips ({t_load:.2f}s)")

    # Verify fold distribution
    fold_trips = df.groupby(fold_col)[group_col].nunique().to_dict()
    print(f"  Trips per fold: {fold_trips}")

    fold_metrics_list: List[Dict[str, Any]] = []
    oof_predictions_list: List[pd.DataFrame] = []

    print("\n[2/3] Executing 5-Fold Out-of-Fold Cross-Validation...")
    header = (
        f"{'Fold':^6} | {'Train (s)':^9} | {'Infer (s)':^9} | "
        f"{'Elig Top-1':^11} | {'Over Top-1':^11} | {'Cand Recall':^11} | "
        f"{'Macro F1':^10} | {'Log Loss':^10}"
    )
    print(header)
    print("-" * len(header))

    total_bench_start = time.time()

    for fold_idx in range(n_splits):
        # 1. Train split: fold != fold_idx AND is_training_eligible == 1
        train_mask = (df[fold_col] != fold_idx) & (df[filter_col] == 1)
        val_mask = df[fold_col] == fold_idx

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col].to_numpy().astype(int)
        val_df = df.loc[val_mask].copy()

        # Verify zero trip leakage
        train_trips = set(df.loc[train_mask, group_col].unique())
        val_trips = set(val_df[group_col].unique())
        overlap = train_trips.intersection(val_trips)
        if overlap:
            raise ValueError(f"Trip leakage in fold {fold_idx}: {overlap}")

        # Instantiate fresh HistGradientBoosting model
        model = get_model("HistGradientBoosting", random_state=random_state)

        # Fit model
        t_fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t_fit_start

        # Evaluate candidate ranking on full validation fold
        eval_metrics = evaluate_candidate_ranking(
            model=model,
            df=val_df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_cols=(group_col, "observation_id"),
            tie_breaker_col="perp_dist",
            ascending_tie_breaker=True,
        )

        eval_metrics["fold"] = fold_idx
        eval_metrics["train_time_sec"] = float(train_time)
        eval_metrics["train_samples"] = int(len(X_train))
        eval_metrics["val_samples"] = int(len(val_df))
        eval_metrics["val_trips"] = int(len(val_trips))

        fold_metrics_list.append(eval_metrics)

        print(
            f"{fold_idx:^6} | {train_time:9.2f} | {eval_metrics['inference_time_sec']:9.4f} | "
            f"{eval_metrics['eligible_top1_accuracy']*100:10.2f}% | "
            f"{eval_metrics['overall_top1_accuracy']*100:10.2f}% | "
            f"{eval_metrics['candidate_generation_recall']*100:10.2f}% | "
            f"{eval_metrics['candidate_macro_f1']:10.4f} | "
            f"{eval_metrics['candidate_log_loss']:10.4f}"
        )

    total_bench_time = time.time() - total_bench_start

    # Aggregate across folds
    agg = aggregate_ranking_cv_metrics(fold_metrics_list)

    print("-" * len(header))
    print(f"\n>>> 5-Fold Summary (HistGradientBoosting, Wall Time: {total_bench_time:.1f}s):")
    print(f"  • Eligible Top-1 Accuracy:       {agg['eligible_top1_accuracy_mean']*100:.2f}% ± {agg['eligible_top1_accuracy_std']*100:.2f}%")
    print(f"  • Overall Top-1 Accuracy:        {agg['overall_top1_accuracy_mean']*100:.2f}% ± {agg['overall_top1_accuracy_std']*100:.2f}%")
    print(f"  • Candidate Generation Recall:   {agg['candidate_generation_recall_mean']*100:.2f}% ± {agg['candidate_generation_recall_std']*100:.2f}%")
    print(f"  • Candidate Macro F1:            {agg['candidate_macro_f1_mean']:.4f} ± {agg['candidate_macro_f1_std']:.4f}")
    print(f"  • Candidate Accuracy:            {agg['candidate_accuracy_mean']*100:.2f}% ± {agg['candidate_accuracy_std']*100:.2f}%")
    print(f"  • Candidate Precision:           {agg['candidate_precision_mean']:.4f} ± {agg['candidate_precision_std']:.4f}")
    print(f"  • Candidate Recall:              {agg['candidate_recall_mean']:.4f} ± {agg['candidate_recall_std']:.4f}")
    print(f"  • Candidate Log Loss:            {agg['candidate_log_loss_mean']:.4f} ± {agg['candidate_log_loss_std']:.4f}")
    print(f"  • Avg Training Time / Fold:      {agg['train_time_sec_mean']:.2f}s ± {agg['train_time_sec_std']:.2f}s")
    print(f"  • Latency per Candidate:         {agg['inference_latency_ms_per_candidate_mean']:.4f} ms")
    print(f"  • Latency per Observation:       {agg['inference_latency_ms_per_observation_mean']:.4f} ms")
    print(f"  • Total Observations Evaluated:  {agg['num_observations_total_sum']:,}")
    print(f"  • Total Eligible Observations:   {agg['num_observations_eligible_sum']:,}")
    print(f"  • Total Candidate Misses:        {agg['num_candidate_misses_sum']:,}")
    print(f"  • Total Correct Top-1 Matches:   {agg['num_correct_top1_sum']:,}")
    print("=" * 80)

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Construct tabular summary DataFrame
    fold_rows = []
    for f in fold_metrics_list:
        fold_rows.append(
            {
                "fold": f["fold"],
                "train_samples": f["train_samples"],
                "val_samples": f["val_samples"],
                "num_obs_total": f["num_observations_total"],
                "num_obs_eligible": f["num_observations_eligible"],
                "num_cand_misses": f["num_candidate_misses"],
                "num_correct_top1": f["num_correct_top1"],
                "eligible_top1_acc": round(f["eligible_top1_accuracy"], 4),
                "overall_top1_acc": round(f["overall_top1_accuracy"], 4),
                "cand_recall_rate": round(f["candidate_generation_recall"], 4),
                "cand_macro_f1": round(f["candidate_macro_f1"], 4),
                "cand_accuracy": round(f["candidate_accuracy"], 4),
                "cand_precision": round(f["candidate_precision"], 4),
                "cand_recall": round(f["candidate_recall"], 4),
                "cand_log_loss": round(f["candidate_log_loss"], 4),
                "train_time_sec": round(f["train_time_sec"], 2),
                "infer_time_sec": round(f["inference_time_sec"], 3),
                "latency_ms_per_cand": round(f["inference_latency_ms_per_candidate"], 4),
                "latency_ms_per_obs": round(f["inference_latency_ms_per_observation"], 4),
            }
        )

    # Add aggregate row
    summary_df = pd.DataFrame(fold_rows)
    summary_df.to_csv(output_csv, index=False)
    print(f"\n[3/3] Saved fold ranking metrics -> {output_csv}")

    # JSON export with full structure
    json_data = {
        "model_name": "HistGradientBoosting",
        "evaluation_type": "5-Fold Out-of-Fold Candidate Ranking",
        "dataset": str(data_path),
        "total_wall_clock_time_sec": round(total_bench_time, 2),
        "aggregate_metrics": agg,
        "fold_metrics": fold_metrics_list,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved detailed JSON report      -> {output_json}")

    return {
        "aggregate": agg,
        "folds": fold_metrics_list,
        "summary_table": summary_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PathSnap HistGradientBoosting Candidate Ranking Benchmark Runner"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_features.csv",
        help="Path to processed train_features.csv",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=REPO_ROOT / "configs" / "feature_schema.json",
        help="Path to feature_schema.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "histgbdt_ranking_evaluation.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "histgbdt_ranking_evaluation.json",
        help="Path to output JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files if they exist",
    )

    args = parser.parse_args()

    run_histgbdt_ranking_benchmark(
        data_path=args.data_path,
        schema_path=args.schema_path,
        output_csv=args.output_csv,
        output_json=args.output_json,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
