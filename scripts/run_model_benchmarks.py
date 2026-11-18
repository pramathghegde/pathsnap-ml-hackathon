"""
PathSnap Automated Model Benchmarking Runner.

Executes sequential, leakage-free 5-fold GroupKFold baseline benchmarking across:
1. Logistic Regression
2. HistGradientBoosting
3. Decision Tree
4. Random Forest

Features:
- Loads canonical configuration from configs/feature_schema.json
- Filters training rows strictly to is_training_eligible == 1
- Uses exact nine numerical features from schema
- Precomputes identical GroupKFold splits strictly on trip_id
- Evaluates models sequentially (one at a time) to prevent resource contention
- Formats and prints per-fold and aggregated performance metrics
- Saves final comparative benchmarks to experiments/results/baseline_model_comparison.csv
  and baseline_model_comparison.json without overwriting existing files
- Excludes KNN from full-data benchmark due to computational scaling constraints
"""

import argparse
import gc
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

from src.models.train import run_training_pipeline

BENCHMARK_MODELS = [
    "Logistic Regression",
    "HistGradientBoosting",
    "Decision Tree",
    "Random Forest",
]


def print_fold_table(model_name: str, folds: List[Dict[str, Any]]) -> None:
    """Print a clean, aligned tabular view of fold-level metrics."""
    print(f"\n--- {model_name}: Fold-Level Evaluation ---")
    header = (
        f"{'Fold':^6} | {'Train (s)':^10} | {'Infer (s)':^10} | "
        f"{'Latency (ms)':^12} | {'Accuracy':^10} | {'Macro F1':^10} | {'Log Loss':^10}"
    )
    print(header)
    print("-" * len(header))

    for f in folds:
        fold_id = f.get("fold", "-")
        t_train = f.get("train_time_sec", 0.0)
        t_infer = f.get("inference_time_sec", 0.0)
        lat_ms = f.get("inference_latency_ms_per_sample", 0.0)
        acc = f.get("accuracy", 0.0)
        f1 = f.get("macro_f1", 0.0)
        ll = f.get("log_loss", np.nan)
        ll_str = f"{ll:10.6f}" if not np.isnan(ll) else f"{'INVALID':^10}"

        print(
            f"{fold_id:^6} | {t_train:10.2f} | {t_infer:10.4f} | "
            f"{lat_ms:12.6f} | {acc:10.6f} | {f1:10.6f} | {ll_str}"
        )


def print_aggregate_summary(model_name: str, agg: Dict[str, Any], wall_time: float) -> None:
    """Print mean and std summary metrics for an evaluated model."""
    print(f"\n>>> {model_name} Summary (5-Fold GroupKFold, Wall Time: {wall_time:.1f}s):")
    print(f"  • Macro F1:          {agg['macro_f1_mean']:.6f} ± {agg['macro_f1_std']:.6f}")
    print(f"  • Accuracy:          {agg['accuracy_mean']:.6f} ± {agg['accuracy_std']:.6f}")
    ll_mean = agg.get("log_loss_mean", np.nan)
    ll_std = agg.get("log_loss_std", np.nan)
    if not np.isnan(ll_mean):
        print(f"  • Log Loss:          {ll_mean:.6f} ± {ll_std:.6f}")
    else:
        print(f"  • Log Loss:          INVALID ({agg.get('log_loss_status', 'N/A')})")
    print(f"  • Train Time / Fold: {agg['train_time_sec_mean']:.2f}s ± {agg['train_time_sec_std']:.2f}s")
    print(f"  • Inference Latency: {agg['inference_latency_ms_per_sample_mean']:.6f} ms/sample")
    print("-" * 75)


def run_benchmark(
    data_path: Path,
    schema_path: Path,
    output_csv: Path,
    output_json: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Execute the sequential baseline model benchmark.

    Args:
        data_path: Path to processed train_features.csv.
        schema_path: Path to configs/feature_schema.json.
        output_csv: Path for saving combined benchmark CSV.
        output_json: Path for saving combined benchmark JSON.
        overwrite: Whether to overwrite existing result files.

    Returns:
        DataFrame containing comparative model benchmark metrics.
    """
    # 1. Check overwrite protection
    if not overwrite:
        for p in (output_csv, output_json):
            if p.exists():
                raise FileExistsError(
                    f"Output file '{p}' already exists. Pass overwrite=True to replace."
                )

    # 2. Load feature schema
    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema["features"]
    target_col = schema.get("target", "is_match")
    group_col = schema.get("group_column", "trip_id")
    n_splits = schema.get("n_splits", 5)
    filter_col = schema.get("training_filter_column", "is_training_eligible")

    print("=" * 80)
    print("PATHSNAP: AUTOMATED MODEL BENCHMARKING ENGINE")
    print("=" * 80)
    print(f"Data Source:       {data_path}")
    print(f"Feature Schema:    {schema_path}")
    print(f"Target Column:     '{target_col}'")
    print(f"Grouping Column:   '{group_col}' (GroupKFold, n_splits={n_splits})")
    print(f"Eligibility Filter:'{filter_col} == 1'")
    print(f"Predictive Feats:  {len(feature_cols)} features: {feature_cols}")
    print(f"Models to Run:     {BENCHMARK_MODELS}")
    print(f"KNN Status:        EXCLUDED (Prediction scaling requires pilot study)")
    print("=" * 80)

    # 3. Load processed training dataset
    if not data_path.is_file():
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    print(f"\n[1/3] Loading dataset from {data_path}...")
    t_load_start = time.time()
    df_raw = pd.read_csv(data_path)
    t_load = time.time() - t_load_start
    print(f"  Loaded raw matrix: {len(df_raw):,} rows, {len(df_raw.columns)} cols ({t_load:.1f}s)")

    # 4. Filter to eligible observations
    if filter_col in df_raw.columns:
        df_eligible = df_raw[df_raw[filter_col] == 1].copy().reset_index(drop=True)
        print(
            f"  Applied filter '{filter_col} == 1': "
            f"{len(df_eligible):,} eligible rows ({len(df_raw) - len(df_eligible):,} rows excluded)"
        )
    else:
        print(f"  Warning: Filter column '{filter_col}' not found. Using full dataset.")
        df_eligible = df_raw

    unique_trips = df_eligible[group_col].nunique()
    print(f"  Active training partition: {len(df_eligible):,} rows across {unique_trips} unique trips.")
    del df_raw
    gc.collect()

    # 5. Run sequential benchmarks
    print(f"\n[2/3] Executing 5-Fold GroupKFold baseline experiments sequentially...")
    all_model_results: Dict[str, Any] = {}
    combined_rows: List[Dict[str, Any]] = []
    total_benchmark_start = time.time()

    for idx, model_name in enumerate(BENCHMARK_MODELS, start=1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(BENCHMARK_MODELS)}] BENCHMARKING: {model_name}")
        print(f"{'='*80}")

        t_model_start = time.time()

        # Run GroupKFold pipeline for this single model
        pipeline_res = run_training_pipeline(
            df=df_eligible,
            target_col=target_col,
            group_col=group_col,
            feature_cols=feature_cols,
            n_splits=n_splits,
            random_state=42,
            class_weight=None,
            refit_final=False,
            save_artifacts=False,
            selected_models=[model_name],
        )

        wall_time = time.time() - t_model_start
        model_data = pipeline_res["models"][model_name]
        all_model_results[model_name] = model_data

        # Print detailed fold and summary metrics
        print_fold_table(model_name, model_data["folds"])
        print_aggregate_summary(model_name, model_data["aggregate"], wall_time)

        # Prepare summary row
        row = {"model_name": model_name, "wall_clock_time_sec": round(wall_time, 2)}
        for k, v in model_data["aggregate"].items():
            if not isinstance(v, list):
                row[k] = v
        combined_rows.append(row)

        # Trigger garbage collection between heavy models
        gc.collect()

    total_wall_time = time.time() - total_benchmark_start
    print("\n" + "=" * 80)
    print(f"ALL {len(BENCHMARK_MODELS)} MODELS COMPLETED IN {total_wall_time:.1f}s ({total_wall_time/60:.2f} mins)!")
    print("=" * 80)

    # 6. Comparative Summary Table
    df_comparison = pd.DataFrame(combined_rows)
    # Sort by Macro F1 descending
    df_comparison = df_comparison.sort_values(by="macro_f1_mean", ascending=False).reset_index(drop=True)

    print("\n[3/3] FINAL COMPARATIVE BENCHMARK RANKING (Ranked by Macro F1):")
    display_cols = [
        "model_name",
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "log_loss_mean",
        "train_time_sec_mean",
        "inference_latency_ms_per_sample_mean",
        "wall_clock_time_sec",
    ]
    present_cols = [c for c in display_cols if c in df_comparison.columns]
    print(df_comparison[present_cols].to_string(index=False))

    # 7. Persist Comparative Benchmark Artifacts
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    df_comparison.to_csv(output_csv, index=False)
    print(f"\nSaved Comparative Benchmark Table  -> {output_csv}")

    json_payload = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_wall_clock_time_sec": round(total_wall_time, 2),
            "num_models_evaluated": len(BENCHMARK_MODELS),
            "training_samples": len(df_eligible),
            "unique_trips": unique_trips,
            "n_splits": n_splits,
            "class_weight": None,
            "random_state": 42,
            "knn_status": "EXCLUDED (Pending Pilot Study)",
        },
        "comparative_summary": combined_rows,
        "detailed_results": all_model_results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved Detailed Benchmark Breakdown -> {output_json}")

    return df_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PathSnap Automated Model Benchmarking Engine"
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
        default=REPO_ROOT / "experiments" / "results" / "baseline_model_comparison.csv",
        help="Destination path for benchmark comparison CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "baseline_model_comparison.json",
        help="Destination path for benchmark comparison JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files if they exist",
    )

    args = parser.parse_args()

    run_benchmark(
        data_path=args.data_path,
        schema_path=args.schema_path,
        output_csv=args.output_csv,
        output_json=args.output_json,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
