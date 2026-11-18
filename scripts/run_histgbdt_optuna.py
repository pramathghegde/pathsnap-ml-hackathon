"""
Script to execute Phase 2: Bayesian Hyperparameter Optimization for HistGradientBoosting.

Usage:
    py -3.14 scripts/run_histgbdt_optuna.py [--n-trials 30] [--seed 42] [--overwrite]
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import pandas as pd

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.optimize import HistGBDTOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Bayesian Hyperparameter Optimization for PathSnap HistGradientBoosting."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_features.csv",
        help="Path to training features CSV.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=REPO_ROOT / "configs" / "feature_schema.json",
        help="Path to feature schema JSON.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "histgbdt_ranking_evaluation.json",
        help="Path to Phase 1 baseline evaluation JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "histgbdt_optuna_trials.csv",
        help="Output CSV path for trial logs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "histgbdt_optuna_summary.json",
        help="Output JSON path for summary report.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to run (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TPESampler (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trial and summary artifacts.",
    )

    args = parser.parse_args()

    # Overwrite protection
    if not args.overwrite:
        for p in (args.output_csv, args.output_json):
            if p.exists():
                raise FileExistsError(
                    f"Output file '{p}' already exists. Pass --overwrite to replace."
                )

    if not args.data_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")
    if not args.schema_path.is_file():
        raise FileNotFoundError(f"Schema not found at {args.schema_path}")

    # Load schema
    with open(args.schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema["features"]
    target_col = schema.get("target", "is_match")
    group_col = schema.get("group_column", "trip_id")
    filter_col = schema.get("training_filter_column", "is_training_eligible")
    fold_col = schema.get("fold_column", "fold")
    n_splits = schema.get("n_splits", 5)

    # Load baseline metrics if available
    baseline_metrics = None
    if args.baseline_json.is_file():
        with open(args.baseline_json, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            baseline_metrics = base_data.get("aggregate_metrics", {})

    print("=" * 80)
    print("PATHSNAP: BAYESIAN HYPERPARAMETER OPTIMIZATION (HISTGRADIENTBOOSTING)")
    print("=" * 80)
    print(f"Data Path:          {args.data_path}")
    print(f"Feature Count:      {len(feature_cols)} features: {feature_cols}")
    print(f"Target Column:      '{target_col}'")
    print(f"Group Column:       '{group_col}' ({n_splits} Folds)")
    print(f"Training Filter:    '{filter_col} == 1' (Strictly eligible rows)")
    print(f"Optimization Goal:  Maximize Mean 5-Fold Eligible Top-1 Accuracy")
    print(f"Trial Budget:       {args.n_trials} trials (TPE Sampler, Seed={args.seed})")
    print("=" * 80)

    # Load data once
    print(f"\n[1/3] Loading dataset from {args.data_path}...")
    t0 = time.time()
    df = pd.read_csv(
        args.data_path,
        dtype={
            group_col: str,
            "observation_id": int,
            "candidate_arc_id": int,
            fold_col: int,
            target_col: int,
            filter_col: int,
        },
    )
    t_load = time.time() - t0
    print(f"  Loaded {len(df):,} candidate rows across {df[group_col].nunique()} trips ({t_load:.2f}s)")

    # Instantiate optimizer
    print(f"\n[2/3] Initializing HistGBDTOptimizer and pre-partitioning folds...")
    optimizer = HistGBDTOptimizer(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        filter_col=filter_col,
        fold_col=fold_col,
        n_splits=n_splits,
        seed=args.seed,
        tie_breaker_col="perp_dist",
        ascending_tie_breaker=True,
    )
    print(f"  Validated {n_splits} folds with zero trip leakage.")

    # Run Optuna study
    print(f"\n[3/3] Running Optuna study with {args.n_trials} trials...")
    t_study_start = time.time()

    # Progress header
    header = (
        f"{'Trial':^6} | {'Status':^8} | {'Mean Top-1':^11} | {'Overall Top-1':^13} | "
        f"{'Macro F1':^10} | {'Log Loss':^10} | {'Time (s)':^8}"
    )
    print("\n" + header)
    print("-" * len(header))

    def print_callback(study, trial):
        status = trial.state.name
        val_str = (
            f"{trial.value * 100:.2f}%" if trial.value is not None else "N/A"
        )
        overall_top1 = trial.user_attrs.get("overall_top1_acc_mean", None)
        overall_str = f"{overall_top1 * 100:.2f}%" if overall_top1 else "N/A"
        macro_f1 = trial.user_attrs.get("cand_macro_f1_mean", None)
        f1_str = f"{macro_f1:.4f}" if macro_f1 else "N/A"
        log_loss_val = trial.user_attrs.get("cand_log_loss_mean", None)
        ll_str = f"{log_loss_val:.4f}" if log_loss_val else "N/A"
        dur = trial.user_attrs.get("trial_wall_time_sec", 0.0)

        print(
            f"{trial.number:^6} | {status:^8} | {val_str:^11} | {overall_str:^13} | "
            f"{f1_str:^10} | {ll_str:^10} | {dur:8.1f}"
        )

    study = optimizer.run_optimization(
        n_trials=args.n_trials,
        enqueue_baseline=True,
        study_name="histgbdt_ranking_optimization",
        callbacks=[print_callback],
    )
    total_study_time = time.time() - t_study_start
    print("-" * len(header))

    # Export artifacts
    print(f"\nPersisting artifacts...")
    optimizer.export_artifacts(
        study=study,
        output_csv=args.output_csv,
        output_json=args.output_json,
        baseline_metrics=baseline_metrics,
        overwrite=args.overwrite,
    )
    print(f"  Saved trials CSV  -> {args.output_csv}")
    print(f"  Saved summary JSON -> {args.output_json}")

    # Summary report
    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total Study Time:        {total_study_time:.2f}s ({total_study_time/60:.2f} min)")
    print(f"Total Trials:            {len(study.trials)}")
    print(f"Completed Trials:        {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
    print(f"Failed Trials:           {len([t for t in study.trials if t.state.name == 'FAIL'])}")
    print(f"Pruned Trials:           {len([t for t in study.trials if t.state.name == 'PRUNED'])}")
    print(f"Best Trial Number:       #{best_trial.number}")
    print(f"Best Mean Top-1 Acc:     {best_trial.value * 100:.3f}%")
    print(f"Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  • {k:20s}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
