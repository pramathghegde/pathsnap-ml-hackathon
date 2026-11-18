"""
Detailed 5-fold validation of Trial 6 against Phase 1 Baseline.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.optimize import HistGBDTOptimizer
DATA_PATH = REPO_ROOT / "data" / "processed" / "train_features.csv"
SCHEMA_PATH = REPO_ROOT / "configs" / "feature_schema.json"
BASELINE_PATH = REPO_ROOT / "experiments" / "results" / "histgbdt_ranking_evaluation.json"

def main():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    features = schema["features"]
    target_col = schema.get("target", "is_match")
    group_col = schema.get("group_column", "trip_id")
    filter_col = schema.get("training_filter_column", "is_training_eligible")
    fold_col = schema.get("fold_column", "fold")
    n_splits = schema.get("n_splits", 5)

    print("Loading dataset...")
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            group_col: str,
            "observation_id": int,
            "candidate_arc_id": int,
            fold_col: int,
            target_col: int,
            filter_col: int,
        },
    )

    optimizer = HistGBDTOptimizer(
        df=df,
        feature_cols=features,
        target_col=target_col,
        group_col=group_col,
        filter_col=filter_col,
        fold_col=fold_col,
        n_splits=n_splits,
        seed=42,
    )

    trial6_params = {
        "learning_rate": 0.04228762725326872,
        "max_iter": 400,
        "max_leaf_nodes": 46,
        "max_depth": None,
        "min_samples_leaf": 28,
        "l2_regularization": 0.3142918568673425,
        "max_bins": 186,
    }

    print("Evaluating Trial 6 across 5 folds...")
    primary_obj, metrics, fold_metrics, oof_df = optimizer.evaluate_hyperparameters(trial6_params)

    # Compare with baseline
    base_folds = baseline_data["fold_metrics"]
    base_agg = baseline_data["aggregate_metrics"]

    print("\n" + "=" * 90)
    print("FOLD-BY-FOLD COMPARISON: BASELINE vs. TRIAL 6 (TUNED)")
    print("=" * 90)
    header = f"{'Fold':^6} | {'Base Elig Top1':^16} | {'Tuned Elig Top1':^16} | {'Diff':^10} | {'Base Over Top1':^16} | {'Tuned Over Top1':^16}"
    print(header)
    print("-" * len(header))

    for k in range(n_splits):
        b_e = base_folds[k]["eligible_top1_accuracy"] * 100
        t_e = fold_metrics[k]["eligible_top1_accuracy"] * 100
        diff = t_e - b_e
        b_o = base_folds[k]["overall_top1_accuracy"] * 100
        t_o = fold_metrics[k]["overall_top1_accuracy"] * 100
        print(f"{k:^6} | {b_e:14.2f}% | {t_e:14.2f}% | {diff:+8.2f}% | {b_o:14.2f}% | {t_o:14.2f}%")

    print("-" * len(header))
    b_mean = base_agg["eligible_top1_accuracy_mean"] * 100
    t_mean = metrics["eligible_top1_acc_mean"] * 100
    b_std = base_agg["eligible_top1_accuracy_std"] * 100
    t_std = metrics["eligible_top1_acc_std"] * 100
    print(f"Mean ± Std: {b_mean:.2f}% ± {b_std:.2f}% vs. {t_mean:.2f}% ± {t_std:.2f}% (Diff: {t_mean - b_mean:+.3f}%)")

    # Pooled comparison
    base_pooled_e = (base_agg["num_correct_top1_sum"] / base_agg["num_observations_eligible_sum"]) * 100
    tuned_pooled_e = metrics["pooled_eligible_top1_acc"] * 100
    print(f"Pooled Elig Top1: {base_pooled_e:.2f}% vs. {tuned_pooled_e:.2f}% (Diff: {tuned_pooled_e - base_pooled_e:+.3f}%)")
    print(f"Candidate Macro F1: {base_agg['candidate_macro_f1_mean']:.4f} vs. {metrics['cand_macro_f1_mean']:.4f}")
    print(f"Candidate Log Loss: {base_agg['candidate_log_loss_mean']:.4f} vs. {metrics['cand_log_loss_mean']:.4f}")
    print(f"Candidate Recall:   {base_agg['candidate_generation_recall_mean']*100:.2f}% vs. {metrics['cand_recall_mean']*100:.2f}%")
    print(f"Mean Train Time:    {base_agg['train_time_sec_mean']:.2f}s vs. {metrics['train_time_sec_mean']:.2f}s")
    print(f"Mean Latency / Obs: {base_agg['inference_latency_ms_per_observation_mean']:.4f}ms vs. {metrics['latency_ms_per_obs_mean']:.4f}ms")
    print("=" * 90)

    # Save validation details for report
    result_payload = {
        "trial_number": 6,
        "params": trial6_params,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "baseline_comparison": {
            "baseline_mean_eligible_top1": b_mean / 100.0,
            "tuned_mean_eligible_top1": t_mean / 100.0,
            "baseline_pooled_eligible_top1": base_pooled_e / 100.0,
            "tuned_pooled_eligible_top1": tuned_pooled_e / 100.0,
            "baseline_macro_f1": base_agg["candidate_macro_f1_mean"],
            "tuned_macro_f1": metrics["cand_macro_f1_mean"],
            "baseline_log_loss": base_agg["candidate_log_loss_mean"],
            "tuned_log_loss": metrics["cand_log_loss_mean"],
        }
    }
    with open(REPO_ROOT / "experiments" / "results" / "trial6_validation_details.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print("Saved trial 6 detailed validation to experiments/results/trial6_validation_details.json")

if __name__ == "__main__":
    main()
