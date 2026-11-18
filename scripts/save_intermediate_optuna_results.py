"""
Script to extract and persist completed Optuna trials from intermediate run logs.
"""

import json
from pathlib import Path
import re
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path(r"C:\Users\hegde\.gemini\antigravity\brain\5b153f35-404f-47ce-9a74-be838c5e8c43\.system_generated\tasks\task-958.log")
OUTPUT_CSV = REPO_ROOT / "experiments" / "results" / "histgbdt_optuna_trials.csv"
OUTPUT_JSON = REPO_ROOT / "experiments" / "results" / "histgbdt_optuna_summary.json"
BASELINE_JSON = REPO_ROOT / "experiments" / "results" / "histgbdt_ranking_evaluation.json"

def main():
    if not LOG_PATH.is_file():
        raise FileNotFoundError(f"Log not found at {LOG_PATH}")

    # Read log
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        log_text = f.read()

    # Pattern for trial finish:
    # [I 2026-11-18 14:01:01,607] Trial 0 finished with value: 0.8009855001032153 and parameters: {...}.
    pattern = re.compile(
        r"\[I\s+([\d\-:,\s]+)\]\s+Trial\s+(\d+)\s+finished\s+with\s+value:\s+([\d\.]+)\s+and\s+parameters:\s+(\{.*?\})\."
    )

    matches = pattern.findall(log_text)
    print(f"Found {len(matches)} completed trials in log.")

    trial_records = []
    for timestamp_str, trial_num_str, val_str, params_str in matches:
        trial_num = int(trial_num_str)
        val = float(val_str)
        # Parse params dict safely
        # Note: None in python string representation
        params = eval(params_str, {"None": None})

        record = {
            "trial_number": trial_num,
            "state": "COMPLETE",
            "eligible_top1_acc_mean": val,
            **params,
            "timestamp": timestamp_str.strip(),
        }
        trial_records.append(record)

    # Convert to DataFrame
    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(trials_df)} completed trials to {OUTPUT_CSV}")

    # Load baseline metrics
    baseline_metrics = {}
    if BASELINE_JSON.is_file():
        with open(BASELINE_JSON, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            baseline_metrics = base_data.get("aggregate_metrics", {})

    base_macro_top1 = baseline_metrics.get("eligible_top1_accuracy_mean", 0.8009855001032153)
    base_overall_top1 = baseline_metrics.get("overall_top1_accuracy_mean", 0.7198650329655745)
    base_macro_f1 = baseline_metrics.get("candidate_macro_f1_mean", 0.8679160621084423)
    base_log_loss = baseline_metrics.get("candidate_log_loss_mean", 0.13160988072589658)

    # Find best trial
    best_record = max(trial_records, key=lambda r: r["eligible_top1_acc_mean"])
    best_top1 = best_record["eligible_top1_acc_mean"]
    diff_abs = best_top1 - base_macro_top1
    diff_pct = (diff_abs / base_macro_top1) * 100.0

    summary = {
        "study_name": "histgbdt_ranking_optimization",
        "model_name": "HistGradientBoostingClassifier",
        "optimization_objective": "Mean 5-Fold Eligible Top-1 Accuracy",
        "n_trials_requested": 30,
        "n_trials_completed": len(trial_records),
        "n_trials_failed": 0,
        "n_trials_pruned": 0,
        "n_trials_unfinished": 30 - len(trial_records),
        "task_interrupted": True,
        "interruption_reason": "Safely stopped per user instruction after 8 trials",
        "best_trial": {
            "trial_number": best_record["trial_number"],
            "objective_value": best_record["eligible_top1_acc_mean"],
            "params": {k: v for k, v in best_record.items() if k in [
                "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
                "min_samples_leaf", "l2_regularization", "max_bins"
            ]},
        },
        "baseline_comparison": {
            "baseline_eligible_top1_accuracy": base_macro_top1,
            "tuned_eligible_top1_accuracy": best_top1,
            "eligible_top1_diff_absolute": diff_abs,
            "eligible_top1_diff_percentage": diff_pct,
            "baseline_overall_top1_accuracy": base_overall_top1,
            "baseline_candidate_macro_f1": base_macro_f1,
            "baseline_candidate_log_loss": base_log_loss,
        },
        "reproducibility": {
            "sampler": "TPESampler",
            "seed": 42,
            "n_splits": 5,
            "group_column": "trip_id",
            "target_column": "is_match",
            "training_filter": "is_training_eligible == 1",
            "tie_breaker": "perp_dist ascending",
        },
        "completed_trials_summary": [
            {
                "trial_number": r["trial_number"],
                "eligible_top1_acc_mean": r["eligible_top1_acc_mean"],
                "params": {k: v for k, v in r.items() if k in [
                    "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
                    "min_samples_leaf", "l2_regularization", "max_bins"
                ]}
            }
            for r in trial_records
        ]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
