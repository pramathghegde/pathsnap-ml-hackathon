"""
Bayesian Hyperparameter Optimization for PathSnap Candidate Ranking Models.

Provides:
- sample_histgbdt_params: Defines the search space for HistGradientBoostingClassifier.
- create_histgbdt_study: Instantiates an Optuna study with a seeded TPESampler.
- HistGBDTOptimizer: Leakage-free GroupKFold cross-validation Bayesian optimizer
  targeting observation-level Eligible Top-1 candidate ranking accuracy.
"""

import copy
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial, TrialState
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.models.evaluate import (
    compute_observation_ranking_metrics,
    evaluate_candidate_ranking,
)


# Default search space bounds for HistGradientBoostingClassifier
DEFAULT_HISTGBDT_SEARCH_SPACE = {
    "learning_rate": (0.02, 0.20),
    "max_iter": (100, 500, 50),
    "max_leaf_nodes": (15, 63),
    "max_depth_choices": [None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "min_samples_leaf": (20, 100),
    "l2_regularization": (0.0, 10.0),
    "max_bins": (64, 255),
}

DEFAULT_BASELINE_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_bins": 255,
}


def sample_histgbdt_params(
    trial: Trial,
    search_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Sample hyperparameters for HistGradientBoostingClassifier from an Optuna trial.

    Search space:
    - learning_rate: float in [0.02, 0.20], log-scale
    - max_iter: integer in [100, 500], step 50
    - max_leaf_nodes: integer in [15, 63]
    - max_depth: categorical in [None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    - min_samples_leaf: integer in [20, 100]
    - l2_regularization: float in [0.0, 10.0]
    - max_bins: integer in [64, 255]

    Args:
        trial: Optuna Trial instance.
        search_space: Optional custom search space bounds dictionary.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    cfg = search_space or DEFAULT_HISTGBDT_SEARCH_SPACE

    lr_min, lr_max = cfg["learning_rate"]
    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

    iter_min, iter_max, iter_step = cfg["max_iter"]
    max_iter = trial.suggest_int("max_iter", iter_min, iter_max, step=iter_step)

    leaf_min, leaf_max = cfg["max_leaf_nodes"]
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", leaf_min, leaf_max)

    depth_choices = cfg["max_depth_choices"]
    max_depth = trial.suggest_categorical("max_depth", depth_choices)

    msl_min, msl_max = cfg["min_samples_leaf"]
    min_samples_leaf = trial.suggest_int("min_samples_leaf", msl_min, msl_max)

    l2_min, l2_max = cfg["l2_regularization"]
    l2_regularization = trial.suggest_float("l2_regularization", l2_min, l2_max)

    bins_min, bins_max = cfg["max_bins"]
    max_bins = trial.suggest_int("max_bins", bins_min, bins_max)

    return {
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "max_bins": max_bins,
    }


def create_histgbdt_study(
    study_name: str = "histgbdt_ranking_optimization",
    seed: int = 42,
    direction: str = "maximize",
) -> optuna.Study:
    """
    Create a deterministic Optuna study using the Tree-structured Parzen Estimator (TPE).

    Args:
        study_name: Unique name for the study.
        seed: Random seed for TPESampler reproducibility.
        direction: Optimization direction ('maximize' for ranking accuracy).

    Returns:
        Configured Optuna Study instance.
    """
    sampler = TPESampler(seed=seed)
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
    )


class HistGBDTOptimizer:
    """
    Bayesian hyperparameter optimizer for HistGradientBoosting candidate ranking.

    Ensures:
    1. Pre-partitioned 5-fold GroupKFold validation slices grouped strictly by trip_id.
    2. Zero trip leakage across folds with explicit assertion checks.
    3. Training strictly on eligible observations (is_training_eligible == 1).
    4. Validation on 100% of candidate observations (including candidate generation misses).
    5. Observation-level Top-1 Eligible Accuracy as the primary objective.
    6. Computation of both macro-averaged and pooled ranking and classification metrics.
    7. Transparent recording of completed, failed, and pruned trials.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "is_match",
        group_col: str = "trip_id",
        filter_col: str = "is_training_eligible",
        fold_col: str = "fold",
        n_splits: int = 5,
        seed: int = 42,
        tie_breaker_col: str = "perp_dist",
        ascending_tie_breaker: bool = True,
    ) -> None:
        self.df = df
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.group_col = group_col
        self.filter_col = filter_col
        self.fold_col = fold_col
        self.n_splits = n_splits
        self.seed = seed
        self.tie_breaker_col = tie_breaker_col
        self.ascending_tie_breaker = ascending_tie_breaker

        self.trial_records: List[Dict[str, Any]] = []

        # Validate required columns
        req_cols = (
            self.feature_cols
            + [self.target_col, self.group_col, self.fold_col, self.filter_col]
        )
        if self.tie_breaker_col:
            req_cols.append(self.tie_breaker_col)
        missing = [c for c in req_cols if c not in self.df.columns]
        if missing:
            raise KeyError(f"Missing required columns in dataset: {missing}")

        # Precompute and verify 5-fold data partitions once
        self._precompute_fold_partitions()

    def _precompute_fold_partitions(self) -> None:
        """Pre-extract fold slices and verify zero group leakage across folds."""
        self.fold_data: List[Dict[str, Any]] = []

        for fold_idx in range(self.n_splits):
            train_mask = (self.df[self.fold_col] != fold_idx) & (
                self.df[self.filter_col] == 1
            )
            val_mask = self.df[self.fold_col] == fold_idx

            train_trips = set(self.df.loc[train_mask, self.group_col].unique())
            val_trips = set(self.df.loc[val_mask, self.group_col].unique())
            overlap = train_trips.intersection(val_trips)
            if overlap:
                raise ValueError(
                    f"Trip leakage detected in fold {fold_idx}: overlapping trips {overlap}"
                )

            X_train = self.df.loc[train_mask, self.feature_cols]
            y_train = self.df.loc[train_mask, self.target_col].to_numpy().astype(int)
            val_df = self.df.loc[val_mask].copy()

            self.fold_data.append(
                {
                    "fold": fold_idx,
                    "X_train": X_train,
                    "y_train": y_train,
                    "val_df": val_df,
                    "train_samples": len(X_train),
                    "val_samples": len(val_df),
                    "val_trips": len(val_trips),
                }
            )

    def evaluate_hyperparameters(
        self,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
        """
        Evaluate a single set of hyperparameters across all cross-validation folds.

        Args:
            params: Dictionary of HistGradientBoosting hyperparameters.

        Returns:
            Tuple of:
            - primary_objective (mean Eligible Top-1 Accuracy)
            - macro_metrics dictionary
            - fold_metrics list
            - pooled_oof_df with predicted probabilities
        """
        fold_metrics_list: List[Dict[str, Any]] = []
        oof_frames: List[pd.DataFrame] = []

        total_train_time = 0.0
        total_infer_time = 0.0

        for fold_item in self.fold_data:
            fold_idx = fold_item["fold"]
            X_train = fold_item["X_train"]
            y_train = fold_item["y_train"]
            val_df = fold_item["val_df"]

            # Instantiate model with trial parameters and fixed seed
            model = HistGradientBoostingClassifier(
                **params,
                random_state=self.seed,
            )

            # Fit model and profile training time
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            total_train_time += train_time

            # Evaluate candidate ranking and candidate classification
            eval_metrics = evaluate_candidate_ranking(
                model=model,
                df=val_df,
                feature_cols=self.feature_cols,
                target_col=self.target_col,
                group_cols=(self.group_col, "observation_id"),
                tie_breaker_col=self.tie_breaker_col,
                ascending_tie_breaker=self.ascending_tie_breaker,
            )

            eval_metrics["fold"] = fold_idx
            eval_metrics["train_time_sec"] = float(train_time)
            eval_metrics["train_samples"] = fold_item["train_samples"]
            eval_metrics["val_samples"] = fold_item["val_samples"]
            eval_metrics["val_trips"] = fold_item["val_trips"]
            total_infer_time += eval_metrics["inference_time_sec"]

            fold_metrics_list.append(eval_metrics)

            # Capture out-of-fold scored frame for pooled evaluation
            pos_class_idx = 1
            if hasattr(model, "classes_") and 1 in model.classes_:
                pos_class_idx = int(np.where(model.classes_ == 1)[0][0])
            val_scores = model.predict_proba(val_df[self.feature_cols])[:, pos_class_idx]

            cols_to_keep = [self.group_col, "observation_id", self.target_col]
            if self.tie_breaker_col in val_df.columns:
                cols_to_keep.append(self.tie_breaker_col)

            val_scored = val_df[cols_to_keep].copy()
            val_scored["_pred_score"] = val_scores
            oof_frames.append(val_scored)

        # Concatenate out-of-fold predictions
        pooled_oof_df = pd.concat(oof_frames, ignore_index=True)

        # Calculate macro-averaged metrics (mean and std across folds)
        macro_metrics = {
            "eligible_top1_acc_mean": float(
                np.mean([f["eligible_top1_accuracy"] for f in fold_metrics_list])
            ),
            "eligible_top1_acc_std": float(
                np.std(
                    [f["eligible_top1_accuracy"] for f in fold_metrics_list], ddof=1
                )
            ),
            "overall_top1_acc_mean": float(
                np.mean([f["overall_top1_accuracy"] for f in fold_metrics_list])
            ),
            "overall_top1_acc_std": float(
                np.std(
                    [f["overall_top1_accuracy"] for f in fold_metrics_list], ddof=1
                )
            ),
            "cand_recall_mean": float(
                np.mean([f["candidate_generation_recall"] for f in fold_metrics_list])
            ),
            "cand_recall_std": float(
                np.std(
                    [f["candidate_generation_recall"] for f in fold_metrics_list],
                    ddof=1,
                )
            ),
            "cand_macro_f1_mean": float(
                np.mean([f["candidate_macro_f1"] for f in fold_metrics_list])
            ),
            "cand_macro_f1_std": float(
                np.std([f["candidate_macro_f1"] for f in fold_metrics_list], ddof=1)
            ),
            "cand_accuracy_mean": float(
                np.mean([f["candidate_accuracy"] for f in fold_metrics_list])
            ),
            "cand_accuracy_std": float(
                np.std([f["candidate_accuracy"] for f in fold_metrics_list], ddof=1)
            ),
            "cand_precision_mean": float(
                np.mean([f["candidate_precision"] for f in fold_metrics_list])
            ),
            "cand_precision_std": float(
                np.std([f["candidate_precision"] for f in fold_metrics_list], ddof=1)
            ),
            "cand_sensitivity_mean": float(
                np.mean([f["candidate_recall"] for f in fold_metrics_list])
            ),
            "cand_sensitivity_std": float(
                np.std([f["candidate_recall"] for f in fold_metrics_list], ddof=1)
            ),
            "cand_log_loss_mean": float(
                np.mean([f["candidate_log_loss"] for f in fold_metrics_list])
            ),
            "cand_log_loss_std": float(
                np.std([f["candidate_log_loss"] for f in fold_metrics_list], ddof=1)
            ),
            "train_time_sec_mean": float(
                np.mean([f["train_time_sec"] for f in fold_metrics_list])
            ),
            "train_time_sec_std": float(
                np.std([f["train_time_sec"] for f in fold_metrics_list], ddof=1)
            ),
            "total_train_time_sec": float(total_train_time),
            "total_infer_time_sec": float(total_infer_time),
            "latency_ms_per_obs_mean": float(
                np.mean(
                    [
                        f["inference_latency_ms_per_observation"]
                        for f in fold_metrics_list
                    ]
                )
            ),
            "latency_ms_per_obs_std": float(
                np.std(
                    [
                        f["inference_latency_ms_per_observation"]
                        for f in fold_metrics_list
                    ],
                    ddof=1,
                )
            ),
        }

        # Calculate pooled metrics across all out-of-fold observations
        pooled_ranking = compute_observation_ranking_metrics(
            df=pooled_oof_df,
            proba_col="_pred_score",
            target_col=self.target_col,
            group_cols=(self.group_col, "observation_id"),
            tie_breaker_col=self.tie_breaker_col,
            ascending_tie_breaker=self.ascending_tie_breaker,
        )

        y_true_all = pooled_oof_df[self.target_col].to_numpy().astype(int)
        scores_all = pooled_oof_df["_pred_score"].to_numpy()
        y_pred_all = (scores_all >= 0.5).astype(int)

        pooled_metrics = {
            "pooled_eligible_top1_acc": float(pooled_ranking["eligible_top1_accuracy"]),
            "pooled_overall_top1_acc": float(pooled_ranking["overall_top1_accuracy"]),
            "pooled_cand_recall": float(pooled_ranking["candidate_generation_recall"]),
            "pooled_num_obs_total": int(pooled_ranking["num_observations_total"]),
            "pooled_num_obs_eligible": int(pooled_ranking["num_observations_eligible"]),
            "pooled_num_cand_misses": int(pooled_ranking["num_candidate_misses"]),
            "pooled_num_correct_top1": int(pooled_ranking["num_correct_top1"]),
            "pooled_cand_accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "pooled_cand_macro_f1": float(
                f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
            ),
            "pooled_cand_log_loss": float(
                log_loss(y_true_all, scores_all, labels=[0, 1])
            ),
        }

        macro_metrics.update(pooled_metrics)
        primary_objective = macro_metrics["eligible_top1_acc_mean"]

        return primary_objective, macro_metrics, fold_metrics_list, pooled_oof_df

    def objective(self, trial: Trial) -> float:
        """Optuna objective function wrapping 5-fold evaluation with error tracking."""
        params = sample_histgbdt_params(trial)
        t_trial_start = time.time()

        try:
            (
                primary_objective,
                metrics,
                fold_metrics,
                _,
            ) = self.evaluate_hyperparameters(params)

            trial_wall_time = time.time() - t_trial_start

            # Store user attributes on Optuna trial
            for k, v in metrics.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("trial_wall_time_sec", trial_wall_time)
            trial.set_user_attr("state", "COMPLETE")

            # Record trial row
            record = {
                "trial_number": trial.number,
                "state": "COMPLETE",
                **params,
                **metrics,
                "trial_wall_time_sec": trial_wall_time,
                "error_message": "",
            }
            self.trial_records.append(record)

            return primary_objective

        except Exception as exc:
            trial_wall_time = time.time() - t_trial_start
            err_msg = f"{type(exc).__name__}: {str(exc)}"
            trial.set_user_attr("state", "FAIL")
            trial.set_user_attr("error_message", err_msg)
            trial.set_user_attr("trial_wall_time_sec", trial_wall_time)

            record = {
                "trial_number": trial.number,
                "state": "FAIL",
                **params,
                "trial_wall_time_sec": trial_wall_time,
                "error_message": err_msg,
            }
            self.trial_records.append(record)
            raise exc

    def run_optimization(
        self,
        n_trials: int = 30,
        timeout: Optional[float] = None,
        enqueue_baseline: bool = True,
        study_name: str = "histgbdt_ranking_optimization",
        callbacks: Optional[List[Callable]] = None,
    ) -> optuna.Study:
        """
        Execute Bayesian optimization study using Optuna TPESampler.

        Args:
            n_trials: Target number of trials.
            timeout: Optional maximum execution time in seconds.
            enqueue_baseline: Whether to enqueue baseline hyperparameters as Trial 0.
            study_name: Name of Optuna study.
            callbacks: Optional list of Optuna callback functions.

        Returns:
            Completed Optuna Study instance.
        """
        study = create_histgbdt_study(
            study_name=study_name,
            seed=self.seed,
            direction="maximize",
        )

        if enqueue_baseline:
            study.enqueue_trial(copy.deepcopy(DEFAULT_BASELINE_PARAMS))

        # Catch exceptions so failed trials are recorded rather than aborting the study
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            catch=(Exception,),
        )

        return study

    def export_artifacts(
        self,
        study: optuna.Study,
        output_csv: Path,
        output_json: Path,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Persist Optuna trials CSV and comprehensive summary JSON.

        Args:
            study: Completed Optuna study.
            output_csv: Destination path for trials CSV.
            output_json: Destination path for summary JSON.
            baseline_metrics: Optional baseline reference metrics for comparison.
            overwrite: Whether to overwrite existing files.
        """
        if not overwrite:
            for p in (output_csv, output_json):
                if p.exists():
                    raise FileExistsError(
                        f"Target artifact '{p}' already exists. Pass overwrite=True to replace."
                    )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        # 1. Export trials CSV
        trials_df = pd.DataFrame(self.trial_records)
        trials_df.to_csv(output_csv, index=False)

        # 2. Extract best trial information
        completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]
        failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]

        best_trial_info = {}
        if completed_trials:
            best_trial = study.best_trial
            best_trial_info = {
                "trial_number": int(best_trial.number),
                "objective_value": float(best_trial.value),
                "params": best_trial.params,
                "user_attrs": {
                    k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else str(v))
                    for k, v in best_trial.user_attrs.items()
                },
            }

        # 3. Construct comparison with baseline
        comparison = {}
        if baseline_metrics and completed_trials:
            base_macro_top1 = baseline_metrics.get(
                "eligible_top1_accuracy_mean", 0.8009855
            )
            base_overall_top1 = baseline_metrics.get(
                "overall_top1_accuracy_mean", 0.7198650
            )
            base_macro_f1 = baseline_metrics.get("candidate_macro_f1_mean", 0.8679161)
            base_log_loss = baseline_metrics.get("candidate_log_loss_mean", 0.1316099)

            best_attrs = best_trial_info.get("user_attrs", {})
            tuned_macro_top1 = best_attrs.get("eligible_top1_acc_mean", 0.0)
            tuned_overall_top1 = best_attrs.get("overall_top1_acc_mean", 0.0)
            tuned_macro_f1 = best_attrs.get("cand_macro_f1_mean", 0.0)
            tuned_log_loss = best_attrs.get("cand_log_loss_mean", 0.0)

            comparison = {
                "baseline_eligible_top1_accuracy": base_macro_top1,
                "tuned_eligible_top1_accuracy": tuned_macro_top1,
                "eligible_top1_diff_absolute": tuned_macro_top1 - base_macro_top1,
                "eligible_top1_diff_percentage": (
                    ((tuned_macro_top1 - base_macro_top1) / base_macro_top1) * 100.0
                    if base_macro_top1 > 0
                    else 0.0
                ),
                "baseline_overall_top1_accuracy": base_overall_top1,
                "tuned_overall_top1_accuracy": tuned_overall_top1,
                "baseline_candidate_macro_f1": base_macro_f1,
                "tuned_candidate_macro_f1": tuned_macro_f1,
                "baseline_candidate_log_loss": base_log_loss,
                "tuned_candidate_log_loss": tuned_log_loss,
            }

        summary_payload = {
            "study_name": study.study_name,
            "model_name": "HistGradientBoostingClassifier",
            "optimization_objective": "Mean 5-Fold Eligible Top-1 Accuracy",
            "n_trials_requested": len(study.trials),
            "n_trials_completed": len(completed_trials),
            "n_trials_failed": len(failed_trials),
            "n_trials_pruned": len(pruned_trials),
            "best_trial": best_trial_info,
            "baseline_comparison": comparison,
            "reproducibility": {
                "sampler": "TPESampler",
                "seed": self.seed,
                "n_splits": self.n_splits,
                "group_column": self.group_col,
                "target_column": self.target_col,
                "training_filter": f"{self.filter_col} == 1",
                "tie_breaker": f"{self.tie_breaker_col} ascending",
            },
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
