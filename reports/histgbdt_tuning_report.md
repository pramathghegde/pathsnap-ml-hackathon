# PathSnap: Bayesian Hyperparameter Optimization Report (Phase 2)

**Role:** Member 2 – Machine Learning Lead  
**Module:** `src/models/optimize.py`  
**Optimization Driver:** `scripts/run_histgbdt_optuna.py`  
**Test Suite:** `tests/test_optuna_tuning.py`  
**Date:** November 18, 2026  
**Status:** Frozen at 8 Completed Trials (Verified & Documented)

---

## 1. Executive Summary & Study Status

Phase 2 established a Bayesian Hyperparameter Optimization engine using **Optuna** and the **Tree-structured Parzen Estimator (TPE)** to optimize `HistGradientBoostingClassifier` directly for observation-level **Eligible Top-1 Candidate Ranking Accuracy**.

### Experiment Freezing Decision
Due to strict project hackathon time constraints and the computational cost of full-matrix cross-validation (~1.42 million training candidates per fold across 80 trips), **the optimization experiment was formally frozen at 8 completed trials** out of the 30 originally planned:
- **Total Requested Budget:** 30 trials
- **Completed Trials Preserved:** **8 trials** (Trials 0 to 7)
- **Failed Trials:** 0
- **Pruned Trials:** 0
- **Unfinished / Cancelled Trials:** 22
- **Best Completed Trial:** **Trial 6**
- **Non-Optimality Disclaimer:** **The result is an empirically validated local improvement and is NOT claimed to be a global optimum** over the 7-dimensional search space. However, it delivers consistent, measurable gains across all validation folds over the baseline.

---

## 2. Experimental Setup & Search Space

### 2.1 Search Space Configuration
| Hyperparameter | Type | Search Range | Sampling Distribution |
|:---|:---:|:---:|:---:|
| `learning_rate` | float | [0.02, 0.20] | Log-uniform |
| `max_iter` | int | [100, 500] | Discrete (step = 50) |
| `max_leaf_nodes` | int | [15, 63] | Uniform integer |
| `max_depth` | categorical | [None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | Categorical with `None` support |
| `min_samples_leaf` | int | [20, 100] | Uniform integer |
| `l2_regularization` | float | [0.0, 10.0] | Uniform float |
| `max_bins` | int | [64, 255] | Uniform integer |

### 2.2 Cross-Validation & Data Protection Protocol
- **Dataset:** `data/processed/train_features.csv` (80 training trips, 208,244 GPS observations, 1,565,762 candidate rows).
- **Validation Partitioning:** 5-fold `GroupKFold` grouped strictly by `trip_id`. Zero trip leakage asserted across all folds.
- **Training Population:** Filtered strictly to `is_training_eligible == 1` (~1.14M rows per fold). All-negative choice sets excluded from training.
- **Validation Population:** Evaluated on 100% of observations in the fold (~313k rows per fold), preserving all candidate generation misses.
- **Strict Data Protection:** No access to `evaluator_test` ground-truth labels. `student_test` never used for hyperparameter tuning.

---

## 3. Preserved Completed Trials Table (Trials 0 to 7)

| Trial # | Status | Mean Eligible Top-1 | Learning Rate | Max Iter | Max Leaf Nodes | Max Depth | Min Samples Leaf | L2 Reg | Max Bins |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** (Baseline Anchor) | COMPLETE | **80.099%** | 0.1000 | 100 | 31 | None | 20 | 0.000 | 255 |
| **1** | COMPLETE | **80.264%** | 0.0474 | 500 | 50 | 10 | 34 | 1.834 | 122 |
| **2** | COMPLETE | **80.385%** | 0.0670 | 250 | 29 | 7 | 33 | 0.651 | 246 |
| **3** | COMPLETE | **80.261%** | 0.1848 | 450 | 29 | 8 | 64 | 1.849 | 250 |
| **4** | COMPLETE | **80.320%** | 0.1192 | 500 | 58 | 3 | 63 | 1.409 | 218 |
| **5** | COMPLETE | **80.336%** | 0.0237 | 500 | 52 | 11 | 46 | 0.636 | 123 |
| **6** (**Best Configuration**) | COMPLETE | **80.387%** | **0.0423** | **400** | **46** | **None** | **28** | **0.314** | **186** |
| **7** | COMPLETE | **80.302%** | 0.0412 | 300 | 59 | 9 | 85 | 1.866 | 235 |

---

## 4. Best Configuration (Trial 6) Detailed 5-Fold Validation

Trial 6 was evaluated across all five GroupKFold validation folds and compared directly against the Phase 1 baseline.

### 4.1 Fold-by-Fold Validation Results

| Fold | Baseline Eligible Top-1 | Tuned Eligible Top-1 (Trial 6) | Absolute Difference | Baseline Overall Top-1 | Tuned Overall Top-1 | Candidate Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 80.88% | **81.65%** | **+0.77%** | 68.20% | **68.85%** | 84.33% |
| **1** | 85.24% | **85.57%** | **+0.34%** | 83.95% | **84.29%** | 98.50% |
| **2** | 79.46% | **79.58%** | **+0.12%** | 77.86% | **77.97%** | 97.99% |
| **3** | 74.90% | **75.08%** | **+0.18%** | 52.79% | **52.91%** | 70.48% |
| **4** | 80.02% | **80.06%** | **+0.04%** | 77.13% | **77.17%** | 96.39% |

### 4.2 Aggregate Comparison vs. Baseline Anchor

| Metric | Phase 1 Baseline (Trial 0) | Tuned Model (Trial 6) | Absolute Change | Relative Change |
|:---|:---:|:---:|:---:|:---:|
| **Mean Eligible Top-1 Accuracy** | **80.10% ± 3.30%** | **80.39% ± 3.79%** | **+0.289%** | **+0.36%** |
| **Pooled Eligible Top-1 Accuracy** | **79.94%** | **80.20%** | **+0.259%** | **+0.32%** |
| **Mean Overall Top-1 Accuracy** | **71.99% ± 10.84%** | **72.24% ± 12.12%** | **+0.250%** | **+0.35%** |
| **Candidate Macro F1** | **0.8679 ± 0.0235** | **0.8693 ± 0.0238** | **+0.0014** | **+0.16%** |
| **Candidate Log Loss** | **0.1316 ± 0.0205** | **0.1281 ± 0.0198** | **-0.0035** | **-2.66% (Improved)** |
| **Candidate Generation Recall** | **89.53% ± 10.85%** | **89.53% ± 10.85%** | $0.000\%$ | Identical |
| **Mean Training Time / Fold** | **4.43s** | **19.47s** | $+15.04\text{s}$ | Within budget |
| **Mean Inference Latency / Obs** | **0.0131 ms** | **0.0641 ms** | $+0.0510\text{ ms}$ | Real-time ready |

---

## 5. Artifacts and Preservation Inventory

All Phase 2 optimization records, parameters, and results are fully preserved:

1. **`experiments/results/histgbdt_optuna_trials.csv`**:
   - Tabular record of all 8 completed trials, their hyperparameter combinations, objective scores, and timestamps.
2. **`experiments/results/histgbdt_optuna_summary.json`**:
   - Machine-readable summary of study metadata, best trial attributes, baseline comparisons, and study termination reason.
3. **`experiments/results/trial6_validation_details.json`**:
   - Complete 5-fold fold-by-fold validation metrics and pooled statistics for Trial 6.
4. **`src/models/optimize.py`**:
   - Reusable Bayesian optimization engine with TPE sampling, leakage checks, and objective scoring.
5. **`tests/test_optuna_tuning.py`**:
   - 8 unit tests validating parameter sampling, TPE reproducibility, trip isolation, and failure handling (all passing).
