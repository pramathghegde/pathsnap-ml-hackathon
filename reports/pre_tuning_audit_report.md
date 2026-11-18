# PathSnap: Pre-Tuning Machine Learning Pipeline Audit Report

**Auditor:** Senior Machine Learning Engineer & ML Research Auditor  
**Repository:** `D:\Engineering\PESU\SEM 5\Machine Learning\Hackthon\pathsnap-ml-hackathon`  
**Branch:** `feat/model-training-continuation`  
**Git Baseline Checkpoint:** `dfcb98d – Complete Phase 1 candidate ranking evaluation`  
**Audit Date:** November 18, 2026  
**Final Audit Verdict:** **VERIFIED & SOUND — APPROVED FOR COMPLETE HYPERPARAMETER OPTIMIZATION**  

---

## 1. Executive Summary

This independent technical audit evaluated the complete Member 2 Machine Learning pipeline in the PathSnap repository, covering all stages implemented prior to and during Bayesian hyperparameter optimization. The audit assessed dataset validation, model architectures, group-aware cross-validation, baseline metrics, candidate ranking evaluation, data leakage safeguards, and baseline-to-optimization consistency.

### Summary Assessment Table
| Audit Stage | Status | Findings / Issues | Severity |
|:---|:---:|:---|:---:|
| **1. Dataset Validation & Schema** | **PASS** | Schema strictly adhered to; zero nulls/infs; correct eligibility filtering. | None |
| **2. Model Implementations** | **PASS** | 5 distinct model families; proper scaling pipelines; dynamic positive class indexing. | None |
| **3. Cross-Validation Protocol** | **PASS** | Strict 5-fold GroupKFold by `trip_id`; zero trip leakage across all folds. | None |
| **4. Baseline Evaluation** | **PASS** | Baseline metrics verified against raw artifacts; KNN scaling pilot justified. | None |
| **5. Candidate Ranking Evaluation** | **PASS** | Mathematically rigorous Eligible vs. Overall Top-1 metrics; deterministic tie-breaking. | None |
| **6. Data Leakage Protection** | **PASS** | `student_test` and `evaluator_test` strictly untouched; no train-to-val feature contamination. | None |
| **7. Baseline-to-Optimization Consistency** | **PASS** | Exact anchor alignment ($80.09855\%$); identical splits and objective function. | None |

---

## 2. Detailed Stage-by-Stage Audit Findings

### Stage 1: Dataset Validation & Feature Schema
- **Data Source:** `data/processed/train_features.csv` ($1,565,762$ candidate rows across 80 training trips).
- **Feature Schema:** Defined in `configs/feature_schema.json`. Exactly 9 predictive features:
  `['perp_dist', 'proj_ratio', 'delta_heading', 'arc_length', 'gps_speed', 'speed_ratio', 'traversal_rate', 'candidate_rank', 'is_adjacent']`.
- **Data Integrity Verification:**
  - Null value count: **0** across all 9 features on all $1,565,762$ rows.
  - Infinite value count: **0** across all 9 features on all $1,565,762$ rows.
- **Class Distribution:**
  - Full Dataset: Negative ($0$): $1,382,700$ ($88.31\%$), Positive ($1$): $183,062$ ($11.69\%$). Class imbalance: $7.55:1$.
  - Eligible Training Partition (`is_training_eligible == 1`): Exactly $1,423,994$ rows (Negative: $1,240,932$, Positive: $183,062$, Imbalance: $6.78:1$).
  - Ineligible Observations (`is_training_eligible == 0`): $141,768$ rows, containing **100% negative labels** ($0$ positives). These represent spatial indexing misses where ground-truth arcs fell outside the 50m search radius.
- **Input Exclusions:** All identifiers (`trip_id`, `observation_id`, `candidate_arc_id`), timestamps, partition flags (`fold`), audit flags (`has_positive_candidate`, `is_training_eligible`), and ground-truth targets (`is_match`) are strictly excluded from predictive feature matrices ($X$).

---

### Stage 2: Model Architecture & Preprocessing
Five algorithm families are implemented in `src/models/model_factory.py`:
1. **Logistic Regression (Linear / Probabilistic):**
   - Wrapped in `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])`.
   - Scaler fitted strictly on training folds within pipeline; zero validation data contamination.
2. **K-Nearest Neighbors (Instance-Based):**
   - Wrapped in `Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance'))])`.
   - Evaluated on a representative 12-trip pilot study (`experiments/results/knn_pilot.csv`) due to $\mathcal{O}(N \times D)$ prediction complexity on 1.4M rows ($3.25 \times 10^{11}$ pairwise comparisons per fold).
3. **Decision Tree (Non-Parametric Tree):**
   - `DecisionTreeClassifier(max_depth=None, random_state=42)`.
4. **Random Forest (Bagging Ensemble):**
   - `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`.
5. **HistGradientBoosting (Gradient Boosting):**
   - `HistGradientBoostingClassifier(max_iter=100, random_state=42)`.
   - Native histogram binning; scale-invariant.
- **Positive-Class Probability Extraction:** Dynamically resolves class 1 column using `model.classes_` in `evaluate.py`, `optimize.py`, and `predict.py` rather than assuming binary index 1.

---

### Stage 3: Cross-Validation & Group Isolation
- **Partitioning Strategy:** 5-fold `GroupKFold` grouped strictly by `trip_id`.
- **Trip Isolation Verification:**
  - Fold 0: 16 trips (0 overlap with Folds 1..4).
  - Fold 1: 16 trips (0 overlap with Folds 0, 2..4).
  - Fold 2: 16 trips (0 overlap with Folds 0, 1, 3..4).
  - Fold 3: 16 trips (0 overlap with Folds 0..2, 4).
  - Fold 4: 16 trips (0 overlap with Folds 0..3).
  - Multi-fold trip count: **0** across all 80 trips.
- **Runtime Leakage Assertions:** `src/models/train.py`, `scripts/run_histgbdt_ranking_benchmark.py`, and `src/models/optimize.py` assert `assert len(train_trips.intersection(val_trips)) == 0` prior to model fitting.

---

### Stage 4: Baseline Evaluation Verification
The initial candidate-level benchmark results stored in `experiments/results/baseline_model_comparison.csv` were verified:
- **HistGradientBoosting:** Macro F1 = $0.8782 \pm 0.0183$, Accuracy = $0.9459 \pm 0.0115$, Log Loss = $0.1323 \pm 0.0204$, Train Time = $9.76\text{s/fold}$, Latency = $0.0036\text{ ms/candidate}$.
- **Random Forest:** Macro F1 = $0.8749 \pm 0.0181$, Accuracy = $0.9450 \pm 0.0114$, Log Loss = $0.1951 \pm 0.0388$, Train Time = $166.43\text{s/fold}$, Latency = $0.0075\text{ ms/candidate}$.
- **Logistic Regression:** Macro F1 = $0.8557 \pm 0.0184$, Accuracy = $0.9378 \pm 0.0114$, Log Loss = $0.1691 \pm 0.0217$, Train Time = $1.48\text{s/fold}$, Latency = $0.0001\text{ ms/candidate}$.
- **Decision Tree:** Macro F1 = $0.8299 \pm 0.0120$, Accuracy = $0.9239 \pm 0.0098$, Log Loss = $2.7415 \pm 0.3525$, Train Time = $13.33\text{s/fold}$, Latency = $0.0002\text{ ms/candidate}$.
- **KNN Pilot:** Macro F1 = $0.8767 \pm 0.0241$, Accuracy = $0.9503 \pm 0.0069$, Log Loss = $0.6400 \pm 0.1582$ on 12 trips.

---

### Stage 5: Candidate Ranking Evaluation Methodology
The candidate ranking evaluation module (`src/models/evaluate.py`) addresses the candidate classification vs. observation ranking mismatch:
1. **Eligible Observation Top-1 Accuracy:**
   $$\text{Eligible Top-1} = \frac{\sum_{o_i \in \text{Eligible}} \mathbb{I}(y(a_i^*) = 1)}{N_{\text{eligible observations}}} = 80.10\% \pm 3.30\%$$
   Measures model ranking discrimination strictly on choice sets containing the true segment.
2. **Overall Top-1 Accuracy:**
   $$\text{Overall Top-1} = \frac{\sum_{o_i} \mathbb{I}(y(a_i^*) = 1)}{N_{\text{total observations}}} = 71.99\% \pm 10.84\%$$
   Reflects true trajectory reconstruction quality, bounded by spatial recall.
3. **Candidate Generation Recall:**
   $$\text{Candidate Recall} = \frac{N_{\text{eligible observations}}}{N_{\text{total observations}}} = 89.53\% \pm 10.85\% \quad (183,062 / 208,244 = 87.907\% \text{ pooled})$$
4. **Mathematical Identity:**
   $$\text{Overall Top-1} \equiv \text{Eligible Top-1} \times \text{Candidate Recall}$$
   Verified both on fold-level and pooled levels.
5. **Deterministic Tie-Breaking:**
   When model probabilities tie, secondary ordering uses `perp_dist` ascending with stable sort (`mergesort`), guaranteeing deterministic, reproducible rankings.
6. **Candidate Miss Handling:**
   Observations where candidate generation missed the correct road segment are retained in the validation population with all candidates labeled $0$. They contribute to `num_candidate_misses` and penalize `overall_top1_accuracy`, while eligible accuracy remains an unpolluted measure of model discrimination.

---

### Stage 6: Data Leakage Protection
- **Student-Test Data Isolation:** Grep audit across all source files confirms that `student_test_features.csv` and `student_test` directories are referenced **only** in data preprocessing/EDA inspection scripts. No modeling script (`train.py`, `run_model_benchmarks.py`, `optimize.py`, `train_final_model.py`, `predict.py`) loads or evaluates on student-test data.
- **Evaluator Ground-Truth Protection:** Evaluator test partitions and reference routes were **never accessed or read**.
- **Preprocessing Leakage Prevention:** Feature scaling (for Logistic Regression and KNN) is encapsulated inside `Pipeline`. Transformation parameters ($\mu, \sigma$) are computed exclusively on training folds.

---

### Stage 7: Baseline-to-Optimization Consistency
- **Baseline Alignment:**
  - Phase 1 Baseline Score: **80.09855001032153%** Eligible Top-1.
  - Optuna Anchor (Trial 0): **80.09855001032153%** Eligible Top-1.
  - Value matches to 16 decimal places, verifying identical data slices, feature sets, and estimator parameterization.
- **Optimization Objective:** `src/models/optimize.py` explicitly returns `macro_metrics["eligible_top1_acc_mean"]` under `direction="maximize"`.
- **Protocol Symmetry:** The Bayesian optimization engine uses the identical 5-fold GroupKFold assignments, filters training to `is_training_eligible == 1`, evaluates on 100% of validation observations, and applies deterministic tie-breaking.

---

## 3. Methodological Observations & Notes

### Observation 1: Validation Set Scope Difference Between Initial Benchmarks and Ranking Benchmarks
- **Context:** In `scripts/run_model_benchmarks.py`, the validation set was filtered to `is_training_eligible == 1` before partitioning, producing candidate-level Macro F1 of $0.8782$. In Phase 1 ranking benchmarks (`scripts/run_histgbdt_ranking_benchmark.py`), validation includes 100% of candidate rows ($1,565,762$), resulting in candidate Macro F1 of $0.8679$.
- **Audit Assessment:** Methodologically valid. The Phase 1 ranking benchmark is more rigorous because it evaluates the full operational observation population (including candidate misses).
- **Severity:** Informational / Documentation note. No code changes required.

### Observation 2: KNN Scalability Boundary
- **Context:** KNN was evaluated on a 12-trip pilot subset rather than the full 1.4M candidate matrix.
- **Audit Assessment:** Sound engineering decision. Full-scale KNN would require $3.25 \times 10^{11}$ distance computations per fold, making cross-validation computationally infeasible without approximate nearest-neighbor indexing.
- **Severity:** Informational. Appropriately documented in `experiments/results/knn_pilot.csv`.

### Observation 3: Hypothesis DLL Block on Windows
- **Context:** Windows Application Control blocked `hypothesis._native.pyd` during raw pytest invocations.
- **Audit Assessment:** Resolved by configuring `-p no:hypothesispytest` in `pyproject.toml` `addopts`. Full test suite runs cleanly: **47 passed in 6.59s**.
- **Severity:** Low / Environmental. Fully mitigated.

---

## 4. Auditor Recommendation

**Verdict: APPROVED FOR COMPLETE HYPERPARAMETER SEARCH.**

The Member 2 ML pipeline satisfies all methodological requirements:
1. Cross-validation is leakage-free and strictly group-isolated by trip ID.
2. Training data is correctly filtered to eligible choice sets without distorting validation populations.
3. The optimization objective is observation-level Eligible Top-1 accuracy, directly aligning model tuning with map-matching route reconstruction.
4. Test coverage is comprehensive (47 passing tests across all modules).
5. Data protection boundaries are strictly maintained.

No blocking issues were identified. The repository is in an optimal state to proceed with extended Optuna hyperparameter tuning.
