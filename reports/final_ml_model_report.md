# PathSnap: Final Production ML Model Report (Phase 3)

**Role:** Member 2 – Machine Learning Lead  
**Model Family:** Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`)  
**Artifact Directory:** `artifacts/models/`  
**Model Artifact:** `artifacts/models/pathsnap_histgbdt.joblib` (0.91 MB)  
**Metadata Sidecar:** `artifacts/models/pathsnap_histgbdt_metadata.json`  
**Inference Module:** `src/models/predict.py`  
**Test Suite:** `tests/test_predict.py` (47 total project tests passing)  
**Date:** November 18, 2026  
**Status:** Validated, Serialized & Integration Ready  

---

## 1. Executive Summary & Model Selection

For the PathSnap distributed GPS map matching system, the primary machine learning objective is to serve as the **Observation Emission Probability Estimator**: given candidate road segments retrieved within a spatial proximity radius of each GPS point, the model evaluates engineered geometric and kinematic features, computes calibrated emission match probabilities $p(\text{is\_match} = 1)$, and ranks candidates descending to identify the true road segment.

After benchmarking across five distinct algorithm families (Logistic Regression, KNN, Decision Tree, Random Forest, and HistGradientBoosting) and executing Bayesian hyperparameter optimization using Optuna (TPE Sampler), **HistGradientBoostingClassifier (Trial 6)** was selected as the final production model for integration into Laptop 1's FastAPI serving pipeline.

> [!NOTE]
> Due to hackathon project time constraints and the computational demands of evaluating 1.42M training candidate rows per fold, the Optuna study was frozen at 8 completed trials (out of 30 originally planned). Trial 6 is the best-performing configuration among all completed trials. This result is an empirically validated local improvement over default parameters and is not claimed to be a global optimum over the 7-dimensional search space.

### Why HistGradientBoosting Was Selected:
1. **Superior Discriminative Ranking Accuracy:** Achieves **80.39% ± 3.79%** Eligible Top-1 Accuracy across 5 folds, outperforming all other linear, tree, and ensemble baselines.
2. **Sub-Millisecond Inference Latency:** Evaluates candidates at **0.064 ms per observation** (~15,600 GPS observations per second on a single CPU core), easily satisfying real-time latency budgets for streaming GPS trajectories.
3. **Consistent Fold-by-Fold Superiority:** In out-of-fold cross-validation, Trial 6 improved Eligible Top-1 accuracy across **100% of validation folds** over the default baseline.
4. **Compact Model Footprint:** Serialized model size is only **0.91 MB**, allowing zero-overhead deployment and instantaneous cold-start loading into memory.
5. **Native Numerical Binning:** Histogram-based partitioning handles continuous spatial distributions (`perp_dist`, `proj_ratio`) with optimal quantization without requiring feature scaling pipelines.

---

## 2. Selected Hyperparameter Configuration (Trial 6)

The production model was trained using the top-performing configuration discovered by Bayesian optimization:

```json
{
  "learning_rate": 0.04228762725326872,
  "max_iter": 400,
  "max_leaf_nodes": 46,
  "max_depth": null,
  "min_samples_leaf": 28,
  "l2_regularization": 0.3142918568673425,
  "max_bins": 186,
  "random_state": 42
}
```

### Architectural Rationale:
- **`learning_rate = 0.0423` & `max_iter = 400`:** A moderate learning rate combined with 400 boosting stages prevents aggressive early tree splits on noisy GPS pings, building a smoother decision surface.
- **`max_leaf_nodes = 46` with `max_depth = None`:** Expands expressive capacity beyond the default 31 leaf nodes to capture non-linear interactions between `perp_dist` and `delta_heading` across complex multi-arc intersections.
- **`min_samples_leaf = 28` & `l2_regularization = 0.314`:** Regularizes leaf values, mitigating overfitting on heavily sampled highway segments.
- **`max_bins = 186`:** Provides fine-grained binning resolution for metric distances while maintaining high cache locality during tree traversal.

---

## 3. Baseline vs. Tuned Model Cross-Validation Comparison

Both models were evaluated on the identical 5-fold GroupKFold protocol (grouped strictly by `trip_id`, 80 training trips, 208,244 GPS observations, 1,565,762 candidate rows).

### 3.1 Fold-by-Fold Performance Comparison

| Fold | Baseline Eligible Top-1 | Tuned Eligible Top-1 (Trial 6) | Absolute Difference | Baseline Overall Top-1 | Tuned Overall Top-1 | Candidate Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 80.88% | **81.65%** | **+0.77%** | 68.20% | **68.85%** | 84.33% |
| **1** | 85.24% | **85.57%** | **+0.34%** | 83.95% | **84.29%** | 98.50% |
| **2** | 79.46% | **79.58%** | **+0.12%** | 77.86% | **77.97%** | 97.99% |
| **3** | 74.90% | **75.08%** | **+0.18%** | 52.79% | **52.91%** | 70.48% |
| **4** | 80.02% | **80.06%** | **+0.04%** | 77.13% | **77.17%** | 96.39% |

> **Key Finding:** In all five folds, Trial 6 strictly outperformed the baseline model. Improvement was particularly pronounced in Fold 0 (+0.77%) and Fold 1 (+0.34%).

### 3.2 Aggregate Cross-Validation Summary

| Metric | Phase 1 Baseline | Phase 3 Tuned Model | Absolute Change | Relative Change |
|:---|:---:|:---:|:---:|:---:|
| **Mean Eligible Top-1 Accuracy** | **80.10% ± 3.30%** | **80.39% ± 3.79%** | **+0.289%** | **+0.36%** |
| **Pooled Eligible Top-1 Accuracy** | **79.94%** | **80.20%** | **+0.259%** | **+0.32%** |
| **Mean Overall Top-1 Accuracy** | **71.99% ± 10.84%** | **72.24% ± 12.12%** | **+0.250%** | **+0.35%** |
| **Candidate Macro F1** | **0.8679 ± 0.0235** | **0.8693 ± 0.0238** | **+0.0014** | **+0.16%** |
| **Candidate Log Loss** | **0.1316 ± 0.0205** | **0.1281 ± 0.0198** | **-0.0035** | **-2.66% (Improved)** |
| **Candidate Generation Recall** | **89.53% ± 10.85%** | **89.53% ± 10.85%** | $0.000\%$ | Unchanged (Fixed) |
| **Mean Inference Latency / Obs** | **0.0131 ms** | **0.0641 ms** | $+0.0510\text{ ms}$ | Real-time budget met |

---

## 4. Final Model Training & Dataset Specifications

The final model was refitted on 100% of permitted training observations:

- **Source Dataset:** `data/processed/train_features.csv`
- **Dataset Hash (SHA-256):** `4b69908baaccdd1731acde8d789df265a66aff7f4c3110ff643e42e430f7ff90`
- **Training Population Filter:** `is_training_eligible == 1` (excludes all-negative choice sets to prevent biased negative gradients)
- **Total Eligible Rows:** $1,423,994$ candidate rows across all 80 training trips
  - Positive match candidates ($y = 1$): $183,062$ (12.86%)
  - Negative candidate segments ($y = 0$): $1,240,932$ (87.14%)
  - Negative-to-Positive Imbalance Ratio: $6.78 : 1$
- **Training Duration:** $27.98\text{ seconds}$ on CPU (`n_iter_ = 400`)
- **Evaluator Ground-Truth Protection:** Evaluator test partitions were **never accessed or used**. Student test data was **never used for training or model selection**.

---

## 5. Predictive Feature Schema

The model consumes exactly the nine standardized features defined in `configs/feature_schema.json`:

| Index | Feature Name | Physical Unit | Description |
|:---:|:---|:---:|:---|
| 0 | `perp_dist` | meters | Perpendicular orthogonal distance from GPS ping to candidate segment |
| 1 | `proj_ratio` | ratio $\in [0, 1]$ | Position of orthogonal projection along segment length |
| 2 | `delta_heading` | radians $\in [0, \pi]$ | Angular difference between GPS velocity heading and arc orientation |
| 3 | `arc_length` | meters | Physical length of candidate road segment |
| 4 | `gps_speed` | m/s | Estimated instantaneous speed between consecutive GPS pings |
| 5 | `speed_ratio` | ratio | Ratio of GPS speed to urban reference speed ($v_{\text{ref}} = 13.89\text{ m/s}$) |
| 6 | `traversal_rate` | $\text{s}^{-1}$ | Segment traversal frequency ($\text{gps\_speed} / \max(1.0, \text{arc\_length})$) |
| 7 | `candidate_rank` | ordinal integer | Proximity rank of candidate among retrieved arcs ordered by `perp_dist` |
| 8 | `is_adjacent` | binary $\{0, 1\}$ | Indicator whether candidate shares a vertex with step $t-1$ top candidate |

---

## 6. Inference Module Interface (`src/models/predict.py`)

A production-grade predictor class `PathSnapPredictor` was implemented to serve emission probabilities to Member 3 (HMM/Viterbi) and Member 4 (FastAPI backend).

### Key Features:
1. **Dynamic Positive Class Resolution:** Selects class 1 probability via `model.classes_` dynamically rather than hardcoding column index 1.
2. **Feature Re-ordering Tolerance:** Re-aligns input columns automatically to match trained feature ordering, ensuring immunity against column reordering in API payloads.
3. **Missing Feature Detection:** Raises descriptive `ValueError` if any mandatory feature is missing from candidate payloads.
4. **Identifier Preservation:** Carries `trip_id`, `observation_id`, and `candidate_arc_id` through scoring and ranking outputs.
5. **Top-1 Selection:** Implements deterministic tie-breaking via `perp_dist` ascending.

### Usage Example:
```python
from src.models.predict import load_predictor
import pandas as pd

# Load serialized model and metadata
predictor = load_predictor()

# Score candidates DataFrame (from spatial indexer)
candidate_df = pd.read_csv("candidates_batch.csv")

# 1. Get raw positive-class probabilities
probabilities = predictor.predict_proba(candidate_df)

# 2. Get candidates with preserved IDs and match_probability
scored_candidates = predictor.predict_candidates(candidate_df)

# 3. Select Top-1 candidate per GPS observation
top1_routes = predictor.select_top1_candidates(candidate_df)
```

---

## 7. Known Limitations & Recommendations

1. **Upper Bound Imposed by Spatial Recall:** Overall Top-1 accuracy is bounded by candidate generation recall ($89.53\%$). For the $12.09\%$ of observations where candidate generation misses the true arc, the ML model cannot correct the error independently.
2. **Optuna Search Scope:** 8 trials demonstrated consistent improvement across all folds, but do not represent a global optimum over the 7-dimensional search space. Additional trials could be run in future phases if compute allows.
3. **Transition Dynamics Hand-off:** Observation-level ranking operates as an independent emission model. Temporal consistency across consecutive GPS points will be resolved downstream by Member 3's Viterbi decoding algorithm using the HMM transition matrix.
