# PathSnap ML Input Data Contract
**Interface & Integration Specification between Member 1 (Data Engineering & Spatial Features) and Member 2 (Machine Learning)**

*Document Version:* 1.1.0  
*Status:* Draft for Team Review  
*Author:* Member 2 (Machine Learning)  
*Stakeholders:* Member 1 (Data Engineering), Member 2 (Machine Learning), Member 3 (HMM Decoding & Bayesian Optimization)  

---

## 1. Executive Summary & Purpose

This document establishes the interface boundary between **Member 1** (upstream ingestion, candidate generation, and feature extraction) and **Member 2** (downstream model training, group-aware cross-validation, and benchmarking).

To ensure transparency across team responsibilities, this document explicitly separates:
1. **Confirmed Requirements**: Constraints strictly enforced by the current Python implementation (`src/models/train.py`, `src/models/evaluate.py`, `src/models/model_factory.py`).
2. **Proposed Schema Conventions**: Structural recommendations for column naming, types, and file organization.
3. **Decisions Requiring Team Agreement**: Open architectural questions that must be collectively resolved with **Member 1** and **Member 3** before production dataset generation and model training begin.

---

## 2. Confirmed Implementation Requirements (Strict Python Constraints)

The following requirements are implemented and validated by `validate_training_inputs()` in `src/models/train.py` and `evaluate_model()` in `src/models/evaluate.py`. Any non-compliance will raise an immediate, informative `ValueError`.

### 2.1 DataFrame Format & Integrity
- **Data Structure**: Input must be delivered as a `pandas.DataFrame` (or stored in an interoperable format such as Apache Parquet readable directly via `pandas.read_parquet`).
- **Non-Emptiness**: The DataFrame must not be empty (`len(df) > 0`).

### 2.2 Grouping Column (`group_col`)
- **Default**: `'trip_id'` (configurable via parameter).
- **Missing Values**: Must contain **zero** `NaN`, `None`, or empty values.
- **Group Cardinality**: Must contain at least `n_splits` distinct groups (default: $\ge 5$ unique trips).
- **Group Purity**: Every observation originating from the same physical trip must share the exact same `trip_id`. Observations from the same trip must never be assigned different group IDs.

### 2.3 Target Column (`target_col`)
- **Configurable**: Configured via `target_col: str` (e.g., `'label'` or `'is_match'`).
- **Missing Values**: Must contain **zero** `NaN`, `None`, or missing values.
- **Class Cardinality**: Must contain at least **2 distinct classes** across the dataset.
- **Classification Capabilities**:
  - `src/models/train.py`, `model_factory.py`, and `evaluate.py` support both binary and multiclass target formulations.
  - Estimators output calibrated probability matrices via `predict_proba(X)` across `model.classes_`.
  - Discrete predictions are derived deterministically via `model.classes_[np.argmax(y_proba, axis=1)]`.
  - Probabilities are validated for 2D shape, finiteness, non-negativity, and row-sum unity ($\sum_k P_k \approx 1.0$).

### 2.4 Feature Columns (`feature_cols`)
- **Automatic Resolution**:
  - If `feature_cols=None`, `train.py` automatically resolves features as all columns in `df.columns` **excluding** recognized metadata columns:
    ```python
    metadata_cols = {group_col, target_col, "point_idx", "candidate_arc_id", "timestamp"}
    ```
  - If explicitly passed as `feature_cols: list[str]`, all listed columns must exist in `df.columns`.
- **Numerical Data Types**: All feature columns must be purely numeric (`float64`, `float32`, `int64`, `int32`). No object, string, boolean, or datetime types are permitted in feature columns.
- **Missing & Infinite Values**: **Zero missing values permitted**. Every feature column must have 0 `NaN`, 0 `None`, and 0 `Inf` values. Upstream imputation, clipping, or sentinel assignment must be completed by Member 1.

### 2.5 Cross-Validation & Unseen Validation Class Behavior
- `GroupKFold(n_splits=n_splits)` partitions data strictly by `group_col` (`trip_id`), guaranteeing zero trip overlap between train and validation splits.
- **Unseen Class Detection**: In multiclass targets, if a validation fold contains a target class that was absent from that fold's training trips, `train.py` detects it via `UnseenClassError`.
  - `log_loss` is marked invalid (`NaN`) for that fold with an explanatory diagnostic note, because cross-entropy is mathematically undefined for classes outside the estimator's probability support.
  - Accuracy and Macro F1 continue to be evaluated on that fold.
  - **Distribution Reality**: Merely having a class appear in two trips does not guarantee that every training fold will contain that class (for instance, in 5-fold CV, both trips might land in the validation fold or together in one fold). Achieving fully valid log loss across all folds requires that every target class is well-represented across multiple trips throughout the training set.

### 2.6 Partition Isolation & `student_test` Rules
- **Schema Alignment**: The `student_test` dataset (10 trips in `data/raw/student_test/`) must be processed with the exact same feature engineering pipeline and column schema as the training dataset.
- **No Data Leakage**: `student_test` data (and its ground-truth `.route` labels) must **never** be used for model fitting, feature scaling calibration, or cross-validation model selection. It is strictly reserved for post-training validation and comparative benchmarking.
- **Evaluator Testing**: `evaluator_test` data is kept strictly confidential and server-side on Laptop 1; it is never touched during offline experimentation.

---

## 3. Proposed Schema Conventions (For Discussion)

To decouple metadata from ML features, the following column structure and conventions are proposed for team adoption:

### 3.1 Proposed Column Taxonomy

| Category | Column Name | Type | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Group Identifier** | `trip_id` | `str` or `int` | Unique trip folder/file name (e.g., `'00000001'`). | **Confirmed Requirement** |
| **Observation Index** | `point_idx` | `int` | Zero-based index $t \in [0, N-1]$ of the GPS point in `.track`. | **Proposed Convention** |
| **Candidate Identifier**| `candidate_arc_id` | `int` | Candidate arc row index from `.arcs`. Identifies candidate segment. | **Proposed Convention** |
| **Temporal Stamp** | `timestamp` | `float` | Elapsed seconds from `.track` (3rd column). | Optional |
| **Target Label** | `label` | `int` | Ground truth target (binary `0`/`1` or multiclass ID). | **Confirmed Requirement** |
| **Candidate Features** | `feat_*` | `float` | Numerical features describing candidate-observation geometry. | **Proposed Convention** |

### 3.2 Illustrative DataFrame Example (Proposed Binary Convention)

Below is an illustrative slice under the **proposed Binary Candidate Classification convention**, where each row represents one candidate arc evaluated for a specific GPS observation:

| `trip_id` | `point_idx` | `candidate_arc_id` | `timestamp` | `feat_dist_orthogonal` | `feat_heading_diff` | `feat_speed_ratio` | `feat_proj_u` | `label` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `"00000001"` | `0` | `2641` | `1.0` | `4.21` | `0.08` | `0.95` | `0.45` | **`1`** |
| `"00000001"` | `0` | `2642` | `1.0` | `18.75` | `1.52` | `0.40` | `0.10` | **`0`** |
| `"00000001"` | `0` | `64970` | `1.0` | `32.10` | `2.85` | `0.12` | `0.02` | **`0`** |
| `"00000001"` | `1` | `2641` | `2.0` | `12.50` | `0.15` | `0.88` | `0.85` | **`0`** |
| `"00000001"` | `1` | `64973` | `2.0` | `3.10` | `0.05` | `0.98` | `0.20` | **`1`** |
| `"00000002"` | `0` | `10540` | `1.0` | `5.80` | `0.12` | `0.91` | `0.30` | **`1`** |
| `"00000002"` | `0` | `10541` | `1.0` | `22.40` | `1.80` | `0.35` | `0.05` | **`0`** |

*Note: In the table above, `feat_*` columns are placeholder values for illustration only. Specific feature formulas and names are owned by Member 1.*

---

## 4. Technical Decisions Requiring Team Agreement (Members 1, 2, and 3)

The following core architectural questions are unresolved and require explicit consensus before production data processing:

### Decision 1: Target Formulation (Binary Candidate Classification vs. Pointwise Multiclass)
- **Status**: **Proposed Recommendation** (NOT yet an agreed team decision).
- **Proposal (Option A - Binary Candidate Classification)**:
  - Each row is an `(observation_point, candidate_arc)` pair.
  - The model learns $P(\text{is\_match} = 1 \mid \text{features})$, ranking candidates locally.
  - *Rationale*: A raw `.arcs` file contains 100,000+ distinct road segments across the network. Predicting raw `arc_id` directly across 100,000 classes causes catastrophic class sparsity, massive memory footprints, and constant unseen-class errors in GroupKFold. Binary candidate scoring scales cleanly and directly supplies the emission probability $P(z_t \mid s_t = a_i)$ required by Member 3's HMM Viterbi decoder.
- **Alternative (Option B - Pointwise Multiclass Classification)**:
  - Each row is a single GPS point ($t$); target is the `arc_id`.
  - *Limitation*: Can only predict among a fixed set of arcs seen during training. Fails on any trajectory traversing an arc not seen in the training set.
- **Team Discussion Point**: Does Member 1 intend to structure the processed dataset as candidate pairs (Option A) or pointwise records (Option B)?

---

### Decision 2: Ground-Truth Sequence-to-Trajectory Alignment
- **Status**: **Unresolved Decision** requiring agreement with Member 1.
- **The Challenge**:
  - The `.track` file contains a chronological sequence of $N$ GPS coordinates sampled at nominally 1 Hz.
  - The `.route` file contains a chronological sequence of $M$ traversed `arc_id`s.
  - **There is no inherent 1-to-1 correspondence between `.track` points and `.route` entries**:
    - A vehicle moving slowly or traversing a long arterial road will generate multiple consecutive GPS points on the same arc ($N > M$).
    - A vehicle traversing short intersection arcs within a single 1-second interval may pass through arcs without an intervening GPS measurement ($M > N$).
    - Severe GPS multipath drift may cause an observation to be spatially closer to an adjacent parallel street than the true traversed segment.
- **Team Discussion Point**: What alignment algorithm will Member 1 use to assign the true ground-truth `arc_id` ($a^*_t$) to each individual GPS observation point $p_t$?
  - *Option 2.1*: Spatial-temporal map matching projection (projecting each GPS point orthogonally onto the ordered sequence of reference arcs in `.route`).
  - *Option 2.2*: Dynamic Time Warping (DTW) or hidden shortest path alignment between timestamps and route segments.
  - *Option 2.3*: Discrete segment traversal interval estimation based on cumulative arc lengths and inferred vehicle speed.

---

### Decision 3: Upstream Candidate Recall Ceiling
- **Status**: **Upstream Limitation & Architectural Boundary**.
- **The Recall Ceiling**:
  - For each GPS point $p_t$, Member 1 generates a candidate arc set $A_t = \{a_{t,1}, \dots, a_{t,K}\}$ (e.g., via spatial KD-Tree / R-Tree query with radius $R \approx 50\text{ m}$).
  - **Critical Limitation**: If the true ground-truth arc $a^*_t$ is absent from the candidate set $A_t$ generated by Member 1, **the downstream ML classifier cannot predict it**, and every candidate row for that point will have `label = 0`.
  - The spatial recall of Member 1's candidate generator forms the hard theoretical upper bound on both ML candidate classification accuracy and Member 3's HMM route reconstruction accuracy.
- **Team Discussion Point**:
  - What spatial search strategy will Member 1 implement (fixed radius $R = 50\text{ m}$ vs. fixed Top-$K$ nearest segments)?
  - **Zero-Candidate Fallback**: If large GPS noise places a point where 0 candidate arcs fall within radius $R$, how does Member 1 handle it?
    - Dynamically expand search radius until at least 1 arc is retrieved?
    - Guarantee that the single nearest spatial arc is always included as candidate 0?
    *(Member 2 recommends always ensuring at least 1 candidate arc per GPS point to prevent dropped trajectory points).*

---

### Decision 4: Missing Value & Edge-Case Imputation Strategy
- **Status**: **Unresolved Preprocessing Protocol**.
- Certain physical features produce mathematical edge cases:
  - **Stationary Vehicle ($v \approx 0\text{ m/s}$)**: GPS heading is undefined or dominated by sensor noise. What value will Member 1 impute for angular heading difference ($|\theta_{\text{gps}} - \theta_{\text{arc}}|$) during standstill? (e.g., sentinel `0.0`, forward-fill previous valid heading, or an indicator flag `is_stationary`).
  - **Projection Outside Arc Extents ($u < 0$ or $u > 1$)**: When an orthogonal projection falls beyond segment endpoints, does Member 1 clamp $u \in [0, 1]$ and use distance to the nearest node?
- *Requirement*: The final feature matrix delivered to Member 2 must contain **zero `NaN` or infinite values**.

---

### Decision 5: Dataset Packaging & Delivery Format
- **Status**: **Recommendation for Agreement**.
- **Format**: Apache Parquet (`.parquet`) is strongly recommended over `.csv` because:
  - Preserves exact 32-bit/64-bit numerical precision and integer column types without parsing overhead.
  - Considerably smaller disk footprint and memory overhead.
- **Proposed File Paths**:
  - Training features: `data/processed/train_features.parquet`
  - Student test features: `data/processed/student_test_features.parquet`

---

## 5. Verification Checklist for Member 1 Prior to Handover

Before handing over the processed dataset for model benchmarking:

- [ ] File loads directly via `pd.read_parquet()` or `pd.read_csv()`.
- [ ] Contains all 80 training trips in the training feature file.
- [ ] `trip_id` column contains exact string/integer trip identifiers matching raw folder names.
- [ ] Exactly 0 `NaN`, 0 `None`, and 0 `Inf` values across all feature and target columns.
- [ ] All feature columns are numeric (`float` or `int`).
- [ ] Metadata columns (`point_idx`, `candidate_arc_id`) are preserved so candidate predictions can be reconnected to trajectory sequences by Member 3.
- [ ] `student_test_features` uses the exact same feature columns and transformations as `train_features`.
