# PathSnap — Distributed Map-Matching & GPS Trajectory Reconstruction

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)
[![Institution](https://img.shields.io/badge/PES%20University-UE24CS352A%20ML%20Hackathon-red.svg)](https://www.pes.edu/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit%20%7C%20NetworkX-orange.svg)](https://fastapi.tiangolo.com/)

> **PathSnap** is an end-to-end distributed map-matching and sequence reconstruction system developed for the PES University Machine Learning Hackathon (Course UE24CS352A). It accurately matches noisy, sparse, 1 Hz urban GPS trajectories to directed road networks by coupling a **Bayesian-optimized gradient boosted decision tree classifier** with a **log-domain Hidden Markov Model (HMM) Viterbi sequence decoder**, served across a **physical two-laptop local network architecture**.

---

## 1. Project Title and Description

**PathSnap** resolves the fundamental challenge of urban vehicle localization. Consumer GPS telemetry in metropolitan environments is corrupted by multipath signal reflections, satellite occlusion in urban canyons, atmospheric delay, clock drift, and temporal sensor dropouts. These inaccuracies cause raw GPS coordinates to drift into buildings, cross physical dividers, or project onto wrong parallel roads.

PathSnap solves this problem through a staged probabilistic and topological pipeline:
1. **Spatial Candidate Generation**: A 2D spatial grid index rapidly retrieves potential road segments within a metric radius ($R \approx 50\text{ m}$).
2. **Kinematic & Topological Feature Engineering**: 9 invariant physical and geometric features are extracted per candidate-observation pair.
3. **Discriminative ML Candidate Classification**: A tuned `HistGradientBoostingClassifier` evaluates candidate match probabilities ($p(\text{is\_match}=1)$).
4. **HMM Sequence Decoding**: An exponential transition model combined with Bayesian emission likelihoods resolves the globally optimal trajectory via the log-domain Viterbi algorithm.
5. **Topological Gap Filling**: Shortest-path graph interpolation eliminates sensor dropout gaps to produce a continuous road traversal sequence.
6. **Distributed Serving**: A high-performance FastAPI service on Laptop 1 serves an interactive Folium geospatial dashboard on Laptop 2 across a local Wi-Fi network.

---

## 2. Problem Statement

Given:
1. A temporal sequence of noisy GPS observations $T = \{(\text{lon}_t, \text{lat}_t, \tau_t)\}_{t=1}^N$ sampled at approximately 1 Hz.
2. A directed road network graph $G = (V, E)$, where vertices $V$ denote intersections/nodes and directed edges $E$ denote traversable road segments (arcs) with metric lengths.

The objective is to reconstruct the true sequence of traversed directed road segments $R = (a_1, a_2, \dots, a_M)$ ($a_k \in E$) that satisfies:
- **Spatial Proximity**: The reconstructed path follows the observed GPS coordinates within measurement uncertainty.
- **Topological Continuity**: Consecutive segments $(a_k, a_{k+1})$ must be connected in the directed graph ($v(a_k) = u(a_{k+1})$) without impossible network jumps.
- **Directional Consistency**: Segment orientation must align with vehicle heading and respect one-way restrictions.
- **Robustness to Dropouts**: Gaps between non-adjacent candidate transitions must be infilled via shortest-path routing.

---

## 3. Key Objectives and Features

- **Spatial Indexing**: 2D Cartesian spatial grid index for sub-millisecond road candidate retrieval.
- **Multi-Family ML Benchmarks**: Systematic 5-fold `GroupKFold` cross-validation across 5 algorithm families on 80 training trips ($1.42\text{M}$ candidate rows).
- **Bayesian Hyperparameter Optimization**: Optuna-driven tuning of `HistGradientBoostingClassifier` optimizing observation-level Top-1 ranking accuracy.
- **HMM Viterbi Decoder**: Numerically stable log-domain dynamic programming trellis with Newson & Krumm (2009) network distance transition modeling.
- **100% Topological Continuity**: Shortest-path intermediate arc interpolation resolves observation gaps.
- **Physical Two-Laptop Architecture**: Strict separation of concerns between backend compute/evaluation (Laptop 1) and evaluator presentation (Laptop 2).
- **Zero Data Leakage**: Ground-truth reference routes remain strictly isolated on Laptop 1. Laptop 2 never receives reference sequences or raw labels.
- **Interactive Geospatial Visualization**: Folium maps with toggleable layers for raw GPS points, reconstructed route polylines, and start/destination pins.

---

## 4. System Architecture

PathSnap decouples candidate scoring from sequence decoding:

```
Raw GPS Points (.track) + Road Network (.nodes, .arcs)
                      |
                      v
      [ Spatial Grid Index Retrieval (R = 50m) ]
                      |
                      v
      [ 9-Feature Candidate Extraction ]
      (perp_dist, proj_ratio, delta_heading, arc_length,
       gps_speed, speed_ratio, traversal_rate, rank, is_adjacent)
                      |
                      v
      [ HistGradientBoosting Classifier ]
      Evaluates candidate posterior: p(is_match = 1 | x)
                      |
                      v
      [ Bayesian Emission Transformation ]
      log p(o_t | a_j) with clipping (eps = 1e-7) & normalization
                      |
                      v
      [ NetworkX Graph Transition Model ]
      log p(a_k | a_j) = -log(beta) - |d_net - d_gps| / beta
                      |
                      v
      [ Log-Domain Viterbi MAP Decoder ]
      A* = argmax sum [ log p(o_t | a_t) + log p(a_t | a_{t-1}) ]
                      |
                      v
      [ Deduplication & Shortest-Path Gap Filling ]
      Produces continuous, fully traversable route sequence
                      |
                      v
      [ FastAPI JSON Endpoint & Streamlit Map Dashboard ]
```

---

## 5. Two-Laptop Deployment Architecture

The system is deployed across two distinct physical machines connected over a local area network (LAN):

```
+-------------------------------------------------------------------------+
|                       LAPTOP 2: EVALUATOR CLIENT                        |
|                                                                         |
|  - Streamlit UI Dashboard (ui/app.py)                                   |
|  - Evaluator selects Trip ID from 10 test trips                         |
|  - Sends HTTP POST request to Laptop 1: /api/v1/predict                 |
|  - Renders interactive Folium map (GPS points, predicted road route)    |
|  - Displays continuity score, arc count, and server latency             |
|  - ZERO local dataset files, trained models, or reference routes        |
+-------------------------------------------------------------------------+
                                    |
                                    | HTTP REST API (Port 8000)
                                    | PATHSNAP_API_URL: http://<LAPTOP_1_IP>:8000
                                    v
+-------------------------------------------------------------------------+
|                      LAPTOP 1: BACKEND MODEL SERVER                     |
|                                                                         |
|  - FastAPI Backend (api/main.py) listening on 0.0.0.0:8000              |
|  - Road network graph cache (NetworkX directed graph)                   |
|  - Spatial candidate retrieval & feature extraction engine              |
|  - Trained HistGradientBoosting Classifier (artifacts/models/)          |
|  - HMM emission likelihood & transition modeling                        |
|  - Viterbi dynamic programming trellis decoder                          |
|  - Shortest-path gap filling & continuity calculation                   |
|  - Private server-side evaluation against isolated ground truth         |
|  - Reference .route files strictly confined to Laptop 1                 |
+-------------------------------------------------------------------------+
```

---

## 6. Repository Directory Structure

```
pathsnap-ml-hackathon/
├── .gitignore                          # Excludes raw data, venvs, caches, large binaries
├── pyproject.toml                      # Package metadata, pytest configuration, tool settings
├── requirements.txt                    # Pinned production dependencies
├── README.md                           # Master project specification & documentation
│
├── api/                                # FastAPI Backend Service (Laptop 1)
│   ├── __init__.py                     # Package initialization
│   ├── main.py                         # FastAPI server with /health and /api/v1/predict
│   ├── routes.py                       # Modular endpoint routing
│   ├── schemas.py                      # Pydantic request/response schemas
│   └── services.py                     # TripManager and ReconstructionService orchestrators
│
├── artifacts/                          # Serialized Model Artifacts
│   ├── .gitkeep
│   └── models/
│       ├── pathsnap_histgbdt.joblib    # Trained production HistGradientBoosting model (957 KB)
│       └── pathsnap_histgbdt_metadata.json # Complete training parameters & CV metrics
│
├── configs/                            # Configuration Files
│   └── feature_schema.json             # Feature names, types, and physical units
│
├── data/                               # Dataset Storage (Local Only, Git-Ignored)
│   ├── raw/
│   │   ├── train/                      # 80 training trips (.track, .nodes, .arcs, .route)
│   │   ├── student_test/               # 10 student testing trips
│   │   └── evaluator_test/             # 10 evaluator demonstration trips
│   └── processed/
│       └── train_features.csv          # Precomputed 1.42M row feature matrix
│
├── docs/                               # Technical Guides & Specifications
│   ├── architecture.md                 # System architecture and pipeline design
│   ├── hmm_route_reconstruction_guide.md # Mathematical formulation of HMM and Viterbi
│   ├── ml_inference_integration_guide.md # Model loading and prediction contracts
│   ├── ml_input_contract.md            # Schema validation rules for ML models
│   └── two_laptop_deployment_guide.md  # Step-by-step physical LAN deployment runbook
│
├── experiments/                        # Experiment Tracking & Logs
│   └── results/
│       ├── baseline_model_comparison.csv # 5 model benchmark metrics across 5 folds
│       ├── baseline_model_comparison.json# Full raw benchmark JSON
│       ├── histgbdt_optuna_trials.csv  # 8 Optuna Bayesian optimization trials
│       ├── histgbdt_ranking_evaluation.csv # Phase 1 Top-1 observation ranking metrics
│       ├── knn_pilot.csv               # Pilot KNN experiment results
│       └── knn_pilot.json              # Pilot KNN detailed records
│
├── notebooks/                          # Exploratory Data Analysis
│   └── 01_dataset_exploration.ipynb    # Trajectory statistics and network exploration
│
├── reports/                            # Technical Reports & Audit Documentation
│   ├── api_performance_benchmark.csv   # Latency and continuity benchmark across trips
│   ├── candidate_ranking_evaluation.md # Observation-level Top-1 ranking report
│   ├── eda_findings.md                 # Data distributions, velocity profiles, sampling rates
│   ├── final_integration_readiness_audit.md # Full architecture verification audit
│   ├── final_ml_model_report.md        # Comprehensive report on production HistGBDT
│   ├── histgbdt_tuning_report.md       # Bayesian optimization analysis
│   ├── pipeline_audit.md               # Data pipeline audit
│   ├── pre_tuning_audit_report.md      # ML methodology pre-tuning audit
│   └── figures/                        # EDA plots, topology graphs, and map HTMLs
│
├── scripts/                            # Pipeline Utilities & Launchers
│   ├── build_features.py               # Feature extraction script for raw trips
│   ├── build_notebook.py               # Notebook generation utility
│   ├── run_backend.bat / .ps1          # Windows launchers for Laptop 1 FastAPI server
│   ├── run_frontend.bat / .ps1         # Windows launchers for Laptop 2 Streamlit dashboard
│   ├── run_eda.py                      # Exploratory data analysis generation
│   ├── run_histgbdt_optuna.py          # Optuna Bayesian hyperparameter optimization
│   ├── run_histgbdt_ranking_benchmark.py # Observation ranking evaluation script
│   ├── run_knn_pilot.py                # KNN pilot benchmark script
│   ├── run_model_benchmarks.py         # 5 model family cross-validation benchmark
│   ├── train_final_model.py            # Production model training on full eligible set
│   └── validate_best_trial.py          # 5-fold CV validation of best Optuna trial
│
├── src/                                # Core Library Package
│   ├── data/                           # Ingestion, validation, and graph construction
│   │   ├── alignment.py                # Monotonic GPS-to-route alignment & projection
│   │   ├── inspector.py                # Dataset validation and integrity auditing
│   │   └── loader.py                   # Typed parsers for .track, .nodes, .arcs, .route
│   ├── decoding/                       # HMM Sequence Decoding Engine
│   │   ├── emission.py                 # Bayesian emission likelihood calibration
│   │   ├── reconstruction.py           # Route deduplication, gap filling & continuity
│   │   ├── transition.py               # NetworkX graph transition modeling
│   │   └── viterbi.py                  # Log-domain Viterbi trellis dynamic programming
│   ├── evaluation/                     # Metric Calculation
│   │   └── route_metrics.py            # Sequence Levenshtein, Arc F1, Continuity
│   ├── features/                       # Spatial Indexing & Feature Extraction
│   │   └── builder.py                  # SpatialGridIndex and 9-feature generator
│   ├── models/                         # Model Implementation & Inference
│   │   ├── evaluate.py                 # 5-fold GroupKFold evaluation with ranking metrics
│   │   ├── model_factory.py            # Factory for LR, DT, RF, HistGBDT, KNN
│   │   ├── optimize.py                 # Optuna objective functions & search spaces
│   │   ├── predict.py                  # Production PathSnapPredictor inference engine
│   │   └── train.py                    # Training pipelines and cross-validation runners
│   └── visualization/                  # Map Rendering & Plotting
│       ├── eda_plots.py                # Figure generators for technical reports
│       └── map_renderer.py             # Multi-layer Folium map generation
│
├── tests/                              # Automated Test Suite (73 Tests)
│   ├── test_api.py                     # FastAPI endpoints and client integration
│   ├── test_decoding.py                # Emission, transition, Viterbi, gap filling
│   ├── test_evaluate.py                # Evaluation metrics and GroupKFold grouping
│   ├── test_optuna_tuning.py           # Optimization search space and objective tests
│   ├── test_predict.py                 # Predictor schema validation and reproducibility
│   ├── test_ranking_evaluate.py        # Top-1 candidate ranking evaluation
│   ├── test_route_metrics.py           # Sequence edit distance, Arc F1, route continuity
│   └── test_train.py                   # Model factory, training pipelines, determinism
│
└── ui/                                 # Streamlit Evaluator Dashboard (Laptop 2)
    └── app.py                          # Evaluator client dashboard with Folium map rendering
```

---

## 7. Dataset Description and Partitions

The dataset comprises urban vehicle tracking data partitioned into discrete trips. Each trip is contained in a dedicated directory with four plain-text files:

| File Extension | Content Description | Format & Schema | Key Considerations |
| :--- | :--- | :--- | :--- |
| **`.track`** | Raw GPS trajectory points | 3 whitespace-separated columns:<br>`<lon> <lat> <timestamp_seconds>` | Nominally 1 Hz sampling rate (~97% of intervals are exactly 1 s). Angular WGS84 coordinates requiring spherical projection. |
| **`.nodes`** | Road network vertices / intersections | 2 whitespace-separated columns:<br>`<lon> <lat>` | Zero-based line index defines the implicit `node_id`. |
| **`.arcs`** | Directed road network segments | 2 integer columns:<br>`<from_node_id> <to_node_id>` | Zero-based line index defines the implicit `arc_id`. Forms directed graph edges $E$. |
| **`.route`** | Ground-truth traversed route | 1 integer column:<br>`<arc_id>` | Chronological sequence of traversed road segments. Used exclusively for model evaluation. |

### Dataset Partitions and Isolation Rules

| Partition | Trip Count | Storage Location | Ground Truth Access |
| :--- | :---: | :--- | :--- |
| **Training** | **80 trips** | `data/raw/train/` | Full access. Used for feature extraction, 5-fold `GroupKFold` CV, model training, and Optuna tuning. |
| **Student Testing** | **10 trips** | `data/raw/student_test/` (or `Student folder/student_test/`) | Offline baseline verification and benchmarking. Never used for model training or tuning. |
| **Evaluator Testing** | **10 trips** | `data/raw/evaluator_test/` | **Strictly confidential server-side on Laptop 1**. Ground-truth `.route` files are never exposed to Laptop 2. |

---

## 8. Data Preprocessing and Feature Engineering

### 8.1 Spatial Grid Indexing
Road segments are indexed in a 2D metric Cartesian grid (`SpatialGridIndex` in `src/features/builder.py`) with cell size $50\text{ m}$. For any GPS point $P(\text{lon}, \text{lat})$, neighboring cells overlapping a search radius $R = 50\text{ m}$ are queried in $O(1)$ expected time, with automatic fallback expansion to $2R$ if no candidates are found.

### 8.2 The 9 Predictive Features
For every candidate road segment $a = (u, v)$ retrieved for observation $o_t$, the pipeline computes:

| Feature Name | Type | Physical Units | Mathematical Definition & Physical Interpretation |
| :--- | :---: | :---: | :--- |
| `perp_dist` | `float` | meters ($\text{m}$) | Orthogonal distance from GPS observation to candidate road segment in local projected metric space. Primary spatial proximity signal. |
| `proj_ratio` | `float` | dimensionless | Orthogonal projection scalar $u \in [0, 1]$ indicating relative position along segment from node $u$ ($0.0$) to node $v$ ($1.0$). Clamped to $[0, 1]$. |
| `delta_heading` | `float` | radians ($\text{rad}$) | Minimal angular heading discrepancy $| \theta_{\text{gps}} - \theta_{\text{arc}} | \in [0, \pi]$ between GPS movement vector and road segment bearing. Penalizes wrong-way and perpendicular segments. |
| `arc_length` | `float` | meters ($\text{m}$) | Metric physical length of the candidate road segment. |
| `gps_speed` | `float` | meters/sec ($\text{m/s}$) | Inferred vehicle speed between consecutive GPS timestamps: $v = \Delta d_{\text{gps}} / \Delta t$. |
| `speed_ratio` | `float` | dimensionless | Ratio of inferred GPS speed to reference urban velocity ($v / v_{\text{ref}}$, where $v_{\text{ref}} = 13.8889\text{ m/s} \approx 50\text{ km/h}$). |
| `traversal_rate` | `float` | inverse seconds ($\text{s}^{-1}$) | Frequency of segment traversal at current speed: $v_{\text{gps}} / \max(1.0, \text{arc\_length})$. Distinguishes highway transit from slow urban crawl. |
| `candidate_rank` | `int` | ordinal ($1, 2, \dots$) | 1-based proximity rank of the candidate sorted by perpendicular distance within the observation's candidate pool. |
| `is_adjacent` | `int` | binary ($0$ or $1$) | Topological continuity indicator: $1$ if candidate arc shares an endpoint with any of the previous observation's top-3 candidate arcs; $0$ otherwise. |

---

## 9. Machine Learning Models Experimented With

Five distinct model families were implemented in `src/models/model_factory.py` and evaluated across 5 folds using `GroupKFold` grouped by `trip_id`:

1. **HistGradientBoostingClassifier** (Gradient Boosting): Bin-based gradient boosting optimized for dense tabular features with monotonic and regularized splits.
2. **RandomForestClassifier** (Bagging / Tree Ensembles): Ensemble of 100 decorrelated decision trees with bootstrap aggregation.
3. **LogisticRegression** (Linear / Probabilistic): L2-regularized linear model with standard feature scaling.
4. **DecisionTreeClassifier** (Single Tree): Non-parametric greedy recursive partitioning tree baseline.
5. **KNeighborsClassifier** (Instance-Based): Non-parametric distance-weighted nearest neighbors ($k = 15$).

---

## 10. Model Comparison and Final Model Selection

### 10.1 5-Fold GroupKFold Benchmark (80 Training Trips, 1.42M Samples)

| Model Family | Candidate Macro F1 | Candidate Accuracy | Candidate Log Loss | Train Time / Fold | Inference Latency / Sample | Total Benchmark Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | **0.8782 ± 0.0183** | **94.59% ± 1.15%** | **0.1323 ± 0.0204** | **9.76 s** | **0.0036 ms** | **55.36 s** |
| **Random Forest** | 0.8749 ± 0.0181 | 94.50% ± 1.14% | 0.1951 ± 0.0388 | 166.43 s | 0.0075 ms | 846.17 s |
| **Logistic Regression** | 0.8557 ± 0.0184 | 93.78% ± 1.14% | 0.1691 ± 0.0217 | 1.48 s | 0.00013 ms | 9.71 s |
| **Decision Tree** | 0.8299 ± 0.0120 | 92.39% ± 0.98% | 2.7415 ± 0.3525 | 13.33 s | 0.00024 ms | 68.30 s |
| **KNN (Pilot Scale)\*** | 0.8767 ± 0.0241 | 95.03% ± 0.69% | 0.6400 ± 0.1582 | 0.02 s | 0.0196 ms | 3.41 s |

*\*KNN was benchmarked on a stratified pilot scale (12 trips, 10% sampled observations, 14,573 rows). Full-scale evaluation was omitted because $O(N)$ brute-force distance computation would take $>2$ hours per fold and preclude sub-millisecond real-time serving.*

### 10.2 Selection Rationale
`HistGradientBoostingClassifier` was selected as the final production model because it achieved:
- The highest candidate-level Macro F1 ($0.8782$).
- The lowest candidate Log Loss ($0.1323$), providing well-calibrated probabilities for downstream HMM emission modeling.
- Ultra-low inference latency ($0.0036\text{ ms}$ per candidate).
- Fast training ($9.76\text{ s}$ per fold), facilitating Bayesian hyperparameter optimization.

---

## 11. Bayesian Hyperparameter Optimization

Bayesian hyperparameter optimization was conducted using **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler (`scripts/run_histgbdt_optuna.py`). The objective function evaluated 5-fold `GroupKFold` cross-validation directly on observation-level **Eligible Top-1 Accuracy** (the metric that directly determines correct road selection).

### 11.1 Search Space and Trial Results

| Hyperparameter | Search Space | Trial 0 (Baseline) | Trial 6 (Best Selected) |
| :--- | :---: | :---: | :---: |
| `learning_rate` | Log-uniform $[0.01, 0.2]$ | $0.1000$ | **$0.04229$** |
| `max_iter` | Int $[100, 500]$ (step 50) | $100$ | **$400$** |
| `max_leaf_nodes` | Int $[15, 63]$ | $31$ | **$46$** |
| `max_depth` | Categorical $[None, 3, 5, 7, 9, 12]$ | $None$ | **$None$** |
| `min_samples_leaf` | Int $[10, 100]$ | $20$ | **$28$** |
| `l2_regularization` | Log-uniform $[10^{-3}, 10.0]$ | $0.0$ | **$0.31429$** |
| `max_bins` | Int $[64, 255]$ | $255$ | **$186$** |
| **Eligible Top-1 Accuracy** | **Objective** | **80.099%** | **80.387%** |

### 11.2 Production Model Checkpoint
Due to computational time constraints, the search was frozen at **8 completed trials** (Trial 6 confirmed best). The production model was refit on all eligible training samples ($1,423,994$ rows) using Trial 6 hyperparameters and serialized to:
- Model: `artifacts/models/pathsnap_histgbdt.joblib` ($957\text{ KB}$)
- Metadata: `artifacts/models/pathsnap_histgbdt_metadata.json`

---

## 12. HMM and Viterbi Route Reconstruction

While the ML classifier scores individual road segments, independent Top-1 selection produces disjoint trajectories with zigzagging jumps. PathSnap couples the classifier with a **Hidden Markov Model (HMM)** sequence decoder.

### 12.1 Emission Likelihood Modeling
Under Bayes' rule with a uniform spatial candidate prior $p(a_j) = \text{const}$:
$$p(o_t \mid a_j) = \frac{p(a_j \mid o_t) \cdot p(o_t)}{p(a_j)} \propto p(\text{is\_match}=1 \mid \mathbf{x}_{t, j})$$

To guarantee numerical stability:
1. Probabilities are clipped to $[\epsilon, 1.0]$ with $\epsilon = 10^{-7}$.
2. Probabilities are normalized across candidates for each observation: $\tilde{p}_j = p_j / \sum_k p_k$.
3. Log likelihood is computed: $\log p(o_t \mid a_j) = \log \tilde{p}_j$.
4. For candidate dropouts, a calibrated Gaussian fallback is applied: $\log p_{\text{spatial}} = -\frac{1}{2}\log(2\pi \sigma_z^2) - \frac{d_{\text{perp}}^2}{2\sigma_z^2}$ ($\sigma_z = 8.0\text{ m}$).

### 12.2 Graph-Based Transition Likelihood Modeling
Following Newson & Krumm (2009), the transition likelihood between candidate $a_{t-1, i}$ and $a_{t, j}$ models the discrepancy between the network shortest-path distance $d_{\text{net}}$ and observed GPS displacement $d_{\text{gps}}$:

$$\log p(a_{t, j} \mid a_{t-1, i}) = -\log \beta - \frac{|d_{\text{net}}(a_i, a_j) - d_{\text{gps}}|}{\beta} \quad (\beta = 15.0\text{ m})$$

- **Same Arc ($a_i == a_j$):** If $d_{\text{gps}} \le \text{length}(a_i)$, $\Delta d = 0.0$ (no false penalty for traveling along the segment).
- **Direct Adjacency ($v_i == u_j$):** $d_{\text{net}} = \text{length}(a_j)$, $O(1)$ fast path with 0 graph hops.
- **Multi-Hop Path ($v_i \ne u_j$):** Bounded Dijkstra computes shortest path length in `nx.DiGraph` using edge attribute `length_m`.
- **Disconnected Transitions:** Assigned `disconnect_penalty = -100.0`.

### 12.3 High-Performance Graph Search Optimization
- **Euclidean Pre-Rejection Filter ($O(1)$):** If $dx^2 + dy^2 > \text{cutoff}^2$, the pair is immediately rejected without calling Dijkstra.
- **Cutoff-Bounded Dijkstra:** `nx.single_source_dijkstra` restricts graph search to $\min(1000\text{m}, \max(300\text{m}, 4 \cdot d_{\text{gps}}))$.
- **Pairwise Path Memoization:** Caches all computed $(u, v)$ distances within the trip.

### 12.4 Log-Domain Viterbi Trellis
The Maximum A Posteriori (MAP) sequence $A^* = (a_0^*, \dots, a_{T-1}^*)$ is decoded via dynamic programming:
$$V_t(a_{t, j}) = \log p(o_t \mid a_{t, j}) + \max_{a_{t-1, i}} \left[ V_{t-1}(a_{t-1, i}) + \log p(a_{t, j} \mid a_{t-1, i}, d_{\text{gps}}) \right]$$
Backpointers track optimal predecessors, followed by traceback from $\arg\max_a V_{T-1}(a)$.

### 12.5 Deduplication & Shortest-Path Gap Filling
- Consecutive identical arcs $[a, a, a, b, b, c]$ are collapsed to $[a, b, c]$.
- For any remaining non-adjacent transitions ($v(a_k) \ne u(a_{k+1})$), intermediate shortest-path arcs are interpolated from the road graph, guaranteeing **100% topological route continuity**.

---

## 13. Evaluation Metrics

PathSnap computes rigorous sequence-level and observation-level metrics:

1. **Topological Route Continuity**:
   $$\text{Continuity} = \frac{\sum_{i=1}^{M-1} \mathbb{I}(v(a_i) == u(a_{i+1}))}{M - 1} \times 100\%$$
   *Note: Route continuity measures physical traversability across the road graph; it is not equivalent to ground-truth accuracy.*

2. **Arc Precision, Recall, and F1-Score**:
   $$\text{Precision} = \frac{|R_{\text{pred}} \cap R_{\text{ref}}|}{|R_{\text{pred}}|}, \quad \text{Recall} = \frac{|R_{\text{pred}} \cap R_{\text{ref}}|}{|R_{\text{ref}}|}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

3. **Multiset Arc F1**: Evaluates arc overlap frequency accounting for loops and repeated traversals.
4. **Sequence Levenshtein Edit Distance**: Exact dynamic programming edit distance between ordered lists of traversed arc IDs.
5. **Metric Route Length Error**: Percentage discrepancy between total reconstructed road length and reference length.

---

## 14. FastAPI Backend Documentation (Laptop 1)

The backend service is located in `api/main.py` and runs on **Laptop 1**, binding to `0.0.0.0:8000`.

### 14.1 Lifespan Management
The trained model (`artifacts/models/pathsnap_histgbdt.joblib`) and road network dataset catalog are loaded **once during application startup** via `@asynccontextmanager lifespan`. They are stored in application state and reused across requests.

### 14.2 Endpoints Specification

#### `GET /`
Service health check and API identification.
- **Response `200 OK`**:
  ```json
  {
    "status": "running",
    "service": "PathSnap API"
  }
  ```

#### `GET /health`
Operational readiness probe.
- **Response `200 OK`**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-11-19T12:00:00.000000"
  }
  ```

#### `POST /api/v1/predict`
Main trajectory reconstruction endpoint.
- **Request Body**:
  ```json
  {
    "trip_id": "00000000"
  }
  ```
  *(Accepts an 8-digit string trip ID validated against `^\d{8}$`)*
- **Response `200 OK`**:
  ```json
  {
    "status": "success",
    "trip_id": "00000000",
    "predicted_route": [12, 15, 18, 22, 25],
    "route_geometry": [
      [-122.4012, 37.7854],
      [-122.4025, 37.7865]
    ],
    "metrics": {
      "num_observations": 185,
      "num_route_arcs": 218,
      "continuity_score": 100.0,
      "gap_count": 4,
      "disconnect_count": 0,
      "runtime_ms": 1486.4,
      "viterbi_runtime_ms": 42.1
    },
    "message": "Route reconstructed successfully."
  }
  ```
- **Error Responses**:
  - `404 Not Found`: If `trip_id` does not exist in the dataset partition.
  - `422 Unprocessable Entity`: If `trip_id` format is invalid or route reconstruction fails.
  - `500 Internal Server Error`: Unhandled server exception with logged traceback.

---

## 15. Streamlit Frontend Documentation (Laptop 2)

The evaluator dashboard is located in `ui/app.py` and runs on **Laptop 2**.

### Key Capabilities:
- **Zero Local Model/Data Dependency**: The frontend requires no local models, datasets, or reference route files.
- **Configurable Backend Endpoint**: Reads `PATHSNAP_API_URL` environment variable with an interactive fallback text input.
- **Live Health Status Check**: Pings `GET /health` on Laptop 1 and displays an active connection badge.
- **Interactive Trip Input**: Validates 8-digit trip IDs before triggering reconstruction.
- **KPI Metrics Display**: Renders Route Continuity (%), Reconstructed Arcs, GPS Observation count, Total Runtime (ms), and Viterbi Runtime (ms).
- **Interactive Folium Map**: Automatically converts route geometry `[lon, lat]` into `[lat, lon]`, renders an OpenStreetMap base layer with an emerald green polyline, start pin (green), and end pin (red), fitted to bounding coordinates.

---

## 16. Installation and Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14.
- Git.
- Both laptops connected to the same Wi-Fi router or local network.

### Setup Instructions

```powershell
# 1. Clone repository
git clone https://github.com/pramathghegde/pathsnap-ml-hackathon.git
cd pathsnap-ml-hackathon

# 2. Create Python virtual environment
py -3.14 -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 4. Install production dependencies
pip install -r requirements.txt
```

---

## 17. Running Laptop 1 – Model Server

Laptop 1 hosts the dataset, trained models, and FastAPI backend.

```powershell
# Open terminal in repository root on Laptop 1
# Ensure server binds to 0.0.0.0 so Laptop 2 can access it over LAN
py -3.14 -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or execute the one-click Windows launcher:
.\scripts\run_backend.bat
```

Verify backend in browser: `http://localhost:8000/health`  
Interactive Swagger API docs: `http://localhost:8000/docs`

---

## 18. Running Laptop 2 – Evaluator Client

Laptop 2 runs only the Streamlit UI dashboard.

```powershell
# 1. Identify Laptop 1's LAN IP address (see Section 19)
# Example: 192.168.1.105

# 2. Set environment variable pointing to Laptop 1
$env:PATHSNAP_API_URL = "http://192.168.1.105:8000"

# 3. Launch Streamlit dashboard
py -3.14 -m streamlit run ui/app.py

# Or execute the interactive Windows launcher (prompts for Laptop 1 IP):
.\scripts\run_frontend.bat
```

Open browser on Laptop 2: `http://localhost:8501`

---

## 19. LAN Configuration and Troubleshooting

### 19.1 Identifying Laptop 1's LAN IPv4 Address
On **Laptop 1**, run in PowerShell:
```powershell
ipconfig
```
Look for `IPv4 Address` under the active Wi-Fi or Ethernet adapter (e.g., `192.168.1.105`).

### 19.2 Network Reachability Test
On **Laptop 2**, test connection to Laptop 1:
```powershell
# Test ICMP ping
ping 192.168.1.105

# Test HTTP health endpoint
curl http://192.168.1.105:8000/health
```

### 19.3 Windows Firewall Configuration
If Laptop 2 cannot connect despite being on the same Wi-Fi, Windows Firewall on Laptop 1 may be blocking port 8000. Run this command once in an Administrator PowerShell on Laptop 1:
```powershell
New-NetFirewallRule -DisplayName "PathSnap Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

---

## 20. Running Tests

The test suite is built on `pytest` and verifies data pipelines, ML models, Optuna tuning, HMM decoding, route metrics, and API serving:

```powershell
# Execute the complete test suite
py -3.14 -m pytest

# Run with verbose output
py -3.14 -m pytest -v

# Run API integration tests specifically
py -3.14 -m pytest tests/test_api.py
```

### Test Suite Summary:
- `tests/test_api.py` (5 tests): Home endpoint, health probe, valid prediction, 422 validation, 404 unknown trip.
- Full test suite across repository branches comprises up to **73 tests** verifying ML training, 5-fold CV, ranking evaluation, Optuna tuning, and HMM decoding. All tests pass with 0 failures.

---

## 21. Experimental Results

### 21.1 End-to-End Reconstruction Benchmarks (Student Test Trips)

Evaluated via the API on the 10 student-test trajectories (`reports/api_performance_benchmark.csv`):

| Trip ID | Observations | Traversed Arcs | Topological Continuity | Disconnections | Server Latency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 185 | 218 | **100.00%** | 0 | 1,486 ms |
| `00000007` | 94 | 124 | **100.00%** | 0 | 641 ms |
| `00000009` | 240 | 499 | **100.00%** | 0 | 1,649 ms |
| `00000040` | 312 | 383 | **100.00%** | 0 | 1,990 ms |
| `00000052` | 78 | 93 | **100.00%** | 0 | 445 ms |
| `00000056` | 1,210 | 1,367 | **99.93%** | 1 | 4,958 ms |
| `00000065` | 115 | 159 | **100.00%** | 0 | 1,219 ms |
| `00000076` | 450 | 501 | **100.00%** | 0 | 3,013 ms |
| `00000077` | 512 | 534 | **100.00%** | 0 | 3,624 ms |

- **Mean Topological Route Continuity**: **$99.99\%$** across all test trips.
- **Average Server Response Time**: **$2.11\text{ s}$** per trip.

### 21.2 Comparative Analysis: Naive ML Top-1 vs HMM Viterbi
Benchmarked on training trip `00000001` (185 observations, 165,790 road segments):

| Metric | Naive ML Top-1 (Argmax) | PathSnap HMM Viterbi | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Topological Continuity** | 84.21% | **100.00%** | **+15.79% (Fully continuous)** |
| **Traversed Route Arcs** | 57 fragmented arcs | **542 continuous arcs** | **Full traversable road path** |
| **Unique Arc F1-Score** | 0.7842 | **0.8654** | **+0.0812** |
| **Sequence Edit Distance** | 382 | **214** | **-168 edit operations** |

---

## 22. Visualizations and Screenshots

PathSnap generates multi-layer interactive Folium HTML maps stored in `reports/figures/`:

- **Trip 00000000 Route Map**: [`reports/figures/trip_00000000_map.html`](file:///D:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/reports/figures/trip_00000000_map.html)
- **Trip 00000001 Route Map**: [`reports/figures/trip_00000001_map.html`](file:///D:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/reports/figures/trip_00000001_map.html)

### Key EDA Figures:
- **Candidate Recall Curve**: `reports/figures/candidate_recall_curve.png` (89.53% recall at $R=50\text{m}$).
- **Sampling Rate Distribution**: `reports/figures/sampling_rate_distribution.png` (97.2% exactly 1.0s).
- **Projection Error Distribution**: `reports/figures/projection_error_distribution.png` (median error $4.2\text{m}$).
- **Road Network Topology**: `reports/figures/network_topology.png` (directed urban graph structure).

---

## 23. Limitations and Future Improvements

1. **Computational Scaling for Very Long Trips**: Trips with $>5,000$ observations (e.g. `00000015` with 6,854 points) require $12\text{--}15$ seconds for full feature extraction and Viterbi decoding. A sliding-window Viterbi decoder would bound latency to $O(W)$.
2. **Turn Restriction Constraints**: Transition modeling currently relies on shortest-path edge distances. Incorporating explicit turn-prohibition penalties (e.g., illegal U-turns or disallowed left turns) would further refine urban route accuracy.
3. **Multi-Modal Transport Modes**: Current kinematics assume passenger vehicle dynamics with an urban reference speed of $50\text{ km/h}$. Dynamic mode detection (pedestrian, bicycle, bus) would improve flexibility.

---

## 24. Team Contributions

| Team Member | Hackathon Role | Key Deliverables & Contributions |
| :--- | :--- | :--- |
| **Member 1** | **Data Engineering & Spatial Pipeline** | Ingested `.track`, `.nodes`, `.arcs`, and `.route` datasets. Implemented monotonic GPS-to-route alignment, 2D `SpatialGridIndex`, 9 engineered features, EDA analysis, and candidate recall auditing. |
| **Member 2** | **Machine Learning & Bayesian Optimization** | Implemented 5 model families, 5-fold `GroupKFold` cross-validation, observation-level Top-1 ranking evaluation, Optuna Bayesian optimization (8 trials), and trained the production `HistGradientBoostingClassifier`. |
| **Member 3** | **HMM Decoding & Route Evaluation** | Formulated Bayesian emission likelihoods with clipping and fallback, NetworkX graph transition model, log-domain Viterbi decoder, shortest-path gap filling, sequence Levenshtein distance, and route continuity metrics. |
| **Member 4** | **Distributed Backend, UI & System Integration** | Built the FastAPI backend server (`api/main.py`), Streamlit evaluator dashboard (`ui/app.py`), physical two-laptop network architecture, interactive Folium map integration, launch scripts, and automated test suite. |

---

## 25. Acknowledgements and References

1. **Newson, P., & Krumm, J. (2009)**. *Hidden Markov map matching through noise and sparseness*. Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, 336–343.
2. **Lou, Y., et al. (2009)**. *Map-matching for low-sampling-rate GPS trajectories*. Proceedings of the 17th ACM SIGSPATIAL, 352–361.
3. **Akiba, T., et al. (2019)**. *Optuna: A next-generation hyperparameter optimization framework*. KDD 2019.
4. **Pedregosa, F., et al. (2011)**. *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. **Hagberg, A., Swart, P., & Chult, D. (2008)**. *Exploring network structure, dynamics, and function using NetworkX*. Los Alamos National Laboratory.
6. **PES University Department of Computer Science & Engineering** — Machine Learning Course UE24CS352A Hackathon Faculty & Mentors.
