# PathSnap System Architecture and Data Flow Specification

> **Document Status**: Architectural Specification & Baseline Blueprint  
> **Target Audience**: Development Team, Hackathon Evaluators, System Integrators  
> **Implementation State**: Planned / Pre-Implementation Scaffold  

---

## 1. Executive Summary

**PathSnap** is a distributed machine learning system engineered to resolve noisy, low-frequency GPS trajectories against high-density road network graphs. In dense urban topographies, multipath distortion, GPS signal degradation, and satellite shadowing cause reported coordinates to deviate significantly from the true traversed road segments—frequently placing vehicles onto parallel corridors, water bodies, or adjacent buildings.

PathSnap addresses this via an end-to-end multi-stage pipeline:
1. Spatial candidate generation and topological indexing.
2. Geometric and temporal feature engineering.
3. Multi-class/probabilistic candidate prediction via machine learning classifiers.
4. Hidden Markov Model (HMM) sequence decoding with the Viterbi algorithm to enforce topological continuity.
5. Distributed two-node deployment separating the compute/evaluation engine (FastAPI on Laptop 1) from the evaluator visual interface (Streamlit/Folium on Laptop 2).

---

## 2. Distributed Two-Laptop Deployment Architecture

The system is architected for distributed physical deployment across two independent machines connected over a local area network (LAN / Wi-Fi subnet).

```
+-----------------------------------------------------------------------------------+
|                                 PHYSICAL ARCHITECTURE                             |
+-----------------------------------------------------------------------------------+

   +---------------------------------------+          +--------------------------------------+
   |        LAPTOP 2: EVALUATOR UI         |          |       LAPTOP 1: MODEL & API HOST     |
   |                                       |          |                                      |
   |  +---------------------------------+  |          |  +--------------------------------+  |
   |  |   Evaluator-Facing Interface    |  |          |  |       FastAPI Application      |  |
   |  |   (Streamlit / Web App)         |  |          |  |       (Uvicorn ASGI Server)    |  |
   |  +---------------------------------+  |          |  +--------------------------------+  |
   |                  |                    |          |                  |                   |
   |                  v                    |          |                  v                   |
   |  +---------------------------------+  |          |  +--------------------------------+  |
   |  |     Trip Selection Console      |  |          |  |     Pipeline Controller        |  |
   |  |    (10 Reserved Eval Trips)     |  |          |  |   - Ingestion & Candidates     |  |
   |  +---------------------------------+  |          |  |   - ML Feature Extraction      |  |
   |                  |                    |          |  |   - Tuned Classifier Inference |  |
   |                  |                    |          |  |   - HMM Viterbi Decoder        |  |
   |                  | HTTP POST /predict |          |  +--------------------------------+  |
   |                  | trip_id: "00000040"|          |                  |                   |
   |                  +------------------------------>|                  v                   |
   |                                       |          |  +--------------------------------+  |
   |                                       |          |  |     Server-Side Evaluator      |  |
   |                  +-------------------------------|  |   - Internal Ground Truth      |  |
   |                  | Response JSON:     |          |  |   - Top-1 Accuracy & F1-score  |  |
   |                  | - Predicted Arcs   |          |  |   - Route Continuity Metrics   |  |
   |                  | - Evaluation Score |          |  +--------------------------------+  |
   |                  v                    |          |                                      |
   |  +---------------------------------+  |          |  +--------------------------------+  |
   |  |    Interactive Folium Renderer  |  |          |  |    Artifact Storage & Logs     |  |
   |  |   - Raw GPS Observations        |  |          |  |   - Serialized Models (.joblib)|  |
   |  |   - Road Network Graph          |  |          |  |   - Topology Cache (.parquet)  |  |
   |  |   - Predicted Vehicle Path      |  |          |  |   - Execution Logs (.log)      |  |
   |  +---------------------------------+  |          |  +--------------------------------+  |
   +---------------------------------------+          +--------------------------------------+
```

### Deployment Roles and Responsibilities

| Role | Host Node | Responsibilities | Security / Isolation Policy |
| :--- | :--- | :--- | :--- |
| **Model & API Server** | **Laptop 1** | - Hosts trained classifier weights and HMM decoder.<br>- Executes feature extraction and inference pipeline.<br>- Loads and stores ground-truth reference routes (`.route`).<br>- Computes objective validation metrics server-side.<br>- Emits structured API logs on console for evaluator verification. | **Zero Exposure**: Reference route files and ground-truth labels remain strictly server-side. They are never transmitted to Laptop 2 or exposed to the candidate prediction engine. |
| **User Interface Client** | **Laptop 2** | - Displays the 10 evaluator trip selection interface.<br>- Transmits selected `trip_id` payloads via REST requests.<br>- Receives reconstructed route vectors and performance metrics.<br>- Renders multi-layer interactive Folium maps with layer toggles. | **Client Isolation**: Contains zero machine learning weights, zero pre-computed route predictions, and zero ground-truth data. Communicates exclusively via HTTP. |

---

## 3. End-to-End Data Flow Architecture

The data pipeline progresses through discrete, decoupled phases:

```
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END DATA PIPELINE                                        |
+----------------------------------------------------------------------------------------------------+

 [ Raw Trajectory ]       [ Road Network ]
    (.track file)       (.nodes & .arcs files)
          |                       |
          +----------+------------+
                     |
                     v
   +------------------------------------+
   | 1. Ingestion & Spatial Preprocessing|
   |    - Geodesic / UTM Projection     |
   |    - Temporal 1Hz Verification     |
   |    - Spatial Indexing (KDTree/RTree)|
   +------------------------------------+
                     |
                     v
   +------------------------------------+
   | 2. Candidate Arc Generation        |
   |    - Spatial radius search (e.g.50m)|
   |    - Orthogonal projection to arcs |
   |    - Top-K nearest arc candidates  |
   +------------------------------------+
                     |
                     v
   +------------------------------------+
   | 3. Feature Extraction Pipeline     |
   |    - Perpendicular offset distance |
   |    - Heading difference (|θ_gps-θ_arc|)|
   |    - Longitudinal arc position (u) |
   |    - Speed plausibility vs speed limit|
   |    - Temporal connectivity penalty |
   +------------------------------------+
                     |
                     v
   +------------------------------------+
   | 4. ML Candidate Classification     |
   |    - 5 Evaluated Model Families    |
   |    - Bayesian-Tuned Best Model     |
   |    - Yields: P(arc_i | observation)|
   +------------------------------------+
                     |
                     v
   +------------------------------------+
   | 5. HMM & Viterbi Sequence Decoder  |
   |    - Hidden States: Candidate Arcs |
   |    - Emission: Classifier P(arc)   |
   |    - Transition: Shortest path vs  |
   |      great-circle distance delta   |
   |    - Viterbi dynamic programming   |
   +------------------------------------+
                     |
                     v
   +------------------------------------+
   | 6. Evaluation & Presentation Layer |
   |    - Pointwise Top-1 Accuracy      |
   |    - Macro F1-score & Loss         |
   |    - Route Continuity Ratio        |
   |    - Folium Interactive GeoMap     |
   +------------------------------------+
```

---

## 4. Subsystem Deep-Dive

### 4.1 Data Ingestion & Spatial Coordinate Handling
- **Coordinate System Considerations**: Raw `.track` observations and `.nodes` coordinates are provided as angular spherical coordinates (WGS84: Longitude, Latitude). Direct planar Euclidean distances in decimal degrees introduce significant distortion.
- **Planned Projection**: Calculations will utilize geodesic distances (Vincenty / Haversine) or re-projection into local Universal Transverse Mercator (UTM) cartesian coordinates (measured in meters) prior to metric calculations.
- **Graph Topology**: Road networks are ingested from `.nodes` (geographical coordinates) and `.arcs` (directed edges: `from_node_id`, `to_node_id`). Row index in `.arcs` serves as `arc_id`.

### 4.2 Candidate Arc Generation
For every GPS point $p_t = (\text{lon}_t, \text{lat}_t, \tau_t)$:
1. A spatial query window (e.g., radius $R = 50\text{ m}$) retrieves candidate road segments $A_t = \{a_{t,1}, a_{t,2}, \dots, a_{t,K}\}$.
2. The orthogonal projection point $p'_{t,k}$ on each segment $a_{t,k}$ is computed.
3. If no candidate falls within $R$, the search radius dynamically expands, or a sentinel fallback candidate is flagged to prevent pipeline blockage.

### 4.3 Feature Engineering Pipeline
Features extracted for each observation-candidate pair $(p_t, a_{t,k})$ include:
- **Spatial Orthogonal Distance**: Perpendicular distance from $p_t$ to arc segment $a_{t,k}$.
- **Heading Deviation**: Absolute angular difference $|\theta_{\text{GPS}} - \theta_{\text{arc}}|$ between vehicle movement vector and arc orientation.
- **Speed Consistency**: Ratio of inferred speed $d(p_{t-1}, p_t)/\Delta t$ against expected speed profile.
- **Longitudinal Projection Ratio**: Relative position $u \in [0, 1]$ of the projected point along the candidate segment length.
- **Structural Connectivity Indicators**: Binary indicator if candidate $a_{t,k}$ is directly reachable from previous top candidates $A_{t-1}$.

### 4.4 Machine Learning Classification Layer
- **Requirement**: Five distinct algorithm families must be independently implemented, trained on the 80 training trips, and cross-evaluated:
  1. *Linear / Probabilistic*: Logistic Regression / Gaussian Naive Bayes / SGD Classifier.
  2. *Instance-Based*: Distance-Weighted K-Nearest Neighbors.
  3. *Tree-Based / Ensemble*: Random Forest / Extra Trees Classifier.
  4. *Gradient Boosting*: LightGBM / XGBoost / CatBoost.
  5. *Neural Networks / Kernel Methods*: Multi-Layer Perceptron (MLP) / Support Vector Machine (RBF kernel).
- **Output Constraint**: Models must output well-calibrated class probability distributions $P(y = 1 \mid \mathbf{x})$, serving as emission probabilities for the sequential decoder.

### 4.5 Bayesian Hyperparameter Optimization
- **Strategy**: Bayesian Optimization via Optuna utilizing Tree-structured Parzen Estimators (TPE).
- **Leakage Prevention**: Group-aware $K$-Fold cross-validation (`GroupKFold`) grouped by `trip_id` across the 80 training trips. Under no circumstance are observations from the same trip split between training and validation folds.
- **Objective Function**: Maximize validation Macro F1-score subject to inference latency constraints.

### 4.6 Hidden Markov Model (HMM) & Viterbi Sequence Decoder
Pointwise classifiers frequently output disconnected segments due to localized GPS noise. The HMM resolves this:
- **Hidden States ($S_t$)**: The candidate road segments $\{a_{t,1}, \dots, a_{t,K}\}$ for observation $t$.
- **Emission Probability $P(z_t \mid s_t = a_{t,i})$**:
  $$P(z_t \mid s_t = a_{t,i}) \propto P_{\text{ML}}(a_{t,i} \mid \mathbf{x}_t) \cdot \exp\left(-\frac{\|p_t - p'_{t,i}\|^2}{2\sigma_z^2}\right)$$
  combining the learned classifier confidence with gaussian measurement error.
- **Transition Probability $P(s_t = a_{t,j} \mid s_{t-1} = a_{t-1,i})$**:
  $$P(a_{t,j} \mid a_{t-1,i}) \propto \exp\left(-\frac{|d_{\text{network}}(p'_{t-1,i}, p'_{t,j}) - d_{\text{euclidean}}(p_{t-1}, p_t)|}{\beta}\right)$$
  where $d_{\text{network}}$ is the shortest path graph distance between projected points on the road network. If no valid topological path exists between arcs, transition probability drops to an epsilon penalty.
- **Viterbi Decoding**: Dynamic programming computes the globally optimal hidden state sequence $\hat{S}_{1:T} = \arg\max \prod_{t=1}^T P(z_t \mid s_t) P(s_t \mid s_{t-1})$.

---

## 5. Component Implementation Status Matrix

| Subsystem / Module | Directory / Path | Planned Role | Current Implementation Status |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | `src/data/` | Parsers for `.track`, `.nodes`, `.arcs`, `.route` | 🟡 **Planned** (Scaffolded package) |
| **Feature Extraction**| `src/features/` | Spatial distance, heading, graph connectivity features | 🟡 **Planned** (Scaffolded package) |
| **ML Classifiers** | `src/models/` | 5 distinct model families + Bayesian tuning module | 🟡 **Planned** (Scaffolded package) |
| **Sequence Decoding**| `src/decoding/` | HMM formulation + Viterbi path reconstruction | 🟡 **Planned** (Scaffolded package) |
| **Evaluation Metrics**| `src/evaluation/`| Top-1 accuracy, F1-score, route continuity scorer | 🟡 **Planned** (Scaffolded package) |
| **Map Visualization** | `src/visualization/` | Multi-layer Folium HTML map generator | 🟡 **Planned** (Scaffolded package) |
| **FastAPI Backend** | `api/` | REST inference & validation endpoints (Laptop 1) | 🟡 **Planned** (Scaffolded package) |
| **Evaluator UI** | `ui/` | Interactive trip console & map interface (Laptop 2) | 🟡 **Planned** (Scaffolded package) |
| **Experiment Store** | `experiments/` | Logs, metric CSVs, Optuna trial databases | 🟡 **Planned** (Scaffolded directories) |
| **Configuration** | `configs/` | YAML / JSON pipeline hyperparameter configurations | 🟡 **Planned** (Scaffolded directory) |
| **Test Suite** | `tests/` | Unit, integration, and topology validation tests | 🟡 **Planned** (Scaffolded package) |

---

## 6. Verification and Evaluator Demonstration Protocol

During hackathon judging:
1. **Host Launch (Laptop 1)**: The FastAPI server is initialized with `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Console displays live access logs and model loading confirmations.
2. **Client Launch (Laptop 2)**: The evaluator UI is launched pointing to Laptop 1's IP address.
3. **Trip Execution**: Evaluators select any trip from the 10 reserved evaluation trips (`student_test` / `evaluator_test`).
4. **Live Verification**:
   - UI sends the trip identifier to Laptop 1.
   - Laptop 1 logs show request receipt, feature extraction time, classifier inference time, and Viterbi decoding execution.
   - Ground-truth validation metrics are computed on Laptop 1 and returned alongside the predicted arc sequence.
   - Laptop 2 renders the interactive Folium map displaying the raw GPS points (blue), road network geometry (gray), and predicted trajectory (green/orange).
