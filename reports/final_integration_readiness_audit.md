# PathSnap: Final Integration Readiness & Architecture Audit Report

**Auditor & Architect:** Technical Lead, Senior Software Architect & Integration Engineer  
**Project:** PathSnap – Distributed Map-Matching & Route Reconstruction (PES University ML Hackathon)  
**Repository:** `D:\Engineering\PESU\SEM 5\Machine Learning\Hackthon\pathsnap-ml-hackathon`  
**Current Branch:** `feat/model-training-continuation`  
**Date:** November 18, 2026  
**Overall Readiness Verdict:** **PHASE 1 & 2 COMPLETE — PROCEED IMMEDIATELY TO HMM & DISTRIBUTED INTEGRATION**  

---

## 1. Executive Summary & Component Status Matrix

| Component / Workstream | Owner | Status | Key Deliverables & Evidence | Blockers / Next Steps |
|:---|:---:|:---:|:---|:---|
| **Member 1: Data Engineering & Preprocessing** | Member 1 | **COMPLETED** | `loader.py`, `alignment.py`, `builder.py`, spatial grid indexing, 80 train trips + 10 student-test trips processed. | Ready for backend integration; need to extract `evaluator_test.zip` on Laptop 1. |
| **Member 2: Machine Learning & Inference** | Member 2 | **COMPLETED** | 5 model families, 8 Optuna trials, Trial 6 GBDT serialized (`0.91 MB`), `predict.py` inference engine, 47 tests pass. | Frozen & verified. Ready to supply emission probabilities to Member 3. |
| **Member 3: HMM Decoding & Route Reconstruction** | Member 3 | **NOT IMPLEMENTED** | `src/decoding/` and `src/evaluation/` have only empty scaffolding `__init__.py`. | **Primary Blocker for Route Output:** Must implement emission calibration, transition graph modeling, and Viterbi decoder. |
| **Member 4A: FastAPI Backend (Laptop 1)** | Member 4 | **NOT IMPLEMENTED** | `api/` has only empty scaffolding `__init__.py`. | Needs `fastapi`, `uvicorn`, endpoint implementation, and pipeline orchestration. |
| **Member 4B: Streamlit UI Client (Laptop 2)** | Member 4 | **NOT IMPLEMENTED** | `ui/` has only empty `.gitkeep`. `map_renderer.py` exists in `src/visualization/`. | Needs `streamlit` client app, trip selection, and LAN HTTP client. |

---

## 2. Current Repository & Git State

- **Branch:** `feat/model-training-continuation`
- **Head Commit:** `dfcb98d – Complete Phase 1 candidate ranking evaluation`
- **Working Tree Status:** Clean, stable, zero merge conflicts.
- **Untracked Additions (Phases 2 & 3):**
  - Serialized model: `artifacts/models/pathsnap_histgbdt.joblib` (0.91 MB)
  - Metadata: `artifacts/models/pathsnap_histgbdt_metadata.json`
  - Predictor: `src/models/predict.py`
  - Integration guide: `docs/ml_inference_integration_guide.md`
  - Audit and tuning reports: `reports/*.md`
  - New test suites: `tests/test_optuna_tuning.py`, `tests/test_predict.py`
- **Test Suite Status:** **47 passed in 5.55s** (`py -3.14 -m pytest`).

---

## 3. Detailed Component Audits

### 3.1 Member 1 – Data Engineering (COMPLETED)
- **Data Loaders (`src/data/loader.py`):** Typed parsers for `.track`, `.nodes`, `.arcs`, `.route`, and NetworkX directed graph builder `build_network_graph()`.
- **Spatial Indexing & Candidate Generation (`src/features/builder.py`):** `SpatialGridIndex` implements local 2D metric grid partitioning with 50m radius search.
- **Feature Pipeline (`src/features/builder.py`):** `extract_trip_features()` generates all 9 standardized geometric and kinematic features with rigorous physical units (`perp_dist`, `proj_ratio`, `delta_heading`, `arc_length`, `gps_speed`, `speed_ratio`, `traversal_rate`, `candidate_rank`, `is_adjacent`).
- **Visualization Assets (`src/visualization/`):** `map_renderer.py` provides complete Folium multi-layer mapping for GPS points, corridor road networks, and reference routes.
- **Assessment:** Fully verified and capable of on-the-fly candidate extraction for any raw trip directory.

### 3.2 Member 2 – Machine Learning (COMPLETED & FROZEN)
- **Baseline Models:** 5 distinct algorithm families evaluated in `experiments/results/baseline_model_comparison.csv` and `knn_pilot.csv`.
- **Ranking Evaluation:** Implemented in `src/models/evaluate.py`. Rigorously distinguishes Eligible Top-1 ($80.39\%$) from Overall Top-1 ($72.24\%$) with deterministic tie-breaking.
- **Bayesian Tuning:** 8 completed Optuna trials preserved in `experiments/results/histgbdt_optuna_trials.csv`. Trial 6 selected as best completed configuration (+0.29% gain over baseline).
- **Production Artifacts:** Refitted on $1,423,994$ eligible candidate rows across all 80 training trips. Saved to `artifacts/models/pathsnap_histgbdt.joblib` (0.91 MB).
- **Inference Engine (`src/models/predict.py`):** `PathSnapPredictor` handles dynamic positive class resolution, feature re-ordering immunity, and identifier preservation with $0.064\text{ ms/observation}$ latency.
- **Assessment:** Ready for immediate consumption by Member 3 (HMM) and Member 4 (FastAPI).

### 3.3 Member 3 – HMM & Route Reconstruction (NOT IMPLEMENTED)
- **Current State:** `src/decoding/` and `src/evaluation/` contain only empty `__init__.py` files.
- **Required Modules to Build:**
  1. `src/decoding/emission.py`: Probability transformation from ML classifier match probabilities into HMM emission likelihoods, with spatial distance fallback for candidate misses.
  2. `src/decoding/transition.py`: NetworkX graph routing distance vs. GPS displacement modeling via exponential distribution ($p(d) = \frac{1}{\beta} e^{-|d_{\text{graph}} - d_{\text{gps}}|/\beta}$) and immediate arc connectivity handling.
  3. `src/decoding/viterbi.py`: Log-domain dynamic programming Viterbi decoder tracking optimal path backpointers.
  4. `src/decoding/reconstruction.py`: Route sequence assembly and topological continuity validation.
  5. `src/evaluation/route_metrics.py`: Sequence-level route evaluation metrics (Exact Sequence Match, Arc Precision/Recall/F1, Edit Distance, Route Length Error).

### 3.4 Member 4 – Backend Server & UI Client (NOT IMPLEMENTED)
- **Current State:** `api/` contains only `__init__.py`; `ui/` contains only `.gitkeep`.
- **Required Modules to Build:**
  1. `api/main.py`: FastAPI server with CORS middleware, lifespan initialization loading the ML model and road graph caches.
  2. `api/schemas.py`: Pydantic request/response contracts (`ReconstructionRequest`, `ReconstructionResponse`, `TripSummary`).
  3. `api/routes.py`:
     - `GET /api/v1/health`: System health and model version.
     - `GET /api/v1/trips`: List available evaluator-test trips (ID and bounding box only; no labels).
     - `POST /api/v1/reconstruct`: Executes pipeline on Laptop 1 and returns reconstructed coordinates and evaluation score summary.
     - `GET /api/v1/trips/{trip_id}/map`: Returns interactive Folium map HTML.
  4. `ui/app.py`: Streamlit frontend on Laptop 2 with LAN server IP configuration, trip selector, metrics cards, and embedded Folium map.

---

## 4. Dataset and Data-Protection Status

### 4.1 Dataset Availability
- `data/raw/train/`: 80 trips present (`00000001` .. `00000099`).
- `data/raw/student_test/`: 10 trips present (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`).
- `data/raw/evaluator_test/`: Not extracted yet. Archive `evaluator_test.zip` (2.4 MB, 10 trips: `00000015`, `00000017`, `00000023`, etc.) is present in the parent workspace directory `d:\Engineering\PESU\SEM 5\Machine Learning\Hackthon\`.

### 4.2 Strict Data Protection Rules (Two-Laptop Protocol)
1. **Server-Side Confinement:** `evaluator_test.zip` will be extracted **only on Laptop 1** into `data/raw/evaluator_test/`.
2. **Zero Ground-Truth Transmission:** Laptop 2 will **never receive** reference `.route` files or ground-truth arc arrays over LAN.
3. **Trip ID Only:** Laptop 2 sends only the selected `trip_id` (e.g. `{"trip_id": "00000015"}`).
4. **Server-Side Evaluation:** Evaluation against reference routes occurs strictly on Laptop 1; only high-level evaluation metrics (Precision, Recall, F1) and display coordinates are returned to Laptop 2.

---

## 5. Dependency & Environment Analysis

| Package | Status in Python 3.14 | Role | Installation Action |
|:---|:---:|:---|:---|
| `fastapi` | **MISSING** | Laptop 1 Backend API | Pip install in Prompt 3 (wheel available) |
| `uvicorn` | **MISSING** | ASGI Web Server | Pip install in Prompt 3 (wheel available) |
| `pydantic` | **MISSING** | API Schema Validation | Installed with FastAPI |
| `streamlit` | **MISSING** | Laptop 2 Evaluator UI | Pip install in Prompt 4 (wheel available) |
| `folium` | **AVAILABLE (1.0.0)** | Interactive Map Visualization | Ready |
| `networkx` | **AVAILABLE (3.6.1)** | Road Network Graph & Shortest Paths | Ready |
| `scikit-learn` | **AVAILABLE (1.9.0)** | Machine Learning Estimator | Ready |
| `optuna` | **AVAILABLE (5.1.0)** | Hyperparameter Optimization | Ready |
| `pytest` | **AVAILABLE (9.2.0)** | Test Suite Execution | Ready |
| `joblib` | **AVAILABLE (1.5.3)** | Model Serialization | Ready |
| `pandas` | **AVAILABLE (3.0.5)** | DataFrames & Feature Matrices | Ready |
| `numpy` | **AVAILABLE (2.5.1)** | Numerical Computing | Ready |
| `scipy` | **AVAILABLE (1.18.0)** | Spatial & Probability Distributions | Ready |
| `requests` | **AVAILABLE (2.34.2)** | HTTP LAN Client for Laptop 2 | Ready |

---

## 6. Critical Blockers & Action Items

1. **Blocker 1: Missing HMM Decoding Pipeline (Prompt 2 Priority):**
   - Without Viterbi decoding, route reconstruction falls back to greedy Top-1 selection, which produces disconnected arcs. Member 3's HMM engine must be built next.
2. **Blocker 2: Unextracted Evaluator Test Dataset:**
   - `evaluator_test.zip` must be extracted into `data/raw/evaluator_test/` on Laptop 1 during backend integration.
3. **Blocker 3: Missing Web Packages (`fastapi`, `uvicorn`, `streamlit`):**
   - Must be installed prior to running backend and frontend services.
4. **Blocker 4: Cross-Laptop Network Accessibility:**
   - Laptop 1 must bind to `0.0.0.0:8000` and Windows Defender Firewall must permit incoming TCP connections on port 8000.

---

## 7. Concrete Implementation Plan for Remaining Prompts

```
┌────────────────────────────────────────────────────────────────────────┐
│ PROMPT 2: Member 3 – HMM Decoding & Route Reconstruction              │
│ - Implement src/decoding/emission.py                                   │
│ - Implement src/decoding/transition.py                                 │
│ - Implement src/decoding/viterbi.py                                    │
│ - Implement src/decoding/reconstruction.py                             │
│ - Implement src/evaluation/route_metrics.py                            │
│ - Unit tests in tests/test_decoding.py & tests/test_route_metrics.py    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PROMPT 3: Member 4 – Laptop 1 FastAPI Backend Server                   │
│ - Install fastapi, uvicorn, pydantic                                   │
│ - Extract evaluator_test.zip to data/raw/evaluator_test/               │
│ - Implement api/schemas.py, api/routes.py, api/main.py                 │
│ - Implement full orchestration: Trip -> Features -> ML -> HMM -> Eval  │
│ - Integration tests in tests/test_api.py                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PROMPT 4: Member 4 – Laptop 2 Evaluator UI & LAN Communication         │
│ - Install streamlit                                                    │
│ - Implement ui/app.py with Folium map, trip selector, LAN API client   │
│ - Implement standalone mock mode for offline demonstration resilience │
│ - Verify 2-laptop LAN communication protocol                           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PROMPT 5: Full Integration, End-to-End QA & Hackathon Demo Prep        │
│ - End-to-end evaluation on 10 evaluator-test trips                     │
│ - Standalone launcher scripts: run_server.bat, run_client.bat          │
│ - Networking & troubleshooting runbook                                 │
│ - Final demo presentation script and presentation artifacts            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Files to Create in Subsequent Prompts

| Target File | Target Prompt | Responsibility |
|:---|:---:|:---|
| `src/decoding/emission.py` | Prompt 2 | Calibrated emission likelihoods with spatial miss fallbacks |
| `src/decoding/transition.py` | Prompt 2 | Graph shortest-path transition probability matrix |
| `src/decoding/viterbi.py` | Prompt 2 | Log-domain dynamic programming Viterbi decoder |
| `src/decoding/reconstruction.py` | Prompt 2 | Route sequence builder & topological continuity checker |
| `src/evaluation/route_metrics.py` | Prompt 2 | Sequence match, Arc Precision/Recall/F1, Edit distance |
| `tests/test_decoding.py` | Prompt 2 | Unit tests for emission, transition, Viterbi, reconstruction |
| `tests/test_route_metrics.py` | Prompt 2 | Unit tests for route evaluation metrics |
| `api/schemas.py` | Prompt 3 | Pydantic request and response models |
| `api/routes.py` | Prompt 3 | Endpoints: `/health`, `/trips`, `/reconstruct`, `/map` |
| `api/main.py` | Prompt 3 | FastAPI application instance, CORS, lifespan startup |
| `tests/test_api.py` | Prompt 3 | FastAPI integration test suite using TestClient |
| `ui/app.py` | Prompt 4 | Streamlit dashboard for Laptop 2 |
| `run_server.bat` | Prompt 5 | One-click launcher for Laptop 1 backend |
| `run_client.bat` | Prompt 5 | One-click launcher for Laptop 2 frontend |
| `docs/two_laptop_deployment_guide.md`| Prompt 5 | Networking, IP configuration, and presentation guide |
