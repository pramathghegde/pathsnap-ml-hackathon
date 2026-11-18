# PathSnap – Distributed Map Matching & GPS Trajectory Reconstruction

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Project Status](https://img.shields.io/badge/Status-Initial%20Setup%20%2F%20Scaffold-yellow.svg)](#project-status)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/PES%20University-UE24CS352A%20ML%20Hackathon-red.svg)](https://www.pes.edu/)

> **PathSnap** is an end-to-end distributed machine learning and sequential inference system engineered to reconstruct accurate, physically plausible vehicle trajectories by matching noisy, low-frequency urban GPS measurements to complex road network graphs.

---

## 1. Project Overview

Global Positioning System (GPS) telemetry collected in dense urban corridors suffers from significant spatial inaccuracies caused by multipath signal reflections, satellite occlusion in urban canyons, atmospheric distortion, clock drifts, and low sampling frequencies (nominally 1 Hz). As a consequence, raw coordinate points frequently deviate from actual roadways—appearing to drift across adjacent city blocks, intersect buildings, or jump to parallel streets.

**PathSnap** tackles this fundamental challenge through a hybrid pipeline combining **probabilistic spatial feature engineering**, **multi-class machine learning classification** across diverse algorithm families, **Bayesian hyperparameter optimization**, and **Hidden Markov Model (HMM) sequence decoding via the Viterbi algorithm**. The complete system is deployed across a distributed **two-laptop physical architecture**, separating the compute and evaluation engine (FastAPI backend on Laptop 1) from the interactive evaluator dashboard (Folium geographical visualization on Laptop 2).

---

## 2. Problem Statement

Given:
1. A noisy sequence of temporal GPS observations $T = \{(\text{lon}_t, \text{lat}_t, \tau_t)\}_{t=1}^N$ sampled at approximately 1 Hz.
2. A geographical road network represented as a directed graph $G = (V, E)$, where vertices $V$ denote intersections/nodes and directed edges $E$ denote traversable road segments (arcs).

The objective is to reconstruct the true chronological sequence of traversed road segments $R = \{a_1, a_2, \dots, a_N\}$ where $a_t \in E$, while satisfying:
- **Spatial Proximity**: Predicted segments must align closely with observed GPS coordinates.
- **Topological Continuity**: Consecutive predicted segments $(a_{t-1}, a_t)$ must form valid, physically traversable paths within the road network without impossible jumps.
- **Directional Plausibility**: Predicted segments must honor vehicle heading and one-way constraints.
- **Robustness to Gaps**: The system must gracefully interpolate missing candidates or temporal sensor dropouts.

---

## 3. Project Objectives

1. **Robust Preprocessing & Spatial Indexing**: Implement geodesic coordinate projections and spatial data structures (e.g., KD-Tree / R-Tree) for candidate arc generation within spatial search radii.
2. **Multi-Model Machine Learning Benchmark**: Implement, train, and independently evaluate at least **five distinct ML model families** on the 80 training trips using group-aware cross-validation.
3. **Bayesian Hyperparameter Optimization**: Optimize the primary classification model using Bayesian search strategies (e.g., Tree-structured Parzen Estimator via Optuna) on the 80 training trips without data leakage.
4. **Topological Route Reconstruction via HMM & Viterbi**: Couple candidate classifier emission probabilities with road-network shortest path transition probabilities to enforce continuous route topology.
5. **Distributed Two-Laptop Deployment**: Deploy an enterprise-grade FastAPI service on Laptop 1 and an evaluator-facing UI on Laptop 2, communicating over local network HTTP.
6. **Geographical Folium Visualizations**: Generate interactive, multi-layer folium maps visualizing raw GPS observations, road network geometry, reference routes, and predicted paths.
7. **Strict Leakage Prevention**: Guarantee absolute isolation of evaluator-testing ground-truth data, ensuring that predictions and evaluations remain verifiable and uncompromised.

---

## 4. Proposed System Architecture

PathSnap is designed with clean architectural boundaries separating data parsing, feature extraction, ML inference, sequence decoding, network serving, and presentation.

### Physical Distributed Deployment

```
+-----------------------------------------------------------------------------------------+
|                                 PHYSICAL DEPLOYMENT TOPOLOGY                            |
+-----------------------------------------------------------------------------------------+

        LAPTOP 2 (Evaluator Console)                         LAPTOP 1 (Model & API Host)
  +-------------------------------------+             +-------------------------------------+
  |                                     |             |                                     |
  |  +-------------------------------+  |             |  +-------------------------------+  |
  |  |    Interactive Evaluator UI   |  |             |  |      FastAPI ASGI Server      |  |
  |  |     (Streamlit Web Console)   |  |             |  |     (Uvicorn on Port 8000)    |  |
  |  +---------------+---------------+  |             |  +---------------+---------------+  |
  |                  |                  |             |                  |                  |
  |                  v                  |             |                  v                  |
  |  +-------------------------------+  |             |  +-------------------------------+  |
  |  |     10-Trip Selector Menu     |  |             |  |    Inference Engine Controller |  |
  |  |   (Student & Evaluator Sets)  |  |             |  |    - Candidate Generator      |  |
  |  +---------------+---------------+  |             |  |    - ML Feature Extractor     |  |
  |                  |                  |             |  |    - Tuned Model Predictor    |  |
  |                  | HTTP POST        |             |  |    - HMM Viterbi Decoder      |  |
  |                  | /api/v1/predict  |             |  +---------------+---------------+  |
  |                  +------------------------------------------>|       |                  |
  |                                     |    LAN      |          v       v                  |
  |                                     |  (TCP/IP)   |  +-------------------------------+  |
  |                  +<----------------------------------+  Server-Side Evaluation Engine |  |
  |                  | JSON Response:   |             |  - Internal Ground-Truth Store   |  |
  |                  | - Predicted Arcs |             |  - Top-1 Accuracy & F1-Score     |  |
  |                  | - Metric Summary |             |  - Continuity Ratio              |  |
  |                  v                  |             |  +-------------------------------+  |
  |  +-------------------------------+  |             |                                     |
  |  |  Interactive Folium Renderer  |  |             |  +-------------------------------+  |
  |  |  - Raw GPS (Blue markers)     |  |             |  |     Artifact & Model Store    |  |
  |  |  - Road Network (Gray lines)  |  |             |  |   - Serialized Classifiers    |  |
  |  |  - Predicted Route (Green)    |  |             |  |   - Road Network Graph Cache  |  |
  |  +-------------------------------+  |             |  +-------------------------------+  |
  +-------------------------------------+             +-------------------------------------+
```

### Logical Data Flow Pipeline

```
Raw GPS (.track) + Road Network (.nodes, .arcs)
                     |
                     v
   [ 1. Spatial Preprocessing & Indexing ]
        - WGS84 Geodesic / UTM Projection
        - KDTree / RTree Candidate Search Window (R ~ 50m)
                     |
                     v
   [ 2. Geometric & Topological Feature Engineering ]
        - Perpendicular distance to arc
        - Absolute heading difference (|θ_gps - θ_arc|)
        - Longitudinal projection ratio (u ∈ [0, 1])
        - Inferred velocity vs expected speed profile
                     |
                     v
   [ 3. Machine Learning Candidate Classifier ]
        - Benchmarked across 5 algorithm families
        - Bayesian hyperparameter optimization (Optuna)
        - Output: Calibrated probabilities P(arc_i | point_t)
                     |
                     v
   [ 4. Hidden Markov Model & Viterbi Decoder ]
        - Hidden States: Spatial arc candidates
        - Emission Probability: Classifier confidence + spatial gaussian
        - Transition Probability: Shortest path network distance vs euclidean delta
        - Viterbi Dynamic Programming: Optimal global sequence
                     |
                     v
   [ 5. Serving & Visualization ]
        - FastAPI JSON endpoint (Latency < 200ms)
        - Evaluator UI with Folium multi-layer map rendering
```

---

## 5. Planned Technology Stack

The project technology stack is designed around modularity, reproducibility, and high-performance computing:

| Category | Component / Tool | Planned Role | Status |
| :--- | :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language environment | Active |
| **Scientific Computing** | `numpy`, `scipy`, `pandas` | Vectorized calculations, numerical optimization | Planned |
| **Spatial & Graph Computing** | `shapely`, `pyproj`, `networkx`, `rtree` | Coordinate transformation, spatial indexing, shortest path calculations | Planned |
| **Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` | Multi-model candidate classification (5 distinct model families) | Planned |
| **Hyperparameter Tuning** | `optuna` | Bayesian optimization with Tree-structured Parzen Estimators | Planned |
| **API Backend** | `fastapi`, `uvicorn`, `pydantic` | High-performance REST API with automated OpenAPI docs | Planned |
| **User Interface** | `streamlit` | Evaluator-facing trip selection and metrics dashboard | Planned |
| **Geospatial Visualization**| `folium` | Leaflet-based interactive HTML maps with layer controls | Planned |
| **Testing & Quality** | `pytest`, `black`, `ruff` | Automated unit testing, PEP 8 formatting, static analysis | Scaffolded |

> *Note: Specific dependency versions will be finalized and locked in `requirements.txt` during the initial implementation phase to guarantee cross-platform determinism.*

---

## 6. Dataset Description

The dataset represents realistic urban vehicle tracking data partitioned by individual trips. Each trip is encapsulated within an isolated directory containing four canonical raw files:

### Raw File Specifications

| File Extension | Content Description | Format & Schema | Key Considerations |
| :--- | :--- | :--- | :--- |
| **`.track`** | Raw GPS trajectory points | 3 whitespace-separated columns:<br>`<longitude> <latitude> <timestamp_seconds>` | No headers. Nominally 1 Hz sampling rate (~97% of intervals are exactly 1 s). Angular coordinates requiring geodesic projection. |
| **`.nodes`** | Geographical intersections & vertices | 2 whitespace-separated columns:<br>`<longitude> <latitude>` | No headers. The zero-based row index defines the implicit `node_id`. |
| **`.arcs`** | Directed road segments connecting nodes | 2 integer columns:<br>`<from_node_id> <to_node_id>` | No headers. The zero-based row index serves as the unique `arc_id`. Represents directed graph edges. |
| **`.route`** | Ground-truth reference route | 1 integer column:<br>`<arc_id>` | Chronological sequence of ground-truth `arc_id` values traversed. Supplied for training and student testing only. |

### Dataset Partitions and Isolation Rules

| Partition | Trip Count | Purpose | Access & Leakage Constraints |
| :--- | :---: | :--- | :--- |
| **Training** | **80 trips** | Feature engineering, cross-validation, model training, Bayesian hyperparameter search. | Full access. Group-aware $K$-Fold cross-validation (`GroupKFold` by `trip_id`) required. |
| **Student Testing** | **10 trips** | Development-stage model comparison, baseline tuning, sanity checking. | Used for offline validation and comparative evaluation tables. Never used for model fitting. |
| **Evaluator Testing**| **10 trips** | Final hackathon demonstration and live evaluation. | **Strictly confidential server-side**. Ground-truth `.route` files are never exposed to models or UI. |

---

## 7. Planned ML Experimentation Workflow

To satisfy the hackathon requirements and ensure scientific rigor, our team will execute the following structured workflow:

```
[ Phase 1: EDA & Baseline ]
  ├── Inspect distribution of trajectory lengths, velocities, and sampling intervals
  ├── Analyze coordinate bounds, missing values, and disconnected road components
  └── Build Folium exploratory maps of sample trajectories

[ Phase 2: Feature Engineering & Candidate Search ]
  ├── Implement KD-Tree spatial candidate arc generator (search radius R)
  ├── Extract geometric features: perpendicular offset, heading discrepancy, arc length ratio
  └── Construct graph connectivity features between sequential candidate sets

[ Phase 3: Multi-Model Benchmark (5 Families) ]
  ├── 1. Linear/Probabilistic: Logistic Regression / Gaussian Naive Bayes
  ├── 2. Instance-Based: Distance-Weighted K-Nearest Neighbors (KNN)
  ├── 3. Tree-Based: Random Forest / Extra Trees Classifier
  ├── 4. Gradient Boosting: LightGBM / XGBoost
  ├── 5. Neural Network: Multi-Layer Perceptron (MLP)
  └── Evaluate: CV Macro F1, Top-1 Accuracy, Log Loss, Inference Latency

[ Phase 4: Bayesian Hyperparameter Optimization ]
  ├── Select top-performing model family from Phase 3
  ├── Define hyperparameter search spaces (learning rate, tree depth, regularization, etc.)
  ├── Execute Optuna Bayesian search using GroupKFold on 80 training trips
  └── Refit best configuration and compare tuned vs untuned metrics

[ Phase 5: HMM & Viterbi Sequence Decoding ]
  ├── Construct emission probability matrix from tuned classifier probabilities
  ├── Build transition matrix using road-network shortest path graph distance vs geodesic delta
  ├── Implement Viterbi dynamic programming recurrence with gap handling
  └── Benchmark classifier-only vs classifier + HMM route continuity

[ Phase 6: Distributed Integration & UI ]
  ├── Package trained model and graph topology artifacts into FastAPI service (Laptop 1)
  ├── Build evaluator Streamlit interface with Folium map overlays (Laptop 2)
  └── Conduct end-to-end latency and network resilience testing
```

---

## 8. Team Responsibilities & Work Breakdown

The project tasks are distributed across our four team members according to functional ownership:

| Team Member | Hackathon Role | Primary Responsibilities |
| :--- | :--- | :--- |
| **Team Member 1**<br>`[Placeholder: Name & SRN]` | **ML Lead & Bayesian Optimization** | - Lead selection, training, and benchmarking of the 5 model families.<br>- Implement Bayesian hyperparameter optimization using Optuna.<br>- Track experiment logs, generate training loss curves, and evaluate overfitting/underfitting.<br>- Author Section 3 & 4 of the technical report. |
| **Team Member 2**<br>`[Placeholder: Name & SRN]` | **Data Engineering & Spatial Features** | - Ingest and validate `.track`, `.nodes`, and `.arcs` datasets.<br>- Implement geodesic / UTM coordinate projections and spatial KD-Tree candidate search.<br>- Engineer geometric and temporal candidate features.<br>- Prevent data leakage across trip partitions.<br>- Author Section 2 of the technical report. |
| **Team Member 3**<br>`[Placeholder: Name & SRN]` | **HMM Decoding & Evaluation Metrics** | - Formulate Hidden Markov Model state representations.<br>- Model emission probabilities and graph shortest path transition matrices.<br>- Implement the Viterbi dynamic programming algorithm with gap interpolation.<br>- Compute Top-1 accuracy, Macro F1-score, and route continuity metrics.<br>- Author Section 5 of the technical report. |
| **Team Member 4**<br>`[Placeholder: Name & SRN]` | **Distributed Serving, UI & Folium Visuals** | - Architect and implement the FastAPI REST backend on Laptop 1.<br>- Build the evaluator-facing Streamlit user interface on Laptop 2.<br>- Develop multi-layer Folium geographical maps (GPS points, road network, routes).<br>- Manage two-laptop network deployment and latency optimization.<br>- Author Section 6 & 7 of the technical report. |

---

## 9. Repository Structure

```
pathsnap-ml-hackathon/
├── .env.example               # Template environment configuration
├── .gitignore                  # Comprehensive git exclusion rules
├── pyproject.toml              # Build system, project metadata & tool configurations
├── requirements.txt            # Scaffolding placeholder for dependencies
├── CONTRIBUTING.md             # Branching, commit conventions & PR guidelines
├── README.md                   # Project documentation & master specification
│
├── api/                        # FastAPI backend application (Laptop 1)
│   └── __init__.py             # API package definition
│
├── artifacts/                  # Serialized models, graphs, and scalers (Git-ignored)
│   └── .gitkeep
│
├── configs/                    # Experiment and pipeline configuration YAMLs
│   └── .gitkeep
│
├── data/                       # Dataset directories (Git-ignored; local only)
│   ├── raw/                    # Raw trip packages (80 train, 10 student, 10 eval)
│   │   └── .gitkeep
│   └── processed/              # Extracted feature matrices and cached graphs
│       └── .gitkeep
│
├── docs/                       # Architectural documentation and design specifications
│   └── architecture.md         # Detailed system design & data flow specification
│
├── experiments/                # Experiment tracking and optimization outputs
│   ├── logs/                   # Training and inference logs
│   │   └── .gitkeep
│   ├── optimization/           # Optuna study databases and trial histories
│   │   └── .gitkeep
│   └── results/                # Evaluation metric tables and comparative CSVs
│       └── .gitkeep
│
├── notebooks/                  # Jupyter notebooks for EDA and initial model trials
│   └── .gitkeep
│
├── reports/                    # Hackathon technical reports, figures, and deliverables
│   └── .gitkeep
│
├── scripts/                    # Automation utilities and pipeline entrypoint scripts
│   └── .gitkeep
│
├── src/                        # Core Python library package
│   ├── __init__.py             # Source package root
│   ├── data/                   # Parsers, graph builders, and coordinate converters
│   │   └── __init__.py
│   ├── decoding/               # HMM formulation, transition matrices, Viterbi decoder
│   │   └── __init__.py
│   ├── evaluation/             # Metrics: Top-1 accuracy, F1-score, continuity scorer
│   │   └── __init__.py
│   ├── features/               # Spatial candidate generation and feature extraction
│   │   └── __init__.py
│   ├── models/                 # Classifiers across 5 families and Bayesian tuner
│   │   └── __init__.py
│   └── visualization/          # Folium map generation and plotting scripts
│       └── __init__.py
│
├── tests/                      # Automated unit and integration test suite
│   └── __init__.py             # Test package definition
│
└── ui/                         # Evaluator-facing interface (Laptop 2)
    └── .gitkeep
```

---

## 10. Installation Instructions

> **Note**: Dependency versions are currently pending final environment specification. The instructions below demonstrate the planned environment initialization.

### 1. Clone the Repository
```bash
git clone https://github.com/your-organization/pathsnap-ml-hackathon.git
cd pathsnap-ml-hackathon
```

### 2. Configure Python Virtual Environment
```bash
# Create a dedicated virtual environment (Python 3.10+)
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

### 3. Environment Variables Configuration
```bash
# Copy the template environment configuration
cp .env.example .env

# Populate .env with appropriate local IP addresses and directory paths
```

### 4. Install Dependencies
```bash
# Dependencies installation is pending environment lockdown:
# pip install -r requirements.txt
# pip install -e .
```

---

## 11. Execution Instructions (Planned Workflow)

> **Implementation Status**: Code implementations for inference and UI are currently pending. The commands below reflect the planned operational interface.

### Running the API Host (Laptop 1)
```bash
# Launch the FastAPI service on Laptop 1
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running the Evaluator UI (Laptop 2)
```bash
# Launch the Streamlit evaluator dashboard on Laptop 2
streamlit run ui/app.py --server.port 8501
```

### Running Automated Test Suite
```bash
# Execute unit and integration tests
pytest tests/ -v
```

---

## 12. Project Status & Roadmap

| Milestone | Deliverable | Status |
| :---: | :--- | :---: |
| **Phase 1** | Repository Setup, Scalable Directory Structure, Architecture Docs | 🟢 **Completed** |
| **Phase 2** | Exploratory Data Analysis & Spatial Candidate Generation | 🟡 *Pending* |
| **Phase 3** | Geometric & Topological Feature Engineering Pipeline | 🟡 *Pending* |
| **Phase 4** | Benchmark of 5 Distinct ML Model Families | 🟡 *Pending* |
| **Phase 5** | Bayesian Hyperparameter Optimization via Optuna | 🟡 *Pending* |
| **Phase 6** | HMM Formulation & Viterbi Sequence Decoding | 🟡 *Pending* |
| **Phase 7** | Distributed FastAPI Backend & Evaluator Streamlit UI | 🟡 *Pending* |
| **Phase 8** | Interactive Folium Visualizations & Final Technical Report | 🟡 *Pending* |

---

## 13. Academic Integrity & Ethics

This project strictly adheres to the **PES University Academic Integrity and Plagiarism Policy**. All data preprocessing, feature engineering logic, machine learning models, optimization routines, and sequential decoders are independently designed, implemented, and verified by the team. No synthetic metrics, pre-computed evaluation answers, or external solutions have been incorporated.
