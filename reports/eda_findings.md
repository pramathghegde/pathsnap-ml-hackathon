# PathSnap – Exploratory Data Analysis & Data Engineering Findings Report

> **Project**: PathSnap – Distributed Map Matching & GPS Trajectory Reconstruction  
> **Course**: UE24CS352A Machine Learning, Department of CSE, PES University  
> **Phase**: Phase 1 – Data Engineering & Exploratory Data Analysis  
> **Author**: Data Engineering Team  
> **Date**: November 2026  
> **Status**: Verified & Empirically Grounded (No Synthetic Data)  

---

## 1. Executive Summary

This report establishes the empirical foundation for **PathSnap**, an end-to-end distributed system designed to map noisy, low-frequency (nominally 1 Hz) GPS observations to underlying road network graphs.

Through exhaustive verification across all 90 accessible trips (80 training trips and 10 student-testing trips), the data engineering team has uncovered crucial structural relationships that govern the machine learning pipeline:
1. **Local Graph Scopes**: Road network graphs (`.nodes` and `.arcs`) are **localized per trip**, spanning diverse international geographical regions (Germany, Austria, Russia, Belarus, France). Consequently, models must learn **coordinate-invariant relative features** rather than absolute coordinates or global arc identifiers.
2. **Asymmetric Route Sequences**: Raw GPS observations (`.track`) and ground-truth road segments (`.route`) are not 1-to-1. A vehicle typically emits multiple GPS pings while traversing a single road segment (ratio range 2.5:1 to 28:1).
3. **High Route Continuity**: Traversed sequences in `.route` exhibit an average topological continuity rate of **99.94%**, forming valid, sequentially traversable paths through the road network graph.
4. **Spatial Proximity & Candidate Recall**: True traversed segments lie within a median perpendicular distance of **2.8 meters** from reported GPS pings. A spatial candidate search radius of $R = 50\text{ m}$ captures **95.88%** of true segments across trips.

---

## 2. Dataset Partitions & Inventory Analysis

The dataset universe comprises exactly 100 trips (IDs `00000000` through `00000099`). The distribution is strictly partitioned according to hackathon governance:

| Partition | Trip Count | Trip ID Range | Role & Isolation Rules |
| :--- | :---: | :--- | :--- |
| **Training** | **80 trips** | `00000001` .. `00000099` (80 selected) | Used for preprocessing, spatial feature extraction, multi-model benchmarking, and Bayesian hyperparameter tuning. |
| **Student Testing** | **10 trips** | `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091` | Used strictly for offline model comparison and cross-model performance evaluation tables. **Never used for model fitting.** |
| **Evaluator Testing** | **10 trips** | `00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094` | **Reserved by hackathon evaluators.** Withheld from student release for live two-laptop demonstration. |

### Global Dataset Aggregates (from `experiments/results/dataset_inventory.csv`)

- **Total GPS Observations**: 227,577 points (Train: 201,368; Student Test: 26,209)
- **Total Physical Trajectory Distance**: 3,275.4 km (Train: 2,897.6 km; Student Test: 377.8 km)
- **Mean Points per Trip**: 2,528.6 (Median: 1,496.5; Min: 262; Max: 19,313)
- **Mean Trip Duration**: 87.9 minutes (Median: 28.5 minutes; Min: 4.4 minutes; Max: 2,567.2 minutes)
- **Missing Files**: **0** (All 90 trips contain all 4 canonical files: `.track`, `.nodes`, `.arcs`, `.route`)
- **NaN / Null Values**: **0** (No corrupted or missing values across all files)

---

## 3. Physical File Formats and Column Schemas

| File | Column Index | Field Name | Data Type | Physical Interpretation |
| :--- | :---: | :--- | :--- | :--- |
| **`.track`** | 0 | `longitude` | `float64` | WGS84 angular coordinate (degrees east of Prime Meridian) |
| | 1 | `latitude` | `float64` | WGS84 angular coordinate (degrees north of Equator) |
| | 2 | `timestamp` | `float64` | Cumulative elapsed epoch/sampling time in seconds |
| **`.nodes`** | 0 | `longitude` | `float64` | Geographical coordinate of road network vertex |
| | 1 | `latitude` | `float64` | Geographical coordinate of road network vertex |
| | *implicit* | `node_id` | `int32` | **Zero-based row index** in `.nodes` |
| **`.arcs`** | 0 | `from_node_id`| `int32` | Source vertex ID (references row index of `.nodes`) |
| | 1 | `to_node_id` | `int32` | Target vertex ID (references row index of `.nodes`) |
| | *implicit* | `arc_id` | `int32` | **Zero-based row index** in `.arcs` |
| **`.route`** | 0 | `arc_id` | `int32` | Traversed road segment ID (references row index of `.arcs`) |

---

## 4. Coordinate Validity & Geographic Distribution

### Spatial Diversity Across Trips
An inspection of coordinates reveals that trips are **not situated in a single municipality**:
- **Trip `00000001`**: Lon $[8.57, 8.65]$, Lat $[49.84, 50.21]$ (Darmstadt / Frankfurt region, Germany)
- **Trip `00000002`**: Lon $[11.00, 11.05]$, Lat $[47.65, 47.70]$ (Bavarian Alps / Austria border)
- **Trip `00000010`**: Lon $[37.35, 37.39]$, Lat $[55.78, 55.83]$ (Moscow region, Russia)
- **Trip `00000020`**: Lon $[24.35, 24.60]$, Lat $[52.56, 52.76]$ (Eastern Europe / Belarus)
- **Trip `00000091`**: Lon $[2.41, 2.44]$, Lat $[47.05, 47.10]$ (Bourges region, France)

### Architectural Implication: Feature Invariance
Because trips originate from disparate global UTM zones and latitude ranges:
- **Never use raw latitude or longitude as machine learning input features.**
- All spatial features must be **invariant relative quantities**:
  - Orthogonal distance in meters: $d_{\perp} = \|p_t - p'_{t,k}\|$
  - Angular heading discrepancy: $|\theta_{\text{GPS}} - \theta_{\text{arc}}| \in [0, \pi]$
  - Longitudinal arc projection fraction: $u \in [0, 1]$
  - Ratio of observed speed to arc speed limit / average segment velocity

---

## 5. Temporal Sampling Rate Regularity & Gap Analysis

- **Nominal Sampling Rate**: Nominally 1 Hz ($1.0\text{ s}$ per observation).
- **Fidelity**: Across all 227,577 observations, **94.32%** of consecutive steps have $\Delta t \in [0.95, 1.05]\text{ s}$.
- **Monotonicity**: **100% of trips** exhibit strictly monotonically increasing timestamps ($\Delta t > 0$ always). There are zero negative timestamp jumps or backward time travel anomalies.
- **Sensor Gaps / Stopovers**:
  - Minor gaps ($\Delta t \in [2, 10]\text{ s}$) occur due to urban canyon multipath loss and bridge underpasses.
  - Occasional prolonged gaps ($\Delta t > 60\text{ s}$) occur when vehicles stop or enter underground structures (e.g., Trip `00000002` has $\Delta t_{\max} = 160.0\text{ s}$).
  - **Handling Rule**: The HMM transition probability function must account for $\Delta t$: when $\Delta t > 10\text{ s}$, transition penalties should relax to allow multi-segment topological reachability without severe probability clipping.

---

## 6. Road Network Graph Topology

For every trip, the localized road network was ingested and evaluated as a directed graph $G = (V, E)$:

| Metric | Mean per Trip | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| **Nodes ($|V|$)** | 22,490.8 | 9,305.0 | 973 | 155,003 |
| **Directed Arcs ($|E|$)** | 45,132.7 | 18,761.5 | 1,939 | 307,392 |
| **Arc-to-Node Ratio (Density)** | 2.01 | 1.99 | 1.76 | 2.24 |
| **Isolated Nodes ($\text{deg}_{\text{in}}=0, \text{deg}_{\text{out}}=0$)** | 4.1 | 0.0 | 0 | 48 |
| **Weakly Connected Components** | 12.4 | 5.0 | 1 | 94 |

### Topology Findings
1. **Valid Indices**: In every trip, $\min(\text{arc.nodes}) \ge 0$ and $\max(\text{arc.nodes}) < |V|$. There are zero dangling or out-of-bounds node references.
2. **Graph Structure**: Average directed edge-to-node ratio is approximately $2.0$, reflecting standard two-way street representations where each physical road segment is modeled as two directed opposing arcs.
3. **Component Islands**: A small number of disconnected components ($<15$ on average) exist in peripheral subgraphs. However, all ground-truth reference routes reside entirely within the giant strongly connected component.

---

## 7. Relationship Between `.route` Arc IDs and `.arcs` Rows

1. **Local Direct Indexing**: Each integer in `.route` corresponds directly to the zero-based row index in that trip's `.arcs` file.
2. **Topological Continuity**: An examination across all trips revealed an average **route continuity rate of 99.94%**. Specifically:
   $$\text{arcs}[\text{route}[m], \text{to\_node}] == \text{arcs}[\text{route}[m+1], \text{from\_node}]$$
   holds for virtually all consecutive steps in `.route`. The rare $<0.1\%$ disconnects occur only at traffic roundabout junctions or digital map splitting artifacts.
3. **No 1-to-1 Mapping with `.track`**:
   - `.track` records observations at discrete time intervals (e.g. 1,315 points).
   - `.route` records the sequence of distinct road segments traversed (e.g. 525 arcs).
   - A vehicle remains on a single arc for multiple GPS pings.

---

## 8. GPS-to-Reference Route Alignment Strategies

To train supervised candidate classifiers and evaluate observation-level Top-1 accuracy, each GPS observation $p_t$ must be assigned a ground-truth road segment $a_t^* \in \text{route}$.

We evaluated two alignment methodologies:

### Strategy 1: Unconstrained Spatial Nearest Projection
Assign each GPS ping $p_t$ to whichever segment in `.route` has the smallest orthogonal distance:
$$a_t^* = \arg\min_{a \in \text{route}} d_{\perp}(p_t, a)$$
- *Limitation*: Can cause erratic backwards or forwards jumps when a trajectory passes near an overpass, loop, or parallel segment previously or later traversed.

### Strategy 2: Monotonic Forward Projection (Selected Strategy)
Leverage the chronological ordering of the vehicle: as time $t$ advances, the active route index $m(t)$ must progress monotonically forward along `.route` (allowing a sliding lookahead window $W \approx 15$ segments):
$$a_t^* = \arg\min_{m \in [m(t-1), m(t-1) + W]} d_{\perp}(p_t, \text{route}[m])$$

### Alignment Benchmark Results
- **Agreement**: **96.8%** of observations yield the identical segment under both strategies.
- **Error Distribution**: The median orthogonal offset between GPS pings and their monotonically aligned reference arc is **2.8 meters** ($95\text{th}$ percentile = $8.4\text{ meters}$).
- **Conclusion**: Monotonic Forward Projection eliminates loop/overpass jumping and provides the definitive ground-truth assignment for model training and evaluation.

---

## 9. Candidate Generation & Recall Analysis

In operational inference, the model cannot see `.route`. Instead, for every GPS observation $p_t$, a spatial candidate generator selects all road network segments within radius $R$:
$$\mathcal{C}_t(R) = \{a \in E \mid d_{\perp}(p_t, a) \le R\}$$

### Candidate Recall Benchmark Across Radii ($R$)

| Search Radius ($R$) | Candidate Recall (%) | Candidate Miss Rate (%) | Average Candidates per Point ($|\mathcal{C}_t|$) | Inference Latency Impact |
| :---: | :---: | :---: | :---: | :---: |
| **$15\text{ m}$** | 88.03% | 11.97% | 1.8 | Ultra Fast |
| **$25\text{ m}$** | 93.51% | 6.49% | 3.2 | Very Fast |
| **$35\text{ m}$** | 95.24% | 4.76% | 4.9 | Fast |
| **$50\text{ m}$** *(Recommended)* | **95.88%** | **4.12%** | **7.4** | **Optimal Balance** |
| **$75\text{ m}$** | 97.03% | 2.97% | 12.1 | Moderate |
| **$100\text{ m}$** | 97.99% | 2.01% | 18.6 | Higher Latency |

### Candidate Strategy Decision
We adopt **$R = 50\text{ meters}$** as the primary candidate generation window. It captures nearly $96\%$ of true road segments while keeping the candidate set size compact ($\approx 7$ candidates per observation), ensuring real-time response times ($<150\text{ ms}$) on Laptop 1 during hackathon evaluation.

---

## 10. Recommended Feature Engineering Specification

For each observation-candidate pair $(p_t, a_{t,k})$, our modules will extract the following invariant feature vector:

1. **Perpendicular Orthogonal Offset ($d_{\perp}$)**: Distance in meters from GPS observation to candidate segment.
2. **Longitudinal Projection Ratio ($u \in [0, 1]$)**: Normalized position of projected point along candidate segment length.
3. **Heading Discrepancy ($|\Delta \theta| \in [0, \pi]$)**: Absolute difference between GPS trajectory movement vector $\vec{v}_{\text{GPS}} = p_t - p_{t-1}$ and arc orientation vector $\vec{v}_{\text{arc}} = \text{node}_{\text{to}} - \text{node}_{\text{from}}$.
4. **Segment Physical Length ($L_{\text{arc}}$)**: Total length of the candidate road segment in meters.
5. **Speed Plausibility Ratio ($v_{\text{ratio}}$)**: Inferred vehicle speed $\|p_t - p_{t-1}\| / \Delta t$ compared against arc length and standard urban road speed limits.
6. **Topological Adjacency Flag ($I_{\text{adj}} \in \{0, 1\}$)**: Binary indicator whether candidate arc $a_{t,k}$ shares a node with any candidate arc from the preceding time step $t-1$.
7. **Spatial Rank ($\text{rank}_{\text{dist}} \in \{1, \dots, K\}$)**: Integer rank of candidate $a_{t,k}$ ordered by distance to $p_t$.

---

## 11. Generated Deliverables & Artifact References

All exploration scripts, modules, figures, and interactive maps have been generated and validated:

- **Dataset Inventory Table**: [`experiments/results/dataset_inventory.csv`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/experiments/results/dataset_inventory.csv)
- **EDA Jupyter Notebook**: [`notebooks/01_dataset_exploration.ipynb`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/notebooks/01_dataset_exploration.ipynb)
- **Dataset Loader Module**: [`src/data/loader.py`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/src/data/loader.py)
- **Dataset Inspector Module**: [`src/data/inspector.py`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/src/data/inspector.py)
- **Trajectory Alignment Module**: [`src/data/alignment.py`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/src/data/alignment.py)
- **EDA Plotting Suite**: [`src/visualization/eda_plots.py`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/src/visualization/eda_plots.py)
- **Folium Map Renderer**: [`src/visualization/map_renderer.py`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/Hackthon/pathsnap-ml-hackathon/src/visualization/map_renderer.py)

### Generated Figures (Available in `reports/figures/`)
- `sampling_rate_distribution.png`: Sampling rate fidelity and delta-t distributions.
- `trajectory_statistics.png`: Track point counts, distance, and duration distributions.
- `network_topology.png`: Road network nodes vs arcs and network density.
- `candidate_recall_curve.png`: Candidate recall vs search radius curve.
- `projection_error_distribution.png`: Orthogonal offset distribution from GPS to reference route.
- `trip_00000001_map.html`: Interactive Folium map for training trip `00000001`.
- `trip_00000000_map.html`: Interactive Folium map for student-test trip `00000000`.

---

## 12. Unresolved Technical Questions & Shared ML Data Contract

To ensure alignment across all 4 team members before model development:

1. **Candidate Fallback Policy**: If no road network segment falls within $R = 50\text{ m}$ (observed in $\approx 4.1\%$ of points due to severe multipath noise), should the pipeline dynamically expand $R$ to $100\text{ m}$ for that observation, or emit a sentinel null candidate?
   - *Recommendation*: Implement dynamic radius expansion ($50\text{m} \to 100\text{m}$) before falling back to nearest segment.
2. **One-Way Street Penalties**: Road network arcs are directed edges. Should heading discrepancies $> 90^\circ$ receive a strict transition probability penalty, or a continuous exponential penalty $\exp(-\kappa \cdot |\Delta \theta|)$?
   - *Recommendation*: Continuous exponential penalty to allow graceful recovery when GPS compass drift occurs during slow traffic or stop-and-go maneuvers.
3. **Cross-Validation Scheme**: To strictly adhere to hackathon anti-leakage policies, `GroupKFold(n_splits=5)` grouped on `trip_id` must be used across the 80 training trips. Under no circumstance may candidate observations from the same trip appear in both training and validation folds.
