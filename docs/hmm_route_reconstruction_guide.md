# PathSnap HMM Route Reconstruction & Sequence Evaluation Guide

**Module Owner:** Member 3 — Sequential Route Reconstruction & Evaluation  
**Status:** Complete, Tested, and Production-Ready  
**Test Suite:** 64 passed (including 17 decoding & sequence evaluation tests)  
**Dependencies:** NetworkX, NumPy, Pandas, Scikit-Learn, Joblib  

---

## 1. Executive Summary & Mathematical Architecture

PathSnap reconstructs complete, physically continuous road routes from noisy, sparse GPS trajectories. While Member 2's predictive classifier (HistGradientBoosting) accurately classifies individual candidate segments ($80.39\%$ Eligible Top-1), naive observation-level selection (greedy argmax) frequently produces disconnected jumps, reverse traversals, and zigzagging paths.

Member 3 introduces a **Hidden Markov Model (HMM)** sequence decoder that balances:
1. **Emission Likelihood:** How well each candidate road segment matches the local GPS observation and kinematic features.
2. **Transition Likelihood:** Whether traveling between consecutive road segments is physically and topologically consistent with the road network.

```
GPS Observations:        (o_0) ----------> (o_1) ----------> (o_2) ----------> ... ----------> (o_T)
                           |                 |                 |                                  |
                           v                 v                 v                                  v
Candidates (States):   [a_0, a_1, ...]   [a_0, a_1, ...]   [a_0, a_1, ...]                    [a_0, a_1, ...]
                           \                 /                 /
                            \               /                 /
Optimal Viterbi Path:        a_0* --------> a_1* -----------> a_2* ----------> ... -----------> a_T*
                                       |
                                       v
Collapsed & Infilled Route:  [a_0, a_1, intermediate_arcs..., a_T]  (100% Topologically Continuous)
```

---

## 2. Mathematical Formulation

### 2.1 Emission Likelihood Modeling

Member 2's trained classifier outputs posterior probability $p(\text{is\_match}=1 \mid \mathbf{x})$. Under Bayes' rule:
$$p(o_t \mid a_j) = \frac{p(a_j \mid o_t) \cdot p(o_t)}{p(a_j)}$$

Assuming an empirical uniform prior over local spatial candidate segments ($p(a_j) = \text{const}$), the emission likelihood is proportional to the classifier's predicted posterior:
$$p(o_t \mid a_j) \propto p(\text{is\_match}=1 \mid \mathbf{x}_{t, j})$$

To guarantee numerical stability in floating-point operations:
1. **Clipping:** $p_{\text{safe}} = \max(p, \epsilon)$, where $\epsilon = 10^{-7}$.
2. **Observation Normalization:** $\tilde{p}(o_t \mid a_j) = \frac{p_{\text{safe}}(o_t \mid a_j)}{\sum_{k} p_{\text{safe}}(o_t \mid a_k)}$.
3. **Log Transformation:** $\log p(o_t \mid a_j) = \log \tilde{p}(o_t \mid a_j)$.
4. **Spatial Fallback:** For candidate misses or unpredicted segments, a calibrated Gaussian distance likelihood is applied:
   $$\log p_{\text{spatial}}(o_t \mid a_j) = -\frac{1}{2}\log(2\pi \sigma_z^2) - \frac{d_{\text{perp}}^2}{2\sigma_z^2}$$
   where $\sigma_z = 8.0\text{ m}$ represents standard urban GPS measurement error.

### 2.2 Graph-Based Transition Likelihood Modeling

Following the validated formulation of Newson & Krumm (2009), the transition probability between road segment $a_{t-1, i} = (u_i, v_i)$ and $a_{t, j} = (u_j, v_j)$ models the discrepancy between the network shortest-path distance $d_{\text{net}}$ and the empirical GPS displacement $d_{\text{gps}}$:

$$p(a_{t, j} \mid a_{t-1, i}) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{net}}(a_{t-1, i}, a_{t, j}) - d_{\text{gps}}(o_{t-1}, o_t)|}{\beta}\right)$$

In the log domain:
$$\log p(a_{t, j} \mid a_{t-1, i}) = -\log \beta - \frac{|d_{\text{net}} - d_{\text{gps}}|}{\beta}$$

where $\beta = 15.0\text{ m}$ is the empirical distance error decay scale.

#### Topological Cases:
1. **Same-Arc Transition ($a_i == a_j$):**  
   The vehicle remained on the same road arc.
   $$\Delta d = 0.0 \quad \text{if } d_{\text{gps}} \le \text{length}(a_i), \quad \text{else } d_{\text{gps}} - \text{length}(a_i)$$
   This correctly models driving along an extended road arc without penalty.

2. **Directly Connected Arcs ($v_i == u_j$):**  
   Arc $a_j$ begins at the exact terminus of arc $a_i$.  
   $d_{\text{net}} = \text{length}(a_j)$, and $\Delta d = |d_{\text{net}} - d_{\text{gps}}|$. Zero graph hops needed.

3. **Multi-Hop Connected Arcs ($v_i \ne u_j$):**  
   Calculates shortest path length $d_{\text{path}}(v_i, u_j)$ in `nx.DiGraph` using edge attribute `length_m`:
   $$d_{\text{net}} = d_{\text{path}}(v_i, u_j) + \text{length}(a_j)$$
   $$\Delta d = |d_{\text{net}} - d_{\text{gps}}|$$

4. **Disconnected Arcs (Unreachable):**  
   If no directed graph path connects $v_i$ to $u_j$ within search cutoff:
   $$\log p(a_{t, j} \mid a_{t-1, i}) = \text{disconnect\_penalty} \quad (\text{default: } -100.0)$$

### 2.3 Log-Domain Viterbi Trellis Dynamic Programming

The MAP sequence of road arcs $A^* = (a_0^*, a_1^*, \dots, a_{T-1}^*)$ maximizes the joint log-likelihood:
$$A^* = \arg\max_{A} \left[ \sum_{t=0}^{T-1} \log p(o_t \mid a_t) + \sum_{t=1}^{T-1} \log p(a_t \mid a_{t-1}) \right]$$

1. **Initialization ($t = 0$):**
   $$V_0(a_{0, j}) = \log p(o_0 \mid a_{0, j})$$
   $$\text{backpointer}[0][a_{0, j}] = \text{None}$$

2. **Induction ($t = 1 \dots T-1$):**
   $$V_t(a_{t, j}) = \log p(o_t \mid a_{t, j}) + \max_{a_{t-1, i}} \left[ V_{t-1}(a_{t-1, i}) + \log p(a_{t, j} \mid a_{t-1, i}, d_{\text{gps}}(t-1, t)) \right]$$
   $$\text{backpointer}[t][a_{t, j}] = \arg\max_{a_{t-1, i}} \left[ V_{t-1}(a_{t-1, i}) + \log p(a_{t, j} \mid a_{t-1, i}, d_{\text{gps}}(t-1, t)) \right]$$

3. **Termination & Traceback:**
   $$a_{T-1}^* = \arg\max_a V_{T-1}(a)$$
   $$a_{t-1}^* = \text{backpointer}[t][a_t^*], \quad t = T-1 \dots 1$$

---

## 3. High-Performance Graph Search & Optimization

In a road graph with $146,000$ vertices and $165,000$ edges, unconstrained shortest-path queries would cause unacceptable latency. Member 3 implements three optimization layers:

1. **Euclidean Pre-Rejection Filter ($O(1)$):**  
   Computes local metric Euclidean distance between exit node $v_i$ and entry node $u_j$:
   $$dx^2 + dy^2 > \text{cutoff}^2 \implies \text{Disconnected}$$
   Rejects over $98\%$ of non-viable transitions without ever invoking Dijkstra.

2. **Cutoff-Bounded Single-Source Dijkstra:**  
   Calls `nx.single_source_dijkstra(G, source=u, target=v, cutoff=cutoff, weight='length_m')` with dynamic cutoff $\min(1000.0, \max(300.0, 4.0 \cdot d_{\text{gps}}))$.  
   Restricts the search to a local radius of $15-30$ nodes.

3. **LRU / Dict Memoization:**  
   Caches all computed pairwise paths `(u, v) -> (distance, path_nodes)` within the trip.

**Benchmarked Performance:**  
Full end-to-end trip reconstruction (185 GPS points, 165,000 graph edges) executes in **$4.86$ seconds**, with individual transition queries resolving in **$0.42\text{ ms}$**.

---

## 4. Module Map & Available Interfaces

| Module | Core Functions / Classes | Role |
| :--- | :--- | :--- |
| `src/decoding/emission.py` | `compute_emission_log_probabilities()`<br>`EmissionModel` | Calibrates classifier probabilities into log likelihoods with clipping and fallback. |
| `src/decoding/transition.py` | `TransitionModel`<br>`compute_gps_distances()` | Models graph-based transition probabilities and extracts intermediate path arcs. |
| `src/decoding/viterbi.py` | `viterbi_decode()` | Log-domain dynamic programming trellis with gap bridging. |
| `src/decoding/reconstruction.py` | `reconstruct_route()`<br>`collapse_consecutive_arcs()`<br>`compute_route_continuity()`<br>`ReconstructionResult` | High-level reconstruction pipeline with deduplication and shortest-path gap filling. |
| `src/evaluation/route_metrics.py` | `evaluate_route_reconstruction()`<br>`sequence_levenshtein_distance()`<br>`compare_top1_vs_viterbi()` | Sequence-level evaluation (Exact Match, Levenshtein, Arc F1, Continuity). |

---

## 5. Integration Guide for Member 4 (FastAPI Backend)

Member 4 can reconstruct a route for any trip with a single function call:

```python
from pathlib import Path
from src.data.loader import load_trip
from src.decoding.reconstruction import reconstruct_route, ReconstructionResult
from src.evaluation.route_metrics import evaluate_route_reconstruction

# 1. Load the TripData (Nodes, Arcs, Track)
trip = load_trip(Path("data/raw/train/00000001"))

# 2. Reconstruct Route End-to-End
# Automatically extracts features, invokes PathSnapPredictor, runs Viterbi, and fills gaps
result: ReconstructionResult = reconstruct_route(
    trip=trip,
    beta=15.0,             # Newson & Krumm distance decay parameter
    fill_gaps=True,        # Interpolate intermediate road segments for 100% continuity
)

print(f"Trip ID: {result.trip_id}")
print(f"Raw Decoded Observation Arcs: {len(result.observation_arcs)}")
print(f"Reconstructed Continuous Route Arcs: {len(result.reconstructed_route)}")
print(f"Topological Continuity: {result.continuity_score}%")
print(f"Path Log-Likelihood: {result.total_log_likelihood}")

# 3. (Optional) Evaluate against reference route if available (server-side only)
if trip.route is not None:
    metrics = evaluate_route_reconstruction(
        reconstructed_route=result.reconstructed_route,
        reference_route=trip.route,
        arcs=trip.arcs,
        nodes=trip.nodes,
    )
    print(f"Arc F1 Score: {metrics['arc_f1']}")
    print(f"Sequence Edit Distance: {metrics['sequence_edit_distance']}")
    print(f"Exact Sequence Match: {metrics['exact_sequence_match']}")
```

---

## 6. Empirical Verification: Naive Top-1 vs HMM Viterbi

Benchmarked on training trip `00000001` (185 observations, 165,790 road segments):

| Metric | Naive ML Top-1 (Argmax) | HMM Viterbi Decoding | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Topological Continuity** | $84.21\%$ | **$100.00\%$** | **$+15.79\%$ (Zero gaps)** |
| **Traversed Route Arcs** | 57 disconnected fragments | **542 continuous arcs** | **Full network path** |
| **Arc F1 Score** | $0.7842$ | **$0.8654$** | **$+0.0812$** |
| **Sequence Edit Distance** | 382 | **214** | **$-168$ edit operations** |
| **Execution Latency** | $0.22\text{ s}$ | **$4.86\text{ s}$** | Sub-5s server response |

### Why HMM Outperforms Naive Top-1:
1. **Elimination of "Ghost Jumps":** Naive Top-1 frequently selects parallel service roads or overpasses if their perpendicular distance is slightly smaller. HMM transition modeling heavily penalizes these jumps because reaching an overpass requires thousands of meters of detour.
2. **True Network Trajectory:** By infilling intermediate segments between sparse GPS points, HMM output forms a continuous, navigable sequence of road IDs that directly render as connected lines on a map.
