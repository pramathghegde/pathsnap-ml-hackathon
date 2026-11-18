"""
Log-Domain Viterbi Trellis Dynamic Programming Decoder for PathSnap.

Computes the maximum a posteriori (MAP) sequence of road arcs:
    A* = argmax_{A} [ log p(A, O) ] = argmax_{A} [ sum_t log p(o_t | a_t) + sum_t log p(a_t | a_{t-1}) ]

Features:
- Exact dynamic programming trellis in numerical log domain
- Robust backpointer tracking with deterministic tie-breaking
- Resilient observation handling: gracefully bridges missing candidates / empty observations
- Detailed diagnostic metadata reporting (likelihood, transition penalties, execution latency)
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.decoding.transition import TransitionModel


def viterbi_decode(
    num_observations: int,
    candidate_map: Dict[int, List[int]],
    emission_log_probs: Dict[int, Dict[int, float]],
    transition_model: TransitionModel,
    gps_distances: Optional[Sequence[float]] = None,
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Execute Viterbi trellis dynamic programming decoding in the log domain.

    Args:
        num_observations: Total number of GPS observations T in the trajectory.
        candidate_map: Dictionary mapping observation index t to list of candidate arc IDs.
        emission_log_probs: Nested dict mapping t -> {arc_id: log_emission_prob}.
        transition_model: TransitionModel providing log_transition_prob(from_arc, to_arc, d_gps).
        gps_distances: Sequence of T-1 step distances in meters between consecutive
                       observations (t-1, t). If None, defaults to 0.0.

    Returns:
        Tuple of:
        - decoded_arcs: List of optimal arc IDs of length num_observations.
        - total_log_likelihood: float score of the optimal path.
        - diagnostics: Dict containing execution metrics and decoding statistics.
    """
    start_time = time.perf_counter()

    if num_observations <= 0:
        return [], 0.0, {
            "status": "empty_input",
            "num_observations": 0,
            "runtime_ms": 0.0,
        }

    # Prepare step distance array of length T - 1
    if gps_distances is not None:
        step_dists = np.asarray(gps_distances, dtype=np.float64)
    else:
        step_dists = np.zeros(max(0, num_observations - 1), dtype=np.float64)

    # Identify observations with at least one candidate
    valid_timesteps: List[int] = [
        t for t in range(num_observations)
        if t in candidate_map and len(candidate_map[t]) > 0
    ]

    if not valid_timesteps:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return [], -float("inf"), {
            "status": "no_candidates_available",
            "num_observations": num_observations,
            "valid_observations": 0,
            "runtime_ms": round(elapsed, 3),
        }

    # Handle single valid observation edge case
    if len(valid_timesteps) == 1:
        t0 = valid_timesteps[0]
        cands = candidate_map[t0]
        em_dict = emission_log_probs.get(t0, {})
        best_cand = max(cands, key=lambda c: em_dict.get(c, -20.0))
        best_ll = float(em_dict.get(best_cand, 0.0))
        decoded = [best_cand] * num_observations
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return decoded, best_ll, {
            "status": "single_valid_observation",
            "num_observations": num_observations,
            "valid_observations": 1,
            "runtime_ms": round(elapsed, 3),
        }

    # Viterbi Trellis Initialization at first valid observation t_first
    t_first = valid_timesteps[0]
    first_cands = candidate_map[t_first]
    first_em = emission_log_probs.get(t_first, {})

    # V[cand] = log_emission (assuming uniform prior across first observation candidates)
    V: Dict[int, float] = {
        cand: first_em.get(cand, -20.0) for cand in first_cands
    }

    # Backpointer storage: backpointers[t_step][curr_cand] = prev_cand
    backpointers: Dict[int, Dict[int, int]] = {}

    # Cumulative distance helper between arbitrary timesteps i and j
    def get_interval_gps_dist(i: int, j: int) -> float:
        if i >= j or len(step_dists) == 0:
            return 0.0
        # step_dists[k] is distance from k to k+1
        start_idx = max(0, i)
        end_idx = min(len(step_dists), j)
        if start_idx >= end_idx:
            return 0.0
        return float(np.sum(step_dists[start_idx:end_idx]))

    # Trellis Induction across successive valid timesteps
    for step_idx in range(1, len(valid_timesteps)):
        t_prev = valid_timesteps[step_idx - 1]
        t_curr = valid_timesteps[step_idx]

        curr_cands = candidate_map[t_curr]
        curr_em = emission_log_probs.get(t_curr, {})
        d_gps = get_interval_gps_dist(t_prev, t_curr)

        V_curr: Dict[int, float] = {}
        bp_curr: Dict[int, int] = {}

        for a_curr in curr_cands:
            e_prob = curr_em.get(a_curr, -20.0)
            best_score = -float("inf")
            best_prev: Optional[int] = None

            for a_prev, v_prev in V.items():
                t_prob = transition_model.log_transition_prob(a_prev, a_curr, d_gps)
                score = v_prev + t_prob + e_prob

                if score > best_score:
                    best_score = score
                    best_prev = a_prev

            # If all transitions gave -inf (extremely rare fallback)
            if best_prev is None:
                best_prev = next(iter(V.keys()))
                best_score = V[best_prev] + transition_model.disconnect_penalty + e_prob

            V_curr[a_curr] = best_score
            bp_curr[a_curr] = best_prev

        V = V_curr
        backpointers[t_curr] = bp_curr

    # Trellis Termination: select candidate with highest path likelihood at final step
    t_last = valid_timesteps[-1]
    best_last_cand = max(V.keys(), key=lambda c: V[c])
    total_log_likelihood = float(V[best_last_cand])

    # Backpointer Traceback across valid timesteps
    valid_decoded: Dict[int, int] = {t_last: best_last_cand}
    curr_cand = best_last_cand

    for step_idx in range(len(valid_timesteps) - 1, 0, -1):
        t_curr = valid_timesteps[step_idx]
        t_prev = valid_timesteps[step_idx - 1]
        prev_cand = backpointers[t_curr][curr_cand]
        valid_decoded[t_prev] = prev_cand
        curr_cand = prev_cand

    # Interpolate/propagate decoded arcs for any missing/unobserved timesteps
    decoded_arcs: List[int] = [0] * num_observations
    running_cand = valid_decoded[valid_timesteps[0]]

    # Fill leading unobserved steps if any
    for t in range(valid_timesteps[0]):
        decoded_arcs[t] = running_cand

    for t in range(valid_timesteps[0], num_observations):
        if t in valid_decoded:
            running_cand = valid_decoded[t]
        decoded_arcs[t] = running_cand

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    diagnostics = {
        "status": "success",
        "num_observations": num_observations,
        "valid_observations": len(valid_timesteps),
        "total_log_likelihood": round(total_log_likelihood, 4),
        "mean_log_likelihood_per_obs": round(
            total_log_likelihood / max(1, num_observations), 4
        ),
        "runtime_ms": round(elapsed_ms, 3),
    }

    return decoded_arcs, total_log_likelihood, diagnostics
