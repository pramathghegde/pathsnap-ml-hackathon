"""
End-to-End Route Reconstruction Engine for PathSnap Trajectory Matching.

Orchestrates the entire trajectory reconstruction workflow:
1. Feature extraction and spatial candidate retrieval
2. Model-based candidate emission scoring via PathSnapPredictor
3. Log-domain emission likelihood transformation
4. Graph-based transition modeling with NetworkX
5. Optimal MAP sequence decoding via Viterbi trellis dynamic programming
6. Consecutive arc deduplication and shortest-path topological gap filling
7. Quantitative continuity verification and diagnostic telemetry
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.loader import TripData
from src.features.builder import extract_trip_features
from src.models.predict import PathSnapPredictor, load_predictor
from src.decoding.emission import compute_emission_log_probabilities
from src.decoding.transition import (
    TransitionModel,
    compute_gps_distances,
    DEFAULT_BETA,
    DEFAULT_DISCONNECT_PENALTY,
)
from src.decoding.viterbi import viterbi_decode


@dataclass
class ReconstructionResult:
    """
    Standard container for reconstructed trajectory route and diagnostic telemetry.
    """
    trip_id: str
    observation_arcs: List[int]        # Decoded arc ID for each GPS observation (length T)
    reconstructed_route: List[int]     # Collapsed, continuous sequence of traversed road arcs
    total_log_likelihood: float        # Total Viterbi sequence log likelihood
    continuity_score: float            # Percentage of consecutive transitions topologically valid [0.0 - 100.0]
    num_observations: int              # Total GPS observation count T
    num_route_arcs: int                # Arc count in final reconstructed route
    gap_count: int                     # Number of topological gaps detected between consecutive observations
    disconnect_count: int              # Number of unresolvable topological disconnects
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def collapse_consecutive_arcs(arc_seq: Sequence[int]) -> List[int]:
    """
    Deduplicate consecutive identical road arcs in an observation sequence:
        [12, 12, 12, 15, 15, 18, 18] -> [12, 15, 18]

    Args:
        arc_seq: Sequence of arc IDs.

    Returns:
        Deduplicated list of arc IDs.
    """
    if not arc_seq:
        return []

    collapsed: List[int] = [int(arc_seq[0])]
    for arc_id in arc_seq[1:]:
        arc_int = int(arc_id)
        if arc_int != collapsed[-1]:
            collapsed.append(arc_int)

    return collapsed


def compute_route_continuity(route_arcs: Sequence[int], arcs: np.ndarray) -> float:
    """
    Compute topological continuity percentage of a route sequence.
    A transition (a_i, a_{i+1}) is continuous if to_node(a_i) == from_node(a_{i+1}).

    Args:
        route_arcs: Sequence of road arc IDs.
        arcs: Road network arcs array of shape (E, 2).

    Returns:
        Continuity percentage in [0.0, 100.0].
    """
    m = len(route_arcs)
    if m <= 1:
        return 100.0

    connected = 0
    total_transitions = m - 1

    for i in range(total_transitions):
        curr_arc = int(route_arcs[i])
        next_arc = int(route_arcs[i + 1])
        v_curr = int(arcs[curr_arc, 1])
        u_next = int(arcs[next_arc, 0])
        if v_curr == u_next:
            connected += 1

    return float((connected / total_transitions) * 100.0)


def reconstruct_route(
    trip: TripData,
    scored_candidates: Optional[pd.DataFrame] = None,
    predictor: Optional[PathSnapPredictor] = None,
    beta: float = DEFAULT_BETA,
    disconnect_penalty: float = DEFAULT_DISCONNECT_PENALTY,
    fill_gaps: bool = True,
    spatial_radius_m: float = 50.0,
    max_candidates_per_obs: int = 15,
) -> ReconstructionResult:
    """
    Execute full map-matching and route reconstruction for a given trip.

    Args:
        trip: TripData instance containing track, nodes, arcs, and metadata.
        scored_candidates: Optional pre-scored candidate DataFrame with match_probability.
                           If None, features are extracted and scored via predictor.
        predictor: Optional PathSnapPredictor instance. Loaded from disk if None.
        beta: Exponential distance error scale (meters) for transition model.
        disconnect_penalty: Log probability penalty for disconnected transitions.
        fill_gaps: If True, interpolates shortest-path intermediate road segments
                   between non-adjacent consecutive observations.
        spatial_radius_m: Candidate search radius in meters.
        max_candidates_per_obs: Maximum candidate road segments per observation.

    Returns:
        ReconstructionResult dataclass.
    """
    start_time = time.perf_counter()

    # Step 1: Feature Extraction & Model Scoring if not provided
    if scored_candidates is None:
        df_features, feat_metrics = extract_trip_features(
            trip=trip,
            spatial_radius_m=spatial_radius_m,
            max_candidates_per_obs=max_candidates_per_obs,
        )

        if df_features.empty:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ReconstructionResult(
                trip_id=trip.trip_id,
                observation_arcs=[],
                reconstructed_route=[],
                total_log_likelihood=-float("inf"),
                continuity_score=0.0,
                num_observations=trip.num_observations,
                num_route_arcs=0,
                gap_count=0,
                disconnect_count=0,
                diagnostics={"status": "empty_candidates", "runtime_ms": round(elapsed, 3)},
            )

        if predictor is None:
            predictor = load_predictor()

        scored_candidates = predictor.predict_candidates(df_features)

    # Step 2: Build candidate mapping
    candidate_map: Dict[int, List[int]] = (
        scored_candidates.groupby("observation_id", sort=True)["candidate_arc_id"]
        .apply(lambda s: [int(x) for x in s])
        .to_dict()
    )

    # Step 3: Compute Emission Log Probabilities
    emission_log_probs = compute_emission_log_probabilities(
        scored_candidates=scored_candidates,
        prob_col="match_probability",
        obs_col="observation_id",
        arc_col="candidate_arc_id",
    )

    # Step 4: Build Road Graph & Transition Model
    transition_model = TransitionModel.from_trip(
        trip=trip,
        beta=beta,
        disconnect_penalty=disconnect_penalty,
    )

    # Step 5: Compute GPS Step Displacements
    gps_distances = compute_gps_distances(trip.track)

    # Step 6: Viterbi Trellis Decoding
    obs_arcs, total_ll, vit_diag = viterbi_decode(
        num_observations=trip.num_observations,
        candidate_map=candidate_map,
        emission_log_probs=emission_log_probs,
        transition_model=transition_model,
        gps_distances=gps_distances,
    )

    if not obs_arcs:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ReconstructionResult(
            trip_id=trip.trip_id,
            observation_arcs=[],
            reconstructed_route=[],
            total_log_likelihood=-float("inf"),
            continuity_score=0.0,
            num_observations=trip.num_observations,
            num_route_arcs=0,
            gap_count=0,
            disconnect_count=0,
            diagnostics={
                "status": "viterbi_failed",
                "viterbi_diagnostics": vit_diag,
                "runtime_ms": round(elapsed, 3),
            },
        )

    # Step 7: Collapse consecutive identical arcs
    collapsed_arcs = collapse_consecutive_arcs(obs_arcs)

    # Step 8: Topological Gap Filling (if requested)
    gap_count = 0
    disconnect_count = 0
    final_route: List[int] = []

    if collapsed_arcs:
        final_route.append(collapsed_arcs[0])

        for k in range(len(collapsed_arcs) - 1):
            curr_arc = collapsed_arcs[k]
            next_arc = collapsed_arcs[k + 1]

            v_curr = int(trip.arcs[curr_arc, 1])
            u_next = int(trip.arcs[next_arc, 0])

            if v_curr == u_next:
                # Directly connected
                final_route.append(next_arc)
            else:
                gap_count += 1
                if fill_gaps:
                    intermediate = transition_model.get_intermediate_path_arcs(curr_arc, next_arc)
                    if intermediate:
                        final_route.extend(intermediate)
                    else:
                        disconnect_count += 1
                    final_route.append(next_arc)
                else:
                    final_route.append(next_arc)

    # Step 9: Measure Route Continuity
    continuity_score = compute_route_continuity(final_route, trip.arcs)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    diagnostics = {
        "status": "success",
        "beta": beta,
        "disconnect_penalty": disconnect_penalty,
        "fill_gaps": fill_gaps,
        "collapsed_arc_count": len(collapsed_arcs),
        "infilled_arc_count": len(final_route) - len(collapsed_arcs),
        "viterbi_runtime_ms": vit_diag.get("runtime_ms", 0.0),
        "total_runtime_ms": round(elapsed_ms, 3),
    }

    return ReconstructionResult(
        trip_id=trip.trip_id,
        observation_arcs=obs_arcs,
        reconstructed_route=final_route,
        total_log_likelihood=round(total_ll, 4),
        continuity_score=round(continuity_score, 2),
        num_observations=trip.num_observations,
        num_route_arcs=len(final_route),
        gap_count=gap_count,
        disconnect_count=disconnect_count,
        diagnostics=diagnostics,
    )
