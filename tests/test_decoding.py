"""
Unit and integration tests for PathSnap HMM sequential decoding and route reconstruction.

Tests:
1. Candidate emission log-probability computation and clipping.
2. Exponential transition modeling on synthetic and real graphs.
3. Viterbi dynamic programming trellis decoding and backpointer tracking.
4. Consecutive arc collapsing and topological gap filling.
5. End-to-end route reconstruction pipeline.
"""

from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.data.loader import TripData, build_network_graph, load_trip
from src.decoding.emission import (
    EmissionModel,
    compute_emission_log_probabilities,
)
from src.decoding.transition import (
    TransitionModel,
    compute_gps_distances,
)
from src.decoding.viterbi import viterbi_decode
from src.decoding.reconstruction import (
    collapse_consecutive_arcs,
    compute_route_continuity,
    reconstruct_route,
)


@pytest.fixture
def synthetic_corridor():
    """
    Creates a simple 4-node, 3-arc linear corridor:
    Nodes: 0 -> 1 -> 2 -> 3
    Arcs:
      0: 0 -> 1 (100m)
      1: 1 -> 2 (100m)
      2: 2 -> 3 (100m)
    """
    nodes = np.array([
        [0.0, 0.0],
        [0.001, 0.0],
        [0.002, 0.0],
        [0.003, 0.0],
    ], dtype=np.float64)
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
    ], dtype=np.int32)
    G = build_network_graph(nodes, arcs)
    return nodes, arcs, G


def test_emission_log_probabilities_clipping_and_normalization():
    """Verify that emission computation properly clips to min_prob and normalizes."""
    df = pd.DataFrame({
        "observation_id": [0, 0, 1, 1],
        "candidate_arc_id": [10, 11, 20, 21],
        "match_probability": [0.8, 0.2, 0.0, 0.5],
    })

    em = compute_emission_log_probabilities(df, min_prob=1e-5, normalize=True)

    assert 0 in em and 1 in em
    # Observation 0: 0.8 and 0.2 -> normalized probs 0.8 and 0.2
    assert np.isclose(np.exp(em[0][10]) + np.exp(em[0][11]), 1.0)
    assert em[0][10] > em[0][11]

    # Observation 1: 0.0 is clipped to min_prob, then normalized
    assert em[1][20] < em[1][21]
    assert np.isclose(np.exp(em[1][20]) + np.exp(em[1][21]), 1.0)


def test_emission_spatial_fallback():
    """Verify spatial distance fallback when probability column is missing."""
    df = pd.DataFrame({
        "observation_id": [0, 0],
        "candidate_arc_id": [5, 6],
        "perp_dist": [2.0, 30.0],
    })

    em = compute_emission_log_probabilities(df, prob_col="non_existent", normalize=True)
    # Closer distance should have higher log emission likelihood
    assert em[0][5] > em[0][6]


def test_transition_model_same_arc(synthetic_corridor):
    """Verify same-arc transition has zero penalty when d_gps <= arc_length."""
    nodes, arcs, G = synthetic_corridor
    tm = TransitionModel(G, arcs, nodes, beta=15.0)

    # Arc 0 has length ~111m
    d_gps = 50.0
    log_p = tm.log_transition_prob(from_arc=0, to_arc=0, d_gps=d_gps)

    # When delta_d == 0, log_p should be exactly -log(beta)
    expected = -np.log(15.0)
    assert np.isclose(log_p, expected)


def test_transition_model_connected_arcs(synthetic_corridor):
    """Verify directly connected arcs evaluate distance difference accurately."""
    nodes, arcs, G = synthetic_corridor
    tm = TransitionModel(G, arcs, nodes, beta=15.0)

    # Arc 0 -> Arc 1 is directly connected (1 -> 1)
    arc1_len = tm.arc_lengths[1]
    # If d_gps matches arc1_len, delta_d should be 0
    log_p = tm.log_transition_prob(from_arc=0, to_arc=1, d_gps=arc1_len)
    assert np.isclose(log_p, -np.log(15.0))

    # If d_gps differs by 30m, delta_d is 30m
    log_p_mismatch = tm.log_transition_prob(from_arc=0, to_arc=1, d_gps=arc1_len + 30.0)
    assert log_p_mismatch < log_p
    assert np.isclose(log_p_mismatch, -np.log(15.0) - (30.0 / 15.0))


def test_transition_model_disconnected_arcs(synthetic_corridor):
    """Verify disconnected or reverse transitions receive the disconnect penalty."""
    nodes, arcs, G = synthetic_corridor
    tm = TransitionModel(G, arcs, nodes, beta=15.0, disconnect_penalty=-200.0)

    # Arc 1 -> Arc 0 is topologically backwards in a one-way corridor
    log_p_rev = tm.log_transition_prob(from_arc=1, to_arc=0, d_gps=50.0)
    assert log_p_rev == -200.0


def test_transition_intermediate_path_reconstruction(synthetic_corridor):
    """Verify multi-hop path returns intermediate arc IDs."""
    nodes, arcs, G = synthetic_corridor
    tm = TransitionModel(G, arcs, nodes, beta=15.0)

    # From Arc 0 (0->1) to Arc 2 (2->3), intermediate is Arc 1 (1->2)
    intermediate = tm.get_intermediate_path_arcs(from_arc=0, to_arc=2)
    assert intermediate == [1]

    # Directly connected has no intermediate
    assert tm.get_intermediate_path_arcs(from_arc=0, to_arc=1) == []
    # Same arc has no intermediate
    assert tm.get_intermediate_path_arcs(from_arc=0, to_arc=0) == []


def test_viterbi_decoding_linear_path(synthetic_corridor):
    """Test Viterbi decoding correctly resolves an unambiguous sequence."""
    nodes, arcs, G = synthetic_corridor
    tm = TransitionModel(G, arcs, nodes, beta=15.0)

    # 3 observations moving along Arc 0 -> Arc 1 -> Arc 2
    candidate_map = {
        0: [0, 1],
        1: [0, 1, 2],
        2: [1, 2],
    }
    # Biased emissions favoring 0 at t=0, 1 at t=1, 2 at t=2
    emission_log_probs = {
        0: {0: 0.0, 1: -5.0},
        1: {0: -5.0, 1: 0.0, 2: -5.0},
        2: {1: -5.0, 2: 0.0},
    }
    gps_distances = [tm.arc_lengths[1], tm.arc_lengths[2]]

    decoded_arcs, total_ll, diag = viterbi_decode(
        num_observations=3,
        candidate_map=candidate_map,
        emission_log_probs=emission_log_probs,
        transition_model=tm,
        gps_distances=gps_distances,
    )

    assert decoded_arcs == [0, 1, 2]
    assert np.isfinite(total_ll)
    assert diag["status"] == "success"


def test_viterbi_overcoming_noisy_emission(synthetic_corridor):
    """
    Test Viterbi transitions smoothing out an isolated noisy emission.
    Suppose at t=1, false candidate 0 has slightly higher emission, but
    transition 0->0->2 requires jumping backwards or disconnects,
    whereas 0->1->2 is perfectly continuous.
    """
    nodes, arcs, G = synthetic_corridor
    tm = TransitionModel(G, arcs, nodes, beta=15.0, disconnect_penalty=-100.0)

    candidate_map = {
        0: [0],
        1: [0, 1],
        2: [2],
    }
    # At t=1, candidate 0 has slight emission edge (-0.1 vs -0.5)
    emission_log_probs = {
        0: {0: 0.0},
        1: {0: -0.1, 1: -0.5},
        2: {2: 0.0},
    }
    # GPS distance is 100m (moving forward along the corridor)
    gps_distances = [100.0, 100.0]

    decoded_arcs, _, _ = viterbi_decode(
        num_observations=3,
        candidate_map=candidate_map,
        emission_log_probs=emission_log_probs,
        transition_model=tm,
        gps_distances=gps_distances,
    )

    # Transition penalty for 0->2 (skipping 1 or disconnecting) should force [0, 1, 2]
    assert decoded_arcs == [0, 1, 2]


def test_viterbi_edge_cases(synthetic_corridor):
    """Verify Viterbi handling of empty, single observation, and missing candidate cases."""
    nodes, arcs, G = synthetic_corridor
    tm = TransitionModel(G, arcs, nodes)

    # Empty
    d, ll, diag = viterbi_decode(0, {}, {}, tm)
    assert d == []
    assert diag["status"] == "empty_input"

    # Single observation
    cand_map = {0: [1, 2]}
    em = {0: {1: -1.0, 2: -0.2}}
    d, ll, diag = viterbi_decode(1, cand_map, em, tm)
    assert d == [2]
    assert diag["status"] == "single_valid_observation"


def test_collapse_consecutive_arcs():
    """Verify sequence deduplication."""
    seq = [1, 1, 2, 2, 2, 3, 1, 1]
    assert collapse_consecutive_arcs(seq) == [1, 2, 3, 1]
    assert collapse_consecutive_arcs([]) == []
    assert collapse_consecutive_arcs([7]) == [7]


def test_compute_route_continuity():
    """Verify continuity computation on connected vs disjoint routes."""
    arcs = np.array([
        [10, 20],  # 0
        [20, 30],  # 1
        [30, 40],  # 2
        [99, 100], # 3 (disconnected)
    ])

    # [0, 1, 2] is 100% connected: 20==20, 30==30
    assert compute_route_continuity([0, 1, 2], arcs) == 100.0

    # [0, 3] has 0% continuity
    assert compute_route_continuity([0, 3], arcs) == 0.0

    # [0, 1, 3] has 1 connected out of 2 transitions -> 50.0%
    assert compute_route_continuity([0, 1, 3], arcs) == 50.0


def test_reconstruct_route_real_trip():
    """Test full reconstruction pipeline on real training trip 00000001."""
    trip_dir = Path("data/raw/train/00000001")
    if not trip_dir.is_dir():
        pytest.skip("Training trip data not found.")

    trip = load_trip(trip_dir, partition="train")

    result = reconstruct_route(
        trip=trip,
        beta=15.0,
        fill_gaps=True,
    )

    assert result.trip_id == "00000001"
    assert len(result.observation_arcs) == trip.num_observations
    assert len(result.reconstructed_route) > 0
    assert result.continuity_score >= 80.0  # Real road networks achieve very high continuity
    assert np.isfinite(result.total_log_likelihood)
    assert result.diagnostics["status"] == "success"
