"""
Unit and integration tests for PathSnap sequence-level route evaluation metrics.

Tests:
1. Sequence Levenshtein edit distance (exact DP).
2. Set-based Arc Precision, Recall, and F1.
3. Multiset-based Arc Precision, Recall, and F1.
4. Comprehensive route reconstruction evaluation.
5. Comparative benchmarking: Naive Top-1 vs HMM Viterbi.
"""

from pathlib import Path
import numpy as np
import pytest

from src.data.loader import TripData, load_trip
from src.decoding.reconstruction import reconstruct_route
from src.evaluation.route_metrics import (
    compare_top1_vs_viterbi,
    compute_arc_precision_recall_f1,
    compute_multiset_arc_precision_recall_f1,
    evaluate_route_reconstruction,
    sequence_levenshtein_distance,
)
from src.features.builder import extract_trip_features
from src.models.predict import load_predictor


def test_sequence_levenshtein_distance():
    """Verify sequence Levenshtein distance on known cases."""
    # Exact match
    assert sequence_levenshtein_distance([1, 2, 3], [1, 2, 3]) == 0
    # Empty
    assert sequence_levenshtein_distance([], [1, 2]) == 2
    assert sequence_levenshtein_distance([1, 2, 3], []) == 3
    assert sequence_levenshtein_distance([], []) == 0
    # Single substitution
    assert sequence_levenshtein_distance([1, 2, 3], [1, 99, 3]) == 1
    # Single insertion
    assert sequence_levenshtein_distance([1, 3], [1, 2, 3]) == 1
    # Single deletion
    assert sequence_levenshtein_distance([1, 2, 3, 4], [1, 2, 4]) == 1
    # Complex case
    # [10, 20, 30, 40] vs [10, 25, 30, 40, 50]: substitution of 20->25, insertion of 50 -> 2
    assert sequence_levenshtein_distance([10, 20, 30, 40], [10, 25, 30, 40, 50]) == 2


def test_arc_precision_recall_f1():
    """Verify unique set-based precision, recall, and F1 calculation."""
    # Perfect match
    p, r, f1 = compute_arc_precision_recall_f1([1, 2, 3], [1, 2, 3])
    assert p == 1.0 and r == 1.0 and f1 == 1.0

    # Completely disjoint
    p, r, f1 = compute_arc_precision_recall_f1([1, 2], [3, 4])
    assert p == 0.0 and r == 0.0 and f1 == 0.0

    # Recon has extra false positives
    # recon: {1, 2, 3, 4}, ref: {1, 2}
    p, r, f1 = compute_arc_precision_recall_f1([1, 2, 3, 4], [1, 2])
    assert p == 2 / 4  # 0.5
    assert r == 2 / 2  # 1.0
    assert np.isclose(f1, 2 * 0.5 * 1.0 / (0.5 + 1.0))

    # Empty cases
    assert compute_arc_precision_recall_f1([], []) == (1.0, 1.0, 1.0)
    assert compute_arc_precision_recall_f1([1], []) == (0.0, 0.0, 0.0)
    assert compute_arc_precision_recall_f1([], [1]) == (0.0, 0.0, 0.0)


def test_multiset_arc_precision_recall_f1():
    """Verify multiset frequency-aware evaluation on repeated arcs."""
    # Loop scenario: arc 1 traversed twice in ref, once in recon
    recon = [1, 2]
    ref = [1, 2, 1]

    p, r, f1 = compute_multiset_arc_precision_recall_f1(recon, ref)
    # shared: 1 (count 1) + 2 (count 1) = 2
    assert p == 2 / 2  # 1.0
    assert r == 2 / 3  # ~0.667
    assert np.isclose(f1, 2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))


def test_evaluate_route_reconstruction_comprehensive():
    """Verify full dictionary output of evaluate_route_reconstruction."""
    recon = [10, 11, 12]
    ref = [10, 11, 12]

    # Dummy arcs: 10: 0->1, 11: 1->2, 12: 2->3
    arcs = np.array([
        [0, 1],   # 0
        [1, 2],   # 1
        [2, 3],   # 2
        [3, 4],   # 3
        [4, 5],   # 4
        [5, 6],   # 5
        [6, 7],   # 6
        [7, 8],   # 7
        [8, 9],   # 8
        [9, 10],  # 9
        [100, 101], # 10
        [101, 102], # 11
        [102, 103], # 12
    ])

    metrics = evaluate_route_reconstruction(recon, ref, arcs=arcs)

    assert metrics["exact_sequence_match"] == 1
    assert metrics["arc_precision"] == 1.0
    assert metrics["arc_recall"] == 1.0
    assert metrics["arc_f1"] == 1.0
    assert metrics["sequence_edit_distance"] == 0
    assert metrics["continuity_score"] == 100.0


def test_compare_top1_vs_viterbi_real_trip():
    """Verify comparative evaluation between naive Top-1 and Viterbi on training trip."""
    trip_dir = Path("data/raw/train/00000001")
    if not trip_dir.is_dir():
        pytest.skip("Training trip data not found.")

    trip = load_trip(trip_dir, partition="train")
    predictor = load_predictor()

    # Extract and score
    df_features, _ = extract_trip_features(trip)
    scored = predictor.predict_candidates(df_features)

    # Naive Top-1: argmax candidate per observation
    top1_df = predictor.select_top1_candidates(df_features)
    top1_arcs = top1_df["candidate_arc_id"].values.tolist()

    # Viterbi reconstruction
    viterbi_result = reconstruct_route(
        trip=trip,
        scored_candidates=scored,
        predictor=predictor,
    )
    viterbi_arcs = viterbi_result.observation_arcs

    # Run comparison against ground truth route
    comp = compare_top1_vs_viterbi(
        trip=trip,
        top1_observation_arcs=top1_arcs,
        viterbi_observation_arcs=viterbi_arcs,
        reference_route=trip.route,
        fill_gaps=True,
    )

    assert "continuity_improvement_pct" in comp
    assert "viterbi_continuity_pct" in comp
    assert comp["viterbi_continuity_pct"] >= comp["top1_continuity_pct"]
    assert "viterbi_arc_f1" in comp
    assert comp["viterbi_arc_f1"] >= 0.70  # High trajectory F1
