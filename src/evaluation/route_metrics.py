"""
Sequence-Level Route Reconstruction Evaluation Metrics for PathSnap.

Computes comprehensive trajectory-level quality and alignment metrics:
- Exact Sequence Match (binary trajectory identity)
- Set-based & Multiset Arc Precision, Recall, and F1-score
- Ordered Sequence Levenshtein Edit Distance (normalized and absolute)
- Metric Route Length Percentage Error
- Topological Route Continuity Verification
- Comparative Benchmarking: Naive ML Top-1 vs HMM Viterbi Decoding
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.data.loader import TripData
from src.decoding.reconstruction import (
    collapse_consecutive_arcs,
    compute_route_continuity,
)
from src.decoding.transition import TransitionModel


def sequence_levenshtein_distance(seq1: Sequence[int], seq2: Sequence[int]) -> int:
    """
    Compute exact Levenshtein edit distance between two ordered sequences of integer arc IDs.

    Uses an optimal O(min(N, M)) memory dynamic programming implementation.

    Args:
        seq1: First sequence of arc IDs.
        seq2: Second sequence of arc IDs.

    Returns:
        Integer edit distance (insertions, deletions, substitutions).
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    # Ensure seq1 is the shorter sequence for O(min(N, M)) space
    if n > m:
        seq1, seq2 = seq2, seq1
        n, m = m, n

    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        curr_row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq1[j - 1] == seq2[i - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,        # deletion
                curr_row[j - 1] + 1,    # insertion
                prev_row[j - 1] + cost, # substitution
            )
        prev_row = curr_row

    return prev_row[n]


def compute_arc_precision_recall_f1(
    reconstructed: Sequence[int], reference: Sequence[int]
) -> Tuple[float, float, float]:
    """
    Compute set-based Precision, Recall, and F1-score over unique arc IDs.

    Args:
        reconstructed: Sequence of reconstructed route arc IDs.
        reference: Sequence of ground-truth reference arc IDs.

    Returns:
        Tuple of (precision, recall, f1) in range [0.0, 1.0].
    """
    set_recon = set(int(a) for a in reconstructed)
    set_ref = set(int(a) for a in reference)

    if not set_recon and not set_ref:
        return 1.0, 1.0, 1.0
    if not set_recon:
        return 0.0, 0.0, 0.0
    if not set_ref:
        return 0.0, 0.0, 0.0

    intersection_size = len(set_recon & set_ref)
    precision = intersection_size / len(set_recon)
    recall = intersection_size / len(set_ref)

    if precision + recall > 0.0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def compute_multiset_arc_precision_recall_f1(
    reconstructed: Sequence[int], reference: Sequence[int]
) -> Tuple[float, float, float]:
    """
    Compute multiset (frequency-aware) Precision, Recall, and F1-score over arc IDs.
    Accounts for loops and repeated traversals of identical road segments.

    Args:
        reconstructed: Sequence of reconstructed route arc IDs.
        reference: Sequence of ground-truth reference arc IDs.

    Returns:
        Tuple of (multiset_precision, multiset_recall, multiset_f1) in [0.0, 1.0].
    """
    n_recon = len(reconstructed)
    n_ref = len(reference)

    if n_recon == 0 and n_ref == 0:
        return 1.0, 1.0, 1.0
    if n_recon == 0:
        return 0.0, 0.0, 0.0
    if n_ref == 0:
        return 0.0, 0.0, 0.0

    count_recon = Counter(int(a) for a in reconstructed)
    count_ref = Counter(int(a) for a in reference)

    shared_count = sum(
        min(c, count_ref[arc_id]) for arc_id, c in count_recon.items()
    )

    precision = shared_count / n_recon
    recall = shared_count / n_ref

    if precision + recall > 0.0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def evaluate_route_reconstruction(
    reconstructed_route: Sequence[int],
    reference_route: Sequence[int],
    arcs: Optional[np.ndarray] = None,
    nodes: Optional[np.ndarray] = None,
    arc_lengths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute all sequence-level evaluation metrics between reconstructed and reference routes.

    Args:
        reconstructed_route: Reconstructed sequence of road arc IDs.
        reference_route: Ground-truth sequence of traversed road arc IDs.
        arcs: Optional road arcs array (E, 2) for continuity calculation.
        nodes: Optional road nodes array (V, 2) for metric length calculation.
        arc_lengths: Optional precomputed arc lengths array (E,).

    Returns:
        Dictionary of sequence metrics.
    """
    list_recon = [int(a) for a in reconstructed_route]
    list_ref = [int(a) for a in reference_route]

    exact_match = int(list_recon == list_ref)
    p, r, f1 = compute_arc_precision_recall_f1(list_recon, list_ref)
    mp, mr, mf1 = compute_multiset_arc_precision_recall_f1(list_recon, list_ref)
    edit_dist = sequence_levenshtein_distance(list_recon, list_ref)

    max_len = max(len(list_recon), len(list_ref))
    norm_edit_dist = (edit_dist / max_len) if max_len > 0 else 0.0

    results: Dict[str, Any] = {
        "exact_sequence_match": exact_match,
        "arc_precision": round(p, 4),
        "arc_recall": round(r, 4),
        "arc_f1": round(f1, 4),
        "multiset_arc_precision": round(mp, 4),
        "multiset_arc_recall": round(mr, 4),
        "multiset_arc_f1": round(mf1, 4),
        "sequence_edit_distance": edit_dist,
        "normalized_edit_distance": round(norm_edit_dist, 4),
        "num_reconstructed_arcs": len(list_recon),
        "num_reference_arcs": len(list_ref),
    }

    # Continuity verification
    if arcs is not None:
        continuity = compute_route_continuity(list_recon, arcs)
        results["continuity_score"] = round(continuity, 2)

    # Metric length calculation
    if arc_lengths is not None or (arcs is not None and nodes is not None):
        if arc_lengths is None:
            # Precompute arc lengths
            mean_lat = float(np.mean(nodes[:, 1]))
            cos_lat = float(np.cos(np.radians(mean_lat)))
            p1 = nodes[arcs[:, 0]]
            p2 = nodes[arcs[:, 1]]
            dx = (p2[:, 0] - p1[:, 0]) * cos_lat * 111320.0
            dy = (p2[:, 1] - p1[:, 1]) * 111320.0
            lengths = np.sqrt(dx * dx + dy * dy)
        else:
            lengths = arc_lengths

        recon_len_m = float(sum(lengths[a] for a in list_recon if a < len(lengths)))
        ref_len_m = float(sum(lengths[a] for a in list_ref if a < len(lengths)))
        len_error_pct = (
            abs(recon_len_m - ref_len_m) / max(1.0, ref_len_m) * 100.0
        )

        results["reconstructed_length_m"] = round(recon_len_m, 2)
        results["reference_length_m"] = round(ref_len_m, 2)
        results["length_error_pct"] = round(len_error_pct, 2)

    return results


def compare_top1_vs_viterbi(
    trip: TripData,
    top1_observation_arcs: Sequence[int],
    viterbi_observation_arcs: Sequence[int],
    reference_route: Optional[Sequence[int]] = None,
    fill_gaps: bool = True,
) -> Dict[str, Any]:
    """
    Compare raw ML Top-1 baseline (independent argmax) against HMM Viterbi decoding.

    Args:
        trip: TripData containing road arcs and nodes.
        top1_observation_arcs: Naive argmax candidate arc per observation.
        viterbi_observation_arcs: Viterbi decoded candidate arc per observation.
        reference_route: Ground-truth route sequence (if available).
        fill_gaps: Whether to perform shortest-path gap filling on routes.

    Returns:
        Comparative dictionary containing metrics for both approaches.
    """
    # 1. Top-1 route post-processing
    top1_collapsed = collapse_consecutive_arcs(top1_observation_arcs)
    top1_continuity = compute_route_continuity(top1_collapsed, trip.arcs)

    # 2. Viterbi route post-processing
    vit_collapsed = collapse_consecutive_arcs(viterbi_observation_arcs)

    if fill_gaps:
        trans_model = TransitionModel.from_trip(trip)
        vit_route: List[int] = []
        if vit_collapsed:
            vit_route.append(vit_collapsed[0])
            for k in range(len(vit_collapsed) - 1):
                c_arc = vit_collapsed[k]
                n_arc = vit_collapsed[k + 1]
                if trip.arcs[c_arc, 1] == trip.arcs[n_arc, 0]:
                    vit_route.append(n_arc)
                else:
                    intermediate = trans_model.get_intermediate_path_arcs(c_arc, n_arc)
                    if intermediate:
                        vit_route.extend(intermediate)
                    vit_route.append(n_arc)
    else:
        vit_route = vit_collapsed

    vit_continuity = compute_route_continuity(vit_route, trip.arcs)

    comparison: Dict[str, Any] = {
        "trip_id": trip.trip_id,
        "num_observations": trip.num_observations,
        "top1_raw_route_arcs": len(top1_collapsed),
        "top1_continuity_pct": round(top1_continuity, 2),
        "viterbi_final_route_arcs": len(vit_route),
        "viterbi_continuity_pct": round(vit_continuity, 2),
        "continuity_improvement_pct": round(vit_continuity - top1_continuity, 2),
    }

    ref = reference_route if reference_route is not None else trip.route
    if ref is not None:
        top1_metrics = evaluate_route_reconstruction(
            top1_collapsed, ref, arcs=trip.arcs, nodes=trip.nodes
        )
        vit_metrics = evaluate_route_reconstruction(
            vit_route, ref, arcs=trip.arcs, nodes=trip.nodes
        )

        comparison["top1_exact_match"] = top1_metrics["exact_sequence_match"]
        comparison["top1_arc_f1"] = top1_metrics["arc_f1"]
        comparison["top1_edit_distance"] = top1_metrics["sequence_edit_distance"]

        comparison["viterbi_exact_match"] = vit_metrics["exact_sequence_match"]
        comparison["viterbi_arc_f1"] = vit_metrics["arc_f1"]
        comparison["viterbi_edit_distance"] = vit_metrics["sequence_edit_distance"]

        comparison["f1_improvement"] = round(
            vit_metrics["arc_f1"] - top1_metrics["arc_f1"], 4
        )
        comparison["edit_distance_reduction"] = (
            top1_metrics["sequence_edit_distance"] - vit_metrics["sequence_edit_distance"]
        )

    return comparison
