"""
Evaluation, validation, and benchmarking module for PathSnap.

Computes candidate ranking metrics (Top-1, Top-K, Macro F1, Log Loss)
and trajectory sequence reconstruction metrics (Levenshtein distance,
Arc Precision/Recall/F1, and topological route continuity).
"""

from src.evaluation.route_metrics import (
    compare_top1_vs_viterbi,
    compute_arc_precision_recall_f1,
    compute_multiset_arc_precision_recall_f1,
    evaluate_route_reconstruction,
    sequence_levenshtein_distance,
)

__all__ = [
    "sequence_levenshtein_distance",
    "compute_arc_precision_recall_f1",
    "compute_multiset_arc_precision_recall_f1",
    "evaluate_route_reconstruction",
    "compare_top1_vs_viterbi",
]
