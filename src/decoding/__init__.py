"""
Hidden Markov Model (HMM) sequence decoding and route reconstruction module.

Exports:
- compute_emission_log_probabilities: Converts ML match probabilities to log likelihoods
- EmissionModel: Configurable emission likelihood model
- TransitionModel: Graph-based transition likelihood modeling with NetworkX
- compute_gps_distances: GPS displacement utility
- viterbi_decode: Log-domain dynamic programming Viterbi trellis decoder
- reconstruct_route: End-to-end trajectory reconstruction pipeline
- collapse_consecutive_arcs: Arc sequence deduplication
- compute_route_continuity: Route topological connectivity verification
- ReconstructionResult: Typed reconstruction output dataclass
"""

from src.decoding.emission import (
    EmissionModel,
    compute_emission_log_probabilities,
)
from src.decoding.transition import (
    TransitionModel,
    compute_gps_distances,
)
from src.decoding.viterbi import (
    viterbi_decode,
)
from src.decoding.reconstruction import (
    ReconstructionResult,
    collapse_consecutive_arcs,
    compute_route_continuity,
    reconstruct_route,
)

__all__ = [
    "EmissionModel",
    "compute_emission_log_probabilities",
    "TransitionModel",
    "compute_gps_distances",
    "viterbi_decode",
    "ReconstructionResult",
    "collapse_consecutive_arcs",
    "compute_route_continuity",
    "reconstruct_route",
]
