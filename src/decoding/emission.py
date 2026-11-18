"""
Candidate Emission Likelihood Module for PathSnap Map Matching.

Transforms predictive machine learning candidate classification probabilities
p(is_match == 1) into calibrated log-likelihood emission probabilities p(o_t | a_j)
for Hidden Markov Model (HMM) sequence decoding.

Features:
- Bayes' rule inversion with uniform spatial candidate prior assumption
- Numerically stable log-domain transformation with configurable clipping (epsilon = 1e-7)
- Per-observation probability normalization: sum_j p(o_t | a_j) = 1.0
- Spatial distance fallback for candidate misses or missing model predictions
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


DEFAULT_MIN_PROB: float = 1e-7
DEFAULT_SPATIAL_SIGMA_Z: float = 8.0  # GPS measurement error standard deviation in meters


@dataclass
class EmissionModel:
    """
    Configurable emission probability model for HMM map matching.
    """
    min_prob: float = DEFAULT_MIN_PROB
    normalize_across_candidates: bool = True
    sigma_z: float = DEFAULT_SPATIAL_SIGMA_Z

    def compute_log_emission_from_prob(self, p: float) -> float:
        """
        Clip probability to [min_prob, 1.0] and return natural logarithm.

        Args:
            p: Predicted match probability.

        Returns:
            Log-domain emission probability.
        """
        safe_p = max(self.min_prob, min(1.0, float(p)))
        return float(np.log(safe_p))

    def compute_spatial_log_emission(self, perp_dist: float) -> float:
        """
        Compute Gaussian emission log likelihood from perpendicular GPS-to-arc distance:
            p(o_t | a_j) = 1 / (sqrt(2*pi) * sigma) * exp(-d_perp^2 / (2 * sigma^2))

        Args:
            perp_dist: Perpendicular distance in meters.

        Returns:
            Log-domain spatial emission likelihood.
        """
        d = max(0.0, float(perp_dist))
        # log(1 / (sqrt(2*pi)*sigma)) - d^2 / (2 * sigma^2)
        norm_const = -0.5 * np.log(2.0 * np.pi * (self.sigma_z ** 2))
        exponent = - (d ** 2) / (2.0 * (self.sigma_z ** 2))
        return float(norm_const + exponent)


def compute_emission_log_probabilities(
    scored_candidates: pd.DataFrame,
    prob_col: str = "match_probability",
    obs_col: str = "observation_id",
    arc_col: str = "candidate_arc_id",
    min_prob: float = DEFAULT_MIN_PROB,
    normalize: bool = True,
    fallback_dist_col: Optional[str] = "perp_dist",
    sigma_z: float = DEFAULT_SPATIAL_SIGMA_Z,
) -> Dict[int, Dict[int, float]]:
    """
    Transform scored candidate DataFrame into nested log emission dictionary:
        {observation_id: {candidate_arc_id: log_probability}}

    Args:
        scored_candidates: DataFrame containing observation and candidate arc IDs,
                           alongside predicted match probabilities.
        prob_col: Column name containing predicted positive class probability.
        obs_col: Column name identifying GPS observation ID.
        arc_col: Column name identifying candidate road arc ID.
        min_prob: Numerical clipping floor to prevent log(0) / -inf.
        normalize: If True, normalizes probabilities across candidates for each
                   observation before computing logarithms.
        fallback_dist_col: Optional column name for perpendicular distance fallback
                           when prob_col is unavailable or NaN.
        sigma_z: GPS measurement standard deviation (meters) for fallback.

    Returns:
        Dictionary mapping each observation ID to a dictionary of {arc_id: log_prob}.
    """
    if scored_candidates is None or scored_candidates.empty:
        return {}

    if obs_col not in scored_candidates.columns:
        raise KeyError(
            f"Observation column '{obs_col}' not found in candidate DataFrame."
        )
    if arc_col not in scored_candidates.columns:
        raise KeyError(
            f"Candidate arc column '{arc_col}' not found in candidate DataFrame."
        )

    has_prob = prob_col in scored_candidates.columns
    has_fallback = (
        fallback_dist_col is not None and fallback_dist_col in scored_candidates.columns
    )

    if not has_prob and not has_fallback:
        raise KeyError(
            f"Neither probability column '{prob_col}' nor fallback distance column "
            f"'{fallback_dist_col}' found in candidate DataFrame."
        )

    emission_dict: Dict[int, Dict[int, float]] = {}

    # Group by observation ID
    grouped = scored_candidates.groupby(obs_col, sort=True)

    for obs_id, group in grouped:
        obs_int = int(obs_id)
        arc_ids = group[arc_col].values.astype(int)
        k = len(arc_ids)

        if k == 0:
            emission_dict[obs_int] = {}
            continue

        if has_prob:
            raw_probs = group[prob_col].values.astype(np.float64)
            # Replace NaNs with min_prob
            raw_probs = np.nan_to_num(raw_probs, nan=min_prob)
            # Clip below to min_prob
            clipped_probs = np.maximum(raw_probs, min_prob)

            if normalize:
                prob_sum = float(np.sum(clipped_probs))
                if prob_sum > 0.0 and np.isfinite(prob_sum):
                    norm_probs = clipped_probs / prob_sum
                else:
                    norm_probs = np.full(k, 1.0 / k, dtype=np.float64)
                # Clip normalized probabilities
                final_probs = np.maximum(norm_probs, min_prob)
                log_probs = np.log(final_probs)
            else:
                log_probs = np.log(clipped_probs)

        else:
            # Fallback to perpendicular distance Gaussian likelihood
            dists = group[fallback_dist_col].values.astype(np.float64)
            dists = np.nan_to_num(dists, nan=50.0)
            log_probs = np.array([
                -0.5 * np.log(2.0 * np.pi * (sigma_z ** 2)) - (d ** 2) / (2.0 * (sigma_z ** 2))
                for d in dists
            ], dtype=np.float64)

            if normalize:
                # Log-sum-exp normalization
                max_log = float(np.max(log_probs))
                sum_exp = float(np.sum(np.exp(log_probs - max_log)))
                log_probs = log_probs - (max_log + np.log(sum_exp))

        obs_dict: Dict[int, float] = {}
        for arc_id, log_p in zip(arc_ids, log_probs):
            obs_dict[int(arc_id)] = float(log_p)

        emission_dict[obs_int] = obs_dict

    return emission_dict
