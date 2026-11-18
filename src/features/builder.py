"""
PathSnap Feature Generation Pipeline Module.

Constructs candidate-level observation features for map matching:
- Uniform 2D Spatial Grid Indexing for sub-millisecond candidate retrieval
- Invariant geometric, kinematic, and topological features with exact physical units
- Audited route alignment with dynamic lookahead
- Candidate miss tracking and training eligibility flags
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.data.loader import TripData
from src.data.alignment import (
    align_gps_to_route_monotonic,
    project_point_to_segment
)

# Reference urban speed in m/s (50 km/h) for dimensionless speed ratio
URBAN_REFERENCE_SPEED_MPS = 13.8889


class SpatialGridIndex:
    """
    Uniform 2D Spatial Grid Index for fast road segment candidate retrieval.
    Transforms coordinates into local metric Cartesian coordinates.
    """

    def __init__(self, nodes: np.ndarray, arcs: np.ndarray, cell_size_m: float = 50.0):
        self.cell_size = cell_size_m
        self.mean_lat = float(np.mean(nodes[:, 1]))
        self.cos_lat = float(np.cos(np.radians(self.mean_lat)))

        # Convert all nodes to local metric coordinates (meters)
        self.mx = nodes[:, 0] * self.cos_lat * 111320.0
        self.my = nodes[:, 1] * 111320.0

        self.arcs = arcs
        self.nodes = nodes
        self.grid: Dict[Tuple[int, int], List[int]] = {}

        # Precompute arc lengths (m) and heading angles (rad)
        self.arc_lengths = np.zeros(len(arcs), dtype=np.float64)
        self.arc_headings = np.zeros(len(arcs), dtype=np.float64)

        for arc_id, (u, v) in enumerate(arcs):
            x1, y1 = self.mx[u], self.my[u]
            x2, y2 = self.mx[v], self.my[v]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.sqrt(dx * dx + dy * dy))
            self.arc_lengths[arc_id] = length
            # Heading in radians [-pi, pi]
            self.arc_headings[arc_id] = float(np.arctan2(dy, dx))

            # Populate grid cells intersecting arc bounding box
            min_cx = int(min(x1, x2) // self.cell_size)
            max_cx = int(max(x1, x2) // self.cell_size)
            min_cy = int(min(y1, y2) // self.cell_size)
            max_cy = int(max(y1, y2) // self.cell_size)

            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    self.grid.setdefault((cx, cy), []).append(arc_id)

    def query_radius(
        self, px_deg: float, py_deg: float, radius_m: float = 50.0
    ) -> List[int]:
        """Query all unique arcs in grid cells overlapping radius_m."""
        px = px_deg * self.cos_lat * 111320.0
        py = py_deg * 111320.0

        cell_radius = int(np.ceil(radius_m / self.cell_size))
        cx = int(px // self.cell_size)
        cy = int(py // self.cell_size)

        candidate_set: Set[int] = set()
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_arcs = self.grid.get((cx + dx, cy + dy))
                if cell_arcs:
                    candidate_set.update(cell_arcs)

        return list(candidate_set)


def compute_angle_difference(angle1: float, angle2: float) -> float:
    """Compute minimal absolute difference between two angles in radians [0, pi]."""
    diff = abs(angle1 - angle2) % (2.0 * np.pi)
    return float(2.0 * np.pi - diff if diff > np.pi else diff)


def extract_trip_features(
    trip: TripData,
    spatial_radius_m: float = 50.0,
    max_candidates_per_obs: int = 15
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """
    Extract candidate-level feature matrix for a single trip.

    Returns:
        DataFrame with candidate rows and metrics dictionary.
    """
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    N = len(track)
    index = SpatialGridIndex(nodes, arcs, cell_size_m=spatial_radius_m)
    cos_lat = index.cos_lat

    # Align ground-truth route to observations if route exists
    true_arc_for_obs = None
    if route is not None:
        true_arc_for_obs, _, _ = align_gps_to_route_monotonic(
            track, nodes, arcs, route
        )

    # Precompute GPS kinematics (velocities and headings)
    gps_speeds = np.zeros(N, dtype=np.float64)
    gps_headings = np.zeros(N, dtype=np.float64)

    for i in range(N):
        if i == 0:
            if N > 1:
                dx = (track[1, 0] - track[0, 0]) * cos_lat * 111320.0
                dy = (track[1, 1] - track[0, 1]) * 111320.0
                dt = max(0.1, track[1, 2] - track[0, 2])
                dist = np.sqrt(dx * dx + dy * dy)
                gps_speeds[0] = dist / dt
                gps_headings[0] = np.arctan2(dy, dx)
        else:
            dx = (track[i, 0] - track[i - 1, 0]) * cos_lat * 111320.0
            dy = (track[i, 1] - track[i - 1, 1]) * 111320.0
            dt = max(0.1, track[i, 2] - track[i - 1, 2])
            dist = np.sqrt(dx * dx + dy * dy)
            gps_speeds[i] = dist / dt
            gps_headings[i] = np.arctan2(dy, dx)

    rows = []
    total_observations = N
    missed_observations = 0
    prev_top_candidate_nodes: Set[int] = set()

    for obs_id in range(N):
        px, py = track[obs_id, 0], track[obs_id, 1]
        v_gps = gps_speeds[obs_id]
        theta_gps = gps_headings[obs_id]

        # 1. Query spatial candidates
        candidate_arc_ids = index.query_radius(px, py, radius_m=spatial_radius_m)

        # Fallback expansion if empty
        if not candidate_arc_ids:
            candidate_arc_ids = index.query_radius(px, py, radius_m=spatial_radius_m * 2.0)

        # 2. Compute candidate geometric metrics
        cand_metrics = []
        for arc_id in candidate_arc_ids:
            u, v = arcs[arc_id]
            ax, ay = nodes[u]
            bx, by = nodes[v]
            d_perp, u_ratio, _, _ = project_point_to_segment(
                px, py, ax, ay, bx, by, cos_lat
            )
            cand_metrics.append((arc_id, d_perp, u_ratio, u, v))

        # Sort candidates by perpendicular distance
        cand_metrics.sort(key=lambda item: item[1])
        cand_metrics = cand_metrics[:max_candidates_per_obs]

        # 3. Check candidate recall against true route arc
        true_arc = true_arc_for_obs[obs_id] if true_arc_for_obs is not None else None
        cand_arcs_present = [item[0] for item in cand_metrics]

        has_positive = bool(true_arc is not None and true_arc in cand_arcs_present)
        if true_arc is not None and not has_positive:
            missed_observations += 1

        curr_cand_nodes: Set[int] = set()

        # 4. Construct candidate feature rows
        for rank, (arc_id, d_perp, u_ratio, u_node, v_node) in enumerate(cand_metrics, start=1):
            theta_arc = index.arc_headings[arc_id]
            delta_heading = compute_angle_difference(theta_gps, theta_arc)
            arc_len = index.arc_lengths[arc_id]

            # Topological adjacency: connects to previous top candidate nodes
            is_adjacent = int(u_node in prev_top_candidate_nodes or v_node in prev_top_candidate_nodes)

            # Plausibility features (with rigorous units):
            # speed_ratio: dimensionless ratio of GPS speed to 50 km/h reference speed
            speed_ratio = v_gps / URBAN_REFERENCE_SPEED_MPS
            # traversal_rate: inverse seconds (s^-1) indicating segment traversal frequency
            traversal_rate = v_gps / max(1.0, arc_len)

            # Label assignment: 1 strictly when candidate matches true arc
            is_match = 1 if (true_arc is not None and arc_id == true_arc) else 0

            rows.append({
                "trip_id": trip.trip_id,
                "observation_id": obs_id,
                "candidate_arc_id": arc_id,
                "perp_dist": round(d_perp, 4),
                "proj_ratio": round(u_ratio, 4),
                "delta_heading": round(delta_heading, 4),
                "arc_length": round(arc_len, 2),
                "gps_speed": round(v_gps, 2),
                "speed_ratio": round(speed_ratio, 4),
                "traversal_rate": round(traversal_rate, 6),
                "candidate_rank": rank,
                "is_adjacent": is_adjacent,
                "is_match": is_match,
                "has_positive_candidate": int(has_positive)
            })

            # Record nodes of top-3 candidates for next step
            if rank <= 3:
                curr_cand_nodes.update([u_node, v_node])

        prev_top_candidate_nodes = curr_cand_nodes

    df_features = pd.DataFrame(rows)
    miss_rate = (missed_observations / total_observations) * 100.0 if total_observations > 0 else 0.0

    metrics = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "total_observations": total_observations,
        "total_candidate_rows": len(df_features),
        "missed_observations": missed_observations,
        "candidate_miss_rate": round(miss_rate, 2),
        "candidate_recall": round(100.0 - miss_rate, 2),
        "avg_candidates_per_obs": round(len(df_features) / total_observations, 2)
    }

    return df_features, metrics
