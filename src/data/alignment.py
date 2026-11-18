"""
PathSnap Alignment and Candidate Generation Module.

Implements spatial candidate generation, candidate-to-observation projection,
and rigorous alignment between GPS observations and reference route sequences.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from src.data.loader import TripData


def project_point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cos_lat: float
) -> Tuple[float, float, float, float]:
    """
    Project point P onto directed segment AB in local metric space.

    Args:
        px, py: GPS point longitude, latitude
        ax, ay: Segment start node longitude, latitude
        bx, by: Segment end node longitude, latitude
        cos_lat: Cosine of local mean latitude for spherical metric scaling

    Returns:
        Tuple of (perpendicular_dist_m, projection_ratio_u, proj_lon, proj_lat)
        where u in [0, 1] represents position along the segment.
    """
    # Convert angular differences to meters
    abx = (bx - ax) * cos_lat * 111320.0
    aby = (by - ay) * 111320.0
    apx = (px - ax) * cos_lat * 111320.0
    apy = (py - ay) * 111320.0

    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-12:
        dist_m = float(np.sqrt(apx * apx + apy * apy))
        return dist_m, 0.0, ax, ay

    # Orthogonal projection scalar u
    u = (apx * abx + apy * aby) / seg_len_sq
    u_clamped = max(0.0, min(1.0, float(u)))

    # Projected point in local metric coordinates
    proj_mx = u_clamped * abx
    proj_my = u_clamped * aby

    # Distance from P to projected point on segment
    dx = apx - proj_mx
    dy = apy - proj_my
    dist_m = float(np.sqrt(dx * dx + dy * dy))

    # Convert projected position back to lon/lat
    proj_lon = ax + (proj_mx / (cos_lat * 111320.0))
    proj_lat = ay + (proj_my / 111320.0)

    return dist_m, u_clamped, proj_lon, proj_lat


def align_gps_to_route_unconstrained(
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align each GPS observation to the spatially closest arc in the reference route.

    Returns:
        matched_arc_ids: shape (N,) int32
        matched_route_indices: shape (N,) int32 (index into route array)
        projection_distances: shape (N,) float64 (meters)
    """
    N = len(track)
    M = len(route)
    matched_arcs = np.zeros(N, dtype=np.int32)
    matched_route_indices = np.zeros(N, dtype=np.int32)
    distances = np.zeros(N, dtype=np.float64)

    mean_lat = np.mean(track[:, 1])
    cos_lat = np.cos(np.radians(mean_lat))

    # Pre-extract segment endpoints for route arcs
    route_endpoints = []
    for arc_id in route:
        u_node, v_node = arcs[arc_id]
        ax, ay = nodes[u_node]
        bx, by = nodes[v_node]
        route_endpoints.append((ax, ay, bx, by))

    for i in range(N):
        px, py = track[i, 0], track[i, 1]
        best_dist = 1e12
        best_r_idx = 0

        for r_idx, (ax, ay, bx, by) in enumerate(route_endpoints):
            d, _, _, _ = project_point_to_segment(px, py, ax, ay, bx, by, cos_lat)
            if d < best_dist:
                best_dist = d
                best_r_idx = r_idx

        matched_arcs[i] = route[best_r_idx]
        matched_route_indices[i] = best_r_idx
        distances[i] = best_dist

    return matched_arcs, matched_route_indices, distances


def align_gps_to_route_monotonic(
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    base_lookahead: int = 15,
    distance_threshold_m: float = 60.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align each GPS observation to route arcs enforcing forward monotonic progression
    with dynamic lookahead and adaptive re-acquisition on large GPS gaps.

    Args:
        track: shape (N, 3) [lon, lat, timestamp]
        nodes: shape (V, 2) [lon, lat]
        arcs: shape (E, 2) [from_node, to_node]
        route: shape (M,) [arc_id]
        base_lookahead: Base forward segment search window
        distance_threshold_m: Max expected orthogonal distance before re-acquisition

    Returns:
        matched_arc_ids: shape (N,) int32
        matched_route_indices: shape (N,) int32
        projection_distances: shape (N,) float64 (meters)
    """
    N = len(track)
    M = len(route)
    matched_arcs = np.zeros(N, dtype=np.int32)
    matched_route_indices = np.zeros(N, dtype=np.int32)
    distances = np.zeros(N, dtype=np.float64)

    mean_lat = np.mean(track[:, 1])
    cos_lat = np.cos(np.radians(mean_lat))

    route_endpoints = []
    for arc_id in route:
        u_node, v_node = arcs[arc_id]
        ax, ay = nodes[u_node]
        bx, by = nodes[v_node]
        route_endpoints.append((ax, ay, bx, by))

    curr_r_idx = 0

    for i in range(N):
        px, py = track[i, 0], track[i, 1]
        dt = (track[i, 2] - track[i - 1, 2]) if i > 0 else 1.0

        # Dynamic lookahead: scale window with elapsed time gap dt
        # Assuming maximum speed ~35 m/s (~125 km/h) and min arc length ~10m
        dynamic_lookahead = max(base_lookahead, int(dt * 3.5))
        window_start = max(0, curr_r_idx - 2)
        window_end = min(M, curr_r_idx + dynamic_lookahead)

        best_dist = 1e12
        best_r_idx = curr_r_idx

        for r_idx in range(window_start, window_end):
            ax, ay, bx, by = route_endpoints[r_idx]
            d, _, _, _ = project_point_to_segment(px, py, ax, ay, bx, by, cos_lat)
            if d < best_dist:
                best_dist = d
                best_r_idx = r_idx

        # Adaptive Re-acquisition: if best distance is poor (> distance_threshold_m)
        # or large time gap occurred, search broader forward route to prevent getting stuck
        if best_dist > distance_threshold_m and dt > 5.0:
            broad_end = min(M, curr_r_idx + max(100, dynamic_lookahead * 3))
            for r_idx in range(curr_r_idx, broad_end):
                ax, ay, bx, by = route_endpoints[r_idx]
                d, _, _, _ = project_point_to_segment(px, py, ax, ay, bx, by, cos_lat)
                if d < best_dist:
                    best_dist = d
                    best_r_idx = r_idx

        curr_r_idx = max(curr_r_idx, best_r_idx)
        matched_arcs[i] = route[best_r_idx]
        matched_route_indices[i] = best_r_idx
        distances[i] = best_dist

    return matched_arcs, matched_route_indices, distances


def evaluate_candidate_recall(
    trip: TripData,
    search_radii: List[float] = [15.0, 25.0, 35.0, 50.0, 75.0, 100.0]
) -> Dict[float, float]:
    """
    Evaluate candidate recall (% of observations where the true route arc
    is included within radius R) across multiple candidate search radii.

    Returns:
        Dictionary mapping radius (meters) -> recall percentage.
    """
    if trip.route is None:
        raise ValueError("Cannot evaluate candidate recall without reference route.")

    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    # Align observations to true route arcs using monotonic alignment
    true_arcs, _, _ = align_gps_to_route_monotonic(track, nodes, arcs, route)

    mean_lat = np.mean(track[:, 1])
    cos_lat = np.cos(np.radians(mean_lat))

    # Compute distances from each GPS point to its aligned true arc
    true_arc_distances = []
    for i in range(len(track)):
        true_arc = true_arcs[i]
        u_node, v_node = arcs[true_arc]
        ax, ay = nodes[u_node]
        bx, by = nodes[v_node]
        px, py = track[i, 0], track[i, 1]
        d, _, _, _ = project_point_to_segment(px, py, ax, ay, bx, by, cos_lat)
        true_arc_distances.append(d)

    true_arc_distances = np.array(true_arc_distances)

    recall_by_radius = {}
    for r in search_radii:
        recall = float((true_arc_distances <= r).mean() * 100.0)
        recall_by_radius[r] = round(recall, 2)

    return recall_by_radius
