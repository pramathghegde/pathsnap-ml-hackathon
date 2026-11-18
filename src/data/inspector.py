"""
PathSnap Dataset Inspector Module.

Performs exhaustive verification of dataset integrity, coordinates,
timestamp regularity, road network graph topology, and route continuity.
Compiles a consolidated dataset inventory across all partitions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import networkx as nx

from src.data.loader import (
    load_track,
    load_nodes,
    load_arcs,
    load_route,
    load_trip,
    TripData
)


def compute_geodesic_distance_m(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """
    Compute geodesic distance in meters using the Haversine formula.
    """
    R = 6371000.0  # Earth mean radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)


def inspect_single_trip(
    trip_dir: Union[str, Path], partition: str = "train"
) -> Dict[str, Union[str, int, float, bool]]:
    """
    Perform deep inspection of a single trip package.

    Returns:
        Dictionary of computed metrics and data-quality indicators.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    # 1. Load data
    trip = load_trip(trip_dir, partition=partition)
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    # 2. Coordinate & value sanity
    nan_track = bool(np.isnan(track).any())
    nan_nodes = bool(np.isnan(nodes).any())
    nan_arcs = bool(np.isnan(arcs).any())

    lon_min, lon_max = float(track[:, 0].min()), float(track[:, 0].max())
    lat_min, lat_max = float(track[:, 1].min()), float(track[:, 1].max())

    valid_coords = (
        -180.0 <= lon_min <= 180.0
        and -180.0 <= lon_max <= 180.0
        and -90.0 <= lat_min <= 90.0
        and -90.0 <= lat_max <= 90.0
    )

    # 3. Timestamp analysis
    timestamps = track[:, 2]
    duration_s = float(timestamps[-1] - timestamps[0])
    dt = np.diff(timestamps)
    is_strictly_monotonic = bool((dt > 0).all())
    dt_mean = float(np.mean(dt))
    dt_median = float(np.median(dt))
    dt_std = float(np.std(dt))
    dt_min = float(np.min(dt))
    dt_max = float(np.max(dt))
    dt_1s_pct = float(np.isclose(dt, 1.0, atol=0.05).mean() * 100.0)

    # Trajectory length and speed
    step_dists = [
        compute_geodesic_distance_m(
            track[i, 0], track[i, 1], track[i + 1, 0], track[i + 1, 1]
        )
        for i in range(len(track) - 1)
    ]
    total_distance_m = float(sum(step_dists))
    trajectory_length_km = total_distance_m / 1000.0
    avg_speed_kmh = (
        (trajectory_length_km / (duration_s / 3600.0)) if duration_s > 0 else 0.0
    )

    # 4. Graph topology inspection
    num_nodes = len(nodes)
    num_arcs = len(arcs)

    # Index boundaries verification
    max_node_in_arcs = int(arcs.max())
    min_node_in_arcs = int(arcs.min())
    arcs_indices_valid = bool(0 <= min_node_in_arcs and max_node_in_arcs < num_nodes)

    # In-degree / out-degree calculation
    in_degrees = np.zeros(num_nodes, dtype=np.int32)
    out_degrees = np.zeros(num_nodes, dtype=np.int32)
    for u, v in arcs:
        out_degrees[u] += 1
        in_degrees[v] += 1

    isolated_nodes = int(((in_degrees == 0) & (out_degrees == 0)).sum())
    source_nodes = int(((in_degrees == 0) & (out_degrees > 0)).sum())
    sink_nodes = int(((in_degrees > 0) & (out_degrees == 0)).sum())

    # Build graph for component analysis
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((int(u), int(v)) for u, v in arcs)
    num_wcc = nx.number_weakly_connected_components(G)
    num_scc = nx.number_strongly_connected_components(G)

    # 5. Route analysis
    num_route = len(route) if route is not None else 0
    route_valid = False
    route_continuous_pct = 0.0

    if route is not None and num_route > 0:
        max_arc_in_route = int(route.max())
        min_arc_in_route = int(route.min())
        route_valid = bool(0 <= min_arc_in_route and max_arc_in_route < num_arcs)

        # Check topological continuity: does arc[t].to_node == arc[t+1].from_node?
        if num_route > 1:
            continuous_steps = sum(
                1 for i in range(num_route - 1)
                if arcs[route[i], 1] == arcs[route[i + 1], 0]
            )
            route_continuous_pct = float(continuous_steps / (num_route - 1) * 100.0)
        else:
            route_continuous_pct = 100.0

    return {
        "trip_id": trip_id,
        "partition": partition,
        "track_points": len(track),
        "duration_seconds": duration_s,
        "dt_mean": round(dt_mean, 4),
        "dt_median": round(dt_median, 4),
        "dt_std": round(dt_std, 4),
        "dt_min": round(dt_min, 4),
        "dt_max": round(dt_max, 4),
        "dt_1s_pct": round(dt_1s_pct, 2),
        "is_monotonic_ts": is_strictly_monotonic,
        "trajectory_length_km": round(trajectory_length_km, 3),
        "avg_speed_kmh": round(avg_speed_kmh, 2),
        "lon_min": round(lon_min, 6),
        "lon_max": round(lon_max, 6),
        "lat_min": round(lat_min, 6),
        "lat_max": round(lat_max, 6),
        "valid_coords": valid_coords,
        "has_nan": nan_track or nan_nodes or nan_arcs,
        "nodes_count": num_nodes,
        "arcs_count": num_arcs,
        "arcs_indices_valid": arcs_indices_valid,
        "isolated_nodes": isolated_nodes,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
        "weakly_connected_components": num_wcc,
        "strongly_connected_components": num_scc,
        "route_arcs": num_route,
        "route_indices_valid": route_valid,
        "route_continuity_pct": round(route_continuous_pct, 2)
    }


def scan_entire_dataset(
    data_raw_dir: Union[str, Path]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Scan all partitions in data_raw_dir and return a consolidated inventory.

    Args:
        data_raw_dir: Path to directory containing partition folders ('train', 'student_test')

    Returns:
        Tuple of (inventory_dataframe, partitions_dict)
    """
    data_raw_dir = Path(data_raw_dir)
    records = []
    partitions_found = {}

    for partition in ["train", "student_test"]:
        part_dir = data_raw_dir / partition
        if not part_dir.is_dir():
            continue

        trip_dirs = sorted([d for d in part_dir.iterdir() if d.is_dir()])
        partitions_found[partition] = [d.name for d in trip_dirs]

        for trip_dir in trip_dirs:
            info = inspect_single_trip(trip_dir, partition=partition)
            records.append(info)

    # Check for reserved trips out of 00000000..00000099
    all_seen = set()
    for trips in partitions_found.values():
        all_seen.update(trips)

    universe_100 = {f"{i:08d}" for i in range(100)}
    reserved_trips = sorted(list(universe_100 - all_seen))
    partitions_found["evaluator_reserved"] = reserved_trips

    df = pd.DataFrame(records)
    return df, partitions_found
