"""
PathSnap Dataset Loader Module.

Provides robust, typed parsers and data structures for raw PathSnap trip files:
- .track: GPS observations (longitude, latitude, timestamp_seconds)
- .nodes: Road network vertices (longitude, latitude)
- .arcs: Road network directed edges (from_node_id, to_node_id)
- .route: Reference ground-truth traversed arc IDs
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import networkx as nx


@dataclass
class TrajectoryPoint:
    """Single GPS observation from a .track file."""
    longitude: float
    latitude: float
    timestamp: float


@dataclass
class RoadNode:
    """Road network vertex from a .nodes file."""
    node_id: int
    longitude: float
    latitude: float


@dataclass
class RoadArc:
    """Directed road network segment from an .arcs file."""
    arc_id: int
    from_node: int
    to_node: int


@dataclass
class TripData:
    """Complete data container for a single PathSnap trip."""
    trip_id: str
    partition: str
    track: np.ndarray          # shape (N, 3): [lon, lat, timestamp]
    nodes: np.ndarray          # shape (V, 2): [lon, lat]
    arcs: np.ndarray           # shape (E, 2): [from_node, to_node]
    route: Optional[np.ndarray] # shape (M,): [arc_id] (None if unlabelled/withheld)

    @property
    def num_observations(self) -> int:
        return len(self.track)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)

    @property
    def num_route_segments(self) -> int:
        return len(self.route) if self.route is not None else 0


def load_track(filepath: Union[str, Path]) -> np.ndarray:
    """
    Parse a .track file.

    Format: 3 whitespace-separated columns (no headers):
    <longitude> <latitude> <timestamp_seconds>

    Returns:
        np.ndarray of shape (N, 3) with float64 values.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Track file not found: {filepath}")
    
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed line {line_num} in {filepath.name}: expected 3 columns, got {len(parts)}"
                )
            data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
    if not data:
        raise ValueError(f"Track file is empty: {filepath}")
        
    return np.array(data, dtype=np.float64)


def load_nodes(filepath: Union[str, Path]) -> np.ndarray:
    """
    Parse a .nodes file.

    Format: 2 whitespace-separated columns (no headers):
    <longitude> <latitude>
    The zero-based line index is the implicit node_id.

    Returns:
        np.ndarray of shape (V, 2) with float64 coordinates.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Nodes file not found: {filepath}")

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line {line_num} in {filepath.name}: expected 2 columns, got {len(parts)}"
                )
            data.append([float(parts[0]), float(parts[1])])

    if not data:
        raise ValueError(f"Nodes file is empty: {filepath}")

    return np.array(data, dtype=np.float64)


def load_arcs(filepath: Union[str, Path]) -> np.ndarray:
    """
    Parse an .arcs file.

    Format: 2 integer columns (no headers):
    <from_node_id> <to_node_id>
    The zero-based line index is the implicit arc_id.

    Returns:
        np.ndarray of shape (E, 2) with int32 node indices.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Arcs file not found: {filepath}")

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line {line_num} in {filepath.name}: expected 2 columns, got {len(parts)}"
                )
            data.append([int(parts[0]), int(parts[1])])

    if not data:
        raise ValueError(f"Arcs file is empty: {filepath}")

    return np.array(data, dtype=np.int32)


def load_route(filepath: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Parse a .route file.

    Format: 1 integer column (no headers):
    <arc_id>
    Represents the ground-truth sequence of traversed arcs.

    Returns:
        np.ndarray of shape (M,) with int32 arc IDs, or None if file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        return None

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(int(line))

    return np.array(data, dtype=np.int32)


def load_trip(trip_dir: Union[str, Path], partition: str = "train") -> TripData:
    """
    Load all components of a single trip directory.

    Args:
        trip_dir: Path to directory containing <trip_id>.{track, nodes, arcs, route}
        partition: Dataset partition name ('train', 'student_test', 'evaluator_test')

    Returns:
        Populated TripData dataclass.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    track = load_track(track_file)
    nodes = load_nodes(nodes_file)
    arcs = load_arcs(arcs_file)
    route = load_route(route_file)

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route
    )


def build_network_graph(nodes: np.ndarray, arcs: np.ndarray) -> nx.DiGraph:
    """
    Construct a NetworkX directed graph from parsed nodes and arcs.

    Each node has attributes: 'lon', 'lat'
    Each edge has attributes: 'arc_id', 'length_m'

    Returns:
        nx.DiGraph instance.
    """
    G = nx.DiGraph()

    # Add all nodes
    for node_id, (lon, lat) in enumerate(nodes):
        G.add_node(node_id, lon=lon, lat=lat)

    # Local spherical metric distance calculation
    mean_lat = np.mean(nodes[:, 1])
    cos_lat = np.cos(np.radians(mean_lat))

    # Add all directed arcs
    for arc_id, (u, v) in enumerate(arcs):
        p1 = nodes[u]
        p2 = nodes[v]
        dx = (p2[0] - p1[0]) * cos_lat * 111320.0
        dy = (p2[1] - p1[1]) * 111320.0
        length_m = float(np.sqrt(dx * dx + dy * dy))
        G.add_edge(int(u), int(v), arc_id=int(arc_id), length_m=length_m)

    return G
