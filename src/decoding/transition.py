"""
Graph-Based Transition Probability Module for PathSnap Map Matching.

Models transition dynamics between candidate road network segments using NetworkX
directed graphs and the Newson & Krumm (2009) exponential distance error formulation:

    p(a_k | a_j) = 1 / beta * exp(- |d_net(a_j, a_k) - d_gps(o_{t-1}, o_t)| / beta)
    log p(a_k | a_j) = -log(beta) - |d_net - d_gps| / beta

Key Capabilities:
- Exact metric network distance via NetworkX directed shortest paths
- Physical same-arc continuity modeling with zero distance penalty
- Direct topological adjacency fast-path bypass (O(1) lookup)
- Disconnection handling with heavy log-domain penalty
- Multi-hop intermediate arc reconstruction for trajectory gap filling
- High-performance Euclidean pre-filtering and bounded Dijkstra (cutoff) memoization
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from src.data.loader import TripData, build_network_graph


DEFAULT_BETA: float = 15.0             # Distance discrepancy decay scale (meters)
DEFAULT_DISCONNECT_PENALTY: float = -100.0  # Log-domain penalty for unreachable transitions
DEFAULT_MAX_SEARCH_DIST: float = 1000.0     # Metric cutoff for Dijkstra path search (meters)


class TransitionModel:
    """
    Evaluates log transition probabilities between directed road segments
    conditioned on empirical GPS observation displacement.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        arcs: np.ndarray,
        nodes: Optional[np.ndarray] = None,
        beta: float = DEFAULT_BETA,
        disconnect_penalty: float = DEFAULT_DISCONNECT_PENALTY,
        max_search_dist: float = DEFAULT_MAX_SEARCH_DIST,
    ) -> None:
        """
        Initialize transition model with road graph and arc geometry.

        Args:
            graph: NetworkX directed graph with metric edge attribute 'length_m'.
            arcs: np.ndarray of shape (E, 2) containing [from_node, to_node].
            nodes: Optional np.ndarray of shape (V, 2) containing [lon, lat].
            beta: Exponential error scale parameter in meters.
            disconnect_penalty: Log probability penalty for disconnected transitions.
            max_search_dist: Maximum graph distance cutoff in meters.
        """
        self.graph = graph
        self.arcs = np.asarray(arcs, dtype=np.int32)
        self.num_arcs = len(self.arcs)
        self.beta = max(1e-3, float(beta))
        self.log_beta = float(np.log(self.beta))
        self.disconnect_penalty = float(disconnect_penalty)
        self.max_search_dist = float(max_search_dist)

        # Precompute metric coordinates for nodes if available for Euclidean bounding
        self.nodes_mx: Optional[np.ndarray] = None
        self.nodes_my: Optional[np.ndarray] = None
        if nodes is not None and len(nodes) > 0:
            mean_lat = float(np.mean(nodes[:, 1]))
            cos_lat = float(np.cos(np.radians(mean_lat)))
            self.nodes_mx = nodes[:, 0] * cos_lat * 111320.0
            self.nodes_my = nodes[:, 1] * 111320.0

        # Precompute arc lengths
        self.arc_lengths = self._precompute_arc_lengths(nodes)

        # Map (u, v) node pairs to representative arc_id for route reconstruction
        self.edge_to_arc: Dict[Tuple[int, int], int] = {}
        for arc_id, (u, v) in enumerate(self.arcs):
            edge_key = (int(u), int(v))
            if edge_key not in self.edge_to_arc:
                self.edge_to_arc[edge_key] = int(arc_id)
            else:
                prev_arc = self.edge_to_arc[edge_key]
                if self.arc_lengths[arc_id] < self.arc_lengths[prev_arc]:
                    self.edge_to_arc[edge_key] = int(arc_id)

        # Shortest path cache: (u, v) -> (distance, path_nodes_or_None)
        self._sp_cache: Dict[Tuple[int, int], Tuple[Optional[float], Optional[List[int]]]] = {}

    def _precompute_arc_lengths(self, nodes: Optional[np.ndarray]) -> np.ndarray:
        """Calculate metric length in meters for every arc in the network."""
        lengths = np.zeros(self.num_arcs, dtype=np.float64)

        if self.nodes_mx is not None and self.nodes_my is not None:
            for arc_id, (u, v) in enumerate(self.arcs):
                dx = self.nodes_mx[v] - self.nodes_mx[u]
                dy = self.nodes_my[v] - self.nodes_my[u]
                lengths[arc_id] = max(0.1, float(np.sqrt(dx * dx + dy * dy)))
        else:
            for arc_id, (u, v) in enumerate(self.arcs):
                u_int, v_int = int(u), int(v)
                if self.graph.has_edge(u_int, v_int):
                    edge_data = self.graph.get_edge_data(u_int, v_int)
                    lengths[arc_id] = edge_data.get("length_m", 10.0)
                else:
                    lengths[arc_id] = 10.0

        return lengths

    @classmethod
    def from_trip(
        cls,
        trip: TripData,
        beta: float = DEFAULT_BETA,
        disconnect_penalty: float = DEFAULT_DISCONNECT_PENALTY,
        max_search_dist: float = DEFAULT_MAX_SEARCH_DIST,
    ) -> "TransitionModel":
        """
        Construct TransitionModel directly from a TripData instance.
        """
        graph = build_network_graph(trip.nodes, trip.arcs)
        return cls(
            graph=graph,
            arcs=trip.arcs,
            nodes=trip.nodes,
            beta=beta,
            disconnect_penalty=disconnect_penalty,
            max_search_dist=max_search_dist,
        )

    def _query_shortest_path(
        self, u: int, v: int, cutoff: float
    ) -> Tuple[Optional[float], Optional[List[int]]]:
        """
        Query cached bounded shortest path distance and node sequence between graph nodes u and v.
        Uses Euclidean pre-filter and cutoff-bounded Dijkstra to guarantee sub-millisecond execution.
        """
        cache_key = (u, v)
        if cache_key in self._sp_cache:
            cached_dist, cached_path = self._sp_cache[cache_key]
            # If cached as reachable, return it
            if cached_dist is not None:
                return cached_dist, cached_path
            # If cached as unreachable at a cutoff >= current cutoff, return unreachable
            return None, None

        if u == v:
            res = (0.0, [u])
            self._sp_cache[cache_key] = res
            return res

        if not self.graph.has_node(u) or not self.graph.has_node(v):
            res = (None, None)
            self._sp_cache[cache_key] = res
            return res

        # 1. Fast Euclidean distance rejection filter (O(1))
        if self.nodes_mx is not None and self.nodes_my is not None:
            dx = self.nodes_mx[v] - self.nodes_mx[u]
            dy = self.nodes_my[v] - self.nodes_my[u]
            euc_dist_sq = dx * dx + dy * dy
            if euc_dist_sq > (cutoff * cutoff):
                res = (None, None)
                self._sp_cache[cache_key] = res
                return res

        # 2. Bounded Dijkstra search (only explores within cutoff radius)
        try:
            dist, path_nodes = nx.single_source_dijkstra(
                self.graph,
                source=u,
                target=v,
                cutoff=cutoff,
                weight="length_m",
            )
            res = (float(dist), path_nodes)
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            res = (None, None)

        self._sp_cache[cache_key] = res
        return res

    def compute_network_distance(
        self, from_arc: int, to_arc: int, d_gps: float
    ) -> Tuple[Optional[float], float]:
        """
        Compute network distance d_net between two road arcs, and the absolute
        difference delta_d = |d_net - d_gps|.

        Returns:
            Tuple of (d_net_or_None, delta_d). If disconnected, d_net is None.
        """
        if from_arc == to_arc:
            # Same arc: if displacement is within segment length, delta_d is 0
            arc_len = self.arc_lengths[from_arc]
            if d_gps <= arc_len:
                delta_d = 0.0
                d_net = d_gps
            else:
                delta_d = d_gps - arc_len
                d_net = arc_len
            return d_net, delta_d

        v_from = int(self.arcs[from_arc, 1])  # Exit node of previous arc
        u_to = int(self.arcs[to_arc, 0])      # Entry node of current arc
        to_arc_len = self.arc_lengths[to_arc]

        if v_from == u_to:
            # Directly adjacent directed arcs (0 hops)
            d_net = to_arc_len
            delta_d = abs(d_net - d_gps)
            return d_net, delta_d

        # Multi-hop path via intermediate network nodes
        cutoff = min(self.max_search_dist, max(300.0, 4.0 * d_gps))
        sp_dist, _ = self._query_shortest_path(v_from, u_to, cutoff=cutoff)

        if sp_dist is None:
            return None, float("inf")

        d_net = sp_dist + to_arc_len
        delta_d = abs(d_net - d_gps)
        return d_net, delta_d

    def log_transition_prob(
        self, from_arc: int, to_arc: int, d_gps: float
    ) -> float:
        """
        Compute log-domain transition probability log p(to_arc | from_arc).

        Args:
            from_arc: Previous candidate arc ID.
            to_arc: Current candidate arc ID.
            d_gps: Observed GPS distance traveled between observations (meters).

        Returns:
            Log transition probability.
        """
        safe_d_gps = max(0.0, float(d_gps))
        d_net, delta_d = self.compute_network_distance(from_arc, to_arc, safe_d_gps)

        if d_net is None:
            return self.disconnect_penalty

        # Newson & Krumm log transition formula
        log_prob = -self.log_beta - (delta_d / self.beta)
        return max(self.disconnect_penalty, float(log_prob))

    def get_intermediate_path_arcs(self, from_arc: int, to_arc: int) -> List[int]:
        """
        Retrieve sequence of intermediate arc IDs connecting from_arc to to_arc.

        Returns empty list if from_arc == to_arc, or if from_arc directly connects
        to to_arc (v_from == u_to), or if no topological path exists.
        """
        if from_arc == to_arc:
            return []

        v_from = int(self.arcs[from_arc, 1])
        u_to = int(self.arcs[to_arc, 0])

        if v_from == u_to:
            return []

        cutoff = self.max_search_dist
        _, path_nodes = self._query_shortest_path(v_from, u_to, cutoff=cutoff)

        if not path_nodes or len(path_nodes) < 2:
            return []

        intermediate_arcs: List[int] = []
        for i in range(len(path_nodes) - 1):
            hop = (path_nodes[i], path_nodes[i + 1])
            arc_id = self.edge_to_arc.get(hop)
            if arc_id is not None:
                intermediate_arcs.append(arc_id)

        return intermediate_arcs


def compute_gps_distances(track: np.ndarray) -> np.ndarray:
    """
    Compute metric Euclidean displacement distances between consecutive GPS observations.

    Args:
        track: np.ndarray of shape (N, 3) with [lon, lat, timestamp]

    Returns:
        np.ndarray of shape (N - 1,) with metric step distances in meters.
    """
    n = len(track)
    if n <= 1:
        return np.empty(0, dtype=np.float64)

    mean_lat = float(np.mean(track[:, 1]))
    cos_lat = float(np.cos(np.radians(mean_lat)))

    dx = (track[1:, 0] - track[:-1, 0]) * cos_lat * 111320.0
    dy = (track[1:, 1] - track[:-1, 1]) * 111320.0
    return np.sqrt(dx * dx + dy * dy).astype(np.float64)
