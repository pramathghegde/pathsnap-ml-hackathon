"""
PathSnap Folium Map Renderer Module.

Generates interactive, multi-layer Folium maps for GPS trajectories and road networks:
- Base cartographic layers
- Road network geometry (filtered to trajectory corridor for optimal rendering performance)
- Ground-truth reference route (highlighted polyline)
- Raw GPS observation points (blue circle markers with metadata tooltips)
- Trajectory origin and destination markers
- Layer controls and measurement tools
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import folium
from folium import plugins
import numpy as np

from src.data.loader import TripData


def render_folium_trajectory_map(
    trip: TripData,
    output_html_path: Union[str, Path],
    max_network_arcs: int = 5000,
    network_buffer_m: float = 600.0
) -> folium.Map:
    """
    Render an interactive Folium map for a single trip.

    Args:
        trip: Populated TripData dataclass
        output_html_path: Filepath where the .html map will be saved
        max_network_arcs: Cap on background road network segments to ensure fluid browser rendering
        network_buffer_m: Spatial buffer around trajectory to include background road segments

    Returns:
        The generated folium.Map object.
    """
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute trajectory center and bounds
    mean_lat = float(np.mean(track[:, 1]))
    mean_lon = float(np.mean(track[:, 0]))
    cos_lat = np.cos(np.radians(mean_lat))

    # Initialize map with clean CartoDB Positron tiles
    m = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True
    )

    # Add OpenStreetMap as alternative tile layer
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Feature groups for toggleable layers
    fg_network = folium.FeatureGroup(name="Road Network Graph (Gray)", show=True)
    fg_route = folium.FeatureGroup(name="Reference Route Ground Truth (Green)", show=True)
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)

    # 1. Background Road Network Layer
    # Filter arcs within network_buffer_m of any GPS point for performance
    traj_lon_min, traj_lon_max = float(track[:, 0].min()), float(track[:, 0].max())
    traj_lat_min, traj_lat_max = float(track[:, 1].min()), float(track[:, 1].max())

    buffer_deg_lat = network_buffer_m / 111320.0
    buffer_deg_lon = network_buffer_m / (cos_lat * 111320.0)

    bbox_lon_min = traj_lon_min - buffer_deg_lon
    bbox_lon_max = traj_lon_max + buffer_deg_lon
    bbox_lat_min = traj_lat_min - buffer_deg_lat
    bbox_lat_max = traj_lat_max + buffer_deg_lat

    network_arcs_drawn = 0
    route_set = set(route) if route is not None else set()

    for arc_id, (u, v) in enumerate(arcs):
        if arc_id in route_set:
            continue  # Drawn separately with high prominence in the route layer

        u_lon, u_lat = nodes[u]
        v_lon, v_lat = nodes[v]

        # Check if endpoints are within corridor bounding box
        if (
            (bbox_lon_min <= u_lon <= bbox_lon_max and bbox_lat_min <= u_lat <= bbox_lat_max)
            or (bbox_lon_min <= v_lon <= bbox_lon_max and bbox_lat_min <= v_lat <= bbox_lat_max)
        ):
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color="#8c8c8c",
                weight=1.5,
                opacity=0.45,
                tooltip=f"Arc ID: {arc_id} ({u} -> {v})"
            ).add_to(fg_network)

            network_arcs_drawn += 1
            if network_arcs_drawn >= max_network_arcs:
                break

    # 2. Reference Route Layer (Traversed Arcs)
    if route is not None and len(route) > 0:
        for seq_idx, arc_id in enumerate(route):
            u, v = arcs[arc_id]
            u_lon, u_lat = nodes[u]
            v_lon, v_lat = nodes[v]

            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color="#2ca02c",
                weight=4.5,
                opacity=0.9,
                tooltip=f"Step {seq_idx+1}/{len(route)}: Arc {arc_id} ({u} -> {v})"
            ).add_to(fg_route)

    # 3. Raw GPS Observations Layer
    for pt_idx, (lon, lat, ts) in enumerate(track):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#08519c",
            fill=True,
            fill_color="#3182bd",
            fill_opacity=0.75,
            weight=1,
            popup=f"Obs #{pt_idx}<br>Time: {ts:.1f}s<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
        ).add_to(fg_gps)

    # Origin and Destination Markers
    start_pt = track[0]
    end_pt = track[-1]

    folium.Marker(
        location=[start_pt[1], start_pt[0]],
        popup=f"Start: Trip {trip.trip_id} (t={start_pt[2]:.1f}s)",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

    folium.Marker(
        location=[end_pt[1], end_pt[0]],
        popup=f"Destination: Trip {trip.trip_id} (t={end_pt[2]:.1f}s)",
        icon=folium.Icon(color="red", icon="stop", prefix="fa")
    ).add_to(m)

    # Add all layers to map
    fg_network.add_to(m)
    fg_route.add_to(m)
    fg_gps.add_to(m)

    # Add interactive plugins
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    plugins.MeasureControl(position="bottomleft", primary_length_unit="meters").add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)

    # Save to standalone HTML file
    m.save(str(output_html_path))
    return m
