"""
Execution script to perform complete dataset exploration, generate inventory CSV,
render statistical visualizations, and produce interactive Folium maps.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.loader import load_trip
from src.data.inspector import scan_entire_dataset
from src.data.alignment import (
    align_gps_to_route_monotonic,
    align_gps_to_route_unconstrained,
    evaluate_candidate_recall
)
from src.visualization.eda_plots import (
    plot_sampling_rate_distribution,
    plot_trajectory_statistics,
    plot_network_topology,
    plot_candidate_recall_curve,
    plot_projection_error_distribution
)
from src.visualization.map_renderer import render_folium_trajectory_map


def main():
    print("=" * 70)
    print("PathSnap: Initiating Comprehensive Dataset Exploration")
    print("=" * 70)

    data_raw_dir = repo_root / "data" / "raw"
    results_dir = repo_root / "experiments" / "results"
    figures_dir = repo_root / "reports" / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scan entire dataset
    print("\n[Step 1/5] Scanning dataset partitions and compiling inventory...")
    inventory_csv_path = results_dir / "dataset_inventory.csv"
    if inventory_csv_path.exists():
        print(f"Loading existing inventory from {inventory_csv_path}")
        df_inventory = pd.read_csv(inventory_csv_path, dtype={"trip_id": str})
        df_inventory["trip_id"] = df_inventory["trip_id"].astype(str).str.zfill(8)
        train_trips = sorted([d.name for d in (data_raw_dir / "train").iterdir() if d.is_dir()])
        test_trips = sorted([d.name for d in (data_raw_dir / "student_test").iterdir() if d.is_dir()])
        universe_100 = {f"{i:08d}" for i in range(100)}
        partitions = {
            "train": train_trips,
            "student_test": test_trips,
            "evaluator_reserved": sorted(list(universe_100 - set(train_trips) - set(test_trips)))
        }
    else:
        df_inventory, partitions = scan_entire_dataset(data_raw_dir)
        df_inventory.to_csv(inventory_csv_path, index=False)
        print(f"Inventory saved successfully -> {inventory_csv_path}")
    print(f"Total trips inventoried: {len(df_inventory)} (Train: {len(partitions['train'])}, Student Test: {len(partitions['student_test'])})")
    print(f"Reserved evaluator trips identified: {len(partitions['evaluator_reserved'])}")

    # 2. Extract global sampling time differences (all_dts) and projection distances
    print("\n[Step 2/5] Analyzing global sampling intervals and route alignments...")
    all_dts = []
    sample_projection_distances = []
    recall_aggregates = {15.0: [], 25.0: [], 35.0: [], 50.0: [], 75.0: [], 100.0: []}

    # Analyze sample of trips across partitions for projection distance and candidate recall
    trips_to_sample = df_inventory["trip_id"].tolist()[:15]  # first 15 trips

    for t_id in df_inventory["trip_id"]:
        part = "train" if t_id in partitions["train"] else "student_test"
        trip_path = data_raw_dir / part / t_id
        trip = load_trip(trip_path, partition=part)
        
        dts = np.diff(trip.track[:, 2])
        all_dts.extend(dts)

        if t_id in trips_to_sample and trip.route is not None:
            # Monotonic alignment
            _, _, dists = align_gps_to_route_monotonic(trip.track, trip.nodes, trip.arcs, trip.route)
            sample_projection_distances.extend(dists)

            # Candidate recall
            rec_dict = evaluate_candidate_recall(trip, search_radii=[15.0, 25.0, 35.0, 50.0, 75.0, 100.0])
            for r, val in rec_dict.items():
                recall_aggregates[r].append(val)

    all_dts = np.array(all_dts)
    sample_projection_distances = np.array(sample_projection_distances)

    mean_recall_by_radius = {
        r: round(float(np.mean(vals)), 2) for r, vals in recall_aggregates.items()
    }
    print(f"Candidate Recall across radii (sample mean): {mean_recall_by_radius}")

    # 3. Generate Statistical Figures
    print("\n[Step 3/5] Generating publication-quality EDA figures...")
    plot_sampling_rate_distribution(
        df_inventory,
        all_dts,
        figures_dir / "sampling_rate_distribution.png"
    )
    print("Saved -> sampling_rate_distribution.png")

    plot_trajectory_statistics(
        df_inventory,
        figures_dir / "trajectory_statistics.png"
    )
    print("Saved -> trajectory_statistics.png")

    plot_network_topology(
        df_inventory,
        figures_dir / "network_topology.png"
    )
    print("Saved -> network_topology.png")

    plot_candidate_recall_curve(
        mean_recall_by_radius,
        figures_dir / "candidate_recall_curve.png"
    )
    print("Saved -> candidate_recall_curve.png")

    plot_projection_error_distribution(
        sample_projection_distances,
        figures_dir / "projection_error_distribution.png"
    )
    print("Saved -> projection_error_distribution.png")

    # 4. Generate Interactive Folium Maps
    print("\n[Step 4/5] Rendering interactive Folium trajectory maps...")
    
    # Trip 00000001 (Train)
    trip_1 = load_trip(data_raw_dir / "train" / "00000001", partition="train")
    map1_path = figures_dir / "trip_00000001_map.html"
    render_folium_trajectory_map(trip_1, map1_path)
    print(f"Generated -> {map1_path} (Points: {len(trip_1.track)}, Route Arcs: {len(trip_1.route)})")

    # Trip 00000000 (Student Test)
    trip_0 = load_trip(data_raw_dir / "student_test" / "00000000", partition="student_test")
    map0_path = figures_dir / "trip_00000000_map.html"
    render_folium_trajectory_map(trip_0, map0_path)
    print(f"Generated -> {map0_path} (Points: {len(trip_0.track)}, Route Arcs: {len(trip_0.route)})")

    # 5. Print Summary Metrics
    print("\n" + "=" * 70)
    print("Dataset Exploration Summary Statistics:")
    print("=" * 70)
    print(f"Total Trips: {len(df_inventory)} (80 Train + 10 Student Test + 10 Evaluator Reserved)")
    print(f"Total GPS Observations: {df_inventory['track_points'].sum():,}")
    print(f"Total Trajectory Distance: {df_inventory['trajectory_length_km'].sum():,.1f} km")
    print(f"Observation Points per Trip: Mean={df_inventory['track_points'].mean():.1f}, Median={df_inventory['track_points'].median():.1f}, Min={df_inventory['track_points'].min()}, Max={df_inventory['track_points'].max()}")
    print(f"Duration per Trip (min): Mean={df_inventory['duration_seconds'].mean()/60.0:.1f}, Median={df_inventory['duration_seconds'].median()/60.0:.1f}, Min={df_inventory['duration_seconds'].min()/60.0:.1f}, Max={df_inventory['duration_seconds'].max()/60.0:.1f}")
    print(f"Nominal 1 Hz Step Percentage: Mean={df_inventory['dt_1s_pct'].mean():.2f}% across all trips")
    print(f"Road Network Nodes per Trip: Mean={df_inventory['nodes_count'].mean():.1f}, Median={df_inventory['nodes_count'].median():.1f}, Max={df_inventory['nodes_count'].max()}")
    print(f"Road Network Arcs per Trip: Mean={df_inventory['arcs_count'].mean():.1f}, Median={df_inventory['arcs_count'].median():.1f}, Max={df_inventory['arcs_count'].max()}")
    print(f"Route Topological Continuity: Mean={df_inventory['route_continuity_pct'].mean():.2f}% (Min={df_inventory['route_continuity_pct'].min()}%, Max={df_inventory['route_continuity_pct'].max()}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
