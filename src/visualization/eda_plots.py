"""
PathSnap EDA Plotting Module.

Generates high-resolution statistical visualizations for dataset exploration:
- Trajectory duration, length, and point distributions
- Timestamp delta (sampling rate) distribution and cumulative frequency
- Road network node and arc count distributions
- Candidate recall vs search radius curve
- Spatial projection error distribution
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style():
    """Configure clean publication styling for plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.dpi": 200
    })


def plot_sampling_rate_distribution(
    inventory_df: pd.DataFrame,
    all_dts: np.ndarray,
    output_path: Union[str, Path]
):
    """
    Plot sampling interval distribution and cumulative percentage.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: dt distribution in [0, 5s]
    ax1 = axes[0]
    clipped_dts = np.clip(all_dts, 0.0, 5.0)
    sns.histplot(
        clipped_dts,
        bins=50,
        kde=False,
        color="#2b5c8f",
        edgecolor="black",
        ax=ax1
    )
    pct_1s = (np.isclose(all_dts, 1.0, atol=0.05).mean()) * 100.0
    ax1.axvline(1.0, color="#d95f02", linestyle="--", linewidth=2, label=f"Nominal 1 Hz ({pct_1s:.1f}%)")
    ax1.set_title(r"GPS Sampling Time Interval Distribution ($\Delta t$)")
    ax1.set_xlabel("Time Delta (seconds, clipped at 5s)")
    ax1.set_ylabel("Observation Frequency")
    ax1.legend(loc="upper right")

    # Subplot 2: 1s compliance per trip
    ax2 = axes[1]
    sns.histplot(
        inventory_df["dt_1s_pct"],
        bins=20,
        color="#2ca02c",
        edgecolor="black",
        ax=ax2
    )
    ax2.set_title("Trip-Level 1 Hz Compliance Rate")
    ax2.set_xlabel(r"Percentage of Steps with $\Delta t \approx 1.0$s (%)")
    ax2.set_ylabel("Number of Trips")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_statistics(
    inventory_df: pd.DataFrame,
    output_path: Union[str, Path]
):
    """
    Plot trajectory lengths, observation counts, and duration distributions.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Track points count
    ax1 = axes[0]
    sns.histplot(
        inventory_df["track_points"],
        bins=25,
        color="#1f77b4",
        edgecolor="black",
        kde=True,
        ax=ax1
    )
    ax1.axvline(inventory_df["track_points"].median(), color="red", linestyle="--", label=f"Median: {int(inventory_df['track_points'].median())}")
    ax1.set_title("GPS Observations per Trip")
    ax1.set_xlabel("Track Points")
    ax1.set_ylabel("Number of Trips")
    ax1.legend()

    # 2. Trajectory distance in km
    ax2 = axes[1]
    sns.histplot(
        inventory_df["trajectory_length_km"],
        bins=25,
        color="#ff7f0e",
        edgecolor="black",
        kde=True,
        ax=ax2
    )
    ax2.axvline(inventory_df["trajectory_length_km"].median(), color="red", linestyle="--", label=f"Median: {inventory_df['trajectory_length_km'].median():.1f} km")
    ax2.set_title("Trajectory Physical Distance")
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Number of Trips")
    ax2.legend()

    # 3. Trajectory duration in minutes
    ax3 = axes[2]
    durations_min = inventory_df["duration_seconds"] / 60.0
    sns.histplot(
        durations_min,
        bins=25,
        color="#2ca02c",
        edgecolor="black",
        kde=True,
        ax=ax3
    )
    ax3.axvline(durations_min.median(), color="red", linestyle="--", label=f"Median: {durations_min.median():.1f} min")
    ax3.set_title("Trip Duration")
    ax3.set_xlabel("Duration (minutes)")
    ax3.set_ylabel("Number of Trips")
    ax3.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_network_topology(
    inventory_df: pd.DataFrame,
    output_path: Union[str, Path]
):
    """
    Plot distribution of road network graph sizes (nodes, arcs, connected components).
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Nodes vs Arcs scatter
    ax1 = axes[0]
    sns.scatterplot(
        data=inventory_df,
        x="nodes_count",
        y="arcs_count",
        hue="partition",
        palette={"train": "#1f77b4", "student_test": "#d62728"},
        s=80,
        alpha=0.85,
        edgecolor="black",
        ax=ax1
    )
    ax1.set_title("Road Network Scale per Trip (Nodes vs Arcs)")
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Number of Arcs")

    # 2. Arcs per Node ratio (Average Out-Degree)
    ax2 = axes[1]
    degree_ratio = inventory_df["arcs_count"] / inventory_df["nodes_count"]
    sns.histplot(
        degree_ratio,
        bins=20,
        color="#9467bd",
        edgecolor="black",
        kde=True,
        ax=ax2
    )
    ax2.axvline(degree_ratio.mean(), color="red", linestyle="--", label=f"Mean Ratio: {degree_ratio.mean():.2f}")
    ax2.set_title("Network Density (Arcs / Nodes Ratio)")
    ax2.set_xlabel("Edge-to-Node Ratio")
    ax2.set_ylabel("Number of Trips")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_recall_curve(
    recall_by_radius: Dict[float, float],
    output_path: Union[str, Path]
):
    """
    Plot candidate recall curve vs spatial search radius R.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    radii = sorted(list(recall_by_radius.keys()))
    recalls = [recall_by_radius[r] for r in radii]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, recalls, marker="o", color="#d95f02", linewidth=2.5, markersize=8)

    for r, rec in zip(radii, recalls):
        ax.annotate(
            f"{rec:.1f}%",
            (r, rec),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold"
        )

    ax.set_title(r"Candidate Arc Recall vs Spatial Search Radius ($R$)")
    ax.set_xlabel("Candidate Search Radius (meters)")
    ax.set_ylabel("Candidate Recall / Ground-Truth Coverage (%)")
    ax.set_ylim(min(recalls) - 5, 102)
    ax.axhline(99.0, color="gray", linestyle=":", label="99% Target Recall")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_projection_error_distribution(
    distances: np.ndarray,
    output_path: Union[str, Path]
):
    """
    Plot orthogonal projection error distribution from raw GPS to reference route.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    clipped_dists = np.clip(distances, 0.0, 30.0)

    sns.histplot(
        clipped_dists,
        bins=40,
        color="#386cb0",
        edgecolor="black",
        kde=True,
        ax=ax
    )

    median_d = np.median(distances)
    p95_d = np.percentile(distances, 95)

    ax.axvline(median_d, color="#e41a1c", linestyle="--", linewidth=2, label=f"Median Offset: {median_d:.1f} m")
    ax.axvline(p95_d, color="#ff7f00", linestyle=":", linewidth=2, label=f"95th Percentile: {p95_d:.1f} m")

    ax.set_title("GPS Observation Orthogonal Offset to Reference Route")
    ax.set_xlabel("Perpendicular Distance (meters, clipped at 30m)")
    ax.set_ylabel("Observation Frequency")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
