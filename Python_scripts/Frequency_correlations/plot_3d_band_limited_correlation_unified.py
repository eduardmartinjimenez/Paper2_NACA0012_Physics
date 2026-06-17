"""
Unified script to visualize band-limited 3D correlation maps.

Handles all combinations of:
  - AOA: 5° or 12°
  - Signal type: wall shear (tau_w) or pressure
  - Frequency bands: low_mid or high
  - Correlation types: wall_only, both_filtered

Configuration:
  Set AOA, SIGNAL, BAND, and CORRELATION_TYPES in the USER CONFIGURATION section below.

Examples:
  AOA = 12
  SIGNAL = "tau_w"
  BAND = "high"
  CORRELATION_TYPES = ["both_filtered"]
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.ticker as ticker
from matplotlib.path import Path


plt.rc("text", usetex=True)
plt.rc("font", size=14, family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")


# ==========================================================================
# USER CONFIGURATION
# ==========================================================================

AOA = 5
SIGNAL = "tau_w"  # "tau_w" or "pressure"
BAND = "high"  # or "low_mid", "high", "unconditional"
CORRELATION_TYPES = ["wall_only", "both_filtered"]  # or ["wall_only"], ["both_filtered"], ["unconditional"]

# Panel layout. The maps use equal aspect ratio, so a tall figure creates
# vertical whitespace between shallow panels.
PANEL_FIGSIZE = (5, 8.5)
PANEL_HSPACE = 0.02
PANEL_H_PAD = 0.02


# ==========================================================================
# Configuration Tables
# ==========================================================================

CONFIG_LOOKUP = {
    (5, "pressure", "low_mid"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA5_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA5_Re50000_all_snapshots_20260604_222641.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["low", "mid"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (5, "pressure", "high"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_3d_high_frequency_correlation_maps_AOA5_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA5_Re50000_high_freq_all_snapshots_20260610_191508.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures_high"
        ),
        "bands": ["high"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (5, "tau_w", "low_mid"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA5_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA5_Re50000_all_snapshots_20260604_222641.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["low", "mid"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (5, "tau_w", "high"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_3d_high_frequency_correlation_maps_AOA5_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA5_Re50000_high_freq_all_snapshots_20260610_191508.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures_high"
        ),
        "bands": ["high"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (12, "pressure", "low_mid"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA12_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA12_Re50000_all_snapshots_20260605_194150.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["low", "mid"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (12, "pressure", "high"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_3d_high_frequency_correlation_maps_AOA12_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA12_Re50000_high_freq_all_snapshots_20260610_190610.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures_high"
        ),
        "bands": ["high"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (12, "tau_w", "low_mid"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA12_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA12_Re50000_all_snapshots_20260605_194150.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["low", "mid"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (12, "tau_w", "high"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_3d_high_frequency_correlation_maps_AOA12_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA12_Re50000_high_freq_all_snapshots_20260610_190610.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures_high"
        ),
        "bands": ["high"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (5, "pressure", "unconditional"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA5_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA5_Re50000_all_snapshots_20260604_222641.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["unconditional"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (5, "tau_w", "unconditional"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA5_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA5_Re50000_all_snapshots_20260604_222641.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["unconditional"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (12, "pressure", "unconditional"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA12_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA12_Re50000_all_snapshots_20260605_194150.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["unconditional"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
    (12, "tau_w", "unconditional"): {
        "corr_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/data/band_limited_and_unconditional_3d_correlation_maps_AOA12_Re50000.h5"
        ),
        "geo_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
            "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
        ),
        "ts_file": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Signal_correlation/3D_time_series/new/"
            "3D_time_series_AoA12_Re50000_all_snapshots_20260605_194150.h5"
        ),
        "output_dir": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Freq_correlations_3D/Figures"
        ),
        "bands": ["unconditional"],
        "viz_xlim_offset": [0.5, 0.5],
        "viz_ylim_offset": [0.05, 0.2],
    },
}

GLOBAL_CONFIG = {
    "re_c": 50000,
    "x_c_locations": [0.50, 0.70, 0.90],
    "correlation_types": ["wall_only", "both_filtered"],
    "delta_z_raw_index": 0,
    "vmin": -1.0,
    "vmax": 1.0,
    "n_levels": 101,
    "clip_to_color_range": True,
    "background_color": "#E6F2FF",
    "corr_cmap": "RdBu_r",
}


# ==========================================================================
# Helpers
# ==========================================================================

def group_name_from_xc(xc: float) -> str:
    return f"x_c_{xc:.2f}"


def rotate_about_reference(x, y, x_ref, y_ref, aoa_rad):
    """Rotate coordinates into the freestream-aligned frame around reference."""
    cos_a = np.cos(aoa_rad)
    sin_a = np.sin(aoa_rad)

    x_centered = x - x_ref
    y_centered = y - y_ref

    x_rot = x_centered * cos_a + y_centered * sin_a + x_ref
    y_rot = -x_centered * sin_a + y_centered * cos_a + y_ref

    return x_rot, y_rot


def load_airfoil_geometry(geo_file):
    """Load and split the airfoil geometry into upper and lower surfaces."""
    if not os.path.exists(geo_file):
        print(f"WARNING: geometry file not found: {geo_file}")
        return None

    with h5py.File(geo_file, "r") as f:
        interface_points = f["interface_points"][:]

    x_interface = interface_points[:, 0]
    y_interface = interface_points[:, 1]

    y_mean = np.mean(y_interface)
    upper_mask = y_interface > y_mean
    lower_mask = ~upper_mask

    return {
        "x_upper": x_interface[upper_mask],
        "y_upper": y_interface[upper_mask],
        "x_lower": x_interface[lower_mask],
        "y_lower": y_interface[lower_mask],
    }


def add_airfoil_overlay(ax, airfoil, x_ref, y_ref, aoa_rad):
    """Add rotated airfoil fill and outline to an axes."""
    if airfoil is None:
        return

    x_upper_rot, y_upper_rot = rotate_about_reference(
        airfoil["x_upper"], airfoil["y_upper"], x_ref, y_ref, aoa_rad
    )
    x_lower_rot, y_lower_rot = rotate_about_reference(
        airfoil["x_lower"], airfoil["y_lower"], x_ref, y_ref, aoa_rad
    )

    sort_upper = np.argsort(x_upper_rot)
    sort_lower = np.argsort(x_lower_rot)

    x_upper_sorted = x_upper_rot[sort_upper]
    y_upper_sorted = y_upper_rot[sort_upper]
    x_lower_sorted = x_lower_rot[sort_lower]
    y_lower_sorted = y_lower_rot[sort_lower]

    x_polygon = np.concatenate([x_upper_sorted, x_lower_sorted[::-1]])
    y_polygon = np.concatenate([y_upper_sorted, y_lower_sorted[::-1]])

    ax.fill(x_polygon, y_polygon, color="0.90", alpha=1.0, zorder=20)
    ax.plot(x_upper_sorted, y_upper_sorted, "k-", linewidth=1.0, zorder=21)
    ax.plot(x_lower_sorted, y_lower_sorted, "k-", linewidth=1.0, zorder=21)


def get_rotated_airfoil_polygon(airfoil, x_ref, y_ref, aoa_rad):
    """Return the rotated airfoil polygon used to mask the triangulation."""
    if airfoil is None:
        return None, None

    x_upper_rot, y_upper_rot = rotate_about_reference(
        airfoil["x_upper"], airfoil["y_upper"], x_ref, y_ref, aoa_rad
    )
    x_lower_rot, y_lower_rot = rotate_about_reference(
        airfoil["x_lower"], airfoil["y_lower"], x_ref, y_ref, aoa_rad
    )

    sort_upper = np.argsort(x_upper_rot)
    sort_lower = np.argsort(x_lower_rot)

    x_upper_sorted = x_upper_rot[sort_upper]
    y_upper_sorted = y_upper_rot[sort_upper]
    x_lower_sorted = x_lower_rot[sort_lower]
    y_lower_sorted = y_lower_rot[sort_lower]

    x_polygon = np.concatenate([x_upper_sorted, x_lower_sorted[::-1]])
    y_polygon = np.concatenate([y_upper_sorted, y_lower_sorted[::-1]])

    return x_polygon, y_polygon


def build_masked_triangulation(x, y, airfoil, x_ref, y_ref, aoa_rad):
    """
    Build a triangulation of the sparse valid fluid points and mask triangles
    whose centroids or edge midpoints fall inside the airfoil polygon.
    """
    tri = mtri.Triangulation(x, y)

    if airfoil is None:
        return tri

    x_poly, y_poly = get_rotated_airfoil_polygon(airfoil, x_ref, y_ref, aoa_rad)

    if x_poly is None:
        return tri

    airfoil_path = Path(np.column_stack([x_poly, y_poly]))

    triangles = tri.triangles
    tx = x[triangles]
    ty = y[triangles]

    centroids = np.column_stack([
        np.mean(tx, axis=1),
        np.mean(ty, axis=1),
    ])

    mid01 = np.column_stack([
        0.5 * (tx[:, 0] + tx[:, 1]),
        0.5 * (ty[:, 0] + ty[:, 1]),
    ])
    mid12 = np.column_stack([
        0.5 * (tx[:, 1] + tx[:, 2]),
        0.5 * (ty[:, 1] + ty[:, 2]),
    ])
    mid20 = np.column_stack([
        0.5 * (tx[:, 2] + tx[:, 0]),
        0.5 * (ty[:, 2] + ty[:, 0]),
    ])

    mask_inside_airfoil = (
        airfoil_path.contains_points(centroids)
        | airfoil_path.contains_points(mid01)
        | airfoil_path.contains_points(mid12)
        | airfoil_path.contains_points(mid20)
    )

    tri.set_mask(mask_inside_airfoil)

    return tri


def get_reference_point(corr_h5, group_name, valid_x, valid_y, ts_file):
    """Get the wall reference point, with fallbacks."""
    grp = corr_h5[group_name]

    if "mesh_x" in grp.attrs and "mesh_y" in grp.attrs:
        return float(grp.attrs["mesh_x"]), float(grp.attrs["mesh_y"])

    if os.path.exists(ts_file):
        with h5py.File(ts_file, "r") as ts_h5:
            if group_name in ts_h5:
                ts_grp = ts_h5[group_name]
                if "mesh_x" in ts_grp.attrs and "mesh_y" in ts_grp.attrs:
                    return float(ts_grp.attrs["mesh_x"]), float(ts_grp.attrs["mesh_y"])

    x_ref = float(grp.attrs.get("x_c_actual", np.nan))
    if "y_surface" in grp.attrs:
        y_ref = float(grp.attrs["y_surface"])
    else:
        y_ref = float(np.nanmedian(valid_y))

    return x_ref, y_ref


def load_panel_data(corr_file, xc, band, correlation_type, signal_name, aoa_rad, ts_file):
    """Load one x/c panel for one frequency band and correlation type."""
    group_name = group_name_from_xc(xc)

    with h5py.File(corr_file, "r") as f:
        if group_name not in f:
            available = [k for k in f.keys() if k.startswith("x_c_")]
            raise KeyError(
                f"Group {group_name} not found in {corr_file}. Available groups: {available}"
            )

        grp = f[group_name]

        valid_x = grp["valid_x"][:].astype(np.float64)
        valid_y = grp["valid_y"][:].astype(np.float64)

        # Handle unconditional correlations (different HDF5 structure)
        if correlation_type == "unconditional":
            dataset_path = f"unconditional/{signal_name}/R"
        else:
            dataset_path = f"{correlation_type}/{signal_name}/{band}_R"

        if dataset_path not in grp:
            raise KeyError(f"Dataset {group_name}/{dataset_path} not found")

        R = grp[dataset_path][:].astype(np.float64)

        if R.ndim != 2:
            raise ValueError(f"Expected R to have shape (Npoints, Nz), got {R.shape}")

        if GLOBAL_CONFIG["delta_z_raw_index"] >= R.shape[1]:
            raise IndexError(
                f"DELTA_Z_RAW_INDEX={GLOBAL_CONFIG['delta_z_raw_index']} outside Nz={R.shape[1]}"
            )

        R_dz0 = R[:, GLOBAL_CONFIG["delta_z_raw_index"]]

        x_ref, y_ref = get_reference_point(f, group_name, valid_x, valid_y, ts_file)

        metadata = {
            "group_name": group_name,
            "x_c_target": float(grp.attrs.get("x_c_target", xc)),
            "x_c_actual": float(grp.attrs.get("x_c_actual", xc)),
            "N_valid_points": int(grp.attrs.get("N_valid_points", len(valid_x))),
            "Nz": int(grp.attrs.get("Nz", R.shape[1])),
            "x_ref": x_ref,
            "y_ref": y_ref,
        }

    valid = np.isfinite(valid_x) & np.isfinite(valid_y) & np.isfinite(R_dz0)

    valid_x = valid_x[valid]
    valid_y = valid_y[valid]
    R_dz0 = R_dz0[valid]

    x_rot, y_rot = rotate_about_reference(
        valid_x, valid_y, metadata["x_ref"], metadata["y_ref"], aoa_rad
    )

    metadata["N_plotted_points"] = len(R_dz0)
    metadata["R_raw_min"] = float(np.nanmin(R_dz0)) if len(R_dz0) else np.nan
    metadata["R_raw_max"] = float(np.nanmax(R_dz0)) if len(R_dz0) else np.nan
    metadata["R_raw_abs_max"] = float(np.nanmax(np.abs(R_dz0))) if len(R_dz0) else np.nan
    metadata["R_raw_mean"] = float(np.nanmean(R_dz0)) if len(R_dz0) else np.nan

    return {
        "x_rot": x_rot,
        "y_rot": y_rot,
        "R_raw": R_dz0,
        "metadata": metadata,
    }


def plot_tricontour_or_scatter(
    ax, x, y, R, levels_r, vmin, vmax, airfoil=None, x_ref=None, y_ref=None, aoa_rad=None
):
    """Plot triangulated filled contours; fall back to scatter if needed."""
    im = None
    try:
        tri = build_masked_triangulation(x, y, airfoil, x_ref, y_ref, aoa_rad)

        im = ax.tricontourf(
            tri,
            R,
            levels=levels_r,
            cmap=GLOBAL_CONFIG["corr_cmap"],
            vmin=vmin,
            vmax=vmax,
            extend="neither",
            zorder=1,
        )

    except Exception as exc:
        print(f"WARNING: tricontourf failed ({exc}). Falling back to scatter.")
        im = ax.scatter(
            x,
            y,
            c=R,
            cmap=GLOBAL_CONFIG["corr_cmap"],
            vmin=vmin,
            vmax=vmax,
            s=20,
            edgecolors="none",
            alpha=0.85,
            zorder=1,
        )

    return im


def band_label_for_title(band):
    if band == "low":
        return "low band"
    if band == "mid":
        return "mid band"
    if band == "unconditional":
        return "unconditional"
    return band


def correlation_type_label(correlation_type):
    if correlation_type == "wall_only":
        return "wall signal filtered"
    if correlation_type == "both_filtered":
        return "wall signal and velocity filtered"
    return correlation_type.replace("_", " ")


def colorbar_title_for_band(band, correlation_type, signal_name):
    """Return compact notation for the colorbar label."""
    if band == "unconditional":
        if signal_name == "tau_w":
            return rf"$R_{{\tau_{{w}}^{{\prime}}u^{{\prime}}}}$"
        else:  # pressure
            return rf"$R_{{p_{{w}}^{{\prime}}u^{{\prime}}}}$"

    band_suffix = {"low": "L", "mid": "M"}.get(band, "B")

    if signal_name == "tau_w":
        if correlation_type == "both_filtered":
            return rf"$R_{{\tau_{{w,{band_suffix}}}^{{\prime}}u_{{{band_suffix}}}^{{\prime}}}}$"
        return rf"$R_{{\tau_{{w,{band_suffix}}}^{{\prime}}u^{{\prime}}}}$"
    else:  # pressure
        if correlation_type == "both_filtered":
            return rf"$R_{{p_{{w,{band_suffix}}}^{{\prime}}u_{{{band_suffix}}}^{{\prime}}}}$"
        return rf"$R_{{p_{{w,{band_suffix}}}^{{\prime}}u^{{\prime}}}}$"


def get_visible_window_values(panel_data, viz_xlim_offset, viz_ylim_offset):
    """
    Collect R values only inside the plotted visualization windows
    across all panels in the current figure.
    """
    values = []

    for data in panel_data:
        md = data["metadata"]

        x_ref = md["x_ref"]
        y_ref = md["y_ref"]

        xlim = [x_ref - viz_xlim_offset[0], x_ref + viz_xlim_offset[1]]
        ylim = [y_ref - viz_ylim_offset[0], y_ref + viz_ylim_offset[1]]

        x = data["x_rot"]
        y = data["y_rot"]
        R = data["R_raw"]

        window_mask = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(R)
            & (x >= xlim[0])
            & (x <= xlim[1])
            & (y >= ylim[0])
            & (y <= ylim[1])
        )

        if np.any(window_mask):
            values.append(R[window_mask])

    if not values:
        raise RuntimeError(
            "No valid R values found inside the plotted windows. "
            "Check VIZ_XLIM_OFFSET and VIZ_YLIM_OFFSET."
        )

    return np.concatenate(values)


# ==========================================================================
# Main
# ==========================================================================

def main(aoa, signal_name, band, correlation_types):
    """Main execution function."""
    if (aoa, signal_name, band) not in CONFIG_LOOKUP:
        available = list(CONFIG_LOOKUP.keys())
        raise ValueError(
            f"Configuration for AOA={aoa}, signal={signal_name}, band={band} not found.\n"
            f"Available: {available}"
        )

    config = CONFIG_LOOKUP[(aoa, signal_name, band)]
    aoa_rad = np.deg2rad(aoa)

    corr_file = config["corr_file"]
    geo_file = config["geo_file"]
    ts_file = config["ts_file"]
    output_dir = config["output_dir"]
    bands = config["bands"]
    viz_xlim_offset = config["viz_xlim_offset"]
    viz_ylim_offset = config["viz_ylim_offset"]

    x_c_locations = GLOBAL_CONFIG["x_c_locations"]

    print("=" * 70)
    print(f"CORRELATION MAPS")
    print(f"AOA: {aoa}°, Signal: {signal_name}, Band: {band}")
    print("=" * 70)
    print(f"Input correlation file: {corr_file}")
    print(f"Output directory:       {output_dir}")

    if not os.path.exists(corr_file):
        raise FileNotFoundError(f"Correlation file not found: {corr_file}")

    os.makedirs(output_dir, exist_ok=True)

    airfoil = load_airfoil_geometry(geo_file)
    if airfoil is not None:
        print("Loaded airfoil geometry.")

    for correlation_type in correlation_types:
        for freq_band in bands:
            print("\n" + "=" * 70)
            print(f"CREATING 3x1 PANEL FOR TYPE: {correlation_type}, BAND: {freq_band}")
            print("=" * 70)

            panel_data = []
            for xc in x_c_locations:
                data = load_panel_data(
                    corr_file, xc, freq_band, correlation_type, signal_name, aoa_rad, ts_file
                )
                panel_data.append(data)
                md = data["metadata"]
                print(
                    f"  x/c={xc:.2f}: "
                    f"R min={md['R_raw_min']:.4e}, "
                    f"R max={md['R_raw_max']:.4e}, "
                    f"max|R|={md['R_raw_abs_max']:.4e}, "
                    f"R mean={md['R_raw_mean']:.4e}, "
                    f"points={md['N_plotted_points']}"
                )

            visible_values = get_visible_window_values(panel_data, viz_xlim_offset, viz_ylim_offset)

            visible_min = float(np.nanmin(visible_values))
            visible_max = float(np.nanmax(visible_values))

            vmin_plot = visible_min
            vmax_plot = visible_max

            levels_r = np.linspace(vmin_plot, vmax_plot, GLOBAL_CONFIG["n_levels"])

            print(f"\n  {correlation_type} / {freq_band} plot limits from visible windows:")
            print(f"    visible raw range: [{visible_min:.4e}, {visible_max:.4e}]")
            print(f"    color range:       [{vmin_plot:.4e}, {vmax_plot:.4e}]")

            for data in panel_data:
                if GLOBAL_CONFIG["clip_to_color_range"]:
                    data["R_plot"] = np.clip(data["R_raw"], vmin_plot, vmax_plot)
                else:
                    data["R_plot"] = data["R_raw"]

            fig, axes = plt.subplots(
                len(panel_data),
                1,
                figsize=PANEL_FIGSIZE,
                constrained_layout=True,
            )
            fig.set_constrained_layout_pads(h_pad=PANEL_H_PAD, hspace=PANEL_HSPACE)

            if len(panel_data) == 1:
                axes = [axes]

            for ax in axes:
                ax.set_facecolor(GLOBAL_CONFIG["background_color"])

            im_last = None

            for row, data in enumerate(panel_data):
                ax = axes[row]
                md = data["metadata"]

                x_ref = md["x_ref"]
                y_ref = md["y_ref"]

                im_last = plot_tricontour_or_scatter(
                    ax,
                    data["x_rot"],
                    data["y_rot"],
                    data["R_plot"],
                    levels_r,
                    vmin_plot,
                    vmax_plot,
                    airfoil=airfoil,
                    x_ref=x_ref,
                    y_ref=y_ref,
                    aoa_rad=aoa_rad,
                )

                add_airfoil_overlay(ax, airfoil, x_ref, y_ref, aoa_rad)

                ax.plot(
                    x_ref,
                    y_ref,
                    marker="o",
                    color="black",
                    markersize=5,
                    zorder=22,
                )

                ax.set_title(rf"$x/c = {x_c_locations[row]:.2f}$", fontsize=12)
                ax.set_xlabel("x/c", fontsize=11)
                ax.set_ylabel("y/c", fontsize=11)

                ax.set_aspect("equal", adjustable="box")

                xlim = [x_ref - viz_xlim_offset[0], x_ref + viz_xlim_offset[1]]
                ylim = [y_ref - viz_ylim_offset[0], y_ref + viz_ylim_offset[1]]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                print(f"\nPanel {row + 1} ({correlation_type}, {freq_band}, x/c={x_c_locations[row]:.2f}):")
                print(f"  Reference point: x={x_ref:.6f}, y={y_ref:.6f}")
                print(f"  Window: x=[{xlim[0]:.6f}, {xlim[1]:.6f}], y=[{ylim[0]:.6f}, {ylim[1]:.6f}]")

            cbar = fig.colorbar(
                im_last,
                ax=axes,
                orientation="horizontal",
                fraction=0.045,
                pad=0.04,
                shrink=0.85,
            )
            cbar.set_ticks(np.linspace(vmin_plot, vmax_plot, 6))
            cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            cbar.set_label(colorbar_title_for_band(freq_band, correlation_type, signal_name), fontsize=14)

            signal_label = "wall-shear" if signal_name == "tau_w" else "wall-pressure"

            if correlation_type == "unconditional":
                title_prefix = "Unconditional"
                filename_prefix = "unconditional"
            else:
                title_prefix = "Band-limited"
                filename_prefix = "band_limited"

            fig.suptitle(
                f"{title_prefix} {signal_label}/velocity correlation, {band_label_for_title(freq_band)}, "
                f"{correlation_type_label(correlation_type)}, AoA={aoa}°",
                fontsize=14,
                fontweight="bold",
            )

            output_png = os.path.join(
                output_dir,
                f"{filename_prefix}_{signal_name}_u_{correlation_type}_{freq_band}_3x1panel_AOA{aoa}.png",
            )
            output_eps = os.path.join(
                output_dir,
                f"{filename_prefix}_{signal_name}_u_{correlation_type}_{freq_band}_3x1panel_AOA{aoa}.eps",
            )

            fig.savefig(output_png, dpi=300, bbox_inches="tight")
            fig.savefig(output_eps, dpi=300, bbox_inches="tight")
            print(f"\nSaved: {output_png}")
            print(f"Saved: {output_eps}")

            plt.show()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main(AOA, SIGNAL, BAND, CORRELATION_TYPES)
