import os
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ============================================================================
# Configuration
# ============================================================================

# Output directory where time series data is saved
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/3D_time_series"
OUTPUT_FILENAME_PREFIX = "3D_time_series_AoA12_Re50000_all_snapshots_*"
# OUTPUT_FILENAME_PREFIX = "3D_time_series_sparse_snapshot_outer_loop_*"


# Spatial subsampling parameters (must match the extraction script)
STRIDE_X = 5
STRIDE_Y = 25
STRIDE_Z = 1

# Mesh data file (for spatial coordinates)
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Load data loader
module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# MANUALLY CONFIGURED PROBE LOCATIONS FOR EACH CHORD LOCATION
# ============================================================================
#
# Define probe coordinates (x, y) for each chord location.
# Modify these lists to select different probes for each x/c location.
#
# Format: PROBE_COORDS_BY_CHORD[x_c_value] = [(x1, y1), (x2, y2), ...]
#

PROBE_COORDS_BY_CHORD = {
    0.5: [
        (0.48, 0.06),
        (0.50, 0.08),
        (0.52, 0.10),
        (0.50, 0.12),
    ],
    0.7: [
        (0.68, 0.05),
        (0.70, 0.08),
        (0.72, 0.10),
        (0.70, 0.12),
    ],
    0.9: [
        (0.88, 0.04),
        (0.90, 0.07),
        (0.92, 0.09),
        (0.90, 0.11),
    ],
}

print("\n" + "=" * 70)
print("3D TIME SERIES VISUALIZATION - MULTIPLE CHORD LOCATIONS")
print("=" * 70)

print(f"\nConfigured probe locations:")
for x_c, probes in PROBE_COORDS_BY_CHORD.items():
    print(f"  x/c = {x_c:.2f}: {len(probes)} probes")

# ============================================================================
# Load Mesh
# ============================================================================

print("\n" + "=" * 70)
print("LOADING MESH")
print("=" * 70)

loader = CompressedSnapshotLoader(MESH_FILE)

# Coordinates:
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

# Get 2D grid for index finding
x_2d = x_data[0, :, :]  # (Ny, Nx)
y_2d = y_data[0, :, :]  # (Ny, Nx)
x_1d = x_2d[0, :]
y_1d = y_2d[:, 0]

print(f"Mesh Shape: x={x_data.shape[2]}, y={y_data.shape[1]}, z={z_data.shape[0]}")

# ============================================================================
# Load Geometrical Data
# ============================================================================

print("\n" + "=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]

# Extract coordinates
x_interface = interface_points[:, 0]
y_interface = interface_points[:, 1]

# Separate upper and lower surfaces
y_mean = np.mean(y_interface)
upper_mask = y_interface > y_mean
lower_mask = ~upper_mask

print(f"  Number of 2D interface points: {len(interface_points)}")

# ============================================================================
# Load Time Series Data from HDF5
# ============================================================================

print("\n" + "=" * 70)
print("LOADING TIME SERIES DATA FROM HDF5")
print("=" * 70)

# Find the most recent HDF5 file
matching_files = sorted(glob(os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_PREFIX}_*.h5")))

if not matching_files:
    raise FileNotFoundError(f"No time series HDF5 files found in {OUTPUT_DIR}")

latest_file = matching_files[-1]
print(f"Loading: {os.path.basename(latest_file)}")

time_series_data = {}
chord_locations = []

with h5py.File(latest_file, 'r') as f:
    # Load metadata
    stride_x = f.attrs['stride_x']
    stride_y = f.attrs['stride_y']
    stride_z = f.attrs['stride_z']

    print(f"\nStrides from file: strideX={stride_x}, strideY={stride_y}, strideZ={stride_z}")

    # Load data for each x/c location
    for group_name in f.keys():
        if not group_name.startswith('x_c_'):
            continue

        grp = f[group_name]
        x_c_key = float(group_name.split('_')[2])
        chord_locations.append(x_c_key)

        time_series_data[x_c_key] = {
            'wall_pressure': grp['wall_pressure'][:],
            'wall_shear_stress': grp['wall_shear_stress'][:],
            'fluid_u_streamwise': grp['fluid_u_streamwise'][:],
            'sparse_grid_info': {
                'ix_indices': grp['ix_indices'][:],
                'iy_indices': grp['iy_indices'][:],
                'valid_ix': grp['valid_ix'][:],
                'valid_iy': grp['valid_iy'][:],
                'valid_x': grp['valid_x'][:],
                'valid_y': grp['valid_y'][:],
                'Nz': grp.attrs['Nz'],
                'N_valid_points': grp.attrs['N_valid_points']
            },
            'x_c_actual': grp.attrs['x_c_actual'],
            'y_surface': grp.attrs['y_surface'],
            'ix_min': grp.attrs['ix_min'],
            'ix_max': grp.attrs['ix_max'],
            'iy_min': grp.attrs['iy_min'],
            'iy_max': grp.attrs['iy_max'],
        }

        print(f"\n  x/c = {x_c_key:.2f}:")
        print(f"    Wall pressure shape: {time_series_data[x_c_key]['wall_pressure'].shape}")
        print(f"    Wall shear stress shape: {time_series_data[x_c_key]['wall_shear_stress'].shape}")
        print(f"    Fluid u streamwise shape: {time_series_data[x_c_key]['fluid_u_streamwise'].shape}")

# Sort chord locations
chord_locations = sorted(chord_locations)

# ============================================================================
# Find Closest Grid Points for Probes - All Chord Locations
# ============================================================================

print("\n" + "=" * 70)
print("FINDING CLOSEST GRID POINTS FOR PROBES - ALL CHORD LOCATIONS")
print("=" * 70)


def find_closest_sparse_point(point, grid_x, grid_y):
    """Find the closest sparse grid point to a given point."""
    x_point, y_point = point
    distances = np.sqrt((grid_x - x_point)**2 + (grid_y - y_point)**2)
    i_closest = np.argmin(distances)
    closest_point = (grid_x[i_closest], grid_y[i_closest])
    distance = distances[i_closest]
    return closest_point, distance, i_closest


# Store probe information for each chord location
probe_info_by_chord = {}

for x_c_target in chord_locations:
    print(f"\n  x/c = {x_c_target:.2f}:")

    # Get valid grid coordinates from the data
    valid_x = time_series_data[x_c_target]['sparse_grid_info']['valid_x']
    valid_y = time_series_data[x_c_target]['sparse_grid_info']['valid_y']

    # Get probes for this chord location (use default if not specified)
    if x_c_target in PROBE_COORDS_BY_CHORD:
        probe_coords = PROBE_COORDS_BY_CHORD[x_c_target]
    else:
        # Default: sample 4 probes in the window
        print(f"    ⚠ No probes defined for x/c={x_c_target:.2f}. Using default sampling.")
        probe_coords = [
            (x_c_target - 0.02, 0.05),
            (x_c_target, 0.08),
            (x_c_target + 0.02, 0.10),
            (x_c_target, 0.12),
        ]

    # Find closest grid points for each probe
    closest_matches = []
    print(f"    Probe#  Probe Coord                Closest Grid Point           Distance  Index")
    print(f"    " + "-" * 81)

    for i, probe_point in enumerate(probe_coords):
        closest_point, distance, valid_idx = find_closest_sparse_point(probe_point, valid_x, valid_y)
        closest_matches.append({
            'probe_point': probe_point,
            'closest_point': closest_point,
            'distance': distance,
            'valid_idx': valid_idx
        })
        print(f"    {i+1:<7} ({probe_point[0]:.4f}, {probe_point[1]:.4f})    ({closest_point[0]:.4f}, {closest_point[1]:.4f})     {distance:.6f}  {valid_idx}")

    probe_info_by_chord[x_c_target] = {
        'probe_coords': probe_coords,
        'closest_matches': closest_matches,
        'valid_grid_x': valid_x,
        'valid_grid_y': valid_y,
    }

# ============================================================================
# FIGURE 1: DOMAIN VISUALIZATION WITH PROBES FOR EACH CHORD LOCATION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING DOMAIN VISUALIZATIONS FOR EACH CHORD LOCATION")
print("=" * 70)

for x_c_target in chord_locations:
    fig_domain, ax = plt.subplots(figsize=(14, 9))

    # Plot interface points (airfoil surface)
    ax.scatter(x_interface[upper_mask], y_interface[upper_mask],
               c='blue', s=2, alpha=0.5, label='Upper surface')
    ax.scatter(x_interface[lower_mask], y_interface[lower_mask],
               c='red', s=2, alpha=0.5, label='Lower surface')

    # Get reference point information
    x_ref_actual = time_series_data[x_c_target]['x_c_actual']
    y_ref_actual = time_series_data[x_c_target]['y_surface']
    ix_min = time_series_data[x_c_target]['ix_min']
    ix_max = time_series_data[x_c_target]['ix_max']
    iy_min = time_series_data[x_c_target]['iy_min']
    iy_max = time_series_data[x_c_target]['iy_max']

    # Highlight reference point
    ax.scatter(x_ref_actual, y_ref_actual, c='green', s=100, marker='*',
               edgecolors='black', linewidths=2, label=f'Reference point (x/c={x_ref_actual:.3f})', zorder=5)

    # Define window bounds
    x_min_crop = x_2d[iy_min, ix_min]
    x_max_crop = x_2d[iy_min, ix_max-1] if ix_max > 0 else x_2d[iy_min, ix_min]
    y_min_crop = y_2d[iy_min, ix_min]
    y_max_crop = y_2d[iy_max-1, ix_min] if iy_max > 0 else y_2d[iy_min, ix_min]

    # Draw correlation window as rectangle
    rect = patches.Rectangle((x_min_crop, y_min_crop),
                              x_max_crop - x_min_crop,
                              y_max_crop - y_min_crop,
                              linewidth=3, edgecolor='green', facecolor='green',
                              alpha=0.2, label='Correlation window')
    ax.add_patch(rect)

    # Draw reference lines
    ax.axvline(x_ref_actual, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y_ref_actual, color='green', linestyle='--', linewidth=1, alpha=0.5)

    # Add sparse grid points visualization
    valid_x = probe_info_by_chord[x_c_target]['valid_grid_x']
    valid_y = probe_info_by_chord[x_c_target]['valid_grid_y']
    ax.scatter(valid_x, valid_y, s=10, c='black', alpha=0.6, marker='s', label='Valid grid points')

    # Plot probe points
    probe_info = probe_info_by_chord[x_c_target]
    probe_coords = probe_info['probe_coords']
    closest_matches = probe_info['closest_matches']

    probe_x = [p[0] for p in probe_coords]
    probe_y = [p[1] for p in probe_coords]
    ax.scatter(probe_x, probe_y, s=80, c='orange', marker='o',
               edgecolors='darkred', linewidths=1.5, label='Probe points', zorder=4)

    # Plot closest matches with connecting lines and labels
    for i, match in enumerate(closest_matches):
        # Draw line from probe point to closest grid point
        ax.plot([match['probe_point'][0], match['closest_point'][0]],
                [match['probe_point'][1], match['closest_point'][1]],
                'gray', linestyle=':', linewidth=0.8, alpha=0.4)
        # Mark closest grid point
        ax.scatter(match['closest_point'][0], match['closest_point'][1],
                   s=50, c='cyan', marker='x', linewidths=2, zorder=3)
        # Add probe number label
        ax.text(match['probe_point'][0], match['probe_point'][1] - 0.015,
                f'{i+1}', fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Labels and formatting
    ax.set_xlabel('x/c', fontsize=14)
    ax.set_ylabel('y/c', fontsize=14)
    ax.set_title(f'Correlation Domain for x/c = {x_c_target:.2f}\n'
                 f'(Sparse Grid, Reference Point, Windows, and Probes)',
                 fontsize=16, fontweight='bold')

    # Custom legend
    x_min_domain, x_max_domain = np.min(x_2d), np.max(x_2d)
    y_min_domain, y_max_domain = np.min(y_2d), np.max(y_2d)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=6, label='Valid grid points', alpha=0.6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markeredgecolor='darkred', markersize=8, label='Probe points'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='cyan', markersize=10, markeredgewidth=2, label='Closest matches'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=0.8, alpha=0.4, label='Search path'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markeredgecolor='black', markersize=12, label='Reference point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Set axis limits
    ax.set_xlim(min(x_min_crop - 0.1, x_min_domain), max(x_max_crop + 0.1, x_max_domain))
    ax.set_ylim(min(y_min_crop - 0.05, y_min_domain), max(y_max_crop + 0.05, y_max_domain))

    plt.tight_layout()
    plt.show()

    print(f"  ✓ Domain visualization for x/c = {x_c_target:.2f} plotted successfully")

# ============================================================================
# FIGURE 2: TEMPORAL SIGNALS FOR EACH CHORD LOCATION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING TEMPORAL SIGNAL PLOTS FOR EACH CHORD LOCATION")
print("=" * 70)

# Color palette for probes
probe_colors_ts = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5'
]

# Z-index to extract (spanwise position)
z_idx = 0

for x_c_target in chord_locations:
    print(f"\n  x/c = {x_c_target:.2f}:")

    # Get probe information
    probe_info = probe_info_by_chord[x_c_target]
    probe_coords = probe_info['probe_coords']
    closest_matches = probe_info['closest_matches']
    n_probes = len(probe_coords)

    # Prepare time array
    Nt_samples = time_series_data[x_c_target]['wall_pressure'].shape[0]
    time_array = np.arange(Nt_samples)

    # Create figure with subplots: 2 for pressure/shear + n_probes for velocity
    fig_signals, axes_signals = plt.subplots(n_probes + 2, 1, figsize=(14, 3 * (n_probes + 2)))

    # Ensure axes is iterable
    axes_signals = list(axes_signals)

    # ====================================================================
    # SUBPLOT 1: PRESSURE FLUCTUATION (Surface Point at z=z_idx)
    # ====================================================================

    ax_pressure = axes_signals[0]

    # Extract signals at z=z_idx
    p_data = time_series_data[x_c_target]['wall_pressure'][:, z_idx]

    # Compute fluctuations (subtract temporal mean)
    p_mean = np.mean(p_data)
    p_prime_z = p_data - p_mean

    # Plot pressure
    ax_pressure.plot(time_array, p_prime_z, linewidth=1.0, alpha=0.85,
                     color='#d62728', label='Pressure', zorder=3)
    ax_pressure.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
    ax_pressure.set_ylabel(r"$p^\prime$ (Pa)", fontsize=11, fontweight='bold')

    ax_pressure.grid(True, alpha=0.3, which='both', zorder=0)
    ax_pressure.set_title(
        f'x/c = {x_c_target:.2f}: Surface Reference - Pressure Fluctuation $p^\prime$ (z-index = {z_idx})',
        fontsize=12, fontweight='bold'
    )
    ax_pressure.legend(loc='upper right', fontsize=9, framealpha=0.9)

    print(f"    Pressure: mean={p_mean:.6e}, std={np.std(p_prime_z):.6e}")

    # ====================================================================
    # SUBPLOT 2: WALL SHEAR STRESS FLUCTUATION (Surface Point at z=z_idx)
    # ====================================================================

    ax_shear = axes_signals[1]

    # Extract wall shear stress at z=z_idx
    tau_data = time_series_data[x_c_target]['wall_shear_stress'][:, z_idx]

    # Compute fluctuations (subtract temporal mean)
    tau_mean = np.mean(tau_data)
    tau_w_z = tau_data - tau_mean

    # Plot wall shear stress
    ax_shear.plot(time_array, tau_w_z, linewidth=1.0, alpha=0.85,
                  color='#1f77b4', label='Wall Shear Stress', zorder=3)
    ax_shear.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
    ax_shear.set_ylabel(r"$\tau_w^\prime$ (Pa)", fontsize=11, fontweight='bold')

    ax_shear.grid(True, alpha=0.3, which='both', zorder=0)
    ax_shear.set_title(
        f'x/c = {x_c_target:.2f}: Surface Reference - Wall Shear Stress Fluctuation $\tau_w^\prime$ (z-index = {z_idx})',
        fontsize=12, fontweight='bold'
    )
    ax_shear.legend(loc='upper right', fontsize=9, framealpha=0.9)

    print(f"    Wall shear stress: mean={tau_mean:.6e}, std={np.std(tau_w_z):.6e}")

    # ====================================================================
    # SUBPLOTS 3+: STREAMWISE VELOCITY FLUCTUATION (Individual Probes)
    # ====================================================================

    print(f"    Extracting velocity signals for {n_probes} probes at z-index {z_idx}...\n")

    for i, (probe_point, match) in enumerate(zip(probe_coords, closest_matches)):
        ax_probe = axes_signals[i + 2]

        # Get probe valid index
        valid_idx = match['valid_idx']

        # Extract u-velocity at this probe location and z-index
        # fluid_u_streamwise shape: (Nt, N_valid_points, Nz)
        u_prime_probe = time_series_data[x_c_target]['fluid_u_streamwise'][:, valid_idx, z_idx]

        # Compute mean and subtract to get fluctuations
        u_mean = np.mean(u_prime_probe)
        u_prime = u_prime_probe - u_mean

        color = probe_colors_ts[i % len(probe_colors_ts)]

        ax_probe.plot(time_array, u_prime, linewidth=0.8, alpha=0.85,
                      color=color, label=f'Probe {i+1}', zorder=3)
        ax_probe.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
        ax_probe.grid(True, alpha=0.3, which='both', zorder=0)

        ax_probe.set_ylabel(r"$u^\prime$ (m/s)", fontsize=11, fontweight='bold')
        ax_probe.set_title(
            f'x/c = {x_c_target:.2f} - Probe {i+1}: ({probe_point[0]:.4f}, {probe_point[1]:.4f}) → '
            f'Grid ({match["closest_point"][0]:.4f}, {match["closest_point"][1]:.4f}) (z-index = {z_idx})',
            fontsize=11, fontweight='bold'
        )
        ax_probe.legend(loc='upper right', fontsize=9, framealpha=0.9)

        print(f"      Probe {i+1}: u_prime mean={u_mean:.6e}, std={np.std(u_prime):.6e}")

    # Set x-label only on last subplot
    axes_signals[-1].set_xlabel('Snapshot Index', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"  ✓ Temporal signals for x/c = {x_c_target:.2f} plotted successfully")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nGenerated visualizations for {len(chord_locations)} chord location(s):")
for x_c in chord_locations:
    print(f"  - x/c = {x_c:.2f}: Domain plot + Temporal signals plot")

print("\nTo modify probe locations, edit PROBE_COORDS_BY_CHORD at the top of this script.")
