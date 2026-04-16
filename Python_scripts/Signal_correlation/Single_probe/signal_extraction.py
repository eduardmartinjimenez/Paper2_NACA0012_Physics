"""
Signal Correlation Analysis - Complete Time Series Extraction
==============================================================

Study correlation, coherence, and cross-spectrum of two signals from one slice.

This script:
1. Loads a specific slice and its mesh for reference
2. Finds the closest surface point on the airfoil (suction side)
3. Plots the surface point and airfoil surface geometry
4. Extracts u-velocity time series from the configured slice at probe locations
5. Saves time series data for further analysis
"""

import os
import sys
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from collections import defaultdict

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

### AOA 12º

# Reference slice data paths (for mesh/geometry reference)
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
# MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_test/"
# MESH_SLICE_NAME = "slice_1-CROP-MESH.h5"
# SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_test/"


# Geometric data (for visualization and surface point detection)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# Output directory
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Physical parameters
rho_ref = 1.0           # Reference density [kg/m³]
u_infty = 1.0           # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord length [m]
Re_c = 50000            # Reynolds number
AOA_deg = 12.0          # Angle of attack [degrees]
AOA_rad = np.radians(AOA_deg)

# Physical time step [CRITICAL - must match simulation]
dt_iteration = 2.0e-06  # Physical time per iteration [s]

# Compute reference dynamic viscosity
mu_ref = rho_ref * u_infty * c / Re_c

# Fixed probe location for signal correlation analysis
# Surface probe is added automatically at the detected surface y of this slice
Y_PROBE_FIXED = 0.09

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    """Check path exists and print confirmation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"✓ {kind} exists: {path}")


def find_all_slices(base_path: str) -> list:
    """Find all available slices in base directory."""
    slices = []
    for item in sorted(os.listdir(base_path)):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and re.match(r'slice_\w+$', item):
            match = re.search(r'slice_(\w+)', item)
            if match:
                slice_name = match.group(1)
                # Try to extract numeric ID for sorting, otherwise use alphabetically
                numeric_id = int(slice_name) if slice_name.isdigit() else float('inf')
                slices.append((numeric_id, item, item_path))
    slices.sort(key=lambda x: x[0])
    return slices


def get_data_files_for_slice(slice_path: str) -> list:
    """Get all velocity data files in slice directory, sorted by iteration."""
    data_files = glob.glob(os.path.join(slice_path, "*-COMP-DATA.h5"))

    def get_iteration(filepath):
        match = re.search(r'_(\d+)-COMP-DATA', filepath)
        return int(match.group(1)) if match else 0

    data_files.sort(key=get_iteration)
    return data_files


def extract_velocity_at_probes(u_data: np.ndarray, probe_indices: list, z_idx: int = 0) -> list:
    """Extract u-velocity at probe locations at a single spanwise location.

    Args:
        u_data: Velocity data with shape (nz, ny, nx)
        probe_indices: List of probe definitions
        z_idx: Spanwise index to extract (default: 0 for first z position)

    Returns:
        List of u-velocity values at specified z location
    """
    values = []
    for probe in probe_indices:
        y_idx = probe['y_idx']
        # u_data shape: (nz, ny, nx) - extract at single z position
        u_value = u_data[z_idx, y_idx, 0]  # At z_idx, at specific y, x=0 (single x-plane)
        values.append(u_value)
    return values


def compute_shear_stress_at_surface(u_data: np.ndarray, v_data: np.ndarray, w_data: np.ndarray,
                                   y_idx: int, mu_ref: float,
                                   normal_at_point: np.ndarray,
                                   distance_at_point: float, z_idx: int = 0) -> float:
    """Compute wall shear stress at a surface point at a single spanwise location.

    Args:
        u_data, v_data, w_data: Velocity components with shape (nz, ny, nx)
        y_idx: y-index of the surface point
        mu_ref: Reference dynamic viscosity
        normal_at_point: Surface normal vector at the point
        distance_at_point: Wall distance at the point
        z_idx: Spanwise index to extract (default: 0 for first z position)

    Returns:
        Shear stress at specified z location
    """
    # Compute tangent vector from the normal (2D normal -> 2D tangent)
    tangent_at_point = np.array([normal_at_point[1], -normal_at_point[0], 0.0])
    tangent_norm = np.linalg.norm(tangent_at_point)
    if tangent_norm == 0.0:
        raise ValueError("Zero tangent norm at surface point")
    tangent_at_point = tangent_at_point / tangent_norm

    # Extract velocity at single z position
    u_val = u_data[z_idx, y_idx, 0]  # At z_idx, at y_idx, x=0
    v_val = v_data[z_idx, y_idx, 0]
    w_val = w_data[z_idx, y_idx, 0]

    # Project velocity onto tangent direction
    u_t_val = (u_val * tangent_at_point[0] +
               v_val * tangent_at_point[1] +
               w_val * tangent_at_point[2])

    # Compute shear stress: τ = μ * du_tangent/dn
    tau_val = mu_ref * u_t_val / distance_at_point

    return tau_val


# ============================================================================
# LOAD GEOMETRY AND MESH
# ============================================================================

print("="*70)
print("LOAD GEOMETRY AND MESH")
print("="*70)

assert_exists(GEO_FILE, "Geometrical data file")
assert_exists(MESH_SLICE_FILE, "Mesh slice file")

# Load geometrical data
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

# Extract suction and pressure side surfaces
suction_side_points = interface_points[interface_points[:, 1] >= 0]
suction_side_indices = np.where(interface_points[:, 1] >= 0)[0]
pressure_side_points = interface_points[interface_points[:, 1] < 0]

print(f"Suction side points: {suction_side_points.shape[0]}")
print(f"Pressure side points: {pressure_side_points.shape[0]}")

# Load mesh
loader = CompressedSnapshotLoader(MESH_SLICE_FILE)
x_data = loader.x[1:-1, :, :]  # Exclude ghost cells at z boundaries
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print(f"Mesh shape (nz, ny, nx): {x_data.shape}")

# ============================================================================
# VERIFY SLICE STRUCTURE
# ============================================================================

# Verify slice is a single x-plane
x_unique_in_mesh = np.unique(x_data)
if len(x_unique_in_mesh) > 1:
    raise ValueError(f"Slice mesh has {len(x_unique_in_mesh)} unique x values. "
                     f"Expected single x-plane (2D slice). "
                     f"x range: {x_unique_in_mesh.min():.6e} to {x_unique_in_mesh.max():.6e}")

# Verify x is constant (within numerical precision)
x_all = x_data.flatten()
x_std = np.std(x_all[~np.isnan(x_all)])
x_rel_std = x_std / np.abs(x_all[~np.isnan(x_all)].mean() + 1e-15)
if x_rel_std > 1e-6:
    raise ValueError(f"x-coordinate varies too much in slice (rel_std={x_rel_std:.6e}). "
                     f"This slice may not be a valid x-plane.")

slice_x = x_data[0, 0, 0]
print(f"\n✓ Slice structure verified: single x-plane (nx=1), ny={x_data.shape[1]}, nz={x_data.shape[0]}")
print(f"  Slice x-coordinate: {slice_x:.6f}")

# Get spanwise parameters
z_unique = np.unique(z_data[:, 0, 0])
nz = z_unique.size
dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
L_z = dz * nz

print(f"  Spanwise domain: nz={nz}, dz={dz:.6e} m, Lz={L_z:.6e} m")

# Build y-grid
y_unique = np.unique(y_data[:, :, 0][0, :])
print(f"  Total y-grid points: {len(y_unique)}, range: {y_unique[0]:.6e} to {y_unique[-1]:.6e}")

# ============================================================================
# FIND CLOSEST INTERFACE POINT ON SUCTION SIDE
# ============================================================================

print("\n" + "="*70)
print("FINDING CLOSEST INTERFACE POINT")
print("="*70)

# Find closest surface point at this slice x-location
x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
closest_surface_point = suction_side_points[closest_idx]
closest_interface_idx = suction_side_indices[closest_idx]
surface_x = closest_surface_point[0]
surface_y = closest_surface_point[1]

# Get geometric data at the closest surface point
surface_normal = proj_normals[closest_interface_idx]
surface_distance = proj_distances[closest_interface_idx]

print(f"  Slice x-coordinate: {slice_x:.6f}")
print(f"  Closest surface point found:")
print(f"    x: {surface_x:.6e}")
print(f"    y: {surface_y:.6e}")
print(f"    Distance from slice x: {x_distances[closest_idx]:.6e}")
print(f"    Surface normal: [{surface_normal[0]:.6f}, {surface_normal[1]:.6f}]")
print(f"    Wall distance: {surface_distance:.6e}")

# ============================================================================
# SELECT PROBE LOCATIONS AT FIXED Y
# ============================================================================

print("\n" + "="*70)
print("SELECTING PROBE LOCATIONS (SURFACE + FIXED PROBE)")
print("="*70)

probe_locations = []
probe_definitions = [
    {'label': 'surface', 'y_target': surface_y},
    {'label': 'probe_x0p9_y0p09', 'y_target': Y_PROBE_FIXED}
]

for i, probe_def in enumerate(probe_definitions):
    y_target = probe_def['y_target']
    # Find closest grid point
    idx_closest = np.argmin(np.abs(y_unique - y_target))
    y_actual = y_unique[idx_closest]

    dist_error = np.abs(y_actual - y_target)

    probe_locations.append({
        'probe_id': i,
        'label': probe_def['label'],
        'y_target': y_target,
        'y_actual': y_actual,
        'y_idx': idx_closest,
        'error': dist_error
    })

    print(f"Probe {i} ({probe_def['label']}): y_target={y_target:.6e} → "
          f"y_actual={y_actual:.6e}, error={dist_error:.6e} (j_idx={idx_closest})")

# ============================================================================
# INFER SLICE ID
# ============================================================================

match = re.search(r'slice_(\w+)', SLICES_PATH)
if match:
    slice_id = f"slice_{match.group(1)}"
else:
    raise ValueError(f"Cannot infer slice_id from path: {SLICES_PATH}")

print(f"\nInferred slice_id from path: {slice_id}")

# ============================================================================
# VISUALIZE SLICE, AIRFOIL SURFACE AND SURFACE POINT
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION: AIRFOIL SURFACE, SURFACE POINT, AND PROBES")
print("="*70)

fig, ax = plt.subplots(figsize=(13, 8))

# Plot airfoil surfaces as scatter
ax.scatter(suction_side_points[:, 0], suction_side_points[:, 1],
          s=20, c='blue', label='Suction side', zorder=3, alpha=0.6)
ax.scatter(pressure_side_points[:, 0], pressure_side_points[:, 1],
          s=20, c='red', label='Pressure side', zorder=3, alpha=0.6)

# Plot slice plane
ax.axvline(x=slice_x, color='green', linewidth=2.5, linestyle='--',
           label=f'Slice plane (x={slice_x:.4f})', zorder=2, alpha=0.8)

# Plot closest surface point
ax.scatter(surface_x, surface_y, s=200, c='orange', marker='*',
          label=f'Surface point (y={surface_y:.4e})',
          zorder=5, edgecolors='black', linewidths=2)

# Indicate on slice line where surface point is
ax.plot(slice_x, surface_y, 'o', markersize=10, color='orange',
        markeredgecolor='black', markeredgewidth=1.5, zorder=4, alpha=0.8)

# Plot probe locations with distinct colors and markers
probe_colors = ['purple', 'cyan']
probe_markers = ['s', '^']  # square and triangle

for probe in probe_locations:
    probe_idx = probe['probe_id']
    y_actual = probe['y_actual']
    color = probe_colors[probe_idx % len(probe_colors)]
    marker = probe_markers[probe_idx % len(probe_markers)]

    # Plot requested location (diamond outline)
    ax.plot(slice_x, probe['y_target'], 'D', markersize=8, color=color,
            markeredgecolor='black', markeredgewidth=1, zorder=5,
            alpha=0.6, label=f"{probe['label']}_target")

    # Plot actual grid location (filled marker)
    ax.plot(slice_x, y_actual, marker, markersize=10, color=color,
            markeredgecolor='black', markeredgewidth=1.5, zorder=5,
            alpha=1.0, label=f"{probe['label']}_actual (j={probe['y_idx']})")

    # Connect if different
    if probe['error'] > 1e-6:
        ax.plot([slice_x, slice_x], [probe['y_target'], y_actual], '--',
                color=color, linewidth=1, alpha=0.5, zorder=4)

ax.set_xlabel('x (chord)', fontsize=12, fontweight='bold')
ax.set_ylabel('y (chord)', fontsize=12, fontweight='bold')
ax.set_title(f'Airfoil Surface, Slice, and Probe Locations ({slice_id})\n'
             f'AOA={AOA_deg}°, slice_x={slice_x:.4f}',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
ax.set_aspect('equal')
ax.margins(0.05)

plt.tight_layout()
surface_viz_file = os.path.join(SAVE_DIR, f"airfoil_surface_point_{slice_id}.png")
plt.savefig(surface_viz_file, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {surface_viz_file}")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY - PHASE 1: SLICE LOADING AND SURFACE DETECTION")
print("="*70)
print(f"Slice ID: {slice_id}")
print(f"Slice x-location: {slice_x:.6f} chord")
print(f"AOA: {AOA_deg}°")
print(f"Mesh resolution: nz={nz}, ny={x_data.shape[1]}")

print(f"\nSurface point (closest to slice):")
print(f"  x: {surface_x:.6e}")
print(f"  y: {surface_y:.6e}")

print(f"\nProbe locations (for correlation analysis):")
for probe in probe_locations:
    print(f"  Probe {probe['probe_id']}:")
    print(f"    y_target: {probe['y_target']:.6e}")
    print(f"    y_actual: {probe['y_actual']:.6e}")
    print(f"    grid index (j): {probe['y_idx']}")
    print(f"    error: {probe['error']:.6e}")

print(f"\nOutput files:")
print(f"  Visualization: {surface_viz_file}")
print("="*70)

print("\nPhase 1 complete! Ready for:")
print("  - Phase 2: Extract time series at surface and fixed probe")
print("  - Phase 3: Compute signal correlation and coherence")
print("  - Phase 4: Cross-spectrum analysis")

# ============================================================================
# PHASE 2: EXTRACT VELOCITY TIME SERIES FROM ALL SLICES
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: EXTRACT VELOCITY TIME SERIES FROM CONFIGURED SLICE")
print("="*70)

# Use only the configured slice path
assert_exists(SLICES_PATH, "Configured slice directory")
slice_name = os.path.basename(os.path.normpath(SLICES_PATH))
slice_path = SLICES_PATH
print(f"\nConfigured slice: {slice_name}")
print(f"Slice path: {slice_path}")

# Storage for time series data (Phase 2a - raw data collection)
timeseries_raw = defaultdict(lambda: {
    'iterations': [],
    'u': [],
    'v': [],
    'w': [],
    'tau_w': []
})
probe_meta = {}

total_timesteps_collected = 0

print(f"\n{'-'*70}")
print(f"Processing {slice_name}")
print(f"{'-'*70}")

try:
    # Load mesh for this slice (to get probe indices)
    print(f"  Loading mesh...")
    mesh_files = glob.glob(os.path.join(slice_path, "*-CROP-MESH.h5"))
    if not mesh_files:
        raise FileNotFoundError(f"No mesh file found in {slice_path}")

    mesh_file = mesh_files[0]
    mesh_loader = CompressedSnapshotLoader(mesh_file)
    x_data = mesh_loader.x[1:-1, :, :]
    y_data = mesh_loader.y[1:-1, :, :]
    print(f"  Mesh shape (nz, ny, nx): {x_data.shape}")

    # Find probe indices for this slice
    print(f"  Finding probe indices...")
    y_unique = np.unique(y_data[:, :, 0][0, :])

    probe_indices = []
    for probe in probe_locations:
        y_target = probe['y_target']
        idx_closest = np.argmin(np.abs(y_unique - y_target))
        y_actual = y_unique[idx_closest]
        error = np.abs(y_actual - y_target)

        probe_indices.append({
            'probe_id': probe['probe_id'],
            'label': probe['label'],
            'y_target': y_target,
            'y_actual': y_actual,
            'y_idx': idx_closest,
            'error': error
        })

        probe_meta[probe['probe_id']] = {
            'label': probe['label'],
            'y_target': y_target,
            'y_actual': y_actual,
            'y_idx': idx_closest
        }

        print(f"    Probe {probe['probe_id']} ({probe['label']}): "
              f"y_target={y_target:.6e} → y_actual={y_actual:.6e}")

    # Get all data files for this slice
    data_files = get_data_files_for_slice(slice_path)
    print(f"  Found {len(data_files)} time steps")

    if len(data_files) == 0:
        raise FileNotFoundError(f"No data files found in {slice_path}")

    # Process each time step
    for file_idx, data_file in enumerate(data_files):
        if (file_idx + 1) % max(1, len(data_files) // 10) == 0:
            print(f"    Progress: {file_idx + 1}/{len(data_files)}")

        # Extract iteration number
        match = re.search(r'_(\d+)-COMP-DATA', data_file)
        iteration = int(match.group(1)) if match else file_idx

        try:
            # Load ALL velocity components for fluctuation computation
            snapshot = mesh_loader.load_snapshot(data_file)
            u_data_full = mesh_loader.reconstruct_field(snapshot["u"])
            v_data_full = mesh_loader.reconstruct_field(snapshot["v"])
            w_data_full = mesh_loader.reconstruct_field(snapshot["w"])

            u_data = u_data_full[1:-1, :, :]  # Exclude ghost cells
            v_data = v_data_full[1:-1, :, :]
            w_data = w_data_full[1:-1, :, :]

            # For each probe, collect u, v, w, and tau_w
            for probe_idx, probe in enumerate(probe_indices):
                probe_id = probe['probe_id']
                y_idx = probe['y_idx']

                # Extract velocity components at z=0, y_idx, x=0
                u_val = u_data[0, y_idx, 0]
                v_val = v_data[0, y_idx, 0]
                w_val = w_data[0, y_idx, 0]

                # Store raw velocity components
                timeseries_raw[probe_id]['iterations'].append(iteration)
                timeseries_raw[probe_id]['u'].append(u_val)
                timeseries_raw[probe_id]['v'].append(v_val)
                timeseries_raw[probe_id]['w'].append(w_val)

                # Compute wall shear stress for surface probe
                if probe['label'] == 'surface':
                    tau_val = compute_shear_stress_at_surface(
                        u_data, v_data, w_data,
                        y_idx,
                        mu_ref,
                        surface_normal,
                        surface_distance,
                        z_idx=0
                    )
                    timeseries_raw[probe_id]['tau_w'].append(tau_val)
                else:
                    # For non-surface probes, store NaN for tau_w
                    timeseries_raw[probe_id]['tau_w'].append(np.nan)

            total_timesteps_collected += 1

        except Exception as e:
            print(f"    ⚠ Error loading {os.path.basename(data_file)}: {e}")
            continue

    print(f"  ✓ Processed {total_timesteps_collected} valid time steps from {slice_name}")

except Exception as e:
    print(f"  ✗ Error processing configured slice: {e}")

# ============================================================================
# PHASE 2B: COMPUTE TEMPORAL MEANS
# ============================================================================

print(f"\n{'='*70}")
print("PHASE 2B: COMPUTE TEMPORAL MEANS")
print(f"{'='*70}")

# Convert raw lists to numpy arrays and compute means
means = {}
for probe_id in sorted(timeseries_raw.keys()):
    u_array = np.array(timeseries_raw[probe_id]['u'])
    v_array = np.array(timeseries_raw[probe_id]['v'])
    w_array = np.array(timeseries_raw[probe_id]['w'])
    tau_array = np.array(timeseries_raw[probe_id]['tau_w'])

    # Compute temporal means
    u_mean = np.mean(u_array)
    v_mean = np.mean(v_array)
    w_mean = np.mean(w_array)
    tau_mean = np.nanmean(tau_array)  # Use nanmean to handle NaNs

    means[probe_id] = {
        'u_mean': u_mean,
        'v_mean': v_mean,
        'w_mean': w_mean,
        'tau_mean': tau_mean
    }

    probe_label = probe_meta[probe_id]['label']
    print(f"\nProbe {probe_id} ({probe_label}):")
    print(f"  <u>  = {u_mean:.6e}")
    print(f"  <v>  = {v_mean:.6e}")
    print(f"  <w>  = {w_mean:.6e}")
    if probe_label == 'surface':
        print(f"  <τ_w> = {tau_mean:.6e}")

# ============================================================================
# PHASE 2B-VALIDATION: TEMPORAL SPACING VERIFICATION
# ============================================================================

print(f"\n{'='*70}")
print("PHASE 2B-VALIDATION: TEMPORAL SPACING VERIFICATION")
print(f"{'='*70}")

def validate_temporal_spacing(iterations_array, dt_iteration, probe_label):
    """
    Validate that temporal spacing is uniform and report statistics.

    Args:
        iterations_array: Array of iteration numbers
        dt_iteration: Physical time per iteration
        probe_label: Label for this probe

    Returns:
        is_uniform: Boolean indicating if spacing is uniform
    """
    if len(iterations_array) < 2:
        print(f"  ⚠ {probe_label}: Only {len(iterations_array)} samples, cannot validate spacing")
        return True

    # Compute iteration differences
    iter_diff = np.diff(iterations_array)

    # Statistics on iteration spacing
    iter_diff_min = np.min(iter_diff)
    iter_diff_max = np.max(iter_diff)
    iter_diff_mean = np.mean(iter_diff)
    iter_diff_std = np.std(iter_diff)

    # Time step statistics
    dt_samples = iter_diff * dt_iteration
    dt_min = np.min(dt_samples)
    dt_max = np.max(dt_samples)
    dt_mean = np.mean(dt_samples)
    dt_std = np.std(dt_samples)

    # Check uniformity: tolerance of 1% relative standard deviation
    uniformity_tolerance = 0.01
    is_uniform = (iter_diff_std / iter_diff_mean) < uniformity_tolerance

    # Print validation results
    print(f"\n  {probe_label}:")
    print(f"    Number of samples: {len(iterations_array)}")
    print(f"    Iteration range: {iterations_array[0]:,d} to {iterations_array[-1]:,d}")
    print(f"    Total iterations spanned: {iterations_array[-1] - iterations_array[0]:,d}")

    print(f"\n    Iteration spacing (between samples):")
    print(f"      Min: {iter_diff_min:,d} iterations")
    print(f"      Max: {iter_diff_max:,d} iterations")
    print(f"      Mean: {iter_diff_mean:.1f} iterations")
    print(f"      Std: {iter_diff_std:.2f} iterations (σ/μ = {iter_diff_std/iter_diff_mean:.4f})")

    print(f"\n    Time sampling statistics:")
    print(f"      Min Δt: {dt_min:.6e} s")
    print(f"      Max Δt: {dt_max:.6e} s")
    print(f"      Mean Δt: {dt_mean:.6e} s")
    print(f"      Std Δt: {dt_std:.6e} s")
    print(f"      Sampling frequency: {1.0/dt_mean:.2f} Hz (approx)")

    # Physical time span
    t_start = iterations_array[0] * dt_iteration
    t_end = iterations_array[-1] * dt_iteration
    t_span = t_end - t_start
    print(f"\n    Physical time span:")
    print(f"      Start: {t_start:.6f} s (iteration {iterations_array[0]:,d})")
    print(f"      End: {t_end:.6f} s (iteration {iterations_array[-1]:,d})")
    print(f"      Duration: {t_span:.6f} s")

    # Check for uniformity
    if is_uniform:
        print(f"    ✓ Temporal spacing is UNIFORM (σ/μ = {iter_diff_std/iter_diff_mean:.4f} < {uniformity_tolerance})")
    else:
        print(f"    ⚠ WARNING: Non-uniform temporal spacing detected")
        print(f"              σ/μ = {iter_diff_std/iter_diff_mean:.4f} > {uniformity_tolerance}")

    # Detect gaps (iterations not consecutive)
    expected_iter_diff = int(np.round(iter_diff_mean))
    gaps = np.where(iter_diff != expected_iter_diff)[0]

    if len(gaps) > 0:
        print(f"    ⚠ {len(gaps)} non-standard spacing(s) detected:")
        for gap_idx in gaps[:5]:  # Show first 5
            print(f"        Between samples {gap_idx} and {gap_idx+1}: {iter_diff[gap_idx]:,d} iterations (expected ~{expected_iter_diff:,d})")
        if len(gaps) > 5:
            print(f"        ... and {len(gaps) - 5} more")
    else:
        print(f"    ✓ All time steps have consistent spacing ({expected_iter_diff:,d} iterations)")

    return is_uniform

# Validate temporal spacing for all probes
spacing_valid = {}
for probe_id in sorted(timeseries_raw.keys()):
    iterations = np.array(timeseries_raw[probe_id]['iterations'])
    probe_label = probe_meta[probe_id]['label']
    is_valid = validate_temporal_spacing(iterations, dt_iteration, f"Probe {probe_id} ({probe_label})")
    spacing_valid[probe_id] = is_valid

print(f"\n{'='*70}")

# ============================================================================
# PHASE 2C: COMPUTE FLUCTUATIONS AND TKE IN FLOW-ALIGNED FRAME
# ============================================================================

print(f"\n{'='*70}")
print("PHASE 2C: COMPUTE FLUCTUATIONS AND TKE (FLOW-ALIGNED FRAME)")
print(f"{'='*70}")

# Storage for final time series (fluctuations)
timeseries_data = defaultdict(lambda: {
    'iterations': [],
    'u_prime': [],
    'v_prime': [],
    'w_prime': [],
    'tau_prime': [],
    'tke': []
})

# Rotation coefficients for flow-aligned frame
cos_aoa = np.cos(AOA_rad)
sin_aoa = np.sin(AOA_rad)

# Process each probe
for probe_id in sorted(timeseries_raw.keys()):
    u_array = np.array(timeseries_raw[probe_id]['u'])
    v_array = np.array(timeseries_raw[probe_id]['v'])
    w_array = np.array(timeseries_raw[probe_id]['w'])
    tau_array = np.array(timeseries_raw[probe_id]['tau_w'])
    iterations = np.array(timeseries_raw[probe_id]['iterations'])

    # Get means
    u_mean = means[probe_id]['u_mean']
    v_mean = means[probe_id]['v_mean']
    w_mean = means[probe_id]['w_mean']
    tau_mean = means[probe_id]['tau_mean']

    # Rotate to flow-aligned frame (grid frame → flow-aligned)
    u_rot = u_array * cos_aoa + v_array * sin_aoa
    v_rot = -u_array * sin_aoa + v_array * cos_aoa
    w_rot = w_array

    u_mean_rot = u_mean * cos_aoa + v_mean * sin_aoa
    v_mean_rot = -u_mean * sin_aoa + v_mean * cos_aoa
    w_mean_rot = w_mean

    # Compute fluctuations in rotated frame
    u_prime = u_rot - u_mean_rot
    v_prime = v_rot - v_mean_rot
    w_prime = w_rot - w_mean_rot
    tau_prime = tau_array - tau_mean

    # Compute TKE
    tke = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)

    # Store fluctuations
    timeseries_data[probe_id]['iterations'] = list(iterations)
    timeseries_data[probe_id]['u_prime'] = list(u_prime)
    timeseries_data[probe_id]['v_prime'] = list(v_prime)
    timeseries_data[probe_id]['w_prime'] = list(w_prime)
    timeseries_data[probe_id]['tau_prime'] = list(tau_prime)
    timeseries_data[probe_id]['tke'] = list(tke)

    # Verification
    probe_label = probe_meta[probe_id]['label']
    print(f"\nProbe {probe_id} ({probe_label}):")
    print(f"  Fluctuation checks:")
    print(f"    mean(u') = {np.mean(u_prime):.6e} (expect ~0)")
    print(f"    mean(v') = {np.mean(v_prime):.6e} (expect ~0)")
    print(f"    mean(w') = {np.mean(w_prime):.6e} (expect ~0)")
    print(f"  TKE stats:")
    print(f"    min = {np.min(tke):.6e}")
    print(f"    max = {np.max(tke):.6e}")
    print(f"    mean = {np.mean(tke):.6e}")



# ============================================================================
# SAVE TIME SERIES DATA
# ============================================================================

print(f"\n{'='*70}")
print("SAVING TIME SERIES DATA")
print(f"{'='*70}")

output_file = os.path.join(SAVE_DIR, f"velocity_timeseries_{slice_name}_test.h5")

with h5py.File(output_file, 'w') as f:
    # Create group for probe data
    probes_group = f.create_group('probes')

    for probe_id in sorted(timeseries_data.keys()):
        probe_group = probes_group.create_group(f'probe_{probe_id}')

        iterations = np.array(timeseries_data[probe_id]['iterations'])
        probe_label = probe_meta[probe_id]['label']

        # Store iterations and time
        probe_group.create_dataset('iterations', data=iterations)
        time_steps = iterations * dt_iteration
        probe_group.create_dataset('time', data=time_steps)

        # Store data based on probe type
        if probe_label == 'surface':
            # For surface probe: only save tau_prime
            tau_prime = np.array(timeseries_data[probe_id]['tau_prime'])
            probe_group.create_dataset('tau_prime', data=tau_prime, compression='gzip')
        else:
            # For other probes: save velocity fluctuations and TKE
            u_prime = np.array(timeseries_data[probe_id]['u_prime'])
            v_prime = np.array(timeseries_data[probe_id]['v_prime'])
            w_prime = np.array(timeseries_data[probe_id]['w_prime'])
            tke = np.array(timeseries_data[probe_id]['tke'])

            probe_group.create_dataset('u_prime', data=u_prime, compression='gzip')
            probe_group.create_dataset('v_prime', data=v_prime, compression='gzip')
            probe_group.create_dataset('w_prime', data=w_prime, compression='gzip')
            probe_group.create_dataset('tke', data=tke, compression='gzip')

        # Store metadata
        probe_group.attrs['num_timesteps'] = len(iterations)
        probe_group.attrs['label'] = probe_label
        probe_group.attrs['y_target'] = probe_meta[probe_id]['y_target']
        probe_group.attrs['y_actual'] = probe_meta[probe_id]['y_actual']
        probe_group.attrs['y_index'] = probe_meta[probe_id]['y_idx']
        probe_group.attrs['dt_iteration'] = dt_iteration
        probe_group.attrs['slice_used'] = slice_name
        probe_group.attrs['aoa_deg'] = AOA_deg
        probe_group.attrs['coordinate_frame'] = 'flow-aligned (rotated by AOA)'

        # Store temporal spacing validation result
        probe_group.attrs['temporal_spacing_uniform'] = spacing_valid[probe_id]

        # Store formulas (only relevant ones)
        if probe_label != 'surface':
            probe_group.attrs['formula_u_prime'] = 'u_flow - <u_flow> (after AOA rotation)'
            probe_group.attrs['formula_v_prime'] = 'v_flow - <v_flow> (after AOA rotation)'
            probe_group.attrs['formula_w_prime'] = 'w - <w>'
            probe_group.attrs['formula_tke'] = '0.5 * (u_prime^2 + v_prime^2 + w_prime^2)'
        else:
            probe_group.attrs['formula_tau_prime'] = 'tau_w - <tau_w>'

        # Store temporal means for reference
        probe_group.attrs['u_mean_rot'] = means[probe_id]['u_mean'] * np.cos(AOA_rad) + means[probe_id]['v_mean'] * np.sin(AOA_rad)
        probe_group.attrs['v_mean_rot'] = -means[probe_id]['u_mean'] * np.sin(AOA_rad) + means[probe_id]['v_mean'] * np.cos(AOA_rad)
        probe_group.attrs['w_mean'] = means[probe_id]['w_mean']
        if probe_label == 'surface':
            probe_group.attrs['tau_mean'] = means[probe_id]['tau_mean']

    # Store metadata at root level
    f.attrs['total_timesteps'] = total_timesteps_collected
    f.attrs['num_probes'] = len(timeseries_data)
    f.attrs['probe_locations_target'] = np.array([probe_meta[i]['y_target'] for i in sorted(probe_meta.keys())])
    f.attrs['probe_locations_actual'] = np.array([probe_meta[i]['y_actual'] for i in sorted(probe_meta.keys())])
    f.attrs['probe_labels'] = np.array([probe_meta[i]['label'] for i in sorted(probe_meta.keys())], dtype='S64')
    f.attrs['dt_iteration'] = dt_iteration
    f.attrs['slice_used'] = slice_name
    f.attrs['aoa_deg'] = AOA_deg
    f.attrs['coordinate_frame'] = 'flow-aligned (rotated by AOA)'
    f.attrs['description'] = 'Velocity fluctuations and TKE time series at probe locations'
    f.attrs['temporal_spacing_uniform'] = all(spacing_valid.values())
    f.attrs['temporal_spacing_validation'] = 'All probes checked for uniform time stepping'

print(f"✓ Time series data saved: {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("FINAL SUMMARY - ALL PHASES COMPLETE")
print(f"{'='*70}")
print(f"Slice processed: {slice_name}")
print(f"Total timesteps collected: {total_timesteps_collected}")
print(f"Number of probes: {len(timeseries_data)}")

for probe_id in sorted(timeseries_data.keys()):
    num_ts = len(timeseries_data[probe_id]['iterations'])
    u_prime = np.array(timeseries_data[probe_id]['u_prime'])
    v_prime = np.array(timeseries_data[probe_id]['v_prime'])
    w_prime = np.array(timeseries_data[probe_id]['w_prime'])
    tau_prime = np.array(timeseries_data[probe_id]['tau_prime'])
    tke = np.array(timeseries_data[probe_id]['tke'])

    print(f"\nProbe {probe_id}:")
    print(f"  Label: {probe_meta[probe_id]['label']}")
    print(f"  Y-target: {probe_meta[probe_id]['y_target']:.6e}")
    print(f"  Y-actual: {probe_meta[probe_id]['y_actual']:.6e}")
    print(f"  Number of time steps: {num_ts}")
    if num_ts > 0:
        print(f"  Time span: {timeseries_data[probe_id]['iterations'][0]} to "
              f"{timeseries_data[probe_id]['iterations'][-1]} iterations")
        print(f"  u' range: {np.min(u_prime):.6e} to {np.max(u_prime):.6e} m/s")
        print(f"  v' range: {np.min(v_prime):.6e} to {np.max(v_prime):.6e} m/s")
        print(f"  w' range: {np.min(w_prime):.6e} to {np.max(w_prime):.6e} m/s")
        print(f"  TKE range: {np.min(tke):.6e} to {np.max(tke):.6e} (m/s)^2")
        if probe_meta[probe_id]['label'] == 'surface':
            print(f"  τ' range: {np.min(tau_prime):.6e} to {np.max(tau_prime):.6e} Pa")
        print(f"  Temporal spacing valid: {'✓ YES' if spacing_valid[probe_id] else '✗ NO (non-uniform)'}")

print(f"\nTemporal Spacing Validation Summary:")
all_valid = all(spacing_valid.values())
if all_valid:
    print(f"  ✓ All probes have UNIFORM temporal spacing (suitable for spectral analysis)")
else:
    print(f"  ⚠ Some probes have NON-UNIFORM temporal spacing (caution with FFT/spectral methods)")

print(f"\nOutput files:")
print(f"  Time series: {output_file}")
print("="*70)

# ============================================================================
# PLOT VELOCITY FLUCTUATIONS AND TKE TIME SERIES
# ============================================================================

print("\n" + "="*70)
print("GENERATING FLUCTUATION AND TKE TIME SERIES PLOTS")
print("="*70)

# Prepare plot data
plot_probes_data = {}
for probe_id in sorted(timeseries_data.keys()):
    iterations = np.array(timeseries_data[probe_id]['iterations'])
    u_prime = np.array(timeseries_data[probe_id]['u_prime'])
    v_prime = np.array(timeseries_data[probe_id]['v_prime'])
    w_prime = np.array(timeseries_data[probe_id]['w_prime'])
    tau_prime = np.array(timeseries_data[probe_id]['tau_prime'])
    tke = np.array(timeseries_data[probe_id]['tke'])
    time_steps = iterations * dt_iteration
    label = probe_meta[probe_id]['label']

    plot_probes_data[probe_id] = {
        'label': label,
        'time': time_steps,
        'u_prime': u_prime,
        'v_prime': v_prime,
        'w_prime': w_prime,
        'tau_prime': tau_prime,
        'tke': tke,
        'y_actual': probe_meta[probe_id]['y_actual']
    }

# ============================================================================
# COMBINED PLOT: ALL SIGNALS IN ONE FIGURE
# ============================================================================

probe_keys = sorted(plot_probes_data.keys())
fig, axes = plt.subplots(5, 1, figsize=(14, 14))

# PLOT 1: Wall Shear Stress Fluctuation (Surface Probe)
ax = axes[0]
for probe_id in probe_keys:
    probe = plot_probes_data[probe_id]
    if probe['label'] == 'surface':
        time = probe['time']
        tau_p = probe['tau_prime']
        label = probe['label']
        ax.plot(time, tau_p, linewidth=0.8, label=f"{label} (τ')", alpha=0.85, color='#d62728')

        # Add statistics box
        stats_text = f"Mean: {np.mean(tau_p):.4e}\nStd: {np.std(tau_p):.4e}\nMin: {np.min(tau_p):.4e}\nMax: {np.max(tau_p):.4e}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8, family='monospace')

ax.set_ylabel("τ' (Pa)", fontsize=10, fontweight='bold')
ax.set_title('Wall Shear Stress Fluctuation (Surface Probe)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', fontsize=9)

# PLOT 2, 3, 4: Velocity Fluctuations (Velocity Probe)
for probe_id in probe_keys:
    probe = plot_probes_data[probe_id]
    if probe['label'] != 'surface':
        time = probe['time']
        u_p = probe['u_prime']
        v_p = probe['v_prime']
        w_p = probe['w_prime']
        label = probe['label']

        # Plot u'
        ax = axes[1]
        ax.plot(time, u_p, linewidth=0.8, color='#1f77b4', alpha=0.85)
        ax.set_ylabel("u' (m/s)", fontsize=10, fontweight='bold')
        ax.set_title(f"Streamwise Velocity Fluctuation (Velocity Probe)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        stats_text = f"Mean: {np.mean(u_p):.4e}\nStd: {np.std(u_p):.4e}\nMin: {np.min(u_p):.4e}\nMax: {np.max(u_p):.4e}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=8, family='monospace')

        # Plot v'
        ax = axes[2]
        ax.plot(time, v_p, linewidth=0.8, color='#2ca02c', alpha=0.85)
        ax.set_ylabel("v' (m/s)", fontsize=10, fontweight='bold')
        ax.set_title(f"Cross-streamwise Velocity Fluctuation (Velocity Probe)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        stats_text = f"Mean: {np.mean(v_p):.4e}\nStd: {np.std(v_p):.4e}\nMin: {np.min(v_p):.4e}\nMax: {np.max(v_p):.4e}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=8, family='monospace')

        # Plot w'
        ax = axes[3]
        ax.plot(time, w_p, linewidth=0.8, color='#ff7f0e', alpha=0.85)
        ax.set_ylabel("w' (m/s)", fontsize=10, fontweight='bold')
        ax.set_title(f"Spanwise Velocity Fluctuation (Velocity Probe)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        stats_text = f"Mean: {np.mean(w_p):.4e}\nStd: {np.std(w_p):.4e}\nMin: {np.min(w_p):.4e}\nMax: {np.max(w_p):.4e}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=8, family='monospace')

# PLOT 5: Turbulent Kinetic Energy (Velocity Probe)
ax = axes[4]
for probe_id in probe_keys:
    probe = plot_probes_data[probe_id]
    if probe['label'] != 'surface':
        time = probe['time']
        tke = probe['tke']
        label = probe['label']
        ax.plot(time, tke, linewidth=0.8, label=f"{label} (TKE)", alpha=0.85, color='#9467bd')

        # Add statistics box
        stats_text = f"Mean: {np.mean(tke):.4e}\nMin: {np.min(tke):.4e}\nMax: {np.max(tke):.4e}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8, family='monospace')

ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
ax.set_ylabel('TKE (m/s)²', fontsize=10, fontweight='bold')
ax.set_title('Turbulent Kinetic Energy (Velocity Probe)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()

print(f"\n{'='*70}")
print("DISPLAYING COMBINED PLOT")
print(f"{'='*70}")
plt.show()

print(f"\n{'='*70}")
print("PLOT DISPLAY COMPLETE")
print(f"{'='*70}")

