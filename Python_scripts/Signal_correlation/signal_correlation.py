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

# Storage for time series data
timeseries_data = defaultdict(lambda: {'iterations': [], 'u_velocity': []})
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
            # Load compressed velocity data using preloaded mesh/topology
            snapshot = mesh_loader.load_snapshot(data_file)
            u_data_full = mesh_loader.reconstruct_field(snapshot["u"])
            u_data = u_data_full[1:-1, :, :]  # Exclude ghost cells

            # Initialize values list for this timestep
            u_values = []

            # For each probe
            for probe_idx, probe in enumerate(probe_indices):
                if probe['label'] == 'surface':
                    # Compute wall shear stress at surface probe (first z position)
                    v_data_full = mesh_loader.reconstruct_field(snapshot["v"])
                    w_data_full = mesh_loader.reconstruct_field(snapshot["w"])
                    v_data = v_data_full[1:-1, :, :]
                    w_data = w_data_full[1:-1, :, :]

                    tau_val = compute_shear_stress_at_surface(
                        u_data, v_data, w_data,
                        probe['y_idx'],
                        mu_ref,
                        surface_normal,
                        surface_distance,
                        z_idx=0  # Extract from first z position
                    )
                    u_values.append(tau_val)
                else:
                    # Extract u-velocity at fixed probe location (first z position)
                    y_idx = probe['y_idx']
                    u_val = u_data[0, y_idx, 0]  # z=0, at y_idx, x=0
                    u_values.append(u_val)

            # Store data for each probe
            for probe_idx, val in enumerate(u_values):
                probe_id = probe_indices[probe_idx]['probe_id']
                timeseries_data[probe_id]['iterations'].append(iteration)
                timeseries_data[probe_id]['u_velocity'].append(val)

            total_timesteps_collected += 1

        except Exception as e:
            print(f"    ⚠ Error loading {os.path.basename(data_file)}: {e}")
            continue

    print(f"  ✓ Processed {total_timesteps_collected} valid time steps from {slice_name}")

except Exception as e:
    print(f"  ✗ Error processing configured slice: {e}")

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
        u_velocity = np.array(timeseries_data[probe_id]['u_velocity'])

        # Store raw data
        probe_group.create_dataset('iterations', data=iterations)
        probe_group.create_dataset('u_velocity', data=u_velocity)

        # Convert iterations to physical time
        time_steps = iterations * dt_iteration
        probe_group.create_dataset('time', data=time_steps)

        # Store metadata
        probe_group.attrs['num_timesteps'] = len(iterations)
        probe_group.attrs['label'] = probe_meta[probe_id]['label']
        probe_group.attrs['y_target'] = probe_meta[probe_id]['y_target']
        probe_group.attrs['y_actual'] = probe_meta[probe_id]['y_actual']
        probe_group.attrs['y_index'] = probe_meta[probe_id]['y_idx']
        probe_group.attrs['dt_iteration'] = dt_iteration
        probe_group.attrs['slice_used'] = slice_name

    # Store metadata at root level
    f.attrs['total_timesteps'] = total_timesteps_collected
    f.attrs['num_probes'] = len(timeseries_data)
    f.attrs['probe_locations_target'] = np.array([probe_meta[i]['y_target'] for i in sorted(probe_meta.keys())])
    f.attrs['probe_locations_actual'] = np.array([probe_meta[i]['y_actual'] for i in sorted(probe_meta.keys())])
    f.attrs['probe_labels'] = np.array([probe_meta[i]['label'] for i in sorted(probe_meta.keys())], dtype='S64')
    f.attrs['dt_iteration'] = dt_iteration
    f.attrs['slice_used'] = slice_name

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
    u_vals = timeseries_data[probe_id]['u_velocity']
    print(f"\nProbe {probe_id}:")
    print(f"  Label: {probe_meta[probe_id]['label']}")
    print(f"  Y-target: {probe_meta[probe_id]['y_target']:.6e}")
    print(f"  Y-actual: {probe_meta[probe_id]['y_actual']:.6e}")
    print(f"  Number of time steps: {num_ts}")
    if num_ts > 0:
        print(f"  Time span: {timeseries_data[probe_id]['iterations'][0]} to "
              f"{timeseries_data[probe_id]['iterations'][-1]} iterations")
        print(f"  U-velocity range: {np.min(u_vals):.6e} to {np.max(u_vals):.6e}")
        print(f"  U-velocity mean: {np.mean(u_vals):.6e}")

print(f"\nOutput files:")
print(f"  Time series: {output_file}")
print("="*70)

# ============================================================================
# PLOT VELOCITY TIME SERIES
# ============================================================================

print("\n" + "="*70)
print("GENERATING VELOCITY/SHEAR STRESS TIME SERIES PLOTS")
print("="*70)

# Prepare plot data
plot_probes_data = {}
for probe_id in sorted(timeseries_data.keys()):
    iterations = np.array(timeseries_data[probe_id]['iterations'])
    values = np.array(timeseries_data[probe_id]['u_velocity'])
    time_steps = iterations * dt_iteration
    label = probe_meta[probe_id]['label']

    # Determine y-label based on probe type
    if label == 'surface':
        y_label = 'Wall shear stress τ_w'
        value_unit = '(Pa)'
    else:
        y_label = 'u-velocity'
        value_unit = '(m/s)'

    plot_probes_data[probe_id] = {
        'label': label,
        'time': time_steps,
        'values': values,
        'y_actual': probe_meta[probe_id]['y_actual'],
        'y_label': y_label,
        'value_unit': value_unit
    }

# ============================================================================
# PLOT 1: Both signals on separate subplots
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
probe_keys = sorted(plot_probes_data.keys())

for ax_idx, probe_id in enumerate(probe_keys):
    probe = plot_probes_data[probe_id]
    time = probe['time']
    values = probe['values']
    label = probe['label']
    y_label = probe['y_label']

    ax = axes[ax_idx]
    ax.plot(time * 1000, values, linewidth=1.0, color=colors[ax_idx], alpha=0.8)

    ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_title(f'Probe: {label} (y={probe["y_actual"]:.6e})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Add statistics box
    stats_text = f"Mean: {np.mean(values):.4e}\nStd: {np.std(values):.4e}\nMin: {np.min(values):.4e}\nMax: {np.max(values):.4e}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace')

plt.tight_layout()
plt.show()

# ============================================================================
# PLOT 3: Normalized signals for direct comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

for probe_idx, probe_id in enumerate(probe_keys):
    probe = plot_probes_data[probe_id]
    time = probe['time']
    values = probe['values']
    label = probe['label']

    # Normalize: (x - mean) / std
    values_normalized = (values - np.mean(values)) / np.std(values)

    ax.plot(time * 1000, values_normalized, linewidth=1.0, label=label,
            color=colors[probe_idx], alpha=0.8)

ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Signal (σ)', fontsize=12, fontweight='bold')
ax.set_title('Normalized Time Series - Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.show()

print("\nPlot generation complete! (test mode - plots displayed, not saved)")
