import os
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import io
import re

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader


# ============================================================================
# Utility Functions
# ============================================================================

def numeric_sort_key(filepath):
    """
    Extract numeric components from filepath for natural (numeric) sorting.
    Handles filenames like 'snapshot_123.h5' or 'batch_456_data_789.h5'.

    Returns a tuple that sorts numerically, with fallback to lexicographic
    for non-numeric parts.
    """
    basename = os.path.basename(filepath)
    # Extract all numeric sequences and non-numeric parts
    parts = re.split(r'(\d+)', basename)
    # Convert numeric parts to int, keep others as strings
    return tuple(int(part) if part.isdigit() else part for part in parts)


def extract_iteration_number(filepath):
    """
    Extract the solver iteration number from a snapshot filename.

    Expected filename pattern:
        3d_NACA0012_Re50000_AoA12_6350000-COMP-DATA.h5

    Returns:
        iteration number as int

    Raises:
        ValueError if the filename does not match the expected pattern.
    """
    basename = os.path.basename(filepath)

    match = re.search(r'AoA\d+_(\d+)-COMP-DATA\.h5$', basename)

    if match is None:
        raise ValueError(f"Could not extract iteration number from filename: {basename}")

    return int(match.group(1))



# ============================================================================
# Configuration
# ============================================================================

# Data directories
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/"

# Pattern to match batch directories
BATCH_PATTERN = "_batch_3065*"

# Mesh data file
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Averaged snapshot file (for mean fields)
LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output configuration
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/3D_time_series"
OUTPUT_FILENAME_PREFIX = "3D_time_series_sparse"

# Cache configuration
SKIP_IF_EXISTS = False  # Set to False to recompute even if file exists

# ============================================================================
# Timing Diagnostics
# ============================================================================

script_start_time = time.time()
timing_log = {}

# ============================================================================
# Analysis Parameters
# ============================================================================

u_infty = 1.0 # Free-stream velocity
AOA = 12  # degrees
AOA_rad = np.deg2rad(AOA)
c = 1.0  # chord length

# Physical parameters
rho_ref = 1.0   # Reference density
Re_c = 50000    # Reynolds number
mu_ref = 1.0 / Re_c # Dynamic viscosity 

# Chord locations for correlation analysis (x/c values)
X_C_LOCATIONS = [0.5, 0.7, 0.9]
# X_C_LOCATIONS = [0.5]

# Spatial subsampling parameters
STRIDE_X = 5
STRIDE_Y = 25
STRIDE_Z = 1

# ============================================================================
# Safety Checks
# ============================================================================

if STRIDE_Z != 1:
    raise ValueError("STRIDE_Z must remain 1 because the full spanwise domain is required for FFT-based Δz correlation.")

if STRIDE_X < 1 or STRIDE_Y < 1:
    raise ValueError("STRIDE_X and STRIDE_Y must be positive integers.")

# ============================================================================
# Load Geometrical Data
# ============================================================================

print("=" * 70)
print("UNCONDITIONAL WALL SHEAR STRESS CORRELATION ANALYSIS")
print("SPATIALLY SUBSAMPLED IN X-Y, FULL SPANWISE DOMAIN")
print("=" * 70)
print(f"\nAnalysis configuration:")
print(f"  Chord locations (x/c): {X_C_LOCATIONS}")
print(f"  Suction side (upper surface): closest point selected")
print(f"  Spatial subsampling: STRIDE_X={STRIDE_X}, STRIDE_Y={STRIDE_Y}, STRIDE_Z={STRIDE_Z}")

print("\n" + "=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]
    proj_normals = f["proj_normals"][:]
    proj_distances = f["proj_distances"][:]

N_surf = len(interface_points)
print(f"  Number of 2D interface points: {N_surf}")

# Extract coordinates
x_interface = interface_points[:, 0]
y_interface = interface_points[:, 1]
x_over_c = x_interface

# Separate upper and lower surfaces
y_mean = np.mean(y_interface)
upper_mask = y_interface > y_mean
lower_mask = ~upper_mask

# ============================================================================
# Identify Closest Point on Suction Side
# ============================================================================

print("\n" + "=" * 70)
print("IDENTIFYING CLOSEST POINT ON SUCTION SIDE")
print("=" * 70)

point_indices = {}

for x_c_target in X_C_LOCATIONS:
    # Find points on upper surface (suction side)
    upper_indices = np.where(upper_mask)[0]

    if len(upper_indices) == 0:
        print(f"  x/c = {x_c_target:.2f}: No points on upper surface!")
        continue

    # Find the closest point to target x/c
    distances = np.abs(x_over_c[upper_indices] - x_c_target)
    closest_idx_in_upper = np.argmin(distances)
    closest_global_idx = upper_indices[closest_idx_in_upper]

    actual_x_c = x_over_c[closest_global_idx]
    actual_y = y_interface[closest_global_idx]

    # Get mesh indices immediately from pre-computed interface indices
    mesh_ix = int(interface_indices_i[closest_global_idx])
    mesh_iy = int(interface_indices_j[closest_global_idx])

    point_indices[x_c_target] = {
        'indices': np.array([closest_global_idx]),
        'x_c_actual': actual_x_c,
        'y': actual_y,
        'mesh_ix': mesh_ix,
        'mesh_iy': mesh_iy,
    }

    print(f"  x/c = {x_c_target:.2f}: index {closest_global_idx} at actual x/c = {actual_x_c:.4f}, y = {actual_y:.4f}")

if len(point_indices) == 0:
    raise RuntimeError("No points found at any specified chord locations!")

# ============================================================================
# Find All Snapshots Data Files
# ============================================================================

print("\n" + "=" * 70)
print("SEARCHING FOR SNAPSHOTS DATA FILES")
print("=" * 70)

batch_snapshot_dirs = sorted(glob(os.path.join(BASE_SNAPSHOT_DIR, BATCH_PATTERN)), key=numeric_sort_key)
print(f"  Found {len(batch_snapshot_dirs)} batch directories")

all_snapshots_files = []
for batch_dir in batch_snapshot_dirs:
    if not os.path.exists(batch_dir):
        continue

    snapshot_files = sorted(glob(os.path.join(batch_dir, "*A.h5")), key=numeric_sort_key)
    all_snapshots_files.extend(snapshot_files)

N_total_snapshots = len(all_snapshots_files)
print(f"  Total snapshots data files: {N_total_snapshots}")

if N_total_snapshots == 0:
    raise RuntimeError("No snapshots data files found!")

# ============================================================================
# Verify Snapshot Order and Extract Sampling Information
# ============================================================================

print("\n" + "=" * 70)
print("VERIFYING SNAPSHOT ORDER AND SAMPLING FREQUENCY")
print("=" * 70)

# Show snapshot file order (first 3 and last 3)
print(f"\n  Snapshot file order (first 3):")
for i in range(min(3, N_total_snapshots)):
    print(f"    {i+1}: {os.path.basename(all_snapshots_files[i])}")

if N_total_snapshots > 6:
    print(f"  ...")
    print(f"\n  Snapshot file order (last 3):")
    for i in range(max(0, N_total_snapshots - 3), N_total_snapshots):
        print(f"    {i+1}: {os.path.basename(all_snapshots_files[i])}")

# Extract frequency sampling from first 2 snapshots
if N_total_snapshots >= 2:
    print(f"\n  Extracting sampling frequency from iteration numbers...")

    try:
        # Extract iteration numbers from filenames
        iteration_numbers = []
        for filepath in all_snapshots_files:
            iter_num = extract_iteration_number(filepath)
            iteration_numbers.append(iter_num)

        iteration_numbers = np.array(iteration_numbers, dtype=int)

        # Check monotonicity
        if not np.all(np.diff(iteration_numbers) > 0):
            raise RuntimeError(
                "Snapshot iteration numbers are not strictly increasing. "
                "Check numeric sorting or filename parsing."
            )

        # Calculate iteration step between consecutive snapshots
        iter_diffs = np.diff(iteration_numbers)

        # Check for consistency
        unique_diffs = np.unique(iter_diffs)

        print(f"    First 5 iteration numbers: {iteration_numbers[:min(5, len(iteration_numbers))].tolist()}")
        print(f"    Last 5 iteration numbers: {iteration_numbers[max(0, len(iteration_numbers)-5):].tolist()}")

        if len(unique_diffs) == 1:
            iter_step = int(unique_diffs[0])
            print(f"    Iteration step (Δiter): {iter_step}")
            print(f"    ✓ Consistent sampling: all {len(iter_diffs)} intervals are {iter_step} iterations")
        else:
            print(f"    ⚠ Inconsistent sampling detected!")
            print(f"    Unique iteration steps: {sorted(unique_diffs)}")
            print(f"    Distribution:")
            for step in sorted(unique_diffs):
                count = np.sum(iter_diffs == step)
                print(f"      Δiter = {step}: {count} occurrences")
            iter_step = int(np.median(iter_diffs))
            print(f"    Using median iteration step: {iter_step}")

    except ValueError as e:
        print(f"    ✗ Error extracting iteration numbers: {e}")
        iteration_numbers = None
    except Exception as e:
        print(f"    ✗ Error processing iteration info: {e}")
        iteration_numbers = None
else:
    print(f"\n  ⚠ Only {N_total_snapshots} snapshot(s) available; cannot compute sampling frequency")
    iteration_numbers = None

# ============================================================================
# Load Mesh 
# ============================================================================

print("\n" + "=" * 70)
print("LOADING MESH")
print("=" * 70)

# Load mesh
loader = CompressedSnapshotLoader(MESH_FILE)

# Coordinates:
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print(f"Mesh Shape: x={x_data.shape[2]}, y={y_data.shape[1]}, z={z_data.shape[0]}")

# Assign mesh coordinates to each reference point
for x_c_target, point_info in point_indices.items():
    mesh_ix = point_info['mesh_ix']
    mesh_iy = point_info['mesh_iy']
    point_info['mesh_x'] = x_data[0, mesh_iy, mesh_ix]
    point_info['mesh_y'] = y_data[0, mesh_iy, mesh_ix]

# ============================================================================
# Define Correlation Spatial Window
# ============================================================================

print("\n" + "=" * 70)
print("DEFINING CORRELATION SPATIAL WINDOW FOR EACH X/C LOCATION")
print("=" * 70)

# Determine domain extents
x_min_domain, x_max_domain = np.min(x_data), np.max(x_data)
y_min_domain, y_max_domain = np.min(y_data), np.max(y_data)

print(f"  Full domain: x=[{x_min_domain:.3f}, {x_max_domain:.3f}], y=[{y_min_domain:.3f}, {y_max_domain:.3f}]")

# Define window parameters (relative to each reference point)
dx_upstream = 0.1
dx_downstream = 0.1
dy_down = 0.01
dy_up = 0.1

# Get 2D grid for index finding
x_2d = x_data[0, :, :]  # (Ny, Nx)
y_2d = y_data[0, :, :]  # (Ny, Nx)
x_1d = x_2d[0, :]
y_1d = y_2d[:, 0]

# Store cropped window info for each x/c location
crop_windows = {}

for x_c_target, point_info in point_indices.items():
    x_ref = point_info['x_c_actual']
    y_ref = point_info['y']

    print(f"\n  x/c = {x_c_target:.2f} (actual: {x_ref:.4f}, y = {y_ref:.4f}):")

    # Define window bounds
    x_min_crop = x_ref - dx_upstream
    x_max_crop = x_ref + dx_downstream
    y_min_crop = max(y_ref - dy_down, y_min_domain)
    y_max_crop = y_ref + dy_up

    # Find indices
    ix_min = np.argmin(np.abs(x_1d - x_min_crop))
    ix_max = np.argmin(np.abs(x_1d - x_max_crop))
    iy_min = np.argmin(np.abs(y_1d - y_min_crop))
    iy_max = np.argmin(np.abs(y_1d - y_max_crop))

    # Compute sparse indices
    ix_indices = np.arange(ix_min, ix_max, STRIDE_X)
    iy_indices = np.arange(iy_min, iy_max, STRIDE_Y)

    # Verify we have valid sparse grids
    if len(ix_indices) == 0 or len(iy_indices) == 0:
        raise RuntimeError(f"Empty sparse crop for x/c={x_c_target}. Check crop bounds and strides.")

    # Full crop dimensions (for reference)
    Nx_full_crop = ix_max - ix_min
    Ny_full_crop = iy_max - iy_min

    # Sparse crop dimensions
    Nx_sparse = len(ix_indices)
    Ny_sparse = len(iy_indices)

    # Nz dimension remains the same (full spanwise domain)
    Nz_phys = x_data.shape[0]

    # Store for this location
    crop_windows[x_c_target] = {
        'ix_min': ix_min,
        'ix_max': ix_max,
        'iy_min': iy_min,
        'iy_max': iy_max,
        'Nx_crop': Nx_sparse,
        'Ny_crop': Ny_sparse,
        'Nz_crop': Nz_phys,
        'stride_x': STRIDE_X,
        'stride_y': STRIDE_Y,
        'stride_z': STRIDE_Z,
        'ix_indices': ix_indices,
        'iy_indices': iy_indices,
    }

    print(f"    Window: x=[{x_min_crop:.3f}, {x_max_crop:.3f}], y=[{y_min_crop:.3f}, {y_max_crop:.3f}]")
    print(f"    Indices: ix=[{ix_min}:{ix_max}], iy=[{iy_min}:{iy_max}]")
    print(f"    Full crop shape:       (Nz={Nz_phys}, Ny={Ny_full_crop}, Nx={Nx_full_crop})")
    print(f"    Subsampled crop shape: (Nz={Nz_phys}, Ny={Ny_sparse}, Nx={Nx_sparse})")
    print(f"    Strides: stride_x={STRIDE_X}, stride_y={STRIDE_Y}, stride_z={STRIDE_Z}")

# ============================================================================
# FIGURE: DOMAIN VISUALIZATION (Sparse Grid, Reference Point, Window)
# ============================================================================

print("\n" + "=" * 70)
print("FIGURE: DOMAIN VISUALIZATION")
print("=" * 70)

fig_domain, ax = plt.subplots(figsize=(14, 9))

# Plot interface points (airfoil surface)
ax.scatter(x_interface[upper_mask], y_interface[upper_mask],
           c='blue', s=2, alpha=0.5, label='Upper surface')
ax.scatter(x_interface[lower_mask], y_interface[lower_mask],
           c='red', s=2, alpha=0.5, label='Lower surface')

# Highlight reference point
ax.scatter(x_ref, y_ref, c='green', s=100, marker='*',
           edgecolors='black', linewidths=2, label=f'Reference point (x/c={x_ref:.3f})', zorder=5)

# Draw correlation window as rectangle
rect = patches.Rectangle((x_min_crop, y_min_crop),
                          x_max_crop - x_min_crop,
                          y_max_crop - y_min_crop,
                          linewidth=3, edgecolor='green', facecolor='green',
                          alpha=0.2, label='Correlation window')
ax.add_patch(rect)

# Draw reference lines
ax.axvline(x_ref, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y_ref, color='green', linestyle='--', linewidth=1, alpha=0.5)

# Add sparse grid points visualization
x_sparse = x_2d[iy_min:iy_max:STRIDE_Y, ix_min:ix_max:STRIDE_X]
y_sparse = y_2d[iy_min:iy_max:STRIDE_Y, ix_min:ix_max:STRIDE_X]
ax.scatter(x_sparse, y_sparse, s=10, c='black', alpha=0.6, marker='s', label='Sparse grid points')

# Labels and formatting
ax.set_xlabel('x/c', fontsize=14)
ax.set_ylabel('y/c', fontsize=14)
ax.set_title(f'Correlation Domain for x/c = {x_c_target:.2f}\n(Sparse Grid and Reference Point)',
             fontsize=16, fontweight='bold')

ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Set axis limits
ax.set_xlim(min(x_min_crop - 0.1, x_min_domain), max(x_max_crop + 0.1, x_max_domain))
ax.set_ylim(min(y_min_crop - 0.05, y_min_domain), max(y_max_crop + 0.05, y_max_domain))

plt.tight_layout()
plt.show()

print("  ✓ Domain visualization plotted successfully")

# ===========================================================================
# Surface reference point is already assigned
# ===========================================================================

print("\n" + "="*70)
print("SURFACE REFERENCE POINT")
print("="*70)

for x_c_target, point_info in point_indices.items():
    print(f"  x/c = {x_c_target:.2f}:")
    print(f"    Interface point: x={point_info['x_c_actual']:.4f}, y={point_info['y']:.4f}")
    print(f"    Mesh indices: ix={point_info['mesh_ix']}, iy={point_info['mesh_iy']}")
    print(f"    Mesh coordinates: x={point_info['mesh_x']:.6f}, y={point_info['mesh_y']:.6f}")


# ==========================================================================
# Extract time series
# ==========================================================================

def load_cached_time_series(output_dir, prefix):
    """
    Check if cached time series file exists and load it.
    Returns (time_series_data, cached_file_path) if found, (None, None) otherwise.
    """
    if not os.path.exists(output_dir):
        return None, None

    # Find all matching files
    matching_files = sorted(glob(os.path.join(output_dir, f"{prefix}_*.h5")))

    if not matching_files:
        return None, None

    # Load the most recent file
    latest_file = matching_files[-1]

    print(f"\n  Found cached time series: {os.path.basename(latest_file)}")

    try:
        cached_data = {}
        with h5py.File(latest_file, 'r') as f:
            # Load metadata
            stride_x = f.attrs['stride_x']
            stride_y = f.attrs['stride_y']
            stride_z = f.attrs['stride_z']

            # Check if strides match current configuration
            if stride_x != STRIDE_X or stride_y != STRIDE_Y or stride_z != STRIDE_Z:
                print(f"  ⚠ Stride mismatch: cached ({stride_x},{stride_y},{stride_z}) vs current ({STRIDE_X},{STRIDE_Y},{STRIDE_Z})")
                return None, None

            # Load data for each x/c location
            for group_name in f.keys():
                grp = f[group_name]
                x_c_key = float(group_name.split('_')[2])

                cached_data[x_c_key] = {
                    'wall_pressure': grp['wall_pressure'][:],
                    'wall_shear_stress': grp['wall_shear_stress'][:],
                    'fluid_u_streamwise': grp['fluid_u_streamwise'][:],
                    'sparse_grid_info': {
                        'ix_indices': grp['ix_indices'][:],
                        'iy_indices': grp['iy_indices'][:],
                        'Nz': grp.attrs['Nz']
                    }
                }

        return cached_data, latest_file

    except Exception as e:
        print(f"  ✗ Error loading cache: {e}")
        return None, None


def compute_tau_w_all_z(u_data: np.ndarray, v_data: np.ndarray, w_data: np.ndarray,
                        y_idx: int, x_idx: int, mu_ref: float,
                        normal_at_point: np.ndarray,
                        distance_at_point: float) -> np.ndarray:
    """
    Compute wall shear stress for ALL z positions at once.

    Args:
        u_data, v_data, w_data: Velocity components with shape (nz, ny, nx)
        y_idx: y-index of the surface point
        x_idx: x-index of the surface point
        mu_ref: Reference dynamic viscosity
        normal_at_point: Surface normal vector
        distance_at_point: Wall distance at the point

    Returns:
        tau_w: Array of shear stress with shape (nz,)
    """
    # Compute tangent vector from normal (2D normal -> 2D tangent)
    tangent = np.array([normal_at_point[1], -normal_at_point[0], 0.0])
    tangent = tangent / np.linalg.norm(tangent)

    # Extract velocity at surface point for ALL z
    u_vals = u_data[:, y_idx, x_idx]
    v_vals = v_data[:, y_idx, x_idx]
    w_vals = w_data[:, y_idx, x_idx]

    # Project velocity onto tangent direction
    u_t_vals = u_vals * tangent[0] + v_vals * tangent[1] + w_vals * tangent[2]

    # Compute shear stress
    tau_w = mu_ref * u_t_vals / distance_at_point

    return tau_w


print("\n" + "=" * 70)
print("EXTRACTING TIME SERIES FROM ALL SNAPSHOTS")
print("=" * 70)

# Check for cached time series
print("\nChecking for cached time series...")
time_series_data, cached_file = load_cached_time_series(OUTPUT_DIR, OUTPUT_FILENAME_PREFIX)

if time_series_data is not None and SKIP_IF_EXISTS:
    print(f"✓ Loaded from cache: {os.path.basename(cached_file)}")
    for x_c_target, ts_data in time_series_data.items():
        print(f"\n  x/c = {x_c_target:.2f}:")
        print(f"    Wall pressure shape: {ts_data['wall_pressure'].shape}")
        print(f"    Wall shear stress shape: {ts_data['wall_shear_stress'].shape}")
        print(f"    Streamwise velocity shape: {ts_data['fluid_u_streamwise'].shape}")
else:
    # Compute fresh if cache not found or SKIP_IF_EXISTS is False
    if time_series_data is not None:
        print("Cache exists but SKIP_IF_EXISTS=False. Recomputing...")
    else:
        print("No cached time series found. Computing fresh...")

    # Initialize storage for time series: {x_c_target: {field_name: array(Nt, Nz)}}
    time_series_data = {}

    # Timing: start extracting all x/c locations
    extract_all_xc_start = time.time()
    timing_log['extract_all_xc_start'] = extract_all_xc_start

    for x_c_target, point_info in point_indices.items():
        # Get mesh reference indices for surface point
        ix_surf = point_info['mesh_ix']
        iy_surf = point_info['mesh_iy']

        # Get geometrical data for this surface point
        surf_idx = point_info['indices'][0]  # Interface point index
        normal_vec = proj_normals[surf_idx]  # Normal vector at surface
        distance_to_wall = proj_distances[surf_idx]  # Distance used for gradient estimate

        # Get sparse grid info for fluid points
        crop_info = crop_windows[x_c_target]
        ix_indices = crop_info['ix_indices']
        iy_indices = crop_info['iy_indices']

        Nt = N_total_snapshots
        # Use physical domain z-dimension (without ghost cells)
        # reconstruct_field() returns full size with ghosts [130],
        # but mesh has physical domain [128], so slice to [1:-1]
        Nz = x_data.shape[0]  # This is already 128 (physical domain)

        print(f"\n  x/c = {x_c_target:.2f}:")
        print(f"    Surface point: ix={ix_surf}, iy={iy_surf}")
        print(f"    Sparse grid: Nx={len(ix_indices)}, Ny={len(iy_indices)}, Nz={Nz}")
        print(f"    Time series shape: (Nt={Nt}, Nz={Nz})")

        # Storage for all snapshots
        wall_pressure_ts = np.zeros((Nt, Nz))      # (Nt, Nz)
        wall_shear_ts = np.zeros((Nt, Nz))         # (Nt, Nz)
        fluid_u_ts = np.zeros((Nt, len(iy_indices), len(ix_indices), Nz))  # (Nt, Ny, Nx, Nz)

        print(f"    Processing {Nt} snapshots...")

        # Timing: track loading and reconstruction
        if x_c_target not in timing_log:
            timing_log[f'snapshot_load_recon_{x_c_target}'] = []

        for snap_idx, snap_file in enumerate(all_snapshots_files):
            if snap_idx % max(1, N_total_snapshots // 10) == 0:
                print(f"      Progress: {snap_idx}/{Nt}", end='\r', flush=True)

            try:
                # Timing: start snapshot load and reconstruction
                load_recon_start = time.time()

                # Load snapshot (suppress verbose output)
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    snapshot = loader.load_snapshot(snap_file)
                finally:
                    sys.stdout = old_stdout

                u = snapshot["u"]
                v = snapshot["v"]
                w = snapshot["w"]
                p = snapshot["p"]
                topo = snapshot["topo"]

                # Reconstruct full 3D fields (includes ghost cells in z: 130 points)
                u_full = loader.reconstruct_field(u)
                v_full = loader.reconstruct_field(v)
                w_full = loader.reconstruct_field(w)
                p_full = loader.reconstruct_field(p)

                # Remove ghost cells to match physical domain [128 points]
                u_full = u_full[1:-1, :, :]
                v_full = v_full[1:-1, :, :]
                w_full = w_full[1:-1, :, :]
                p_full = p_full[1:-1, :, :]

                # Timing: end snapshot load and reconstruction
                load_recon_end = time.time()
                timing_log[f'snapshot_load_recon_{x_c_target}'].append(load_recon_end - load_recon_start)

                # Extract wall pressure and shear stress (all z positions at surface point)
                p_surface_all_z = p_full[:, iy_surf, ix_surf]  # (Nz,)

                # Compute wall shear stress
                tau_surface_all_z = compute_tau_w_all_z(
                    u_full, v_full, w_full,
                    iy_surf, ix_surf,
                    mu_ref,
                    normal_vec,
                    distance_to_wall
                )  # (Nz,)

                # Extract sparse fluid velocity field
                u_sparse = u_full[:, iy_indices[:, np.newaxis], ix_indices[np.newaxis, :]]  # (Nz, Ny, Nx)
                v_sparse = v_full[:, iy_indices[:, np.newaxis], ix_indices[np.newaxis, :]]  # (Nz, Ny, Nx)

                # Rotate to streamwise direction: u_s = u*cos(AOA) + v*sin(AOA)
                u_streamwise = (u_sparse * np.cos(AOA_rad) + v_sparse * np.sin(AOA_rad))
                u_streamwise = u_streamwise.transpose(1, 2, 0)  # Reorder to (Ny, Nx, Nz)

                # Store in time series
                wall_pressure_ts[snap_idx, :] = p_surface_all_z
                wall_shear_ts[snap_idx, :] = tau_surface_all_z
                fluid_u_ts[snap_idx, :, :, :] = u_streamwise
 
            except Exception as e:
                raise RuntimeError(f"Error loading snapshot {snap_file}: {e}") from e

        print(f"      Progress: {Nt}/{Nt} - Complete\n")

        # Store processed time series
        time_series_data[x_c_target] = {
            'wall_pressure': wall_pressure_ts,         # (Nt, Nz)
            'wall_shear_stress': wall_shear_ts,        # (Nt, Nz)
            'fluid_u_streamwise': fluid_u_ts,          # (Nt, Ny, Nx, Nz)
            'sparse_grid_info': {
                'ix_indices': ix_indices,
                'iy_indices': iy_indices,
                'Nz': Nz
            }
        }

        print(f"    Wall pressure shape: {wall_pressure_ts.shape}")
        print(f"    Wall shear stress shape: {wall_shear_ts.shape}")
        print(f"    Sparse velocity shape: {fluid_u_ts.shape}")

    # Timing: end extracting all x/c locations
    extract_all_xc_end = time.time()
    timing_log['extract_all_xc_end'] = extract_all_xc_end

print("\n" + "=" * 70)
print("TIME SERIES EXTRACTION COMPLETE")
print("=" * 70)
print(f"\nExtracted time series for {len(time_series_data)} x/c location(s)")
print("Data structure:")
print("  time_series_data[x_c][field][time_idx, z_idx]")
print("\nFields available:")
print("  - wall_pressure:              (Nt, Nz)")
print("  - wall_shear_stress:          (Nt, Nz)")
print("  - fluid_u_streamwise_velocity: (Nt, Ny_sparse, Nx_sparse, Nz)")

# ==========================================================================
# Save time series to HDF5 file
# ==========================================================================

print("\n" + "=" * 70)
print("SAVING TIME SERIES TO HDF5 FILE")
print("=" * 70)

# Only save if we computed fresh data (not from cache)
if not (time_series_data is not None and SKIP_IF_EXISTS):
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create HDF5 filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_PREFIX}_{timestamp}.h5")

    print(f"\nSaving to: {output_file}")

    with h5py.File(output_file, 'w') as f:
        # Store global metadata
        f.attrs['description'] = 'Raw time series of wall and fluid signals extracted from snapshots'
        f.attrs['analysis_type'] = 'Unconditional correlation - spatially subsampled'
        f.attrs['stride_x'] = STRIDE_X
        f.attrs['stride_y'] = STRIDE_Y
        f.attrs['stride_z'] = STRIDE_Z
        f.attrs['mu_ref'] = mu_ref
        f.attrs['rho_ref'] = rho_ref
        f.attrs['timestamp'] = timestamp

        # Store snapshot iteration numbers
        if iteration_numbers is not None:
            f.create_dataset('snapshot_iterations', data=iteration_numbers)

        # Store data for each x/c location
        for x_c_target, ts_data in time_series_data.items():
            # Create group for this x/c location
            grp = f.create_group(f"x_c_{x_c_target:.2f}")

            # Store point information
            point_info = point_indices[x_c_target]
            grp.attrs['x_c_target'] = x_c_target
            grp.attrs['x_c_actual'] = point_info['x_c_actual']
            grp.attrs['y_surface'] = point_info['y']
            grp.attrs['mesh_ix'] = point_info['mesh_ix']
            grp.attrs['mesh_iy'] = point_info['mesh_iy']
            grp.attrs['mesh_x'] = point_info['mesh_x']
            grp.attrs['mesh_y'] = point_info['mesh_y']

            # Store crop window information
            crop_info = crop_windows[x_c_target]
            grp.attrs['ix_min'] = crop_info['ix_min']
            grp.attrs['ix_max'] = crop_info['ix_max']
            grp.attrs['iy_min'] = crop_info['iy_min']
            grp.attrs['iy_max'] = crop_info['iy_max']

            # Store raw time series data only
            grp.create_dataset('wall_pressure', data=ts_data['wall_pressure'], compression='gzip', compression_opts=4)
            grp.create_dataset('wall_shear_stress', data=ts_data['wall_shear_stress'], compression='gzip', compression_opts=4)
            grp.create_dataset('fluid_u_streamwise', data=ts_data['fluid_u_streamwise'], compression='gzip', compression_opts=4)

            # Store sparse grid indices
            grp.create_dataset('ix_indices', data=ts_data['sparse_grid_info']['ix_indices'])
            grp.create_dataset('iy_indices', data=ts_data['sparse_grid_info']['iy_indices'])

            # Store metadata for this location
            grp.attrs['Nt'] = ts_data['wall_pressure'].shape[0]
            grp.attrs['Nz'] = ts_data['wall_pressure'].shape[1]
            grp.attrs['Ny_sparse'] = ts_data['fluid_u_streamwise'].shape[1]
            grp.attrs['Nx_sparse'] = ts_data['fluid_u_streamwise'].shape[2]

            print(f"\n  Saved data for x/c = {x_c_target:.2f}")
            print(f"    Wall pressure:              {ts_data['wall_pressure'].shape}")
            print(f"    Wall shear stress:          {ts_data['wall_shear_stress'].shape}")
            print(f"    Streamwise velocity:        {ts_data['fluid_u_streamwise'].shape}")

    print(f"\n✓ File saved successfully: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
else:
    print("\n✓ Data loaded from cache. Skipping save.")

# ==========================================================================
# Compute fluctuations
# ==========================================================================

print("\n" + "=" * 70)
print("COMPUTING FLUCTUATIONS (VELOCITY, PRESSURE, SHEAR STRESS)")
print("=" * 70)

for x_c_target, ts_data in time_series_data.items():
    print(f"\n  x/c = {x_c_target:.2f}:")

    # Extract raw time series
    wall_pressure = ts_data['wall_pressure']              # (Nt, Nz)
    wall_shear_stress = ts_data['wall_shear_stress']      # (Nt, Nz)
    fluid_u_streamwise = ts_data['fluid_u_streamwise']    # (Nt, Ny, Nx, Nz)

    # Compute mean fields (time average)
    # Pressure and shear are computed at wall surface (no NaN values)
    mean_pressure = np.mean(wall_pressure, axis=0)           # (Nz,)
    mean_shear_stress = np.mean(wall_shear_stress, axis=0)   # (Nz,)

    # Velocity field contains NaN inside airfoil, use nanmean to ignore them
    mean_u_streamwise = np.nanmean(fluid_u_streamwise, axis=0)  # (Ny, Nx, Nz)

    # Compute fluctuations
    pressure_fluctuations = wall_pressure - mean_pressure[np.newaxis, :]                      # (Nt, Nz)
    shear_stress_fluctuations = wall_shear_stress - mean_shear_stress[np.newaxis, :]         # (Nt, Nz)
    u_streamwise_fluctuations = fluid_u_streamwise - mean_u_streamwise[np.newaxis, :, :, :]  # (Nt, Ny, Nx, Nz)

    # Store fluctuations and means in time_series_data
    ts_data['mean_pressure'] = mean_pressure
    ts_data['mean_shear_stress'] = mean_shear_stress
    ts_data['mean_u_streamwise'] = mean_u_streamwise

    ts_data['pressure_fluctuations'] = pressure_fluctuations
    ts_data['shear_stress_fluctuations'] = shear_stress_fluctuations
    ts_data['u_streamwise_fluctuations'] = u_streamwise_fluctuations

    # Print statistics
    print(f"    Pressure:")
    print(f"      Mean range: [{np.min(mean_pressure):.6e}, {np.max(mean_pressure):.6e}]")
    print(f"      RMS fluctuations: {np.sqrt(np.mean(pressure_fluctuations**2)):.6e}")
    print(f"    Wall shear stress:")
    print(f"      Mean range: [{np.min(mean_shear_stress):.6e}, {np.max(mean_shear_stress):.6e}]")
    print(f"      RMS fluctuations: {np.sqrt(np.mean(shear_stress_fluctuations**2)):.6e}")
    print(f"    Streamwise velocity:")
    print(f"      Mean range: [{np.nanmin(mean_u_streamwise):.6e}, {np.nanmax(mean_u_streamwise):.6e}]")
    print(f"      RMS fluctuations: {np.sqrt(np.nanmean(u_streamwise_fluctuations**2)):.6e}")
    print(f"      Valid points: {np.count_nonzero(~np.isnan(mean_u_streamwise))} / {mean_u_streamwise.size}")

print("\n" + "=" * 70)
print("FLUCTUATION COMPUTATION COMPLETE")
print("=" * 70)
print("\nComputed fluctuation fields:")
print("  - mean_pressure:              (Nz,)")
print("  - mean_shear_stress:          (Nz,)")
print("  - mean_u_streamwise:          (Ny_sparse, Nx_sparse, Nz)")
print("  - pressure_fluctuations:      (Nt, Nz)")
print("  - shear_stress_fluctuations:  (Nt, Nz)")
print("  - u_streamwise_fluctuations:  (Nt, Ny_sparse, Nx_sparse, Nz)")

# ==========================================================================
# Timing Diagnostics Report
# ==========================================================================

print("\n" + "=" * 70)
print("TIMING DIAGNOSTICS REPORT")
print("=" * 70)

# Total script time
script_end_time = time.time()
total_script_time = script_end_time - script_start_time
timing_log['total_script_time'] = total_script_time

print(f"\nTotal script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)")

# Time spent on extracting all x/c locations
if 'extract_all_xc_start' in timing_log and 'extract_all_xc_end' in timing_log:
    extract_all_xc_time = timing_log['extract_all_xc_end'] - timing_log['extract_all_xc_start']
    extract_all_xc_pct = (extract_all_xc_time / total_script_time * 100) if total_script_time > 0 else 0
    timing_log['extract_all_xc_time'] = extract_all_xc_time

    print(f"\nTime spent extracting all x/c locations:")
    print(f"  Total: {extract_all_xc_time:.2f} seconds ({extract_all_xc_time/60:.2f} minutes)")
    print(f"  Percentage of total: {extract_all_xc_pct:.1f}%")

# Time spent on snapshot loading and reconstruction
print(f"\nTime spent loading/reconstructing snapshots (per x/c location):")

total_snapshot_time = 0
for x_c_target in sorted(point_indices.keys()):
    key = f'snapshot_load_recon_{x_c_target}'
    if key in timing_log and len(timing_log[key]) > 0:
        times = timing_log[key]
        sum_time = sum(times)
        avg_time = sum_time / len(times)
        total_snapshot_time += sum_time

        print(f"\n  x/c = {x_c_target:.2f}:")
        print(f"    Total snapshots processed: {len(times)}")
        print(f"    Total time: {sum_time:.2f} seconds")
        print(f"    Average time per snapshot: {avg_time:.4f} seconds")
        print(f"    Min time: {min(times):.4f} seconds")
        print(f"    Max time: {max(times):.4f} seconds")

        if avg_time > 0:
            throughput = 1.0 / avg_time
            print(f"    Snapshots per second: {throughput:.2f}")

if total_snapshot_time > 0:
    snapshot_pct = (total_snapshot_time / total_script_time * 100) if total_script_time > 0 else 0
    print(f"\nTotal snapshot loading/reconstruction time across all x/c locations:")
    print(f"  Total: {total_snapshot_time:.2f} seconds ({total_snapshot_time/60:.2f} minutes)")
    print(f"  Percentage of total: {snapshot_pct:.1f}%")

print("\n" + "=" * 70)

