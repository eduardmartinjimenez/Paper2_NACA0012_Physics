"""
Multi-Probe Spectral Analysis - Full Spanwise Domain
====================================================

Extends multi-probe analysis to the full spanwise domain using periodicity.
Extracts signals for 4 probes at fixed y-coordinates across all z-positions.

This script:
1. Loads correlation data and visualizes 2D map with airfoil geometry
2. Identifies 4 probe locations (fixed y) and surface reference point
3. Extracts signals τ'(t,z) and u'_i(t,z) for all z-positions across all snapshots
4. Caches raw time series data for each probe
5. Computes spectral quantities (PSD, cross-spectrum, coherence) for each z
6. Averages spectral results over z for improved statistics
7. Plots:
   - Correlation map with overlaid airfoil geometry and probes
   - Temporal signals at z-index 0 (single spanwise plane) for surface and probes
   - Z-averaged PSDs, coherence, and cross-spectrum magnitudes
"""

import os
import sys
import re
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

# Correlation data path (for visualization)
CORR_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/test_3/"
    "wall_shear_correlation_xc_0.500_alpha_1.0_all_fft.h5"
)

# Slice data paths (for signal extraction across full spanwise domain)
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_5/"
MESH_SLICE_NAME = "slice_5-CROP-MESH.h5"
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_5/"

# Geometric data
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# Probe y-coordinates (same as multi_probe_spectral_analysis.py)
PROBE_Y_COORDS = [0.057, 0.08, 0.13, 0.22]

# Output directory
OUTPUT_DIR = None

# Physical parameters
rho_ref = 1.0           # Reference density [kg/m³]
u_infty = 1.0           # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord [m]
Re_c = 50000            # Reynolds number
AOA_deg = 12.0          # Angle of attack [degrees]
AOA_rad = np.radians(AOA_deg)

dt_iteration = 2.0e-06  # Physical time per iteration [s]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity

# Spectral analysis parameters (Welch's method)
NPERSEG = 4096           # Segment length for Welch's method
NOVERLAP = NPERSEG // 2  # 50% overlap
WINDOW = 'hann'          # Window function

# Data loader module
module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from data_loader_functions import CompressedSnapshotLoader
except ImportError as e:
    print(f"Error importing data_loader_functions: {e}")
    sys.exit(1)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_tau_w_all_z(u_data, v_data, w_data, y_idx, mu_ref,
                        normal_at_point, distance_at_point):
    """
    Compute wall shear stress for ALL z positions at once.

    Args:
        u_data, v_data, w_data: Velocity components with shape (nz, ny, nx)
        y_idx: y-index of the surface point
        mu_ref: Reference dynamic viscosity
        normal_at_point: Surface normal vector
        distance_at_point: Wall distance at the point

    Returns:
        tau_w: Array of shear stress with shape (nz,)
    """
    # Compute tangent vector from normal
    tangent = np.array([normal_at_point[1], -normal_at_point[0], 0.0])
    tangent = tangent / np.linalg.norm(tangent)

    # Extract velocity at surface y_idx for ALL z
    u_vals = u_data[:, y_idx, 0]
    v_vals = v_data[:, y_idx, 0]
    w_vals = w_data[:, y_idx, 0]

    # Project velocity onto tangent direction
    u_t_vals = u_vals * tangent[0] + v_vals * tangent[1] + w_vals * tangent[2]

    # Compute shear stress
    tau_w = mu_ref * u_t_vals / distance_at_point

    return tau_w


def compute_psd_welch(signal_data, fs, window='hann', nperseg=None, noverlap=None):
    """Compute Power Spectral Density using Welch's method."""
    valid_idx = ~np.isnan(signal_data)
    signal_clean = signal_data[valid_idx]
    signal_centered = signal_clean - np.mean(signal_clean)
    f, psd = signal.welch(
        signal_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    return f, psd


def compute_cross_spectrum_welch(signal1, signal2, fs, window='hann', nperseg=None, noverlap=None):
    """Compute Cross-Spectrum magnitude between two signals using Welch's method."""
    valid_idx = ~(np.isnan(signal1) | np.isnan(signal2))
    signal1_clean = signal1[valid_idx]
    signal2_clean = signal2[valid_idx]
    signal1_centered = signal1_clean - np.mean(signal1_clean)
    signal2_centered = signal2_clean - np.mean(signal2_clean)
    f, csd = signal.csd(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    return f, np.abs(csd)


def compute_coherence_welch(signal1, signal2, fs, window='hann', nperseg=None, noverlap=None):
    """Compute magnitude-squared coherence between two signals."""
    valid_idx = ~(np.isnan(signal1) | np.isnan(signal2))
    signal1_clean = signal1[valid_idx]
    signal2_clean = signal2[valid_idx]
    signal1_centered = signal1_clean - np.mean(signal1_clean)
    signal2_centered = signal2_clean - np.mean(signal2_clean)
    f, coh = signal.coherence(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )
    return f, coh


def nondimensionalize_frequency(frequency_array, U_inf, c):
    """Convert dimensional frequency to nondimensional: f* = f * c / U_inf"""
    return frequency_array * c / U_inf


# ============================================================================
# LOAD AND VISUALIZE CORRELATION MAP
# ============================================================================

print("="*70)
print("FIGURE 1: LOAD AND VISUALIZE 2D CORRELATION MAP")
print("="*70)

if not os.path.exists(CORR_FILE):
    raise FileNotFoundError(f"Correlation file not found: {CORR_FILE}")

with h5py.File(CORR_FILE, 'r') as f:
    # Get reference x-coordinate from file attributes
    xc_ref = f.attrs['x_c_actual']

    # Load coordinate grids and correlation data
    x_grid = f['x'][...]  # Shape: (nz, ny, nx)
    y_grid = f['y'][...]
    z_grid = f['z'][...]

    # Use R_all (all points correlation)
    corr_3d = f['R_all'][...]  # Shape: (nz, ny, nx)

print(f"✓ Correlation data loaded")
print(f"  xc_ref = {xc_ref:.6f}")
print(f"  Correlation 3D shape (nz, ny, nx): {corr_3d.shape}")

# For visualization, extract correlation at z=0 (first z-plane)
corr_2d = corr_3d[0, :, :]
x_cells = x_grid[0, 0, :]  # x values along x-direction
y_cells = y_grid[0, :, 0]  # y values along y-direction

print(f"  2D slice (z=0) shape: {corr_2d.shape}")
print(f"  x range: [{x_cells[0]:.6f}, {x_cells[-1]:.6f}]")
print(f"  y range: [{y_cells[0]:.6f}, {y_cells[-1]:.6f}]")

# Generate slice identifier from xc_ref
xc_str = f"{xc_ref:.1f}".replace('.', '_')
slice_name = f"xc_{xc_str}"
print(f"  Slice: {slice_name}")

# ============================================================================
# LOAD GEOMETRY AND MESH
# ============================================================================

print("\n" + "="*70)
print("LOAD GEOMETRY AND MESH")
print("="*70)

if not os.path.exists(GEO_FILE):
    raise FileNotFoundError(f"Geometric data file not found: {GEO_FILE}")
if not os.path.exists(MESH_SLICE_FILE):
    raise FileNotFoundError(f"Mesh slice file not found: {MESH_SLICE_FILE}")

# Load geometrical data
with h5py.File(GEO_FILE, 'r') as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

# Extract suction side
suction_side_points = interface_points[interface_points[:, 1] >= 0]
suction_side_indices = np.where(interface_points[:, 1] >= 0)[0]

# Load mesh
loader = CompressedSnapshotLoader(MESH_SLICE_FILE)
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print(f"✓ Mesh loaded: shape (nz, ny, nx) = {x_data.shape}")
nz, ny, nx = x_data.shape

# Get mesh parameters
y_unique = np.unique(y_data[:, :, 0][0, :])
z_unique = np.unique(z_data[:, 0, 0])
print(f"  Y-grid: {len(y_unique)} points")
print(f"  Z-grid: {len(z_unique)} points (nz={nz})")

# ============================================================================
# VALIDATE SLICE X-LOCATION MATCHES CORRELATION FILE
# ============================================================================

print("\n" + "="*70)
print("VALIDATE SLICE X-LOCATION")
print("="*70)

slice_x = x_data[0, 0, 0]
print(f"Slice x-location from mesh: {slice_x:.6f}")
print(f"Reference x-location from correlation file: {xc_ref:.6f}")

if not np.isclose(slice_x, xc_ref, atol=1e-6):
    raise ValueError(
        f"❌ MISMATCH between correlation x-location ({xc_ref:.6f}) "
        f"and extracted slice x-location ({slice_x:.6f}). "
        f"You must use a slice that matches the correlation file x-location. "
        f"Difference: {abs(slice_x - xc_ref):.6e}"
    )

print(f"✓ Slice x-location matches correlation file")

# ============================================================================
# FIND SURFACE REFERENCE POINT
# ============================================================================

print("\n" + "="*70)
print("FIND SURFACE REFERENCE POINT")
print("="*70)

x_distances = np.abs(suction_side_points[:, 0] - xc_ref)
closest_idx = np.argmin(x_distances)
closest_surface_point = suction_side_points[closest_idx]
closest_interface_idx = suction_side_indices[closest_idx]

ref_probe_x = closest_surface_point[0]
ref_probe_y = closest_surface_point[1]

surface_normal = proj_normals[closest_interface_idx]
surface_distance = proj_distances[closest_interface_idx]

print(f"Surface reference point: x={ref_probe_x:.6f}, y={ref_probe_y:.6f}")
print(f"Wall distance: {surface_distance:.6e}")

# Find closest mesh point
ref_probe_idx = np.argmin(np.abs(y_unique - ref_probe_y))
ref_probe_info = {
    'y_target': ref_probe_y,
    'y_actual': y_unique[ref_probe_idx],
    'y_idx': ref_probe_idx
}

# ============================================================================
# SELECT PROBE LOCATIONS
# ============================================================================

print("\n" + "="*70)
print("SELECT PROBE LOCATIONS")
print("="*70)

probe_info = []
for i, y_target in enumerate(PROBE_Y_COORDS):
    idx_closest = np.argmin(np.abs(y_unique - y_target))
    y_actual = y_unique[idx_closest]
    error = np.abs(y_actual - y_target)

    probe_info.append({
        'probe_id': i,
        'y_target': y_target,
        'y_actual': y_actual,
        'y_idx': idx_closest,
        'error': error
    })
    print(f"Probe {i}: y_target={y_target:.5f} → y_actual={y_actual:.5f} (error={error:.6e})")

# ============================================================================
# VISUALIZATION 1: CORRELATION MAP WITH AIRFOIL AND PROBES
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION 1: CORRELATION MAP")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 8))

# Plot correlation map as background
im = ax.contourf(x_cells, y_cells, corr_2d, levels=20, cmap='RdBu_r', alpha=0.9)
cbar = plt.colorbar(im, ax=ax, label=r'$R_{\tau_w^\prime u^\prime}$')

# Plot suction side airfoil geometry
# ax.plot(suction_side_points[:, 0], suction_side_points[:, 1], 'b-', linewidth=2.5,
#         label='Suction side', zorder=10)

# Mark reference vertical line
ax.axvline(x=xc_ref, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
          label=f'Reference x/c={xc_ref:.4f}')

# Mark surface reference point
ax.scatter(ref_probe_x, ref_probe_info['y_actual'], c='orange', s=300, marker='*',
          edgecolors='black', linewidths=2.5, zorder=15,
          label=f'Surface ref (y={ref_probe_info["y_actual"]:.4f})')

# Plot probe locations
probe_colors = ['purple', 'cyan', 'magenta', 'brown']
probe_markers = ['s', 'o', '^', 'v']

for i, probe in enumerate(probe_info):
    color = probe_colors[i % len(probe_colors)]
    marker = probe_markers[i % len(probe_markers)]
    ax.scatter(xc_ref, probe['y_actual'], c=color, s=180, marker=marker,
              edgecolors='black', linewidths=1.5, zorder=14,
              label=f"Probe {i}: y={probe['y_actual']:.5f}")

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.01, 0.3)
ax.set_xlabel('x/c', fontsize=13, fontweight='bold')
ax.set_ylabel('y/c', fontsize=13, fontweight='bold')
ax.set_title(r'Spatial $R_{\tau_w^\prime u^\prime}$ Correlation with Airfoil Geometry and Probes ($\Delta z = 0$)',
            fontsize=13, fontweight='bold')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.2, which='both')
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

plt.tight_layout()
if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_2d_map.png"),
                dpi=150, bbox_inches='tight')
    print(f"  Saved: correlation_2d_map.png")

plt.show()

# ============================================================================
# PHASE 1: GET DATA FILES
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: GET DATA FILES")
print("="*70)

def get_iteration(filepath):
    match = re.search(r'_(\d+)-COMP-DATA', filepath)
    return int(match.group(1)) if match else 0

data_files = sorted(glob.glob(os.path.join(SLICES_PATH, "*-COMP-DATA.h5")),
                   key=get_iteration)

if not data_files:
    raise FileNotFoundError(f"No data files found in {SLICES_PATH}")

print(f"Found {len(data_files)} time steps")

# ============================================================================
# PHASE 2: DEFINE CACHE PATHS
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: SIGNAL CACHING")
print("="*70)

timeseries_cache_dir = os.path.join(os.path.dirname(CORR_FILE), "timeseries_spanwise_cache")
timeseries_cache_file = os.path.join(timeseries_cache_dir, f"timeseries_{slice_name}.h5")

spectral_cache_dir = os.path.join(os.path.dirname(CORR_FILE), "spectral_spanwise_cache")
spectral_cache_file = os.path.join(spectral_cache_dir, f"spectral_spanwise_{slice_name}.h5")

# Helper functions for caching
def load_timeseries_from_cache(cache_file):
    """Load pre-computed time series from HDF5 cache."""
    try:
        with h5py.File(cache_file, 'r') as f:
            timeseries_dict = {}
            for key in f.keys():
                try:
                    # Try to access as a group (has .keys() method)
                    subkeys = list(f[key].keys())
                    if subkeys:
                        # It's a group
                        timeseries_dict[key] = {}
                        for subkey in subkeys:
                            timeseries_dict[key][subkey] = f[key][subkey][...]
                    else:
                        # Empty group, treat as dataset
                        timeseries_dict[key] = f[key][...]
                except (AttributeError, TypeError):
                    # It's a dataset
                    timeseries_dict[key] = f[key][...]
            return timeseries_dict
    except Exception as e:
        print(f"  ⚠ Error loading timeseries cache: {e}")
        return None


def save_timeseries_to_cache(timeseries_dict, cache_file):
    """Save time series to HDF5 cache."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with h5py.File(cache_file, 'w') as f:
            for key, data_dict in timeseries_dict.items():
                if isinstance(data_dict, dict):
                    group = f.create_group(key)
                    for subkey, data_array in data_dict.items():
                        # Convert lists to arrays before saving
                        if isinstance(data_array, list):
                            data_array = np.array(data_array)
                        group.create_dataset(subkey, data=data_array, compression='gzip')
                elif isinstance(data_dict, list):
                    # Save lists as arrays
                    f.create_dataset(key, data=np.array(data_dict), compression='gzip')
                else:
                    # Save arrays directly
                    f.create_dataset(key, data=data_dict, compression='gzip')
        print(f"  ✓ Time series cached to: {timeseries_cache_file}")
        return True
    except Exception as e:
        print(f"  ⚠ Error saving timeseries cache: {e}")
        return False


def load_spectral_from_cache(cache_file):
    """Load pre-computed spectral results from HDF5 cache."""
    try:
        with h5py.File(cache_file, 'r') as f:
            spectral_dict = {}
            for key in f.keys():
                try:
                    # Try to access as a group
                    subkeys = list(f[key].keys())
                    if subkeys:
                        # It's a group
                        spectral_dict[key] = {}
                        for subkey in subkeys:
                            spectral_dict[key][subkey] = f[key][subkey][...]
                    else:
                        # Empty group, treat as dataset
                        spectral_dict[key] = f[key][...]
                except (AttributeError, TypeError):
                    # It's a dataset
                    spectral_dict[key] = f[key][...]
            return spectral_dict
    except Exception as e:
        print(f"  ⚠ Error loading spectral cache: {e}")
        return None


def save_spectral_to_cache(spectral_dict, cache_file):
    """Save spectral results to HDF5 cache."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with h5py.File(cache_file, 'w') as f:
            for key, data_dict in spectral_dict.items():
                group = f.create_group(key)
                for subkey, data_array in data_dict.items():
                    group.create_dataset(subkey, data=data_array, compression='gzip')
        print(f"  ✓ Spectral results cached to: {spectral_cache_file}")
        return True
    except Exception as e:
        print(f"  ⚠ Error saving spectral cache: {e}")
        return False


# ============================================================================
# PHASE 3: LOAD OR EXTRACT TIME SERIES
# ============================================================================

print("\n" + "="*70)
print("PHASE 3: EXTRACT TIME SERIES ACROSS FULL SPANWISE DOMAIN")
print("="*70)

timeseries_data = None
if os.path.exists(timeseries_cache_file):
    print(f"Time series cache found: {timeseries_cache_file}")
    print("Loading cached time series...")
    timeseries_data = load_timeseries_from_cache(timeseries_cache_file)
    if timeseries_data:
        print(f"  ✓ Loaded time series for {len(timeseries_data)} signals")

if timeseries_data is None:
    print(f"\nExtracting time series from {len(data_files)} snapshots...")
    print(f"Extracting across full spanwise domain (nz={nz})...")

    timeseries_data = {
        'iterations': [],
        'ref_point': {'tau_w_z': []},  # Will be (Nt, Nz) - raw wall shear stress
    }

    # Initialize probe data
    for i in range(len(probe_info)):
        timeseries_data[f'probe_{i}'] = {'u_flow_z': []}  # raw flow-aligned velocity

    cos_aoa = np.cos(AOA_rad)
    sin_aoa = np.sin(AOA_rad)

    for file_idx, data_file in enumerate(data_files):
        if (file_idx + 1) % max(1, len(data_files) // 10) == 0 or file_idx == 0:
            print(f"  Progress: {file_idx + 1}/{len(data_files)}")

        # Extract iteration
        match = re.search(r'_(\d+)-COMP-DATA', data_file)
        iteration = int(match.group(1)) if match else file_idx

        try:
            snapshot = loader.load_snapshot(data_file)
            u_data = loader.reconstruct_field(snapshot["u"])[1:-1, :, :]
            v_data = loader.reconstruct_field(snapshot["v"])[1:-1, :, :]
            w_data = loader.reconstruct_field(snapshot["w"])[1:-1, :, :]

            # Extract tau for surface point across all z (raw, before mean subtraction)
            surface_y_idx = ref_probe_info['y_idx']
            tau_w_vals = compute_tau_w_all_z(u_data, v_data, w_data,
                                             surface_y_idx, mu_ref,
                                             surface_normal, surface_distance)

            # Store iteration and surface tau
            timeseries_data['iterations'].append(iteration)
            timeseries_data['ref_point']['tau_w_z'].append(tau_w_vals)

            # Extract u for each probe across all z (raw, before mean subtraction)
            for i, probe in enumerate(probe_info):
                probe_y_idx = probe['y_idx']
                u_vals = u_data[:, probe_y_idx, 0]
                v_vals = v_data[:, probe_y_idx, 0]

                # Rotate to flow-aligned frame
                u_flow_vals = u_vals * cos_aoa + v_vals * sin_aoa

                # Store
                timeseries_data[f'probe_{i}']['u_flow_z'].append(u_flow_vals)

        except Exception as e:
            print(f"  ⚠ Error loading snapshot {file_idx}: {e}")
            continue

    # Convert to arrays
    print("\nConverting to arrays...")
    iterations = np.array(timeseries_data['iterations'])
    timeseries_data['time'] = iterations * dt_iteration

    timeseries_data['ref_point']['tau_w_z'] = np.array(timeseries_data['ref_point']['tau_w_z'])

    for i in range(len(probe_info)):
        timeseries_data[f'probe_{i}']['u_flow_z'] = np.array(timeseries_data[f'probe_{i}']['u_flow_z'])

    Nt = len(iterations)
    print(f"  Time series shape: (Nt={Nt}, Nz={nz})")

    # Compute temporal means and fluctuations (now call them tau_prime and u_prime)
    print("Computing fluctuations...")
    ref_tau_mean = np.mean(timeseries_data['ref_point']['tau_w_z'], axis=0)
    timeseries_data['ref_point']['tau_prime'] = timeseries_data['ref_point']['tau_w_z'] - ref_tau_mean

    for i in range(len(probe_info)):
        u_mean = np.mean(timeseries_data[f'probe_{i}']['u_flow_z'], axis=0)
        timeseries_data[f'probe_{i}']['u_prime'] = timeseries_data[f'probe_{i}']['u_flow_z'] - u_mean

    # Save cache
    save_timeseries_to_cache(timeseries_data, timeseries_cache_file)

# ============================================================================
# PHASE 4: COMPUTE SAMPLING FREQUENCY
# ============================================================================

print("\n" + "="*70)
print("PHASE 4: COMPUTE SAMPLING FREQUENCY")
print("="*70)

# Compute time array if not already present
if 'time' not in timeseries_data:
    # Try to get iterations
    if 'iterations' in timeseries_data and isinstance(timeseries_data['iterations'], np.ndarray):
        iterations = timeseries_data['iterations']
    elif 'iterations' in timeseries_data and isinstance(timeseries_data['iterations'], list):
        iterations = np.array(timeseries_data['iterations'])
    else:
        # Fallback: extract from any dataset
        ref_data_shape = timeseries_data['ref_point']['tau_w_z'].shape
        iterations = np.arange(ref_data_shape[0])
    timeseries_data['time'] = iterations * dt_iteration

time = timeseries_data['time']
if isinstance(time, list):
    time = np.array(time)
    timeseries_data['time'] = time
dt_mean = np.mean(np.diff(time))
fs = 1.0 / dt_mean
Nt = len(time)

print(f"Sampling frequency: {fs:.2f} Hz")
print(f"Total time span: {time[-1] - time[0]:.2f} s")
print(f"Number of samples: {Nt}")

# ============================================================================
# FIGURE 2: TEMPORAL SIGNALS AT Z=0 (5x1 LAYOUT)
# ============================================================================

print("\n" + "="*70)
print("FIGURE 2: TEMPORAL SIGNALS AT Z-INDEX 0 (SINGLE SPANWISE PLANE)")
print("="*70)

# Extract instantaneous signals at z=0
z_idx = 0
tau_prime_z0 = timeseries_data['ref_point']['tau_prime'][:, z_idx]

fig_signals, axes_signals = plt.subplots(5, 1, figsize=(14, 15))

# Probe colors (same as correlation map)
probe_colors_ts = ['purple', 'cyan', 'magenta', 'brown']

# ========================================================================
# SUBPLOT 1: WALL SHEAR STRESS FLUCTUATION (Surface Point at z=0)
# ========================================================================
ax_tau = axes_signals[0]

ax_tau.plot(time, tau_prime_z0, linewidth=0.8, alpha=0.85,
            color='#d62728', label='Surface ref', zorder=3)
ax_tau.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
ax_tau.grid(True, alpha=0.3, which='both', zorder=0)

ax_tau.set_ylabel(r"$\tau^\prime$ (Pa)", fontsize=11, fontweight='bold')
ax_tau.set_title(r'Wall Shear Stress $\tau^\prime$ (Surface, z-index = 0)',
                fontsize=11, fontweight='bold')
ax_tau.legend(loc='upper right', fontsize=9, framealpha=0.9)

# ========================================================================
# SUBPLOTS 2-5: STREAMWISE VELOCITY FLUCTUATION (Individual Probes at z-index 0)
# ========================================================================
for i, probe in enumerate(probe_info):
    ax_probe = axes_signals[i + 1]
    probe_key = f'probe_{i}'

    if probe_key in timeseries_data:
        # Extract signal at z=0
        u_prime_z0 = timeseries_data[probe_key]['u_prime'][:, z_idx]

        color = probe_colors_ts[i % len(probe_colors_ts)]

        ax_probe.plot(time, u_prime_z0, linewidth=0.8, alpha=0.85,
                     color=color, label=f'Probe {i}', zorder=3)
        ax_probe.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
        ax_probe.grid(True, alpha=0.3, which='both', zorder=0)

        ax_probe.set_ylabel(r"$u^\prime$ (m/s)", fontsize=11, fontweight='bold')
        ax_probe.set_title(f'Probe {i}: y={probe["y_actual"]:.5f} (z-index = 0)',
                          fontsize=11, fontweight='bold')
        ax_probe.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Set x-label only on last subplot
axes_signals[-1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')

plt.tight_layout()
if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_signals.savefig(os.path.join(OUTPUT_DIR, "timeseries_z0.png"),
                       dpi=150, bbox_inches='tight')
    print(f"  Saved: timeseries_z0.png")

plt.show()
print("  ✓ Time series signals (z=0) plotted successfully")

# ============================================================================
# PHASE 5: COMPUTE WELCH PARAMETERS
# ============================================================================

print("\n" + "="*70)
print("PHASE 5: VALIDATE WELCH PARAMETERS")
print("="*70)

nperseg = NPERSEG
noverlap = NOVERLAP

if nperseg > Nt:
    nperseg = Nt // 2
    noverlap = nperseg // 2
    print(f"⚠ nperseg reduced to {nperseg} (original {NPERSEG} > Nt={Nt})")

print(f"Using Welch parameters:")
print(f"  nperseg: {nperseg}")
print(f"  noverlap: {noverlap}")
print(f"  window: {WINDOW}")
print(f"  Frequency resolution: {fs / nperseg:.6e} Hz")

# ============================================================================
# PHASE 6: LOAD OR COMPUTE SPECTRAL ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("PHASE 6: SPECTRAL ANALYSIS (COMPUTE FOR EACH Z, THEN AVERAGE)")
print("="*70)

spectral_data = None
if os.path.exists(spectral_cache_file):
    print(f"Spectral cache found: {spectral_cache_file}")
    print("Loading cached spectral results...")
    spectral_data = load_spectral_from_cache(spectral_cache_file)
    if spectral_data:
        print(f"  ✓ Loaded spectral results")

if spectral_data is None:
    print(f"\nComputing spectral quantities for each of {nz} z-positions...")

    spectral_data = {'metadata': {}}

    # Compute PSD and cross-spectrum/coherence for surface reference
    print(f"\nComputing spectral data for surface reference...")
    psd_tau_all_z = []
    csd_mag_all_z = []
    coherence_all_z = []
    f = None

    for iz in range(nz):
        tau_z = timeseries_data['ref_point']['tau_prime'][:, iz]

        # PSD for tau
        f_psd, psd_tau_z = compute_psd_welch(tau_z, fs, window=WINDOW,
                                             nperseg=nperseg, noverlap=noverlap)

        psd_tau_all_z.append(psd_tau_z)

        if f is None:
            f = f_psd

        if (iz + 1) % max(1, nz // 5) == 0 or iz == 0:
            print(f"  z-index {iz}/{nz-1}: Processed")

    # Convert to arrays and average over z
    psd_tau_all_z = np.array(psd_tau_all_z)
    psd_tau_mean = np.mean(psd_tau_all_z, axis=0)

    spectral_data['ref_point'] = {
        'frequency': f,
        'psd': psd_tau_mean
    }

    print(f"  ✓ Surface reference PSD computed and z-averaged")

    # Compute for each probe
    probe_colors = ['purple', 'cyan', 'magenta', 'brown']

    for i, probe in enumerate(probe_info):
        print(f"\nComputing spectral data for Probe {i}...")
        psd_u_all_z = []
        csd_mag_all_z = []
        coherence_all_z = []

        for iz in range(nz):
            tau_z = timeseries_data['ref_point']['tau_prime'][:, iz]
            u_z = timeseries_data[f'probe_{i}']['u_prime'][:, iz]

            # PSD for u
            f_psd, psd_u_z = compute_psd_welch(u_z, fs, window=WINDOW,
                                               nperseg=nperseg, noverlap=noverlap)

            # Cross-spectrum
            f_csd, csd_mag_z = compute_cross_spectrum_welch(
                tau_z, u_z, fs, window=WINDOW, nperseg=nperseg, noverlap=noverlap
            )

            # Coherence
            f_coh, coh_z = compute_coherence_welch(
                tau_z, u_z, fs, window=WINDOW, nperseg=nperseg, noverlap=noverlap
            )

            psd_u_all_z.append(psd_u_z)
            csd_mag_all_z.append(csd_mag_z)
            coherence_all_z.append(coh_z)

            if (iz + 1) % max(1, nz // 5) == 0 or iz == 0:
                print(f"  z-index {iz}/{nz-1}: Processed")

        # Convert to arrays and average over z
        psd_u_all_z = np.array(psd_u_all_z)
        csd_mag_all_z = np.array(csd_mag_all_z)
        coherence_all_z = np.array(coherence_all_z)

        psd_u_mean = np.mean(psd_u_all_z, axis=0)
        csd_mag_mean = np.mean(csd_mag_all_z, axis=0)
        coherence_mean = np.mean(coherence_all_z, axis=0)

        spectral_data[f'probe_{i}'] = {
            'frequency': f,
            'psd': psd_u_mean,
            'frequency_csd': f_csd,
            'csd_magnitude': csd_mag_mean,
            'frequency_coh': f_coh,
            'coherence': coherence_mean
        }

        print(f"  ✓ Probe {i} spectral data computed and z-averaged")

    # Save spectral cache
    save_spectral_to_cache(spectral_data, spectral_cache_file)

# ============================================================================
# PHASE 7: CONVERT TO NONDIMENSIONAL FREQUENCY
# ============================================================================

print("\n" + "="*70)
print("PHASE 7: CONVERT TO NONDIMENSIONAL FREQUENCY")
print("="*70)

f_nd = nondimensionalize_frequency(spectral_data['ref_point']['frequency'], u_infty, c)
spectral_data['ref_point']['f_nd'] = f_nd

for i in range(len(probe_info)):
    f_csd_nd = nondimensionalize_frequency(spectral_data[f'probe_{i}']['frequency_csd'],
                                           u_infty, c)
    f_coh_nd = nondimensionalize_frequency(spectral_data[f'probe_{i}']['frequency_coh'],
                                          u_infty, c)
    spectral_data[f'probe_{i}']['f_csd_nd'] = f_csd_nd
    spectral_data[f'probe_{i}']['f_coh_nd'] = f_coh_nd

print(f"Frequency range: {f_nd[1]:.6e} to {f_nd[-1]:.6e}")

# ============================================================================
# FIGURE 3: Z-AVERAGED PSDs OVERLAY
# ============================================================================

print("\n" + "="*70)
print("FIGURE 3: Z-AVERAGED PSDs OVERLAY (LOG-LOG)")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 8))

freq_start = 1

# Plot tau' PSD
ax.loglog(f_nd[freq_start:], spectral_data['ref_point']['psd'][freq_start:],
         'o-', linewidth=1.5, markersize=3, label="τ' (Surface, z-averaged over full span)",
         color='#d62728', alpha=0.8)

# Plot u' PSDs for each probe
probe_colors = ['purple', 'cyan', 'magenta', 'brown']
for i in range(len(probe_info)):
    color = probe_colors[i % len(probe_colors)]
    ax.loglog(f_nd[freq_start:], spectral_data[f'probe_{i}']['psd'][freq_start:],
             's-', linewidth=1.5, markersize=3,
             label=f"u' Probe {i} (z-averaged over full span)", color=color, alpha=0.8)

ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=12, fontweight='bold')
ax.set_ylabel("PSD [signal²/Hz]", fontsize=12, fontweight='bold')
ax.set_title(f"Power Spectral Density (z-averaged, nz={nz})", fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

info_text = (
    f"Nt = {Nt}, Nz = {nz}\n"
    f"fs = {fs:.2f} Hz\n"
    f"Δt = {time[-1] - time[0]:.2f} s\n"
    f"nperseg = {nperseg}"
)
ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='bottom', horizontalalignment='left',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
       family='monospace')

plt.tight_layout()
if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "spectral_psd_zavg.png"),
               dpi=150, bbox_inches='tight')
    print(f"  Saved: spectral_psd_zavg.png")

plt.show()

# ============================================================================
# FIGURE 4: Z-AVERAGED COHERENCE OVERLAY
# ============================================================================

print("\n" + "="*70)
print("FIGURE 4: Z-AVERAGED COHERENCE OVERLAY (SEMILOG)")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(len(probe_info)):
    color = probe_colors[i % len(probe_colors)]
    ax.semilogx(spectral_data[f'probe_{i}']['f_coh_nd'][freq_start:],
               spectral_data[f'probe_{i}']['coherence'][freq_start:],
               'o-', linewidth=1.5, markersize=3,
               label=f"γ²(τ', u' probe {i}, z-averaged over full span)", color=color, alpha=0.8)

ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=12, fontweight='bold')
ax.set_ylabel("Magnitude-Squared Coherence γ² [-]", fontsize=12, fontweight='bold')
ax.set_title(f"Coherence: τ' vs u' (z-averaged over full span, nz={nz})", fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)

plt.tight_layout()
if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "spectral_coherence_zavg.png"),
               dpi=150, bbox_inches='tight')
    print(f"  Saved: spectral_coherence_zavg.png")

plt.show()

# ============================================================================
# FIGURE 5: Z-AVERAGED CROSS-SPECTRUM MAGNITUDE OVERLAY
# ============================================================================

print("\n" + "="*70)
print("FIGURE 5: Z-AVERAGED CROSS-SPECTRUM MAGNITUDE OVERLAY (LOG-LOG)")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(len(probe_info)):
    color = probe_colors[i % len(probe_colors)]
    ax.loglog(spectral_data[f'probe_{i}']['f_csd_nd'][freq_start:],
             spectral_data[f'probe_{i}']['csd_magnitude'][freq_start:],
             'o-', linewidth=1.5, markersize=3,
             label=f"|S_τu|(probe {i}, z-avg)", color=color, alpha=0.8)

ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=12, fontweight='bold')
ax.set_ylabel("|S_τu(f)| [signal product / Hz]", fontsize=12, fontweight='bold')
ax.set_title(f"Cross-Spectrum Magnitude: τ' vs u' (z-averaged, nz={nz})",
            fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

plt.tight_layout()
if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "spectral_crossspectrum_zavg.png"),
               dpi=150, bbox_inches='tight')
    print(f"  Saved: spectral_crossspectrum_zavg.png")

plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE: MULTI-PROBE FULL SPANWISE SPECTRAL ANALYSIS")
print("="*70)
print(f"\nDomain parameters:")
print(f"  Temporal: Nt={Nt}, fs={fs:.2f} Hz, Δt={time[-1]-time[0]:.2f} s")
print(f"  Spanwise: Nz={nz}, Full periodic domain")
print(f"  Probes: {len(probe_info)} at fixed y-coordinates")
print(f"\nWelch parameters:")
print(f"  nperseg={nperseg}, noverlap={noverlap}, window={WINDOW}")
print(f"  Frequency resolution: {fs/nperseg:.6e} Hz")
print(f"\nCache locations:")
print(f"  Time series: {timeseries_cache_file}")
print(f"  Spectral: {spectral_cache_file}")
print(f"\nGenerated figures:")
print(f"  1. Correlation map with airfoil geometry and probes")
print(f"  2. Temporal signals at z-index 0 (single spanwise plane) for surface and probes")
print(f"  3. Z-averaged PSDs for all signals")
print(f"  4. Z-averaged coherence (τ' vs each probe)")
print(f"  5. Z-averaged cross-spectrum magnitudes")
print("="*70)
