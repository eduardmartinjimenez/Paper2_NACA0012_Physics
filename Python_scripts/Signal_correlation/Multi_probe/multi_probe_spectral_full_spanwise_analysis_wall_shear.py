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

# AoA 12º

# # Correlation data path (for visualization)
# CORR_FILE = (
#     "/home/jofre/Members/Eduard/Paper2/Simulations/"
#     "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
#     "Wall_shear_correlations/test_5/"
#     "wall_shear_correlation_unconditional_xc_0.900.h5"
# )

# # Slice data paths (for signal extraction across full spanwise domain)
# MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
# MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
# SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"

# # Geometric data
# GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
# GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
# GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# # Cache paths
# TIMESERIES_CACHE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_3/timeseries_spanwise_cache/"

# # Probe y-coordinates (same as multi_probe_spectral_analysis.py)
# # PROBE_Y_COORDS = [0.065, 0.07, 0.08, 0.09, 0.1, 0.11, 0.13, 0.15]  # x/c = 0.3
# # PROBE_Y_COORDS = [0.055, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]  # x/c = 0.5
# # PROBE_Y_COORDS = [0.039, 0.045, 0.055, 0.065, 0.075, 0.085, 0.10, 0.12, 0.14, 0.16, 0.18]  # x/c = 0.7
# PROBE_Y_COORDS = [0.02, 0.025, 0.035, 0.045, 0.055, 0.065, 0.085, 0.105, 0.125, 0.145, 0.165]  # x/c = 0.9


# # Output directory
# OUTPUT_DIR = None

# # Physical parameters
# rho_ref = 1.0           # Reference density [kg/m³]
# u_infty = 1.0           # Free-stream velocity [m/s]
# c = 1.0                 # Airfoil chord [m]
# Re_c = 50000            # Reynolds number
# AOA_deg = 12.0          # Angle of attack [degrees]
# AOA_rad = np.radians(AOA_deg)


# AoA 5º

# # Correlation data path (for visualization)
CORR_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/test_2/"
    "wall_shear_correlation_unconditional_xc_0.900.h5"
)

# Slice data paths (for signal extraction across full spanwise domain)
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_9/"

# Geometric data
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# Cache paths
TIMESERIES_CACHE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_1/timeseries_spanwise_cache/"

# Probe y-coordinates (same as multi_probe_spectral_analysis.py)
# PROBE_Y_COORDS = [0.054, 0.057, 0.06, 0.065, 0.07, 0.075, 0.08]  # x/c = 0.5
# PROBE_Y_COORDS = [0.038, 0.042, 0.046, 0.05, 0.055, 0.06,  0.065, 0.07]  # x/c = 0.7
PROBE_Y_COORDS = [0.0165, 0.0205, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08]  # x/c = 0.9

# Physical parameters
rho_ref = 1.0           # Reference density [kg/m³]
u_infty = 1.0           # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord [m]
Re_c = 50000            # Reynolds number
AOA_deg = 5.0          # Angle of attack [degrees]
AOA_rad = np.radians(AOA_deg)

dt_iteration = 2.0e-06  # Physical time per iteration [s]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity

# Spectral analysis parameters (Welch's method)
# NPERSEG: Segment length (larger → better frequency resolution, fewer segments)
#         Preprocessing is controlled by DETREND_TYPE below
NPERSEG = 4096           # Segment length for Welch's method
# NPERSEG = 2048           # Segment length for Welch's method
NOVERLAP = NPERSEG // 2  # 50% overlap
WINDOW = 'hann'          # Window function

# Signal preprocessing configuration
# Used consistently in Welch, PSD, CSD, coherence, variance recovery, and sensitivity analysis
DETREND_TYPE = 'linear'  # 'linear', 'constant', or None

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


def preprocess_signal_for_welch(x, detrend_type='linear'):
    """
    Preprocess signal before Welch/CSD computations.

    Converts to NumPy array, removes NaNs, and applies detrending:
    - If detrend_type == 'linear': apply scipy.signal.detrend with type='linear'
    - If detrend_type == 'constant': subtract the mean
    - If detrend_type is None: return cleaned signal unchanged

    Used consistently across Welch, PSD, CSD, coherence, variance recovery,
    and sensitivity analysis.

    Args:
        x: Input signal (array-like)
        detrend_type: 'linear' (default), 'constant', or None

    Returns:
        Preprocessed signal (1D NumPy array)

    Raises:
        ValueError: If detrend_type is not in {'linear', 'constant', None}
    """
    # Convert to NumPy array
    x_arr = np.asarray(x, dtype=float)

    # Remove NaNs
    valid_idx = ~np.isnan(x_arr)
    x_clean = x_arr[valid_idx]

    # Apply detrending
    if detrend_type == 'linear':
        x_preprocessed = signal.detrend(x_clean, type='linear')
    elif detrend_type == 'constant':
        x_preprocessed = x_clean - np.mean(x_clean)
    elif detrend_type is None:
        x_preprocessed = x_clean
    else:
        raise ValueError(
            f"Unsupported detrend_type: {detrend_type}. "
            f"Must be 'linear', 'constant', or None."
        )

    return x_preprocessed


def compute_spectra_welch(signal1, signal2, fs, window='hann', nperseg=None, noverlap=None, detrend='linear'):
    """
    Compute autospectra S_11, S_22 and complex cross-spectrum S_12 using Welch's method.

    Both signals are preprocessed consistently using preprocess_signal_for_welch()
    with the specified detrend type. NaNs are removed jointly to ensure time alignment
    between signal1 and signal2. This ensures Welch, PSD, CSD, and coherence
    all use the same signal preprocessing rule.

    Args:
        signal1, signal2: Input signals
        fs: Sampling frequency
        window, nperseg, noverlap: Welch parameters
        detrend: 'linear' (default), 'constant', or None. Controls preprocessing before Welch

    Returns:
        f: Frequency array
        S_11: Autospectrum of signal1
        S_22: Autospectrum of signal2
        S_12: Complex cross-spectrum (preserves phase information)
    """
    # Convert both signals to NumPy arrays
    signal1_arr = np.asarray(signal1, dtype=float)
    signal2_arr = np.asarray(signal2, dtype=float)

    # Build joint valid mask: keep only samples where BOTH signals are not NaN
    valid_mask = ~(np.isnan(signal1_arr) | np.isnan(signal2_arr))

    # Apply joint mask to both signals for time alignment
    signal1_masked = signal1_arr[valid_mask]
    signal2_masked = signal2_arr[valid_mask]

    # Preprocess both aligned signals using helper function
    signal1_preprocessed = preprocess_signal_for_welch(signal1_masked, detrend_type=detrend)
    signal2_preprocessed = preprocess_signal_for_welch(signal2_masked, detrend_type=detrend)

    # Compute autospectra
    f, S_11 = signal.welch(
        signal1_preprocessed,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    _, S_22 = signal.welch(
        signal2_preprocessed,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    # Compute complex cross-spectrum (preserves phase)
    _, S_12 = signal.csd(
        signal1_preprocessed,
        signal2_preprocessed,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    return f, S_11, S_22, S_12


def compute_psd_variance_recovery(signal, psd, frequency):
    """
    Compute variance from time domain and frequency domain (PSD integration).

    Args:
        signal: Zero-mean real signal (1D array)
        psd: One-sided Power Spectral Density from Welch (1D array)
        frequency: Frequency array corresponding to PSD (1D array)

    Returns:
        dict: {
            'variance_time': time-domain variance,
            'variance_freq': frequency-domain variance (integrated PSD),
            'relative_error_percent': relative error in percent
        }
    """
    # Time-domain variance
    variance_time = np.var(signal)

    # Frequency-domain variance via rectangular integration (sum rule) of one-sided PSD
    # More appropriate for discrete spectral data: Var = sum(PSD(f) * Δf)
    # Rectangular sum is better for Welch discretized spectra than trapezoidal
    df = frequency[1] - frequency[0]  # Frequency bin width (constant for Welch)
    variance_freq = np.sum(psd) * df

    # Relative error
    if variance_time > 1e-14:
        relative_error = 100.0 * np.abs(variance_freq - variance_time) / variance_time
    else:
        relative_error = 0.0 if variance_freq < 1e-14 else 100.0

    return {
        'variance_time': variance_time,
        'variance_freq': variance_freq,
        'relative_error_percent': relative_error
    }


def quality_label(rel_error_pct):
    """Qualitative label based on relative error percentage."""
    if rel_error_pct < 5.0:
        return "good"
    elif rel_error_pct < 10.0:
        return "acceptable"
    else:
        return "warning"


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
    corr_3d = f['R'][...]  # Shape: (nz, ny, nx)

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

# Mark reference vertical line
ax.axvline(x=xc_ref, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
          label=f'Reference x/c={xc_ref:.4f}')

# Mark surface reference point
ax.scatter(ref_probe_x, ref_probe_info['y_actual'], c='orange', s=300, marker='*',
          edgecolors='black', linewidths=2.5, zorder=15,
          label=f'Surface ref (y={ref_probe_info["y_actual"]:.4f})')

# Plot probe locations
probe_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5'
]

for i, probe in enumerate(probe_info):
    color = probe_colors[i % len(probe_colors)]
    ax.scatter(xc_ref, probe['y_actual'], c=color, s=180, marker='o',
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

# Determine cache directory based on configuration
if TIMESERIES_CACHE_PATH is not None:
    timeseries_cache_dir = TIMESERIES_CACHE_PATH
else:
    timeseries_cache_dir = os.path.join(os.path.dirname(CORR_FILE), "timeseries_spanwise_cache")

timeseries_cache_file = os.path.join(timeseries_cache_dir, f"timeseries_{slice_name}.h5")

print(f"  Cache path: {timeseries_cache_dir}")
print(f"  Cache file: {timeseries_cache_file}")

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


def save_timeseries_to_cache(timeseries_dict, cache_file, metadata=None):
    """Save time series to HDF5 cache with optional metadata."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with h5py.File(cache_file, 'w') as f:
            # Save metadata if provided
            if metadata:
                meta_group = f.create_group('_metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, bytes)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        meta_group.create_dataset(key, data=value, compression='gzip')
                    else:
                        meta_group.attrs[key] = value

            # Save time series data
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


def load_timeseries_metadata(cache_file):
    """Load metadata from cache file."""
    try:
        with h5py.File(cache_file, 'r') as f:
            if '_metadata' not in f:
                return None

            meta_group = f['_metadata']
            metadata = {}

            # Load attributes
            for key in meta_group.attrs:
                metadata[key] = meta_group.attrs[key]

            # Load datasets
            for key in meta_group.keys():
                metadata[key] = meta_group[key][...]

            return metadata
    except Exception as e:
        print(f"  ⚠ Error loading metadata: {e}")
        return None


def validate_cache_metadata(cache_metadata, Nt, nz, probe_coords, xc_ref, ref_y):
    """
    Validate cache metadata against current parameters.

    Args:
        cache_metadata: Dictionary loaded from cache
        Nt: Expected number of time samples
        nz: Expected number of z-planes
        probe_coords: Expected probe y-coordinates
        xc_ref: Expected slice x-location
        ref_y: Expected reference y-location

    Returns:
        (is_valid, messages) tuple where messages is list of validation info
    """
    messages = []
    is_valid = True

    if cache_metadata is None:
        return False, ["No metadata found in cache file"]

    # Check time samples
    if 'Nt' in cache_metadata:
        if cache_metadata['Nt'] != Nt:
            messages.append(f"❌ Nt mismatch: cache={cache_metadata['Nt']}, expected={Nt}")
            is_valid = False
        else:
            messages.append(f"✓ Nt matches: {Nt}")
    else:
        messages.append("⚠ Nt not in metadata")

    # Check z-planes
    if 'nz' in cache_metadata:
        if cache_metadata['nz'] != nz:
            messages.append(f"❌ nz mismatch: cache={cache_metadata['nz']}, expected={nz}")
            is_valid = False
        else:
            messages.append(f"✓ nz matches: {nz}")
    else:
        messages.append("⚠ nz not in metadata")

    # Check probe coordinates
    if 'probe_coords' in cache_metadata:
        cache_coords = cache_metadata['probe_coords']
        if not np.allclose(cache_coords, probe_coords, rtol=1e-5):
            messages.append(f"❌ Probe coordinates mismatch")
            messages.append(f"   Cache: {cache_coords}")
            messages.append(f"   Expected: {probe_coords}")
            is_valid = False
        else:
            messages.append(f"✓ Probe coordinates match: {len(probe_coords)} probes")
    else:
        messages.append("⚠ Probe coordinates not in metadata")

    # Check x-location
    if 'xc_ref' in cache_metadata:
        if not np.isclose(cache_metadata['xc_ref'], xc_ref, rtol=1e-5):
            messages.append(f"❌ xc_ref mismatch: cache={cache_metadata['xc_ref']:.6f}, expected={xc_ref:.6f}")
            is_valid = False
        else:
            messages.append(f"✓ xc_ref matches: {xc_ref:.6f}")
    else:
        messages.append("⚠ xc_ref not in metadata")

    # Check reference y-location
    if 'ref_y' in cache_metadata:
        if not np.isclose(cache_metadata['ref_y'], ref_y, rtol=1e-5):
            messages.append(f"⚠ ref_y differs: cache={cache_metadata['ref_y']:.6f}, current={ref_y:.6f}")
        else:
            messages.append(f"✓ ref_y matches: {ref_y:.6f}")
    else:
        messages.append("⚠ ref_y not in metadata")

    return is_valid, messages


# ============================================================================
# PHASE 3: LOAD OR EXTRACT TIME SERIES
# ============================================================================

print("\n" + "="*70)
print("PHASE 3: EXTRACT TIME SERIES ACROSS FULL SPANWISE DOMAIN")
print("="*70)

timeseries_data = None
if os.path.exists(timeseries_cache_file):
    print(f"Time series cache found: {timeseries_cache_file}")
    print("Checking cache metadata...")

    # Load metadata (may be None for old cache files)
    cache_metadata = load_timeseries_metadata(timeseries_cache_file)

    if cache_metadata is not None:
        # Metadata exists, validate it
        print("Validating cache metadata...")
        probe_coords_array = np.array([p['y_actual'] for p in probe_info])
        is_valid, validation_messages = validate_cache_metadata(
            cache_metadata,
            Nt=len(data_files),
            nz=nz,
            probe_coords=probe_coords_array,
            xc_ref=xc_ref,
            ref_y=ref_probe_info['y_actual']
        )

        # Print validation results
        for msg in validation_messages:
            print(f"  {msg}")

        if is_valid:
            print("✓ Cache validation passed. Loading cached time series...")
            timeseries_data = load_timeseries_from_cache(timeseries_cache_file)
            if timeseries_data:
                print(f"  ✓ Loaded time series for {len(timeseries_data)} signals")
        else:
            print("❌ Cache validation failed. Will re-extract time series.")
            timeseries_data = None
    else:
        # No metadata found - old cache file without metadata
        print("  ⚠ No metadata in cache file (pre-metadata version)")
        print("  Loading cache without validation - please verify manually")
        print("  NOTE: Cache will be re-saved with metadata on next extraction")
        timeseries_data = load_timeseries_from_cache(timeseries_cache_file)
        if timeseries_data:
            print(f"  ✓ Loaded legacy cache for {len(timeseries_data)} signals")
        else:
            timeseries_data = None
else:
    timeseries_data = None

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

    # Save cache with metadata
    cache_metadata = {
        'Nt': len(timeseries_data['iterations']),
        'nz': nz,
        'probe_coords': np.array([p['y_actual'] for p in probe_info]),
        'xc_ref': xc_ref,
        'ref_y': ref_probe_info['y_actual'],
        'dt_iteration': dt_iteration,
        'AOA_deg': AOA_deg,
        'Re_c': Re_c,
    }
    save_timeseries_to_cache(timeseries_data, timeseries_cache_file, metadata=cache_metadata)

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

fig_signals, axes_signals = plt.subplots(len(probe_info) + 1, 1, figsize=(14, 3 * (len(probe_info) + 1)))

# Probe colors (same as correlation map)
probe_colors_ts = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5'
]

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

        color = probe_colors_ts[i % len(probe_colors_ts)]  # Use same color palette

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

print(f"Computing spectral data (cache will be overwritten)...")
if True:
    print(f"\nComputing spectral quantities for each of {nz} z-positions...")

    spectral_data = {'metadata': {}}

    # Compute PSD and cross-spectrum/coherence for surface reference
    print(f"\nComputing spectral data for surface reference...")
    S_tautau_all_z = []
    f = None

    for iz in range(nz):
        tau_z = timeseries_data['ref_point']['tau_prime'][:, iz]

        # Preprocess signal using global DETREND_TYPE configuration
        tau_preprocessed = preprocess_signal_for_welch(tau_z, detrend_type=DETREND_TYPE)

        # PSD for tau (autospectrum)
        f_psd, S_tautau_z = signal.welch(
            tau_preprocessed,
            fs=fs,
            window=WINDOW,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )

        S_tautau_all_z.append(S_tautau_z)

        if f is None:
            f = f_psd

        if (iz + 1) % max(1, nz // 5) == 0 or iz == 0:
            print(f"  z-index {iz}/{nz-1}: Processed")

    # Convert to arrays and average over z
    S_tautau_all_z = np.array(S_tautau_all_z)
    S_tautau_mean = np.mean(S_tautau_all_z, axis=0)

    spectral_data['ref_point'] = {
        'frequency': f,
        'S_tautau_mean': S_tautau_mean,
        'psd': S_tautau_mean
    }

    print(f"  ✓ Surface reference PSD computed and z-averaged")

    # Compute for each probe
    probe_colors = ['purple', 'cyan', 'magenta', 'brown']

    for i, probe in enumerate(probe_info):
        print(f"\nComputing spectral data for Probe {i}...")
        S_tautau_all_z = []
        S_uu_all_z = []
        S_tauu_all_z = []
        f_freq = None

        for iz in range(nz):
            tau_z = timeseries_data['ref_point']['tau_prime'][:, iz]
            u_z = timeseries_data[f'probe_{i}']['u_prime'][:, iz]

            # Compute autospectra and complex cross-spectrum
            f_spec, S_tautau_z, S_uu_z, S_tauu_z = compute_spectra_welch(
                tau_z, u_z, fs, window=WINDOW, nperseg=nperseg, noverlap=noverlap, detrend=DETREND_TYPE
            )

            S_tautau_all_z.append(S_tautau_z)
            S_uu_all_z.append(S_uu_z)
            S_tauu_all_z.append(S_tauu_z)

            if f_freq is None:
                f_freq = f_spec

            if (iz + 1) % max(1, nz // 5) == 0 or iz == 0:
                print(f"  z-index {iz}/{nz-1}: Processed")

        # Convert to arrays: shape (nz, nfreq)
        S_tautau_all_z = np.array(S_tautau_all_z)
        S_uu_all_z = np.array(S_uu_all_z)
        S_tauu_all_z = np.array(S_tauu_all_z)

        # Average autospectra and complex cross-spectrum over z (linear operation)
        S_tautau_mean = np.mean(S_tautau_all_z, axis=0)
        S_uu_mean = np.mean(S_uu_all_z, axis=0)
        S_tauu_mean = np.mean(S_tauu_all_z, axis=0)

        # Compute coherence from averaged spectra (non-linear operation on averaged data)
        coherence_mean = np.abs(S_tauu_mean)**2 / (S_tautau_mean * S_uu_mean)

        # Diagnostic: compute local average coherence (average coherence over z)
        # This shows how the averaging method affects the result
        coherence_all_z_local = np.abs(S_tauu_all_z)**2 / (S_tautau_all_z * S_uu_all_z)
        coherence_mean_local = np.mean(coherence_all_z_local, axis=0)

        # CSD magnitude and phase from averaged spectra
        csd_mag_mean = np.abs(S_tauu_mean)
        csd_phase_mean = np.angle(S_tauu_mean)  # Phase in [-pi, pi]

        # Store both real and imaginary parts of averaged complex cross-spectrum
        spectral_data[f'probe_{i}'] = {
            'frequency': f_freq,
            'psd': S_uu_mean,  # Autospectrum of u
            'S_tautau_mean': S_tautau_mean,
            'S_uu_mean': S_uu_mean,
            'S_tauu_mean_real': np.real(S_tauu_mean),
            'S_tauu_mean_imag': np.imag(S_tauu_mean),
            'frequency_csd': f_freq,
            'csd_magnitude': csd_mag_mean,
            'csd_phase': csd_phase_mean,  # Phase of cross-spectrum in [-pi, pi]
            'frequency_coh': f_freq,
            'coherence': coherence_mean,
            'coherence_mean_local': coherence_mean_local
        }

        # Validation for probe 0 as representative
        if i == 0:
            print(f"  ✓ Probe {i} spectral data computed from averaged spectra")
            print(f"    S_tautau_mean shape: {S_tautau_mean.shape}")
            print(f"    S_uu_mean shape: {S_uu_mean.shape}")
            print(f"    coherence_mean shape: {coherence_mean.shape}")
            print(f"    coherence min/max: {np.min(coherence_mean):.8f} / {np.max(coherence_mean):.8f}")

            # Check if coherence is in [0, 1]
            invalid_coh = np.sum((coherence_mean < -1e-8) | (coherence_mean > 1 + 1e-8))
            if invalid_coh > 0:
                print(f"    ⚠ Warning: {invalid_coh} frequency points with coherence outside [0, 1]")
                print(f"      min coherence: {np.min(coherence_mean):.10f}")
                print(f"      max coherence: {np.max(coherence_mean):.10f}")
            else:
                print(f"    ✓ All coherence values within [0, 1]")

            # Compare corrected vs local-average method
            diff_coh = np.abs(coherence_mean - coherence_mean_local)
            max_diff = np.max(diff_coh)
            mean_diff = np.mean(diff_coh)
            print(f"    Comparison to local-average coherence:")
            print(f"      Max difference: {max_diff:.8e}")
            print(f"      Mean difference: {mean_diff:.8e}")

            # Phase validation
            nan_count = np.sum(np.isnan(csd_phase_mean))
            print(f"    CSD Phase validation:")
            print(f"      NaN count: {nan_count}")
            print(f"      Phase range: [{np.nanmin(csd_phase_mean):.6f}, {np.nanmax(csd_phase_mean):.6f}] rad")
        else:
            print(f"  ✓ Probe {i} spectral data computed from averaged spectra")
            # Phase validation for other probes
            nan_count = np.sum(np.isnan(csd_phase_mean))
            if nan_count > 0:
                print(f"    ⚠ CSD Phase NaN count: {nan_count}")


    # ============================================================================
    # PHASE 6B: WELCH PSD VARIANCE RECOVERY VALIDATION
    # ============================================================================

    print("\n" + "="*70)
    print("PHASE 6B: VARIANCE RECOVERY VALIDATION")
    print("="*70)

    validation_data = {}

    # Select one z-index for validation (middle of domain)
    z_val = nz // 2
    print(f"\nValidating variance recovery at z-index {z_val} (≈ middle domain)...")

    # Define signals to validate: (signal_id, get_signal_fn, description)
    validation_signals = []

    # 1. Surface tau_prime
    tau_val = timeseries_data['ref_point']['tau_prime'][:, z_val]
    validation_signals.append(('surface_tau', tau_val, 'Surface τ_prime'))

    # 2. Probe 0
    u_val_0 = timeseries_data['probe_0']['u_prime'][:, z_val]
    validation_signals.append(('probe_0', u_val_0, 'Probe 0 u_prime'))

    # 3. Middle probe (if exists)
    if len(probe_info) > 2:
        mid_idx = len(probe_info) // 2
        u_val_mid = timeseries_data[f'probe_{mid_idx}']['u_prime'][:, z_val]
        validation_signals.append(
            (f'probe_{mid_idx}', u_val_mid, f'Probe {mid_idx} (middle) u_prime')
        )

    # 4. Outer probe (if exists and different from middle)
    if len(probe_info) > 1:
        outer_idx = len(probe_info) - 1
        u_val_outer = timeseries_data[f'probe_{outer_idx}']['u_prime'][:, z_val]
        validation_signals.append(
            (f'probe_{outer_idx}', u_val_outer, f'Probe {outer_idx} (outer) u_prime')
        )

    # Compute and store validation results
    welch_validation = {}

    for sig_id, signal_data, description in validation_signals:
        # Preprocess signal using global DETREND_TYPE configuration
        signal_preprocessed = preprocess_signal_for_welch(signal_data, detrend_type=DETREND_TYPE)

        # Compute Welch PSD for this signal at the selected z-index
        f_val, psd_val = signal.welch(
            signal_preprocessed,
            fs=fs,
            window=WINDOW,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )

        # Get variance recovery using the preprocessed signal and frequency array from this Welch call
        recovery = compute_psd_variance_recovery(signal_preprocessed, psd_val, f_val)

        # Store
        welch_validation[sig_id] = {
            'description': description,
            'variance_time': recovery['variance_time'],
            'variance_freq': recovery['variance_freq'],
            'relative_error_percent': recovery['relative_error_percent'],
            'quality': quality_label(recovery['relative_error_percent'])
        }

    validation_data['welch_variance_recovery'] = welch_validation

    # Print validation results
    print(f"\n{'Signal':<25} {'Var(time)':<16} {'Var(freq)':<16} {'Error %':<12} {'Quality':<12}")
    print("-" * 85)
    for sig_id, result in welch_validation.items():
        desc = result['description'][:22] if len(result['description']) > 22 else result['description']
        var_t = result['variance_time']
        var_f = result['variance_freq']
        err_pct = result['relative_error_percent']
        qual = result['quality']
        print(f"{desc:<25} {var_t:<16.8e} {var_f:<16.8e} {err_pct:<12.4f} {qual:<12}")

    # Summary
    all_good = all(r['quality'] in ['good', 'acceptable'] for r in welch_validation.values())
    if all_good:
        print("\n✓ Welch variance recovery: all signals acceptable or better")
    else:
        print("\n⚠ Welch variance recovery: some signals show larger errors (see 'warning')")

    # ============================================================================
    # PHASE 6C: NPERSEG SENSITIVITY ANALYSIS
    # ============================================================================

    print("\n" + "="*70)
    print("PHASE 6C: NPERSEG SENSITIVITY ANALYSIS")
    print("="*70)

    test_nperseg_values = [1024, 2048, 4096, 8192]
    nperseg_sensitivity_results = {}

    # Select signals for sensitivity testing (same as main validation)
    sensitivity_test_signals = validation_signals.copy()

    print(f"\nTesting nperseg values: {test_nperseg_values}")
    print(f"Testing on {len(sensitivity_test_signals)} signals")
    print(f"Sampling frequency: {fs:.2f} Hz")

    for test_nperseg in test_nperseg_values:
        # Check if nperseg is valid
        if test_nperseg > Nt:
            print(f"\n⚠ Skipping nperseg={test_nperseg} (exceeds Nt={Nt})")
            continue

        test_noverlap = test_nperseg // 2
        n_segments_test = (Nt - test_nperseg) // (test_nperseg - test_noverlap) + 1

        print(f"\nTesting nperseg={test_nperseg}, noverlap={test_noverlap}")
        print(f"  Number of segments: {n_segments_test}")
        print(f"  Frequency resolution: {fs / test_nperseg:.6e} Hz")

        nperseg_sensitivity_results[test_nperseg] = {}

        for sig_id, signal_data, description in sensitivity_test_signals:
            # Preprocess signal using global DETREND_TYPE configuration
            signal_preprocessed = preprocess_signal_for_welch(signal_data, detrend_type=DETREND_TYPE)

            # Compute Welch PSD with test nperseg and capture its frequency array
            f_test, psd_test = signal.welch(
                signal_preprocessed,
                fs=fs,
                window=WINDOW,
                nperseg=test_nperseg,
                noverlap=test_noverlap,
                scaling='density'
            )

            # Get variance recovery using the preprocessed signal and frequency array from this Welch call
            recovery_test = compute_psd_variance_recovery(signal_preprocessed, psd_test, f_test)

            # Store results
            nperseg_sensitivity_results[test_nperseg][sig_id] = {
                'description': description,
                'variance_time': recovery_test['variance_time'],
                'variance_freq': recovery_test['variance_freq'],
                'relative_error_percent': recovery_test['relative_error_percent'],
                'quality': quality_label(recovery_test['relative_error_percent']),
                'n_segments': n_segments_test
            }

    # Print sensitivity analysis results
    print("\n" + "="*70)
    print("NPERSEG SENSITIVITY RESULTS - VARIANCE RECOVERY")
    print("="*70)

    # Create summary table
    for sig_id, _, sig_desc in sensitivity_test_signals:
        print(f"\n{sig_desc}:")
        print(f"{'nperseg':<10} {'n_seg':<8} {'Var(time)':<16} {'Var(freq)':<16} {'Error %':<12} {'Quality':<12}")
        print("-" * 80)

        for test_nperseg in sorted(nperseg_sensitivity_results.keys()):
            if sig_id in nperseg_sensitivity_results[test_nperseg]:
                result = nperseg_sensitivity_results[test_nperseg][sig_id]
                print(f"{test_nperseg:<10} {result['n_segments']:<8} "
                      f"{result['variance_time']:<16.8e} {result['variance_freq']:<16.8e} "
                      f"{result['relative_error_percent']:<12.4f} {result['quality']:<12}")

    # Summary statistics
    print("\n" + "="*70)
    print("NPERSEG SENSITIVITY - SUMMARY STATISTICS")
    print("="*70)

    summary_stats = {}
    for test_nperseg in sorted(nperseg_sensitivity_results.keys()):
        errors = [nperseg_sensitivity_results[test_nperseg][sig_id]['relative_error_percent']
                  for sig_id, _, _ in sensitivity_test_signals
                  if sig_id in nperseg_sensitivity_results[test_nperseg]]
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            summary_stats[test_nperseg] = {
                'mean_error': mean_error,
                'min_error': min_error,
                'max_error': max_error,
                'n_good': sum(1 for e in errors if e < 5.0),
                'n_acceptable': sum(1 for e in errors if 5.0 <= e < 10.0),
                'n_warning': sum(1 for e in errors if e >= 10.0)
            }

    print(f"\n{'nperseg':<10} {'Mean Err%':<12} {'Min Err%':<12} {'Max Err%':<12} "
          f"{'Good':<6} {'Accept':<6} {'Warn':<6}")
    print("-" * 70)

    optimal_nperseg = None
    optimal_error = float('inf')

    for test_nperseg in sorted(summary_stats.keys()):
        stats = summary_stats[test_nperseg]
        print(f"{test_nperseg:<10} {stats['mean_error']:<12.4f} {stats['min_error']:<12.4f} "
              f"{stats['max_error']:<12.4f} {stats['n_good']:<6} {stats['n_acceptable']:<6} {stats['n_warning']:<6}")

        # Find nperseg with lowest mean error
        if stats['mean_error'] < optimal_error:
            optimal_error = stats['mean_error']
            optimal_nperseg = test_nperseg

    print(f"\n✓ Optimal nperseg (lowest mean error): {optimal_nperseg} with mean error {optimal_error:.4f}%")

    # Create sensitivity figure
    print("\nGenerating NPERSEG sensitivity figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NPERSEG Sensitivity Analysis - Variance Recovery', fontsize=14, fontweight='bold')

    colors_nperseg = {1024: '#1f77b4', 2048: '#ff7f0e', 4096: '#2ca02c', 8192: '#d62728'}
    markers_nperseg = {1024: 'o', 2048: 's', 4096: '^', 8192: 'v'}

    for ax_idx, (sig_id, _, sig_desc) in enumerate(sensitivity_test_signals):
        ax = axes.flat[ax_idx]

        nperseg_vals = []
        errors = []

        for test_nperseg in sorted(nperseg_sensitivity_results.keys()):
            if sig_id in nperseg_sensitivity_results[test_nperseg]:
                errors.append(nperseg_sensitivity_results[test_nperseg][sig_id]['relative_error_percent'])
                nperseg_vals.append(test_nperseg)

        if nperseg_vals:
            # Plot error line
            ax.plot(nperseg_vals, errors, linewidth=2.0, markersize=8, marker='o',
                   color='#1f77b4', alpha=0.8, label='Variance Recovery Error')

            # Add threshold zones
            ax.axhspan(0, 5, alpha=0.1, color='green', label='Good (<5%)')
            ax.axhspan(5, 10, alpha=0.1, color='yellow', label='Acceptable (5-10%)')
            ax.axhspan(10, np.max(errors)*1.1, alpha=0.1, color='red', label='Warning (>10%)')

            # Mark optimal nperseg
            if optimal_nperseg in nperseg_vals:
                opt_idx = nperseg_vals.index(optimal_nperseg)
                ax.plot(optimal_nperseg, errors[opt_idx], marker='*', markersize=20,
                       color='gold', markeredgecolor='black', markeredgewidth=1.5, zorder=10,
                       label=f'Optimal (nperseg={optimal_nperseg})')

            ax.set_xscale('log')
            ax.set_xlabel('nperseg', fontsize=11, fontweight='bold')
            ax.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
            ax.set_title(sig_desc, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=9, loc='best')
            ax.set_ylim([0, np.max(errors) * 1.15])

    plt.tight_layout()
    plt.show()
    print("  ✓ Sensitivity figure generated")

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
         'o-', linewidth=1.5, markersize=3, label="τ' (Surface)",
         color='#d62728', alpha=0.8)

# Plot u' PSDs for each probe
probe_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5'
]
for i in range(len(probe_info)):
    color = probe_colors[i % len(probe_colors)]
    ax.loglog(f_nd[freq_start:], spectral_data[f'probe_{i}']['psd'][freq_start:],
             's-', linewidth=1.5, markersize=3,
             label=f"P{i}(y={probe_info[i]['y_actual']:.3f})", color=color, alpha=0.8)

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

plt.show()

# ============================================================================
# FIGURE 4: Z-AVERAGED COHERENCE OVERLAY
# ============================================================================

print("\n" + "="*70)
print("FIGURE 4: Z-AVERAGED COHERENCE OVERLAY (SEMILOG)")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 8))

probe_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5'
]

for i in range(len(probe_info)):
    color = probe_colors[i % len(probe_colors)]
    ax.semilogx(spectral_data[f'probe_{i}']['f_coh_nd'][freq_start:],
               spectral_data[f'probe_{i}']['coherence'][freq_start:],
               'o-', linewidth=1.5, markersize=3,
               label=f"P{i}(y={probe_info[i]['y_actual']:.3f})", color=color, alpha=0.8)

ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=12, fontweight='bold')
ax.set_ylabel("Magnitude-Squared Coherence γ² [-]", fontsize=12, fontweight='bold')
ax.set_title(f"Coherence: τ' vs u' (z-averaged over full span, nz={nz})", fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.show()

# ============================================================================
# FIGURE 5: Z-AVERAGED CROSS-SPECTRUM MAGNITUDE OVERLAY
# ============================================================================

print("\n" + "="*70)
print("FIGURE 5: Z-AVERAGED CROSS-SPECTRUM MAGNITUDE OVERLAY (LOG-LOG)")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 8))

probe_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5'
]

for i in range(len(probe_info)):
    color = probe_colors[i % len(probe_colors)]
    ax.loglog(spectral_data[f'probe_{i}']['f_csd_nd'][freq_start:],
             spectral_data[f'probe_{i}']['csd_magnitude'][freq_start:],
             'o-', linewidth=1.5, markersize=3,
             label=f"P{i}(y={probe_info[i]['y_actual']:.3f})", color=color, alpha=0.8)

ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=12, fontweight='bold')
ax.set_ylabel("|S_τu(f)| [signal product / Hz]", fontsize=12, fontweight='bold')
ax.set_title(f"Cross-Spectrum Magnitude: τ' vs u' (z-averaged, nz={nz})",
            fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

plt.tight_layout()
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
print(f"\nGenerated figures:")
print(f"  1. Correlation map with airfoil geometry and probes")
print(f"  2. Temporal signals at z-index 0 (single spanwise plane) for surface and probes")
print(f"  3. Z-averaged PSDs for all signals")
print(f"  4. Z-averaged coherence (τ' vs each probe)")
print("="*70)
