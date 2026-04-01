"""
Energy Spectra with Spatial Averaging - Comparison Study
========================================================

Compute energy spectra using TWO METHODS for comparison:
1. Single-point: Traditional approach using single y-grid point (original method)
2. Spatially-averaged: 3-point average (y-1, y, y+1) for smoother results

Analyze the benefits of spatial averaging on spectral estimation quality:
- Reduced high-frequency noise
- Better coherence
- Improved SNR
- Quantifiable metrics comparing both approaches

Key improvements over original:
- Dual extraction paths for direct comparison
- Comprehensive metrics for SNR, coherence, noise floor
- Four comparison visualizations (overlay, ratio, table, SNR analysis)
- Extended HDF5 with metadata tracking spatial averaging parameters
"""

import os
import sys
import re
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

# Slice data paths
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"

# Geometric data (for visualization only)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# Physical parameters
rho_ref = 1.0           # Reference density [kg/m³]
u_infty = 1.0           # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord length [m]
Re_c = 50000            # Reynolds number
AOA_deg = 12.0          # Angle of attack [degrees]
AOA_rad = np.radians(AOA_deg)

# Physical time step [CRITICAL - must match simulation]
dt_iteration = 2.0e-06  # Physical time per iteration [s]

# Probe locations: absolute y-coordinates in domain
Y_LOCATIONS = [0.1]  # Specify actual y-coordinate values

# Output directory
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Energy_spectra/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# SPATIAL AVERAGING CONFIGURATION
# ============================================================================

# Spatial Averaging parameters
ENABLE_SPATIAL_AVERAGING = True        # Toggle spatial averaging method
SPATIAL_AVG_WEIGHTS = [1/3, 1/3, 1/3] # Uniform weighting for 3 points (y-1, y, y+1)

# ============================================================================
# PLOTTING CONFIGURATION (Rodriguez Figure 11 Style)
# ============================================================================

USE_ADDITIVE_OFFSET = True              # True: additive, False: multiplicative
ADDITIVE_OFFSET_SCALE = 0.5             # Factor for vertical offset (0.5-1.0)

# Grid and styling
USE_MAJOR_MINOR_GRID = True             # Separate major/minor grids
REFERENCE_LINE_ALPHA = 0.5              # Kolmogorov line opacity
REFERENCE_LINE_WIDTH = 1.5              # Kolmogorov line thickness

# Figure layout
FIGURE_WIDTH = 14                        # Inches
FIGURE_HEIGHT = 8                        # Inches

# Comparison Plot Configuration
COMPARISON_PLOTS = {
    'overlay_spectra': True,           # Single-point + 3-point overlay
    'spectral_ratio': True,            # E_avg / E_single ratio
    'statistical_table': True,         # Variance, SNR, peaks
    'coherence_analysis': True         # Spectral coherence/smoothing metrics
}
COMPARISON_FIG_WIDTH = 16
COMPARISON_FIG_HEIGHT = 10

# Legend
LEGEND_FONTSIZE = 9
SHOW_PROBE_LABELS = True                # Label each curve with P# and y-value

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    """Check path exists and print confirmation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"{kind} exists: {path}")


def get_adjacent_y_indices(j0: int, y_unique: np.ndarray) -> tuple:
    """
    Find adjacent y-grid indices for 3-point spatial averaging.

    Args:
        j0: Center index (closest to requested y-location)
        y_unique: Array of all unique y-coordinates

    Returns:
        j_minus: Index at y-1 (with boundary handling)
        j0: Center index
        j_plus: Index at y+1 (with boundary handling)
        delta_y_minus: Actual distance from j0 to j_minus
        delta_y_plus: Actual distance from j_plus to j0
    """
    n_y = len(y_unique)

    # Handle boundaries
    j_minus = max(0, j0 - 1)
    j_plus = min(n_y - 1, j0 + 1)

    # Track actual distances
    delta_y_minus = y_unique[j0] - y_unique[j_minus]
    delta_y_plus = y_unique[j_plus] - y_unique[j0]

    return j_minus, j0, j_plus, delta_y_minus, delta_y_plus


def spatially_average_velocity(u_minus: np.ndarray, u_center: np.ndarray,
                               u_plus: np.ndarray, weights: list = None) -> np.ndarray:
    """
    Spatially average 3 adjacent y-grid-point velocity fields.

    Args:
        u_minus: Velocity at y-1 grid point (nz,)
        u_center: Velocity at y grid point (nz,)
        u_plus: Velocity at y+1 grid point (nz,)
        weights: Weighting factors [w_minus, w_center, w_plus] (default: uniform)

    Returns:
        u_avg: Spatially averaged velocity (nz,)
    """
    if weights is None:
        weights = [1/3, 1/3, 1/3]

    u_avg = weights[0] * u_minus + weights[1] * u_center + weights[2] * u_plus
    return u_avg


def estimate_noise_floor(E_spectrum: np.ndarray, freq_band_ratio: float = 0.90) -> float:
    """
    Estimate noise floor from high-frequency region of spectrum.

    Assumes high frequencies (last 10% of spectrum) represent mostly noise.

    Args:
        E_spectrum: Energy spectrum (nfreq,)
        freq_band_ratio: Frequency ratio above which to estimate noise (default: 0.90)

    Returns:
        noise_floor: Estimate of noise level in spectrum
    """
    cutoff_idx = int(len(E_spectrum) * freq_band_ratio)
    noise_floor = np.mean(E_spectrum[cutoff_idx:])
    return noise_floor


def find_spectral_peak(E_spectrum: np.ndarray, frequencies: np.ndarray,
                       ignore_dc_n: int = 5) -> tuple:
    """
    Find dominant frequency peak, excluding DC component.

    Args:
        E_spectrum: Energy spectrum (nfreq,)
        frequencies: Frequency vector (nfreq,)
        ignore_dc_n: Number of DC-adjacent frequencies to ignore (default: 5)

    Returns:
        peak_freq: Dominant frequency
        peak_idx: Index of peak
        peak_energy: Energy at peak
    """
    if len(E_spectrum) <= ignore_dc_n:
        peak_idx = 0
    else:
        peak_idx = np.argmax(E_spectrum[ignore_dc_n:]) + ignore_dc_n

    peak_freq = frequencies[peak_idx]
    peak_energy = E_spectrum[peak_idx]
    return peak_freq, peak_idx, peak_energy


def compute_spectral_coherence(E_single: np.ndarray, E_avg: np.ndarray) -> float:
    """
    Compute coherence between single-point and spatially-averaged spectra.

    Measures how well-correlated the two methods are across frequencies.

    Args:
        E_single: Single-point spectrum (nfreq,)
        E_avg: Spatially-averaged spectrum (nfreq,)

    Returns:
        coherence_score: Coherence measure (0-1, higher is better correlation)
    """
    # Normalize spectra
    E_single_norm = E_single / (np.max(E_single) + 1e-15)
    E_avg_norm = E_avg / (np.max(E_avg) + 1e-15)

    # Compute cross-correlation at zero lag
    correlation = np.mean(E_single_norm * E_avg_norm)

    # Coherence score: measure of similarity (should be close to 1)
    coherence = 2 * correlation / (np.mean(E_single_norm**2) + np.mean(E_avg_norm**2) + 1e-15)
    coherence = np.clip(coherence, 0, 1)

    return coherence


def get_slice_files_sorted(slices_path: str) -> tuple:
    """
    Get snapshot slice files sorted numerically by iteration number (not lexicographic).
    """
    snapshot_files = []
    for file in Path(slices_path).glob("slice_*-COMP-DATA.h5"):
        if "avg" not in file.name:
            snapshot_files.append(str(file))

    if not snapshot_files:
        raise FileNotFoundError(f"No snapshot files found in {slices_path}")

    print(f"\nFound {len(snapshot_files)} snapshot files")

    iterations = []
    files_with_iter = []

    for filepath in snapshot_files:
        filename = os.path.basename(filepath)
        match = re.search(r'_(\d+)-COMP-DATA\.h5', filename)
        if not match:
            raise ValueError(f"Cannot extract iteration number from {filename}")
        iter_num = int(match.group(1))
        iterations.append(iter_num)
        files_with_iter.append((iter_num, filepath))

    files_with_iter.sort(key=lambda x: x[0])
    sorted_files = [f[1] for f in files_with_iter]
    iterations = np.array([f[0] for f in files_with_iter])

    if len(iterations) < 2:
        raise ValueError("Need at least 2 snapshot files to compute time step")

    delta_iterations = np.diff(iterations)
    unique_deltas = np.unique(delta_iterations)

    print(f"\nIteration information:")
    print(f"  First 10 iterations: {iterations[:10]}")
    print(f"  Last 10 iterations: {iterations[-10:]}")
    print(f"  Total samples: {len(iterations)}")
    print(f"  Iteration spacing: {unique_deltas}")
    print(f"  Strictly monotonic: {np.all(delta_iterations > 0)}")

    if len(unique_deltas) > 1:
        raise ValueError(f"Iteration spacing is NOT constant: {unique_deltas}")

    delta_iter = int(unique_deltas[0])
    print(f"  Constant delta_iteration: {delta_iter}")

    return sorted_files, iterations, delta_iter

# ============================================================================
# LOAD GEOMETRY AND MESH
# ============================================================================

print("="*70)
print("LOAD GEOMETRY AND MESH")
print("="*70)

assert_exists(GEO_FILE, "Geometrical data file")
assert_exists(MESH_SLICE_FILE, "Mesh slice file")

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)

suction_side_points = interface_points[interface_points[:, 1] >= 0]
pressure_side_points = interface_points[interface_points[:, 1] < 0]

print(f"Suction side points: {suction_side_points.shape[0]}")
print(f"Pressure side points: {pressure_side_points.shape[0]}")

loader = CompressedSnapshotLoader(MESH_SLICE_FILE)
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print(f"Mesh shape (nz, ny, nx): {x_data.shape}")

x_unique_in_mesh = np.unique(x_data)
if len(x_unique_in_mesh) > 1:
    raise ValueError(f"Slice mesh has {len(x_unique_in_mesh)} unique x values. Expected single x-plane")

x_all = x_data.flatten()
x_std = np.std(x_all[~np.isnan(x_all)])
x_rel_std = x_std / np.abs(x_all[~np.isnan(x_all)].mean() + 1e-15)
if x_rel_std > 1e-6:
    raise ValueError(f"x-coordinate varies too much in slice (rel_std={x_rel_std:.6e})")

slice_x = x_data[0, 0, 0]
print(f"Slice structure verified: single x-plane (nx=1), ny={x_data.shape[1]}, nz={x_data.shape[0]}")
print(f"Slice x-coordinate: {slice_x:.6f}")

z_unique = np.unique(z_data[:, 0, 0])
nz = z_unique.size
dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
L_z = dz * nz

print(f"Spanwise domain: nz={nz}, dz={dz:.6e} m, Lz={L_z:.6e} m")

y_unique = np.unique(y_data[:, :, 0][0, :])
print(f"Total y-grid points: {len(y_unique)}, range: {y_unique[0]:.6e} to {y_unique[-1]:.6e}")

x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
interface_y = suction_side_points[closest_idx, 1]
print(f"Interface y at slice x={slice_x:.6f}: y={interface_y:.6e}")

# ============================================================================
# SORT AND VALIDATE SNAPSHOT FILES
# ============================================================================

print("\n" + "="*70)
print("SORT AND VALIDATE SNAPSHOT FILES")
print("="*70)

slice_files, iter_numbers, delta_iter = get_slice_files_sorted(SLICES_PATH)

match = re.search(r'slice_(\d+)', SLICES_PATH)
if match:
    slice_id = f"slice_{match.group(1)}"
else:
    raise ValueError(f"Cannot infer slice_id from path: {SLICES_PATH}")

print(f"Inferred slice_id from path: {slice_id}")

dt_save = delta_iter * dt_iteration
fs = 1.0 / dt_save

print(f"\nPhysical time step computation:")
print(f"  dt_iteration: {dt_iteration:.6e} s/iteration")
print(f"  delta_iter: {delta_iter} iterations")
print(f"  dt_save: {dt_save:.6e} s/snapshot ({dt_save*1000:.6f} ms)")
print(f"  Sampling frequency fs: {fs:.6e} Hz")

n_samples_expected = len(slice_files)
print(f"\nExpected time steps: {n_samples_expected}")
print(f"Total time span: {(n_samples_expected-1)*dt_save:.6e} s")

# ============================================================================
# SELECT PROBE LOCATIONS
# ============================================================================

print("\n" + "="*70)
print("SELECT PROBE LOCATIONS")
print("="*70)

y_locations_idx = []
y_locations_val = []
y_locations_target = []
y_locations_adjacent = []  # Store adjacent indices for spatial averaging

for i, y_target in enumerate(Y_LOCATIONS):
    idx_closest = np.argmin(np.abs(y_unique - y_target))
    j_idx = idx_closest
    actual_y = y_unique[j_idx]

    y_locations_idx.append(j_idx)
    y_locations_val.append(actual_y)
    y_locations_target.append(y_target)

    # Get adjacent indices for spatial averaging
    j_minus, j0, j_plus, delta_y_minus, delta_y_plus = get_adjacent_y_indices(j_idx, y_unique)
    y_locations_adjacent.append((j_minus, j0, j_plus, delta_y_minus, delta_y_plus))

    dist_error = np.abs(actual_y - y_target)
    print(f"Probe {i}: y_target={y_target:.6e} -> y_actual={actual_y:.6e}, "
          f"error={dist_error:.6e} (j_idx={j_idx})")
    print(f"  Adjacent indices: j-={j_minus} (dy={delta_y_minus:.6e}), "
          f"j0={j0}, j+={j_plus} (dy={delta_y_plus:.6e})")

# ============================================================================
# EXTRACT TIME SERIES (DUAL PATHS - SINGLE POINT AND SPATIAL AVERAGE)
# ============================================================================

print("\n" + "="*70)
print("EXTRACT TIME SERIES (SINGLE-POINT AND SPATIAL-AVERAGE METHODS)")
print("="*70)

# Initialize storage for BOTH methods
time_series_data_single = {j_idx: {'u': [], 'v': []} for j_idx in y_locations_idx}
time_series_data_avg = {j_idx: {'u': [], 'v': []} for j_idx in y_locations_idx}

print(f"Extracting time series from {len(slice_files)} snapshots...")
print(f"  METHOD 1: Single-point (traditional)")
print(f"  METHOD 2: 3-point spatial average (y-1, y, y+1)\n")

failed_files = []

for t, slice_file in enumerate(slice_files):
    if not os.path.exists(slice_file):
        failed_files.append(slice_file)
        continue

    try:
        # Load snapshot
        fields = loader.load_snapshot(slice_file)
        u_data = loader.reconstruct_field(fields["u"])[1:-1, :, :]
        v_data = loader.reconstruct_field(fields["v"])[1:-1, :, :]
        w_data = loader.reconstruct_field(fields["w"])[1:-1, :]

        # Rotate to streamwise/cross-stream coordinates
        u_stream = u_data * np.cos(AOA_rad) + v_data * np.sin(AOA_rad)
        v_cross = -u_data * np.sin(AOA_rad) + v_data * np.cos(AOA_rad)

        # Extract at probe locations
        for probe_idx, j_idx in enumerate(y_locations_idx):
            # METHOD 1: Single-point (original)
            u_single = u_stream[:, j_idx, 0]
            v_single = v_cross[:, j_idx, 0]
            time_series_data_single[j_idx]['u'].append(u_single)
            time_series_data_single[j_idx]['v'].append(v_single)

            # METHOD 2: 3-point spatial average
            if ENABLE_SPATIAL_AVERAGING:
                j_minus, j0, j_plus, _, _ = y_locations_adjacent[probe_idx]

                # Extract at all 3 locations
                u_minus = u_stream[:, j_minus, 0]
                u_center = u_stream[:, j0, 0]
                u_plus = u_stream[:, j_plus, 0]

                v_minus = v_cross[:, j_minus, 0]
                v_center = v_cross[:, j0, 0]
                v_plus = v_cross[:, j_plus, 0]

                # Spatially average BEFORE storing
                u_avg = spatially_average_velocity(u_minus, u_center, u_plus, SPATIAL_AVG_WEIGHTS)
                v_avg = spatially_average_velocity(v_minus, v_center, v_plus, SPATIAL_AVG_WEIGHTS)

                time_series_data_avg[j_idx]['u'].append(u_avg)
                time_series_data_avg[j_idx]['v'].append(v_avg)

        if (t + 1) % max(1, len(slice_files) // 10) == 0:
            print(f"  Processed {t + 1}/{len(slice_files)} snapshots")

    except Exception as e:
        failed_files.append((slice_file, str(e)))
        continue

if failed_files:
    raise RuntimeError(f"Failed to load {len(failed_files)} files. Aborting. "
                      f"First error: {failed_files[0]}")

# Verify consistency
n_samples_list_single = [len(time_series_data_single[j_idx]['u']) for j_idx in y_locations_idx]
n_samples_list_avg = [len(time_series_data_avg[j_idx]['u']) for j_idx in y_locations_idx]

if len(set(n_samples_list_single)) > 1:
    raise ValueError(f"Inconsistent sample counts (single-point): {set(n_samples_list_single)}")
if len(set(n_samples_list_avg)) > 1:
    raise ValueError(f"Inconsistent sample counts (spatially-averaged): {set(n_samples_list_avg)}")

n_samples = n_samples_list_single[0]
print(f"\nActual loaded samples: {n_samples} (consistent across all {len(y_locations_idx)} probes)")

# Convert to numpy arrays
for j_idx in y_locations_idx:
    time_series_data_single[j_idx]['u'] = np.array(time_series_data_single[j_idx]['u'])
    time_series_data_single[j_idx]['v'] = np.array(time_series_data_single[j_idx]['v'])

    if ENABLE_SPATIAL_AVERAGING:
        time_series_data_avg[j_idx]['u'] = np.array(time_series_data_avg[j_idx]['u'])
        time_series_data_avg[j_idx]['v'] = np.array(time_series_data_avg[j_idx]['v'])

    shape_single = time_series_data_single[j_idx]['u'].shape
    print(f"Probe at j={j_idx}, y={y_locations_val[y_locations_idx.index(j_idx)]:.6e}: "
          f"single-point shape = {shape_single}")

# ============================================================================
# COMPUTE ENERGY SPECTRA (BOTH METHODS)
# ============================================================================

print("\n" + "="*70)
print("COMPUTE ENERGY SPECTRA (SINGLE-POINT AND SPATIAL-AVERAGE)")
print("="*70)

frequencies = np.fft.rfftfreq(n_samples, d=dt_save)
nfreq = len(frequencies)
df = frequencies[1] - frequencies[0] if nfreq > 1 else 1.0
f_star = frequencies * c / u_infty

print(f"Frequency parameters:")
print(f"  n_samples: {n_samples}")
print(f"  nfreq (from rfft): {nfreq} = {n_samples}//2 + 1")
print(f"  Frequency resolution df: {df:.6e} Hz")
print(f"  Frequency range: {frequencies[1]:.6e} to {frequencies[-1]:.6e} Hz")

# Function to compute spectra (reusable for both methods)
def compute_spectra_for_method(time_series_data, method_name=""):
    """Compute energy spectra for given time series data."""
    energy_spectra = {}

    for j_idx, y_val in zip(y_locations_idx, y_locations_val):
        u_data = time_series_data[j_idx]['u']  # (n_times, n_z)
        v_data = time_series_data[j_idx]['v']

        E_uu_z = np.zeros((nfreq, nz))
        E_vv_z = np.zeros((nfreq, nz))
        var_u_time_z = np.zeros(nz)
        var_v_time_z = np.zeros(nz)
        var_u_spectral_z = np.zeros(nz)
        var_v_spectral_z = np.zeros(nz)

        for iz in range(nz):
            u_fluct = u_data[:, iz]
            v_fluct = v_data[:, iz]

            var_u_time_z[iz] = np.var(u_fluct)
            var_v_time_z[iz] = np.var(v_fluct)

            u_fluct = u_fluct - np.mean(u_fluct)
            v_fluct = v_fluct - np.mean(v_fluct)

            U_rfft = np.fft.rfft(u_fluct)
            V_rfft = np.fft.rfft(v_fluct)

            E_uu_z[:, iz] = (2.0 * dt_save / n_samples) * (np.abs(U_rfft) ** 2)
            E_vv_z[:, iz] = (2.0 * dt_save / n_samples) * (np.abs(V_rfft) ** 2)

            E_uu_z[0, iz] /= 2.0
            E_vv_z[0, iz] /= 2.0

            if n_samples % 2 == 0:
                E_uu_z[-1, iz] /= 2.0
                E_vv_z[-1, iz] /= 2.0

            var_u_spectral_z[iz] = np.sum(E_uu_z[:, iz] * df)
            var_v_spectral_z[iz] = np.sum(E_vv_z[:, iz] * df)

        E_uu = np.mean(E_uu_z, axis=1)
        E_vv = np.mean(E_vv_z, axis=1)

        var_u_time = np.mean(var_u_time_z)
        var_v_time = np.mean(var_v_time_z)
        var_u_spectral = np.mean(var_u_spectral_z)
        var_v_spectral = np.mean(var_v_spectral_z)

        rel_error_u = np.abs(var_u_spectral - var_u_time) / (var_u_time + 1e-15)
        rel_error_v = np.abs(var_v_spectral - var_v_time) / (var_v_time + 1e-15)

        energy_spectra[j_idx] = {
            'y': y_val,
            'E_uu': E_uu,
            'E_vv': E_vv,
            'E_uu_z': E_uu_z,
            'E_vv_z': E_vv_z,
            'frequencies': frequencies,
            'f_star': f_star,
            'var_u_time': var_u_time,
            'var_v_time': var_v_time,
            'var_u_spectral': var_u_spectral,
            'var_v_spectral': var_v_spectral,
            'var_u_time_z': var_u_time_z,
            'var_v_time_z': var_v_time_z,
            'var_u_spectral_z': var_u_spectral_z,
            'var_v_spectral_z': var_v_spectral_z,
            'rel_error_u': rel_error_u,
            'rel_error_v': rel_error_v,
            'u_rms': np.sqrt(var_u_spectral),
            'v_rms': np.sqrt(var_v_spectral)
        }

    return energy_spectra

# Compute spectra for both methods
print(f"\nComputing spectra for METHOD 1 (single-point)...")
energy_spectra_single = compute_spectra_for_method(time_series_data_single, "single-point")

print(f"Computing spectra for METHOD 2 (spatially-averaged)...")
energy_spectra_avg = compute_spectra_for_method(time_series_data_avg, "spatially-averaged")

# Print variance validation
print(f"\nVariance validation (time vs. spectral domain):")
for i, j_idx in enumerate(y_locations_idx[:2]):
    y_val = energy_spectra_single[j_idx]['y']
    var_time_u_s = energy_spectra_single[j_idx]['var_u_time']
    var_spec_u_s = energy_spectra_single[j_idx]['var_u_spectral']
    var_time_u_a = energy_spectra_avg[j_idx]['var_u_time']
    var_spec_u_a = energy_spectra_avg[j_idx]['var_u_spectral']

    print(f"\nProbe {i} (y={y_val:.6e}):")
    print(f"  Single-point: var_time={var_time_u_s:.6e}, var_spectral={var_spec_u_s:.6e}")
    print(f"  Spatial-avg:  var_time={var_time_u_a:.6e}, var_spectral={var_spec_u_a:.6e}")

# ============================================================================
# COMPUTE COMPARISON METRICS
# ============================================================================

print("\n" + "="*70)
print("COMPUTE COMPARISON METRICS")
print("="*70)

comparison_metrics = {}

for i, j_idx in enumerate(y_locations_idx):
    y_val = energy_spectra_single[j_idx]['y']

    E_uu_single = energy_spectra_single[j_idx]['E_uu']
    E_uu_avg = energy_spectra_avg[j_idx]['E_uu']
    E_vv_single = energy_spectra_single[j_idx]['E_vv']
    E_vv_avg = energy_spectra_avg[j_idx]['E_vv']

    # Variance comparison
    var_u_single = energy_spectra_single[j_idx]['var_u_spectral']
    var_u_avg = energy_spectra_avg[j_idx]['var_u_spectral']
    var_reduction_pct_u = 100 * (1 - var_u_avg / (var_u_single + 1e-15))

    var_v_single = energy_spectra_single[j_idx]['var_v_spectral']
    var_v_avg = energy_spectra_avg[j_idx]['var_v_spectral']
    var_reduction_pct_v = 100 * (1 - var_v_avg / (var_v_single + 1e-15))

    # RMS comparison
    rms_u_single = energy_spectra_single[j_idx]['u_rms']
    rms_u_avg = energy_spectra_avg[j_idx]['u_rms']
    rms_v_single = energy_spectra_single[j_idx]['v_rms']
    rms_v_avg = energy_spectra_avg[j_idx]['v_rms']

    # Spectral peak detection
    peak_freq_u_single, _, peak_energy_u_single = find_spectral_peak(E_uu_single, frequencies)
    peak_freq_u_avg, _, peak_energy_u_avg = find_spectral_peak(E_uu_avg, frequencies)
    peak_shift_pct_u = 100 * (peak_freq_u_avg - peak_freq_u_single) / (peak_freq_u_single + 1e-15)

    peak_freq_v_single, _, peak_energy_v_single = find_spectral_peak(E_vv_single, frequencies)
    peak_freq_v_avg, _, peak_energy_v_avg = find_spectral_peak(E_vv_avg, frequencies)
    peak_shift_pct_v = 100 * (peak_freq_v_avg - peak_freq_v_single) / (peak_freq_v_single + 1e-15)

    # Noise floor estimation
    noise_floor_u_single = estimate_noise_floor(E_uu_single)
    noise_floor_u_avg = estimate_noise_floor(E_uu_avg)
    snr_improvement_db_u = 10 * np.log10((noise_floor_u_single + 1e-15) / (noise_floor_u_avg + 1e-15))

    noise_floor_v_single = estimate_noise_floor(E_vv_single)
    noise_floor_v_avg = estimate_noise_floor(E_vv_avg)
    snr_improvement_db_v = 10 * np.log10((noise_floor_v_single + 1e-15) / (noise_floor_v_avg + 1e-15))

    # Spectral coherence
    coherence_u = compute_spectral_coherence(E_uu_single, E_uu_avg)
    coherence_v = compute_spectral_coherence(E_vv_single, E_vv_avg)

    metrics = {
        'y_actual': y_val,
        'y_target': y_locations_target[i],

        # Variance
        'var_u_single': var_u_single,
        'var_u_avg': var_u_avg,
        'var_u_reduction_pct': var_reduction_pct_u,
        'var_v_single': var_v_single,
        'var_v_avg': var_v_avg,
        'var_v_reduction_pct': var_reduction_pct_v,

        # RMS
        'rms_u_single': rms_u_single,
        'rms_u_avg': rms_u_avg,
        'rms_v_single': rms_v_single,
        'rms_v_avg': rms_v_avg,

        # Spectral peaks
        'peak_freq_u_single': peak_freq_u_single,
        'peak_freq_u_avg': peak_freq_u_avg,
        'peak_shift_pct_u': peak_shift_pct_u,
        'peak_energy_u_single': peak_energy_u_single,
        'peak_energy_u_avg': peak_energy_u_avg,

        'peak_freq_v_single': peak_freq_v_single,
        'peak_freq_v_avg': peak_freq_v_avg,
        'peak_shift_pct_v': peak_shift_pct_v,
        'peak_energy_v_single': peak_energy_v_single,
        'peak_energy_v_avg': peak_energy_v_avg,

        # Noise floor & SNR
        'noise_floor_u_single': noise_floor_u_single,
        'noise_floor_u_avg': noise_floor_u_avg,
        'snr_improvement_db_u': snr_improvement_db_u,
        'noise_floor_v_single': noise_floor_v_single,
        'noise_floor_v_avg': noise_floor_v_avg,
        'snr_improvement_db_v': snr_improvement_db_v,

        # Coherence
        'coherence_u': coherence_u,
        'coherence_v': coherence_v,
    }

    comparison_metrics[j_idx] = metrics

# Print metrics summary
print(f"\nComparison Metrics Summary:")
print(f"{'='*100}")
for i, j_idx in enumerate(y_locations_idx):
    m = comparison_metrics[j_idx]
    print(f"\nProbe {i} (y/c = {m['y_actual']:.4f}):")
    print(f"  ------- U-COMPONENT (Streamwise) -------")
    print(f"  Variance reduction: {m['var_u_reduction_pct']:+.2f}%")
    print(f"  SNR improvement: {m['snr_improvement_db_u']:+.2f} dB")
    print(f"  Spectral coherence: {m['coherence_u']:.4f}")
    print(f"  Peak frequency shift: {m['peak_shift_pct_u']:+.2f}%")
    print(f"  ------- V-COMPONENT (Cross-stream) -------")
    print(f"  Variance reduction: {m['var_v_reduction_pct']:+.2f}%")
    print(f"  SNR improvement: {m['snr_improvement_db_v']:+.2f} dB")
    print(f"  Spectral coherence: {m['coherence_v']:.4f}")
    print(f"  Peak frequency shift: {m['peak_shift_pct_v']:+.2f}%")

print("\n" + "="*100)

print(f"\nComparison metrics computed successfully for {len(y_locations_idx)} probes")

# ============================================================================
# CREATE COMPARISON VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("CREATE COMPARISON VISUALIZATIONS")
print("="*70)

# -------- PLOT 1: OVERLAY SPECTRA (Single-point vs Spatially-averaged) --------
if COMPARISON_PLOTS['overlay_spectra']:
    print("\n  Generating: Overlay Spectra Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(COMPARISON_FIG_WIDTH, COMPARISON_FIG_HEIGHT*0.7))

    for plot_idx, j_idx in enumerate(y_locations_idx):
        y_val = energy_spectra_single[j_idx]['y']
        f_star_vals = energy_spectra_single[j_idx]['f_star']
        E_uu_single = energy_spectra_single[j_idx]['E_uu']
        E_uu_avg = energy_spectra_avg[j_idx]['E_uu']
        E_vv_single = energy_spectra_single[j_idx]['E_vv']
        E_vv_avg = energy_spectra_avg[j_idx]['E_vv']

        # Left: E_uu
        ax = axes[0]
        ax.loglog(f_star_vals[1:], E_uu_single[1:], 'b-', linewidth=2.0, alpha=0.7,
                  label=f'Single-point (y/c={y_val:.4f})')
        ax.loglog(f_star_vals[1:], E_uu_avg[1:], 'r-', linewidth=2.0, alpha=0.7,
                  label=f'Spatially-averaged (y/c={y_val:.4f})')

        # Reference -5/3 line
        f_ref = np.logspace(np.log10(frequencies[10]) if len(frequencies) > 10 else np.log10(frequencies[1]),
                            np.log10(frequencies[-1]), 100)
        mid_idx = len(frequencies) // 2
        E_ref = E_uu_single[mid_idx] * (f_ref / f_ref[len(f_ref)//2]) ** (-5.0/3.0)
        ax.loglog(f_ref, E_ref, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
                  label=r'$f^{-5/3}$ (Kolmogorov)')

        ax.set_xlabel(r'$f^* = f \cdot c / U_\infty$ (nondimensional)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'$E_{uu}$ [m$^2$/s$^2$]', fontsize=11, fontweight='bold')
        ax.set_title(f'Streamwise Energy Spectrum (E_uu)', fontsize=12, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.6, color='gray')
        ax.grid(True, which='minor', alpha=0.15, linestyle='--', linewidth=0.4, color='lightgray')
        ax.legend(loc='upper right', fontsize=10, frameon=True)

        # Right: E_vv
        ax = axes[1]
        ax.loglog(f_star_vals[1:], E_vv_single[1:], 'b-', linewidth=2.0, alpha=0.7,
                  label=f'Single-point (y/c={y_val:.4f})')
        ax.loglog(f_star_vals[1:], E_vv_avg[1:], 'r-', linewidth=2.0, alpha=0.7,
                  label=f'Spatially-averaged (y/c={y_val:.4f})')

        E_ref = E_vv_single[mid_idx] * (f_ref / f_ref[len(f_ref)//2]) ** (-5.0/3.0)
        ax.loglog(f_ref, E_ref, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
                  label=r'$f^{-5/3}$ (Kolmogorov)')

        ax.set_xlabel(r'$f^* = f \cdot c / U_\infty$ (nondimensional)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'$E_{vv}$ [m$^2$/s$^2$]', fontsize=11, fontweight='bold')
        ax.set_title(f'Cross-stream Energy Spectrum (E_vv)', fontsize=12, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.6, color='gray')
        ax.grid(True, which='minor', alpha=0.15, linestyle='--', linewidth=0.4, color='lightgray')
        ax.legend(loc='upper right', fontsize=10, frameon=True)

    fig.suptitle(f'Spectral Overlay Comparison - {slice_id} (Single-point vs Spatially-averaged)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    overlay_file = os.path.join(SAVE_DIR, f"comparison_overlay_spectra_{slice_id}.png")
    plt.savefig(overlay_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {overlay_file}")
    plt.show()

# -------- PLOT 2: SPECTRAL RATIO (Average / Single-point) --------
if COMPARISON_PLOTS['spectral_ratio']:
    print("\n  Generating: Spectral Ratio Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(COMPARISON_FIG_WIDTH, COMPARISON_FIG_HEIGHT*0.6))

    for plot_idx, j_idx in enumerate(y_locations_idx):
        y_val = energy_spectra_single[j_idx]['y']
        f_star_vals = energy_spectra_single[j_idx]['f_star']
        E_uu_single = energy_spectra_single[j_idx]['E_uu']
        E_uu_avg = energy_spectra_avg[j_idx]['E_uu']
        E_vv_single = energy_spectra_single[j_idx]['E_vv']
        E_vv_avg = energy_spectra_avg[j_idx]['E_vv']

        # Compute ratios (avoid division by zero)
        ratio_uu = E_uu_avg / (E_uu_single + 1e-15)
        ratio_vv = E_vv_avg / (E_vv_single + 1e-15)

        # Left: E_uu ratio
        ax = axes[0]
        ax.semilogx(f_star_vals[1:], ratio_uu[1:], 'b-', linewidth=2.0, alpha=0.8,
                    label=f'y/c={y_val:.4f}')
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Identity')
        ax.fill_between(f_star_vals[1:], 0.95, 1.05, alpha=0.1, color='green')
        ax.set_xlabel(r'$f^* = f \cdot c / U_\infty$ (nondimensional)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'$E_{uu}^{avg} / E_{uu}^{single}$ (Ratio)', fontsize=11, fontweight='bold')
        ax.set_title(f'Streamwise Spectral Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.6, color='gray')
        ax.grid(True, which='minor', alpha=0.1, linestyle='--', linewidth=0.4, color='lightgray')
        ax.set_ylim([0.5, 1.5])
        ax.legend(loc='best', fontsize=10)

        # Right: E_vv ratio
        ax = axes[1]
        ax.semilogx(f_star_vals[1:], ratio_vv[1:], 'r-', linewidth=2.0, alpha=0.8,
                    label=f'y/c={y_val:.4f}')
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Identity')
        ax.fill_between(f_star_vals[1:], 0.95, 1.05, alpha=0.1, color='green')
        ax.set_xlabel(r'$f^* = f \cdot c / U_\infty$ (nondimensional)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'$E_{vv}^{avg} / E_{vv}^{single}$ (Ratio)', fontsize=11, fontweight='bold')
        ax.set_title(f'Cross-stream Spectral Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.6, color='gray')
        ax.grid(True, which='minor', alpha=0.1, linestyle='--', linewidth=0.4, color='lightgray')
        ax.set_ylim([0.5, 1.5])
        ax.legend(loc='best', fontsize=10)

    fig.suptitle(f'Spectral Ratio (Averaged / Single-point) - {slice_id}',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    ratio_file = os.path.join(SAVE_DIR, f"comparison_spectral_ratio_{slice_id}.png")
    plt.savefig(ratio_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {ratio_file}")
    plt.show()

# -------- PLOT 3: STATISTICAL COMPARISON TABLE --------
if COMPARISON_PLOTS['statistical_table']:
    print("\n  Generating: Statistical Comparison Table...")
    fig, ax = plt.subplots(figsize=(COMPARISON_FIG_WIDTH, max(4, len(y_locations_idx)*0.6)))

    # Prepare table data
    row_labels = []
    table_data = []

    for i, j_idx in enumerate(y_locations_idx):
        m = comparison_metrics[j_idx]
        row_labels.append(f"P{i}: y/c={m['y_actual']:.4f}")

        row = [
            f"{m['var_u_single']:.3e}",
            f"{m['var_u_avg']:.3e}",
            f"{m['var_u_reduction_pct']:+.2f}%",
            f"{m['rms_u_single']:.3e}",
            f"{m['rms_u_avg']:.3e}",
            f"{m['peak_freq_u_single']:.3e}",
            f"{m['peak_freq_u_avg']:.3e}",
            f"{m['peak_shift_pct_u']:+.2f}%",
            f"{m['snr_improvement_db_u']:+.2f} dB",
            f"{m['coherence_u']:.4f}",
        ]
        table_data.append(row)

    col_labels = [
        'Var(u)-Single',
        'Var(u)-Avg',
        'Var Reduction',
        'RMS(u)-Single',
        'RMS(u)-Avg',
        'f_peak(u)-Single',
        'f_peak(u)-Avg',
        'Peak Shift',
        'SNR Improve.',
        'Coherence'
    ]

    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color row labels
    for i in range(len(row_labels)):
        table[(i+1, -1)].set_facecolor('#cccccc')
        table[(i+1, -1)].set_text_props(weight='bold')

    ax.axis('off')
    fig.suptitle(f'Statistical Comparison: Single-point vs Spatially-averaged - {slice_id}',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()
    table_file = os.path.join(SAVE_DIR, f"comparison_statistics_{slice_id}.png")
    plt.savefig(table_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {table_file}")
    plt.show()

# -------- PLOT 4: SNR AND NOISE FLOOR ANALYSIS --------
if COMPARISON_PLOTS['coherence_analysis']:
    print("\n  Generating: SNR and Noise Floor Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(COMPARISON_FIG_WIDTH, COMPARISON_FIG_HEIGHT*0.6))

    j_idx = y_locations_idx[0]  # Plot first probe
    y_val = energy_spectra_single[j_idx]['y']
    f_star_vals = energy_spectra_single[j_idx]['f_star']

    # Left: Noise floor comparison
    ax = axes[0]
    noise_floor_u_s = comparison_metrics[j_idx]['noise_floor_u_single']
    noise_floor_u_a = comparison_metrics[j_idx]['noise_floor_u_avg']
    noise_floor_v_s = comparison_metrics[j_idx]['noise_floor_v_single']
    noise_floor_v_a = comparison_metrics[j_idx]['noise_floor_v_avg']

    bars_x = np.arange(2)
    width = 0.35

    ax.bar(bars_x - width/2, [noise_floor_u_s, noise_floor_v_s], width, label='Single-point',
           color='blue', alpha=0.7)
    ax.bar(bars_x + width/2, [noise_floor_u_a, noise_floor_v_a], width, label='Spatially-averaged',
           color='red', alpha=0.7)

    ax.set_ylabel('Noise Floor [m²/s²]', fontsize=11, fontweight='bold')
    ax.set_title('Noise Floor Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(bars_x)
    ax.set_xticklabels(['E_uu', 'E_vv'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Right: SNR improvement
    ax = axes[1]
    snr_u = comparison_metrics[j_idx]['snr_improvement_db_u']
    snr_v = comparison_metrics[j_idx]['snr_improvement_db_v']
    coherence_u = comparison_metrics[j_idx]['coherence_u']
    coherence_v = comparison_metrics[j_idx]['coherence_v']

    x_pos = np.arange(4)
    ax.bar(x_pos, [snr_u, snr_v, coherence_u*10, coherence_v*10], color=['blue', 'blue', 'red', 'red'],
           alpha=0.7)

    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title('SNR Improvement & Coherence', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['SNR_u\n(dB)', 'SNR_v\n(dB)', 'Coh_u\n(×10)', 'Coh_v\n(×10)'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    fig.suptitle(f'SNR & Coherence Analysis - {slice_id} (y/c={y_val:.4f})',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    snr_file = os.path.join(SAVE_DIR, f"comparison_snr_analysis_{slice_id}.png")
    plt.savefig(snr_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {snr_file}")
    plt.show()

# ============================================================================
# SAVE COMPARISON METRICS TO JSON
# ============================================================================

print("\n" + "="*70)
print("SAVE COMPARISON METRICS TO JSON")
print("="*70)

# Convert numpy types to native Python types for JSON serialization
metrics_json = {}
for j_idx, metrics in comparison_metrics.items():
    j_idx_str = str(int(j_idx))
    metrics_json[j_idx_str] = {}
    for key, val in metrics.items():
        if isinstance(val, (np.integer, np.floating)):
            metrics_json[j_idx_str][key] = float(val)
        else:
            metrics_json[j_idx_str][key] = val

json_file = os.path.join(SAVE_DIR, f"comparison_metrics_{slice_id}.json")
with open(json_file, 'w') as f:
    json.dump(metrics_json, f, indent=2)

print(f"Comparison metrics saved: {json_file}")

# ============================================================================
# SAVE DETAILED RESULTS TO HDF5
# ============================================================================

print("\n" + "="*70)
print("SAVE DETAILED RESULTS TO HDF5")
print("="*70)

# Single-point spectra
output_file_single = os.path.join(SAVE_DIR, f"energy_spectra_data_single_point_{slice_id}.h5")
with h5py.File(output_file_single, "w") as f:
    f.attrs["method"] = "single-point"
    f.attrs["slice_id"] = slice_id
    f.attrs["slice_x"] = slice_x
    f.attrs["AOA_deg"] = AOA_deg
    f.attrs["dt_iteration"] = dt_iteration
    f.attrs["delta_iter"] = delta_iter
    f.attrs["dt_save"] = dt_save
    f.attrs["u_infty"] = u_infty
    f.attrs["c"] = c
    f.attrs["Re"] = Re_c
    f.attrs["n_samples"] = n_samples
    f.attrs["n_z"] = nz
    f.attrs["fs"] = fs
    f.attrs["nfreq"] = nfreq

    for probe_idx, j_idx in enumerate(y_locations_idx):
        y_target = Y_LOCATIONS[probe_idx]
        y_actual = y_locations_val[probe_idx]

        grp_name = f"probe_{probe_idx:02d}"
        grp = f.create_group(grp_name)

        grp.attrs["probe_name"] = f"probe_{probe_idx:02d}"
        grp.attrs["y_target"] = y_target
        grp.attrs["y_actual"] = y_actual
        grp.attrs["y_distance_error"] = np.abs(y_actual - y_target)
        grp.attrs["j_index"] = j_idx

        grp.create_dataset("frequencies", data=energy_spectra_single[j_idx]['frequencies'])
        grp.create_dataset("f_star", data=energy_spectra_single[j_idx]['f_star'])
        grp.create_dataset("E_uu", data=energy_spectra_single[j_idx]['E_uu'])
        grp.create_dataset("E_vv", data=energy_spectra_single[j_idx]['E_vv'])
        grp.create_dataset("E_uu_z", data=energy_spectra_single[j_idx]['E_uu_z'])
        grp.create_dataset("E_vv_z", data=energy_spectra_single[j_idx]['E_vv_z'])

        grp.attrs["var_u_time_mean"] = energy_spectra_single[j_idx]['var_u_time']
        grp.attrs["var_v_time_mean"] = energy_spectra_single[j_idx]['var_v_time']
        grp.attrs["var_u_spectral"] = energy_spectra_single[j_idx]['var_u_spectral']
        grp.attrs["var_v_spectral"] = energy_spectra_single[j_idx]['var_v_spectral']
        grp.attrs["u_rms"] = energy_spectra_single[j_idx]['u_rms']
        grp.attrs["v_rms"] = energy_spectra_single[j_idx]['v_rms']

print(f"Single-point results saved: {output_file_single}")

# Spatially-averaged spectra
output_file_avg = os.path.join(SAVE_DIR, f"energy_spectra_data_spatial_avg_{slice_id}.h5")
with h5py.File(output_file_avg, "w") as f:
    f.attrs["method"] = "spatial-average-3point"
    f.attrs["slice_id"] = slice_id
    f.attrs["slice_x"] = slice_x
    f.attrs["AOA_deg"] = AOA_deg
    f.attrs["dt_iteration"] = dt_iteration
    f.attrs["delta_iter"] = delta_iter
    f.attrs["dt_save"] = dt_save
    f.attrs["u_infty"] = u_infty
    f.attrs["c"] = c
    f.attrs["Re"] = Re_c
    f.attrs["n_samples"] = n_samples
    f.attrs["n_z"] = nz
    f.attrs["fs"] = fs
    f.attrs["nfreq"] = nfreq
    f.attrs["spatial_averaging_enabled"] = ENABLE_SPATIAL_AVERAGING
    f.attrs["weights"] = SPATIAL_AVG_WEIGHTS

    for probe_idx, j_idx in enumerate(y_locations_idx):
        y_target = Y_LOCATIONS[probe_idx]
        y_actual = y_locations_val[probe_idx]
        j_minus, j0, j_plus, delta_y_minus, delta_y_plus = y_locations_adjacent[probe_idx]

        grp_name = f"probe_{probe_idx:02d}"
        grp = f.create_group(grp_name)

        grp.attrs["probe_name"] = f"probe_{probe_idx:02d}_spatially_averaged"
        grp.attrs["y_target"] = y_target
        grp.attrs["y_actual"] = y_actual
        grp.attrs["y_distance_error"] = np.abs(y_actual - y_target)
        grp.attrs["j_index_center"] = j0
        grp.attrs["j_index_minus"] = j_minus
        grp.attrs["j_index_plus"] = j_plus
        grp.attrs["delta_y_minus"] = delta_y_minus
        grp.attrs["delta_y_plus"] = delta_y_plus
        grp.attrs["weight_minus"] = SPATIAL_AVG_WEIGHTS[0]
        grp.attrs["weight_center"] = SPATIAL_AVG_WEIGHTS[1]
        grp.attrs["weight_plus"] = SPATIAL_AVG_WEIGHTS[2]

        grp.create_dataset("frequencies", data=energy_spectra_avg[j_idx]['frequencies'])
        grp.create_dataset("f_star", data=energy_spectra_avg[j_idx]['f_star'])
        grp.create_dataset("E_uu", data=energy_spectra_avg[j_idx]['E_uu'])
        grp.create_dataset("E_vv", data=energy_spectra_avg[j_idx]['E_vv'])
        grp.create_dataset("E_uu_z", data=energy_spectra_avg[j_idx]['E_uu_z'])
        grp.create_dataset("E_vv_z", data=energy_spectra_avg[j_idx]['E_vv_z'])

        grp.attrs["var_u_time_mean"] = energy_spectra_avg[j_idx]['var_u_time']
        grp.attrs["var_v_time_mean"] = energy_spectra_avg[j_idx]['var_v_time']
        grp.attrs["var_u_spectral"] = energy_spectra_avg[j_idx]['var_u_spectral']
        grp.attrs["var_v_spectral"] = energy_spectra_avg[j_idx]['var_v_spectral']
        grp.attrs["u_rms"] = energy_spectra_avg[j_idx]['u_rms']
        grp.attrs["v_rms"] = energy_spectra_avg[j_idx]['v_rms']

print(f"Spatially-averaged results saved: {output_file_avg}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY OF SPATIAL AVERAGING COMPARISON")
print("="*70)
print(f"Slice: {slice_id}")
print(f"Method: Comparison of single-point vs 3-point spatial averaging")
print(f"Number of probes: {len(Y_LOCATIONS)}")
print(f"Number of snapshots: {n_samples}")
print(f"Physical time span: {(n_samples-1)*dt_save:.6e} s")
print(f"Sampling frequency: {fs:.6e} Hz")
print(f"Frequency range: {frequencies[1]:.6e} to {frequencies[-1]:.6e} Hz")
print(f"Spanwise samples per probe: {nz}")
print(f"\nComparison Metrics Generated:")
for i, j_idx in enumerate(y_locations_idx):
    m = comparison_metrics[j_idx]
    print(f"\n  Probe {i} (y/c = {m['y_actual']:.4f}):")
    print(f"    E_uu SNR improvement: {m['snr_improvement_db_u']:+.2f} dB")
    print(f"    E_vv SNR improvement: {m['snr_improvement_db_v']:+.2f} dB")
    print(f"    Spectral coherence (u): {m['coherence_u']:.4f}")
    print(f"    Spectral coherence (v): {m['coherence_v']:.4f}")

print(f"\nOutput files generated:")
print(f"  Single-point spectra (HDF5): {output_file_single}")
print(f"  Spatial-avg spectra (HDF5): {output_file_avg}")
print(f"  Comparison metrics (JSON): {json_file}")
if COMPARISON_PLOTS['overlay_spectra']:
    print(f"  Overlay comparison plot: {overlay_file}")
if COMPARISON_PLOTS['spectral_ratio']:
    print(f"  Spectral ratio plot: {ratio_file}")
if COMPARISON_PLOTS['statistical_table']:
    print(f"  Statistical table plot: {table_file}")
if COMPARISON_PLOTS['coherence_analysis']:
    print(f"  SNR analysis plot: {snr_file}")

print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
