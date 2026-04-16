"""
visualize_2d_correlation_map.py
================================
Load spatial correlation data, visualize 2D correlation map at Dz=0,
and extract velocity signals at probe locations matching the slice geometry.

This script:
1. Loads correlation data to get the reference x/c location (xc_ref)
2. Finds the corresponding slice location
3. Loads the slice mesh to extract surface geometry
4. Selects probes at specified y-coordinates
5. Extracts velocity time series at those probe locations
6. Visualizes the correlation map with reference markers
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

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from data_loader_functions import CompressedSnapshotLoader
except ImportError:
    print("Warning: CompressedSnapshotLoader not available")
    CompressedSnapshotLoader = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Correlation data file (output of wall_shear_correlations_2.py)
CORR_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/test_3/"
    "wall_shear_correlation_xc_0.500_alpha_1.0_all_fft.h5"
)

# Base path for slice data (will be matched to xc_ref from correlation file)
SLICES_BASE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/"

# Probe locations (y-coordinates) to extract from the slice
# These should match y-coordinates that appear in the correlation map
PROBE_Y_COORDS = [0.06, 0.10, 0.15, 0.20]  # Example y-coordinates

# Output directory (set to None to only display interactively)
OUTPUT_DIR = None

# Physical parameters (for signal extraction)
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

# Geometric data (for reference point detection)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# ============================================================================
# LOAD SPATIAL CORRELATION DATA
# ============================================================================
print("=" * 70)
print("2D SPATIAL CORRELATION MAP VISUALIZATION")
print("=" * 70)

print(f"\nLoading spatial correlation from:\n  {CORR_FILE}")

with h5py.File(CORR_FILE, "r") as f:
    R_all  = f['R_all'][:]     # (Nz, Ny_crop, Nx_crop)
    x_corr = f['x'][:]        # (Nz, Ny_crop, Nx_crop)
    y_corr = f['y'][:]
    xc_ref = float(f.attrs['x_c_actual'])
    yc_ref = float(f.attrs['y_actual'])

# Take the Dz=0 slice (first z-index, since z-axis = relative separation)
R_all_2d = R_all[0, :, :]       # (Ny_crop, Nx_crop)
x_2d     = x_corr[0, :, :]
y_2d     = y_corr[0, :, :]

print(f"  R_all shape: {R_all.shape}")
print(f"  Dz=0 slice : {R_all_2d.shape}")
print(f"  Reference pt: x/c={xc_ref:.3f}, y={yc_ref:.4f}")

# ============================================================================
# LOAD GEOMETRICAL DATA FOR REFERENCE POINT DETECTION
# ============================================================================
print("\n" + "="*70)
print("LOADING GEOMETRICAL DATA FOR REFERENCE POINT")
print("="*70)

ref_point_data = {
    'surface_normal': None,
    'wall_distance': None,
    'closest_interface_idx': None
}

try:
    if os.path.exists(GEO_FILE):
        print(f"\nLoading geometrical data from:\n  {GEO_FILE}")
        with h5py.File(GEO_FILE, "r") as f:
            interface_points = f["interface_points"][...].astype(np.float64)
            proj_normals = f["proj_normals"][...].astype(np.float64)
            proj_distances = f["proj_distances"][...].astype(np.float64)

        # Extract suction side (y >= 0) and pressure side (y < 0)
        suction_side_points = interface_points[interface_points[:, 1] >= 0]
        suction_side_indices = np.where(interface_points[:, 1] >= 0)[0]

        print(f"  Suction side points: {suction_side_points.shape[0]}")

        # Find closest point to reference on suction side by x-coordinate
        # (reference point should be at the correlation reference location)
        x_distances = np.abs(suction_side_points[:, 0] - xc_ref)
        closest_idx = np.argmin(x_distances)
        closest_surface_point = suction_side_points[closest_idx]
        closest_interface_idx = suction_side_indices[closest_idx]

        surface_normal = proj_normals[closest_interface_idx]
        surface_distance = proj_distances[closest_interface_idx]

        ref_point_data['surface_normal'] = surface_normal
        ref_point_data['wall_distance'] = surface_distance
        ref_point_data['closest_interface_idx'] = closest_interface_idx

        print(f"  Closest surface point found:")
        print(f"    x: {closest_surface_point[0]:.6f}")
        print(f"    y: {closest_surface_point[1]:.6e}")
        print(f"    Distance from xc_ref: {x_distances[closest_idx]:.6e}")
        print(f"    Surface normal: [{surface_normal[0]:.6f}, {surface_normal[1]:.6f}]")
        print(f"    Wall distance: {surface_distance:.6e}")
    else:
        print(f"⚠ Geometrical data file not found: {GEO_FILE}")
        print(f"Continuing with reference point visualization only")
except Exception as e:
    print(f"⚠ Error loading geometrical data: {e}")
    print(f"Continuing with reference point visualization only")

# ============================================================================
# LOAD SLICE DATA AND SELECT PROBES
# ============================================================================
print("\n" + "="*70)
print("SLICE DATA LOADING AND PROBE SELECTION")
print("="*70)

# Utility functions
def find_closest_slice(xc_ref, slices_base_path):
    """Find the slice directory closest to xc_ref value."""
    slices = []
    for item in os.listdir(slices_base_path):
        item_path = os.path.join(slices_base_path, item)
        if os.path.isdir(item_path) and item.startswith('slice_'):
            slices.append((item, item_path))

    if not slices:
        raise FileNotFoundError(f"No slice directories found in {slices_base_path}")

    # Try to find slice with matching xc_ref in filenames or best guess
    print(f"\nAvailable slices:")
    for name, path in sorted(slices):
        print(f"  {name}")

    # Use the slice that's closest in naming (simple heuristic)
    # For x/c = 0.5, look for slice_5 or similar numeric identifier
    slice_num = int(round(xc_ref * 10))  # e.g., 0.5 -> 5
    target_name = f"slice_{slice_num}"

    for name, path in slices:
        if target_name in name or name.endswith(str(slice_num)):
            print(f"\n✓ Found matching slice: {name}")
            return name, path

    # Fallback: ask user or use first available
    print(f"\n⚠ No exact match for slice at x/c={xc_ref:.3f}")
    print(f"Using first available slice for demonstration")
    return slices[0]

# Find and load slice
probe_info = []
ref_probe_info = None
try:
    slice_name, slice_path = find_closest_slice(xc_ref, SLICES_BASE_PATH)
    print(f"Slice path: {slice_path}")

    # Look for mesh file
    mesh_files = glob.glob(os.path.join(slice_path, "*-CROP-MESH.h5"))
    if not mesh_files:
        raise FileNotFoundError(f"No mesh file found in {slice_path}")

    mesh_file = mesh_files[0]
    print(f"✓ Mesh file: {os.path.basename(mesh_file)}")

    # Load mesh
    if CompressedSnapshotLoader is not None:
        loader = CompressedSnapshotLoader(mesh_file)
        x_data = loader.x[1:-1, :, :]  # Exclude ghost cells
        y_data = loader.y[1:-1, :, :]
        z_data = loader.z[1:-1, :, :]

        print(f"✓ Mesh loaded: shape (nz, ny, nx) = {x_data.shape}")

        # Get mesh parameters
        y_unique = np.unique(y_data[:, :, 0][0, :])
        print(f"  Y-grid: {len(y_unique)} points, range [{y_unique[0]:.6f}, {y_unique[-1]:.6f}]")

        # Select probes at specified y-coordinates
        print(f"\nProbe selection (target y-coordinates):")
        for i, y_target in enumerate(PROBE_Y_COORDS):
            # Find closest grid point
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

            print(f"  Probe {i}: y_target={y_target:.6f} → y_actual={y_actual:.6f} (error={error:.6e})")
    else:
        print("⚠ CompressedSnapshotLoader unavailable, skipping mesh operations")
        probe_info = []

    # ========================================================================
    # FIND CLOSEST MESH POINT FOR REFERENCE PROBE
    # ========================================================================
    print(f"\n{'-'*70}")
    print("REFERENCE PROBE MESH POINT SELECTION")
    print(f"{'-'*70}")

    try:
        if y_unique is not None and len(y_unique) > 0:
            # Find closest mesh point to yc_ref
            idx_ref = np.argmin(np.abs(y_unique - yc_ref))
            y_actual_ref = y_unique[idx_ref]
            error_ref = np.abs(y_actual_ref - yc_ref)

            ref_probe_info = {
                'probe_id': 'reference',
                'label': 'reference',
                'y_target': yc_ref,
                'y_actual': y_actual_ref,
                'y_idx': idx_ref,
                'error': error_ref
            }

            print(f"\nReference probe from correlation file:")
            print(f"  y_target (yc_ref from corr file): {yc_ref:.6e}")
            print(f"  y_actual (closest mesh point):   {y_actual_ref:.6e}")
            print(f"  Grid index (j_idx):              {idx_ref}")
            print(f"  Distance error:                  {error_ref:.6e}")

            if error_ref > 1e-6:
                print(f"  ⚠ Note: Reference point snapped to nearest mesh grid point")
        else:
            print("⚠ y_unique not available, cannot locate reference mesh point")
    except Exception as e:
        print(f"⚠ Error finding reference mesh point: {e}")

except Exception as e:
    print(f"⚠ Error loading slice: {e}")
    print(f"Continuing with correlation map visualization only")
    probe_info = []
    ref_probe_info = None

# ============================================================================
# EXTRACT VELOCITY AND WALL SHEAR STRESS SIGNALS
# ============================================================================
print("\n" + "="*70)
print("SIGNAL EXTRACTION: VELOCITY AND WALL SHEAR STRESS")
print("="*70)

signal_data = {}
total_timesteps = 0

# Define cache file path
cache_dir = os.path.join(os.path.dirname(CORR_FILE), "signal_cache")
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, f"signals_{slice_name}.h5")

# Function to load signals from cache
def load_signals_from_cache(cache_file):
    """Load cached signal data from HDF5 file."""
    try:
        signal_data_loaded = {}
        with h5py.File(cache_file, 'r') as f:
            for key in f.keys():
                signal_data_loaded[key] = {
                    'iterations': f[key]['iterations'][...],
                    'time': f[key]['time'][...],
                    'u_prime': f[key]['u_prime'][...],
                    'v_prime': f[key]['v_prime'][...],
                    'w_prime': f[key]['w_prime'][...],
                    'tau_prime': f[key]['tau_prime'][...]
                }
        return signal_data_loaded
    except Exception as e:
        print(f"  ⚠ Error loading cache: {e}")
        return None

# Function to save signals to cache
def save_signals_to_cache(signal_data, cache_file):
    """Save signal data to HDF5 cache file."""
    try:
        with h5py.File(cache_file, 'w') as f:
            for key, data in signal_data.items():
                group = f.create_group(key)
                group.create_dataset('iterations', data=data['iterations'], compression='gzip')
                group.create_dataset('time', data=data['time'], compression='gzip')
                group.create_dataset('u_prime', data=data['u_prime'], compression='gzip')
                group.create_dataset('v_prime', data=data['v_prime'], compression='gzip')
                group.create_dataset('w_prime', data=data['w_prime'], compression='gzip')
                group.create_dataset('tau_prime', data=data['tau_prime'], compression='gzip')
        print(f"  ✓ Signals cached to: {cache_file}")
        return True
    except Exception as e:
        print(f"  ⚠ Error saving cache: {e}")
        return False

# ============================================================================
# SPECTRAL ANALYSIS HELPER FUNCTIONS
# ============================================================================

def compute_sampling_frequency(time_array):
    """
    Compute sampling frequency from time array.

    Args:
        time_array: Time points [s]

    Returns:
        fs: Sampling frequency [Hz]
    """
    dt = np.diff(time_array)
    dt_mean = np.mean(dt)
    fs = 1.0 / dt_mean
    return fs


def compute_psd_welch(signal_data, fs, window='hann', nperseg=None, noverlap=None):
    """
    Compute Power Spectral Density using Welch's method.

    Args:
        signal_data: Input signal (time series)
        fs: Sampling frequency [Hz]
        window: Window function ('hann', 'hamming', etc.)
        nperseg: Segment length for Welch's method
        noverlap: Overlap between segments

    Returns:
        frequencies: Frequency array [Hz]
        psd: Power Spectral Density
    """
    # Remove NaN values
    valid_idx = ~np.isnan(signal_data)
    signal_clean = signal_data[valid_idx]

    # Remove mean (zero-center the signal)
    signal_centered = signal_clean - np.mean(signal_clean)

    # Compute PSD using Welch's method
    frequencies, psd = signal.welch(
        signal_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    return frequencies, psd


def compute_cross_spectrum_welch(signal1, signal2, fs, window='hann', nperseg=None, noverlap=None):
    """
    Compute Cross-Spectrum between two signals using Welch's method.

    Args:
        signal1: First signal (time series)
        signal2: Second signal (time series)
        fs: Sampling frequency [Hz]
        window: Window function
        nperseg: Segment length
        noverlap: Overlap

    Returns:
        frequencies: Frequency array [Hz]
        cross_spectrum: Complex cross-spectrum
    """
    # Remove NaN values
    valid_idx = ~(np.isnan(signal1) | np.isnan(signal2))
    signal1_clean = signal1[valid_idx]
    signal2_clean = signal2[valid_idx]

    # Remove mean (zero-center both signals)
    signal1_centered = signal1_clean - np.mean(signal1_clean)
    signal2_centered = signal2_clean - np.mean(signal2_clean)

    # Compute cross-spectrum using Welch's method
    frequencies, cross_spectrum = signal.csd(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    return frequencies, cross_spectrum


def compute_coherence_welch(signal1, signal2, fs, window='hann', nperseg=None, noverlap=None):
    """
    Compute Magnitude-Squared Coherence between two signals.

    Args:
        signal1: First signal (time series)
        signal2: Second signal (time series)
        fs: Sampling frequency [Hz]
        window: Window function
        nperseg: Segment length
        noverlap: Overlap

    Returns:
        frequencies: Frequency array [Hz]
        coherence: Magnitude-squared coherence (0 to 1)
    """
    # Remove NaN values
    valid_idx = ~(np.isnan(signal1) | np.isnan(signal2))
    signal1_clean = signal1[valid_idx]
    signal2_clean = signal2[valid_idx]

    # Remove mean (zero-center both signals)
    signal1_centered = signal1_clean - np.mean(signal1_clean)
    signal2_centered = signal2_clean - np.mean(signal2_clean)

    # Compute coherence using Welch's method
    frequencies, coherence = signal.coherence(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )

    return frequencies, coherence


def nondimensionalize_frequency(frequency_array, U_inf, c):
    """
    Convert dimensional frequency to nondimensional frequency.

    f* = f * c / U_inf

    Args:
        frequency_array: Dimensional frequency [Hz]
        U_inf: Free-stream velocity [m/s]
        c: Characteristic length scale (chord) [m]

    Returns:
        f_star: Nondimensional frequency [-]
    """
    f_star = frequency_array * c / U_inf
    return f_star

# Check if cache exists and load if available
if os.path.exists(cache_file):
    print(f"\nCache file found: {cache_file}")
    print(f"Loading cached signals...")
    signal_data = load_signals_from_cache(cache_file)
    if signal_data:
        print(f"  ✓ Loaded {len(signal_data)} cached signal datasets")
        total_timesteps = len(signal_data['ref_point']['iterations']) if 'ref_point' in signal_data else 0
    else:
        print(f"  ⚠ Cache loading failed, will extract from snapshots")
        signal_data = {}

# If no cache, extract signals
if not signal_data and probe_info and ref_probe_info and 'loader' in locals() and loader is not None:
    try:
        # Get all data files from slice directory
        data_files = glob.glob(os.path.join(slice_path, "*-COMP-DATA.h5"))

        def get_iteration(filepath):
            match = re.search(r'_(\d+)-COMP-DATA', filepath)
            return int(match.group(1)) if match else 0

        data_files.sort(key=get_iteration)

        if len(data_files) == 0:
            print("⚠ No data files found in slice directory")
        else:
            print(f"\nExtracting from {len(data_files)} snapshot files...")

            # Initialize signal storage
            raw_signals = defaultdict(lambda: {
                'iterations': [],
                'u': [],
                'v': [],
                'w': [],
                'tau': []
            })

            # Loop through all data files
            for file_idx, data_file in enumerate(data_files):
                if (file_idx + 1) % max(1, len(data_files) // 10) == 0:
                    print(f"  Processed: {file_idx + 1}/{len(data_files)}")

                try:
                    # Extract iteration number
                    match = re.search(r'_(\d+)-COMP-DATA', data_file)
                    iteration = int(match.group(1)) if match else file_idx

                    # Load snapshot
                    snapshot = loader.load_snapshot(data_file)
                    u_data_full = loader.reconstruct_field(snapshot["u"])
                    v_data_full = loader.reconstruct_field(snapshot["v"])
                    w_data_full = loader.reconstruct_field(snapshot["w"])

                    # Remove ghost cells
                    u_data = u_data_full[1:-1, :, :]
                    v_data = v_data_full[1:-1, :, :]
                    w_data = w_data_full[1:-1, :, :]

                    # Extract velocity at each probe
                    for probe in probe_info:
                        probe_id = probe['probe_id']
                        y_idx = probe['y_idx']

                        u_val = u_data[0, y_idx, 0]
                        v_val = v_data[0, y_idx, 0]
                        w_val = w_data[0, y_idx, 0]

                        raw_signals[f'probe_{probe_id}']['iterations'].append(iteration)
                        raw_signals[f'probe_{probe_id}']['u'].append(u_val)
                        raw_signals[f'probe_{probe_id}']['v'].append(v_val)
                        raw_signals[f'probe_{probe_id}']['w'].append(w_val)

                    # Extract velocity and compute shear stress at surface point
                    y_idx_ref = ref_probe_info['y_idx']
                    u_val_ref = u_data[0, y_idx_ref, 0]
                    v_val_ref = v_data[0, y_idx_ref, 0]
                    w_val_ref = w_data[0, y_idx_ref, 0]

                    # Compute wall shear stress
                    if ref_point_data['surface_normal'] is not None and ref_point_data['wall_distance'] is not None:
                        surface_normal = ref_point_data['surface_normal']
                        surface_distance = ref_point_data['wall_distance']

                        # Tangent vector (rotate normal 90 degrees)
                        tangent = np.array([surface_normal[1], -surface_normal[0]])
                        tangent = tangent / np.linalg.norm(tangent)

                        # Project velocity onto tangent
                        u_t = u_val_ref * tangent[0] + v_val_ref * tangent[1]

                        # Shear stress
                        tau_val = mu_ref * u_t / surface_distance
                    else:
                        tau_val = np.nan

                    raw_signals['ref_point']['iterations'].append(iteration)
                    raw_signals['ref_point']['u'].append(u_val_ref)
                    raw_signals['ref_point']['v'].append(v_val_ref)
                    raw_signals['ref_point']['w'].append(w_val_ref)
                    raw_signals['ref_point']['tau'].append(tau_val)

                    total_timesteps += 1

                except Exception as e:
                    print(f"    ⚠ Error processing file {os.path.basename(data_file)}: {e}")
                    continue

            # Compute temporal means
            print(f"\n  ✓ Processed {total_timesteps} valid timesteps")
            print(f"\nComputing temporal statistics...")

            means = {}
            for key in raw_signals.keys():
                u_array = np.array(raw_signals[key]['u'])
                v_array = np.array(raw_signals[key]['v'])
                w_array = np.array(raw_signals[key]['w'])
                tau_array = np.array(raw_signals[key]['tau'])

                means[key] = {
                    'u_mean': np.mean(u_array),
                    'v_mean': np.mean(v_array),
                    'w_mean': np.mean(w_array),
                    'tau_mean': np.nanmean(tau_array)
                }

            # Compute fluctuations in flow-aligned frame
            print(f"Rotating to flow-aligned frame (AOA={AOA_deg}°)...")

            cos_aoa = np.cos(AOA_rad)
            sin_aoa = np.sin(AOA_rad)

            for key in raw_signals.keys():
                u_array = np.array(raw_signals[key]['u'])
                v_array = np.array(raw_signals[key]['v'])
                w_array = np.array(raw_signals[key]['w'])
                tau_array = np.array(raw_signals[key]['tau'])
                iterations_array = np.array(raw_signals[key]['iterations'])

                # Rotate velocity to flow-aligned frame
                u_rot = u_array * cos_aoa + v_array * sin_aoa
                v_rot = -u_array * sin_aoa + v_array * cos_aoa
                w_rot = w_array

                u_mean_rot = means[key]['u_mean'] * cos_aoa + means[key]['v_mean'] * sin_aoa
                v_mean_rot = -means[key]['u_mean'] * sin_aoa + means[key]['v_mean'] * cos_aoa
                w_mean_rot = means[key]['w_mean']

                # Compute fluctuations
                u_prime = u_rot - u_mean_rot
                v_prime = v_rot - v_mean_rot
                w_prime = w_rot - w_mean_rot
                tau_prime = tau_array - means[key]['tau_mean']

                # Store time-series data
                time_steps = iterations_array * dt_iteration

                signal_data[key] = {
                    'iterations': iterations_array,
                    'time': time_steps,
                    'u_prime': u_prime,
                    'v_prime': v_prime,
                    'w_prime': w_prime,
                    'tau_prime': tau_prime
                }

            # Save to cache
            print(f"\nSaving signals to cache...")
            save_signals_to_cache(signal_data, cache_file)

    except Exception as e:
        print(f"⚠ Error during signal extraction: {e}")
        signal_data = {}
elif not signal_data:
    print("⚠ Insufficient data for signal extraction (no probes, ref point, or loader)")


# ============================================================================
# PLOT: 2D Spatial Correlation Map
# ============================================================================
print("\n" + "="*70)
print("GENERATING 2D SPATIAL CORRELATION MAP")
print("="*70)

fig, ax = plt.subplots(figsize=(14, 8))

# ============================================================================
# LAYER 1: CORRELATION MAP BACKGROUND
# ============================================================================

# Clip R_all to [-1, 1] for robust plotting
R_plot = np.clip(R_all_2d, -1.0, 1.0)

# Hide NaN / zero regions inside the airfoil
R_plot = np.ma.masked_where(np.abs(R_plot) < 1e-12, R_plot)

# Symmetric colorbar range based on data
R_abs_max = np.max(np.abs(R_plot))
levels = np.linspace(-R_abs_max, R_abs_max, 41)

# Plot correlation as background
cf = ax.contourf(x_2d, y_2d, R_plot, levels=levels,
                 cmap='RdBu_r', extend='both', zorder=1)
cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label(r'$R_{\tau_w^\prime u^\prime}$  ($\Delta z = 0$)', fontsize=12, fontweight='bold')

# ============================================================================
# LAYER 2: CONTOUR LINES
# ============================================================================
ax.contour(x_2d, y_2d, R_plot,
           levels=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
           colors='k', linewidths=0.5, alpha=0.3, zorder=2)

# ============================================================================
# LAYER 3: AIRFOIL GEOMETRY (SUCTION SIDE ONLY)
# ============================================================================
# if 'suction_side_points' in locals() and suction_side_points is not None:
#     # Filter for points only on suction side (y >= 0)
#     suction_points_filtered = suction_side_points[suction_side_points[:, 1] >= 0]
#     ax.scatter(suction_points_filtered[:, 0], suction_points_filtered[:, 1],
#                s=20, c='blue', alpha=0.6, label='Suction side geometry', zorder=5)

# ============================================================================
# LAYER 4: VERTICAL REFERENCE LINE
# ============================================================================
ax.axvline(xc_ref, color='green', linewidth=2.5, linestyle='--',
           alpha=0.7, zorder=10, label=f'Reference x/c = {xc_ref:.3f}')

# ============================================================================
# LAYER 5: REFERENCE SURFACE POINT (from mesh)
# ============================================================================
if ref_probe_info is not None:
    y_actual_ref = ref_probe_info['y_actual']
    ax.scatter(xc_ref, y_actual_ref, c='orange', s=300, marker='*',
              edgecolors='black', linewidths=2.5, zorder=15,
              label=f'Surface reference point (y={y_actual_ref:.4f})')

# ============================================================================
# LAYER 6: PROBE LOCATIONS
# ============================================================================
if probe_info:
    # Use distinct colors for each probe
    probe_colors = ['purple', 'cyan', 'magenta', 'brown']
    probe_markers = ['s', 'o', '^', 'v']  # square, circle, triangle up, triangle down

    for i, probe in enumerate(probe_info):
        y_actual = probe['y_actual']
        color = probe_colors[i % len(probe_colors)]
        marker = probe_markers[i % len(probe_markers)]

        ax.scatter(xc_ref, y_actual, c=color, s=180, marker=marker,
                  edgecolors='black', linewidths=1.5, zorder=14,
                  label=f"Probe {i}: y={y_actual:.5f}")

# ============================================================================
# LAYER 7: PLOT CONFIGURATION
# ============================================================================

# Set axis limits (zoom to region of interest)
x_min, x_max = -0.2, 1.2
y_min, y_max = -0.01, 0.3

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel('x/c', fontsize=13, fontweight='bold')
ax.set_ylabel('y/c', fontsize=13, fontweight='bold')
ax.set_title(r'Spatial $R_{\tau_w^\prime u^\prime}$ Correlation with Airfoil Geometry and Probes ($\Delta z = 0$)',
            fontsize=13, fontweight='bold')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.2, which='both')

# ============================================================================
# LAYER 8: LEGEND AND ANNOTATIONS
# ============================================================================

ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92, ncol=1,
         edgecolor='black', fancybox=True)

plt.tight_layout()

# ============================================================================
# SAVE OR SHOW
# ============================================================================
print("\n" + "=" * 70)

if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_2d_map.png"),
                dpi=150, bbox_inches='tight')
    print(f"  Saved: correlation_2d_map.png")

plt.show()

# ============================================================================
# PLOT: TIME SERIES SIGNALS (TAU' AND U')
# ============================================================================
if signal_data and 'ref_point' in signal_data and len(signal_data) > 1:
    print("\n" + "="*70)
    print("GENERATING TIME SERIES PLOTS (TAU' AND U')")
    print("="*70)

    # Create figure with 5x1 subplots (tau' + 4 probes)
    num_probes = len(probe_info)
    fig_signals, axes_signals = plt.subplots(num_probes + 1, 1, figsize=(14, 15))

    # Probe colors (same as correlation map)
    probe_colors_ts = ['purple', 'cyan', 'magenta', 'brown']

    # ========================================================================
    # SUBPLOT 1: WALL SHEAR STRESS FLUCTUATION (Surface Point)
    # ========================================================================
    ax_tau = axes_signals[0]

    if 'ref_point' in signal_data:
        ref_data = signal_data['ref_point']
        time_ref = ref_data['time']
        tau_prime_ref = ref_data['tau_prime']

        ax_tau.plot(time_ref, tau_prime_ref, linewidth=0.8, alpha=0.85,
                    color='#d62728', label='Surface ref', zorder=3)
        ax_tau.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
        ax_tau.grid(True, alpha=0.3, which='both', zorder=0)

        ax_tau.set_ylabel(r"$\tau^\prime$ (Pa)", fontsize=11, fontweight='bold')
        ax_tau.set_title(r'Wall Shear Stress $\tau^\prime$ (Surface)',
                        fontsize=11, fontweight='bold')
        ax_tau.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ========================================================================
    # SUBPLOTS 2-5: STREAMWISE VELOCITY FLUCTUATION (Individual Probes)
    # ========================================================================
    for i, probe in enumerate(probe_info):
        ax_probe = axes_signals[i + 1]
        probe_key = f'probe_{probe["probe_id"]}'

        if probe_key in signal_data:
            probe_data = signal_data[probe_key]
            time_probe = probe_data['time']
            u_prime_probe = probe_data['u_prime']

            color = probe_colors_ts[i % len(probe_colors_ts)]

            ax_probe.plot(time_probe, u_prime_probe, linewidth=0.8, alpha=0.85,
                         color=color, label=f'Probe {i}', zorder=3)
            ax_probe.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
            ax_probe.grid(True, alpha=0.3, which='both', zorder=0)

            ax_probe.set_ylabel(r"$u^\prime$ (m/s)", fontsize=11, fontweight='bold')
            ax_probe.set_title(f'Probe {i}: y={probe["y_actual"]:.5f}',
                              fontsize=11, fontweight='bold')
            ax_probe.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Set x-label only on last subplot
    axes_signals[-1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if OUTPUT_DIR is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig_signals.savefig(os.path.join(OUTPUT_DIR, "signal_timeseries.png"),
                           dpi=150, bbox_inches='tight')
        print(f"  Saved: signal_timeseries.png")

    plt.show()
    print("  ✓ Time series plots generated successfully")

else:
    print("\n⚠ Insufficient signal data for time series plots")

# ============================================================================
# SPECTRAL ANALYSIS: PSD, CROSS-SPECTRUM, AND COHERENCE
# ============================================================================

if signal_data and 'ref_point' in signal_data and len(signal_data) > 1:
    print("\n" + "="*70)
    print("SPECTRAL ANALYSIS: PSDs, CROSS-SPECTRUM, AND COHERENCE")
    print("="*70)

    # Define spectral cache path
    spectral_cache_dir = os.path.join(os.path.dirname(CORR_FILE), "spectral_cache")
    spectral_cache_file = os.path.join(spectral_cache_dir, f"spectral_{slice_name}.h5")

    # Helper function to load spectral results from cache
    def load_spectral_from_cache(cache_file):
        """Load pre-computed spectral results from HDF5 cache."""
        try:
            with h5py.File(cache_file, 'r') as f:
                spectral_dict = {}
                for key in f.keys():
                    spectral_dict[key] = {}
                    for subkey in f[key].keys():
                        spectral_dict[key][subkey] = f[key][subkey][...]
                return spectral_dict
        except Exception as e:
            print(f"  ⚠ Error loading spectral cache: {e}")
            return None

    # Helper function to save spectral results to cache
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

    # Try to load spectral cache
    spectral_data = None
    if os.path.exists(spectral_cache_file):
        print(f"\nSpectral cache found: {spectral_cache_file}")
        print(f"Loading cached spectral results...")
        spectral_data = load_spectral_from_cache(spectral_cache_file)
        if spectral_data:
            print(f"  ✓ Loaded spectral results for {len(spectral_data)} signals")

    # Compute spectral data if not cached
    if spectral_data is None:
        print(f"\nComputing spectral analysis (Welch's method)...")
        print(f"  Segment length (nperseg): {NPERSEG}")
        print(f"  Overlap: {NOVERLAP} ({100*NOVERLAP/NPERSEG:.0f}%)")
        print(f"  Window: {WINDOW}")

        # Compute sampling frequency
        ref_time = signal_data['ref_point']['time']
        fs = compute_sampling_frequency(ref_time)
        print(f"  Sampling frequency: {fs:.2f} Hz")

        # Frequency resolution
        freq_resolution = fs / NPERSEG
        print(f"  Frequency resolution: {freq_resolution:.6e} Hz")

        spectral_data = {
            'metadata': {
                'fs': np.array([fs]),
                'nperseg': np.array([NPERSEG]),
                'noverlap': np.array([NOVERLAP]),
                'window': np.array([WINDOW], dtype='S10')
            }
        }

        # Compute PSD for τ' (reference point)
        print(f"\n  Computing PSD for τ' (surface reference)...")
        tau_prime_ref = signal_data['ref_point']['tau_prime']
        f_tau, psd_tau = compute_psd_welch(tau_prime_ref, fs, window=WINDOW,
                                           nperseg=NPERSEG, noverlap=NOVERLAP)
        spectral_data['ref_point'] = {
            'frequency': f_tau,
            'psd': psd_tau
        }
        print(f"    ✓ Frequency range: {f_tau[1]:.6e} to {f_tau[-1]:.2f} Hz")
        print(f"    ✓ PSD range: {np.min(psd_tau):.6e} to {np.max(psd_tau):.6e}")

        # Compute PSD and cross-spectrum/coherence for each probe
        probe_colors_spectral = ['purple', 'cyan', 'magenta', 'brown']

        for i, probe in enumerate(probe_info):
            probe_key = f'probe_{probe["probe_id"]}'
            print(f"\n  Computing spectral data for {probe_key}...")

            if probe_key in signal_data:
                u_prime_probe = signal_data[probe_key]['u_prime']

                # PSD for u'
                f_u, psd_u = compute_psd_welch(u_prime_probe, fs, window=WINDOW,
                                               nperseg=NPERSEG, noverlap=NOVERLAP)

                # Cross-spectrum between τ' and u'
                f_csd, cross_spec = compute_cross_spectrum_welch(
                    tau_prime_ref, u_prime_probe, fs, window=WINDOW,
                    nperseg=NPERSEG, noverlap=NOVERLAP
                )

                # Coherence between τ' and u'
                f_coh, coherence = compute_coherence_welch(
                    tau_prime_ref, u_prime_probe, fs, window=WINDOW,
                    nperseg=NPERSEG, noverlap=NOVERLAP
                )

                spectral_data[probe_key] = {
                    'frequency_psd': f_u,
                    'psd': psd_u,
                    'frequency_csd': f_csd,
                    'cross_spectrum': cross_spec,
                    'frequency_coh': f_coh,
                    'coherence': coherence
                }

                print(f"    ✓ PSD range: {np.min(psd_u):.6e} to {np.max(psd_u):.6e}")
                print(f"    ✓ Coherence range: {np.min(coherence):.6e} to {np.max(coherence):.6e}")

        # Save spectral data to cache
        save_spectral_to_cache(spectral_data, spectral_cache_file)

    # ========================================================================
    # CONVERT TO NONDIMENSIONAL FREQUENCY
    # ========================================================================

    print(f"\n" + "="*70)
    print("CONVERTING TO NONDIMENSIONAL FREQUENCY")
    print("="*70)
    print(f"  f* = f × c / U_inf = f × {c} / {u_infty}")

    fs = spectral_data['metadata']['fs'][0]

    # Nondimensionalize all frequencies
    f_tau_nd = nondimensionalize_frequency(spectral_data['ref_point']['frequency'],
                                            u_infty, c)

    for i, probe in enumerate(probe_info):
        probe_key = f'probe_{probe["probe_id"]}'
        if probe_key in spectral_data:
            f_u_nd = nondimensionalize_frequency(spectral_data[probe_key]['frequency_psd'],
                                                  u_infty, c)
            f_csd_nd = nondimensionalize_frequency(spectral_data[probe_key]['frequency_csd'],
                                                    u_infty, c)
            f_coh_nd = nondimensionalize_frequency(spectral_data[probe_key]['frequency_coh'],
                                                    u_infty, c)

            # Store nondimensional frequencies
            spectral_data[probe_key]['f_psd_nd'] = f_u_nd
            spectral_data[probe_key]['f_csd_nd'] = f_csd_nd
            spectral_data[probe_key]['f_coh_nd'] = f_coh_nd

    spectral_data['ref_point']['f_tau_nd'] = f_tau_nd

    # ========================================================================
    # FIGURE 1: ALL PSDs OVERLAID (LOG-LOG)
    # ========================================================================

    print(f"\n" + "="*70)
    print("GENERATING FIGURE 1: PSD OVERLAY (LOG-LOG)")
    print("="*70)

    fig_psd, ax_psd = plt.subplots(figsize=(12, 8))

    probe_colors_spectral = ['purple', 'cyan', 'magenta', 'brown']

    # Plot τ' PSD
    freq_start = 1  # Skip DC component
    ax_psd.loglog(f_tau_nd[freq_start:], spectral_data['ref_point']['psd'][freq_start:],
                  'o-', linewidth=1.5, markersize=3, label="τ' (Surface)",
                  color='#d62728', alpha=0.8)

    # Plot u' PSDs for each probe
    for i, probe in enumerate(probe_info):
        probe_key = f'probe_{probe["probe_id"]}'
        if probe_key in spectral_data:
            color = probe_colors_spectral[i % len(probe_colors_spectral)]
            ax_psd.loglog(spectral_data[probe_key]['f_psd_nd'][freq_start:],
                         spectral_data[probe_key]['psd'][freq_start:],
                         's-', linewidth=1.5, markersize=3,
                         label=f"u' Probe {i}", color=color, alpha=0.8)

    ax_psd.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]",
                      fontsize=12, fontweight='bold')
    ax_psd.set_ylabel("PSD [signal²/Hz]", fontsize=12, fontweight='bold')
    ax_psd.set_title("Power Spectral Density: τ' and u' (All Probes)",
                     fontsize=13, fontweight='bold')
    ax_psd.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax_psd.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax_psd.legend(fontsize=10, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    if OUTPUT_DIR is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig_psd.savefig(os.path.join(OUTPUT_DIR, "spectral_psd_overlay.png"),
                       dpi=150, bbox_inches='tight')
        print(f"  Saved: spectral_psd_overlay.png")
    plt.show()

    # ========================================================================
    # FIGURE 2: ALL COHERENCES OVERLAID (SEMILOG)
    # ========================================================================

    print(f"\n" + "="*70)
    print("GENERATING FIGURE 2: COHERENCE OVERLAY (SEMILOG)")
    print("="*70)

    fig_coh, ax_coh = plt.subplots(figsize=(12, 8))

    for i, probe in enumerate(probe_info):
        probe_key = f'probe_{probe["probe_id"]}'
        if probe_key in spectral_data:
            color = probe_colors_spectral[i % len(probe_colors_spectral)]
            ax_coh.semilogx(spectral_data[probe_key]['f_coh_nd'][freq_start:],
                           spectral_data[probe_key]['coherence'][freq_start:],
                           'o-', linewidth=1.5, markersize=3,
                           label=f"γ²(τ', u' probe {i})", color=color, alpha=0.8)

    ax_coh.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]",
                      fontsize=12, fontweight='bold')
    ax_coh.set_ylabel("Magnitude-Squared Coherence γ² [-]", fontsize=12, fontweight='bold')
    ax_coh.set_title("Coherence: τ' vs u' (All Probes)", fontsize=13, fontweight='bold')
    ax_coh.set_ylim([0, 1.05])
    ax_coh.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax_coh.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax_coh.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_coh.legend(fontsize=10, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    if OUTPUT_DIR is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig_coh.savefig(os.path.join(OUTPUT_DIR, "spectral_coherence_overlay.png"),
                       dpi=150, bbox_inches='tight')
        print(f"  Saved: spectral_coherence_overlay.png")
    plt.show()

    # ========================================================================
    # FIGURE 3: ALL CROSS-SPECTRUM MAGNITUDES OVERLAID (LOG-LOG)
    # ========================================================================

    print(f"\n" + "="*70)
    print("GENERATING FIGURE 3: CROSS-SPECTRUM MAGNITUDE OVERLAY (LOG-LOG)")
    print("="*70)

    fig_csd, ax_csd = plt.subplots(figsize=(12, 8))

    for i, probe in enumerate(probe_info):
        probe_key = f'probe_{probe["probe_id"]}'
        if probe_key in spectral_data:
            color = probe_colors_spectral[i % len(probe_colors_spectral)]
            cross_spec_mag = np.abs(spectral_data[probe_key]['cross_spectrum'])
            ax_csd.loglog(spectral_data[probe_key]['f_csd_nd'][freq_start:],
                         cross_spec_mag[freq_start:],
                         'o-', linewidth=1.5, markersize=3,
                         label=f"|S_τu|(probe {i})", color=color, alpha=0.8)

    ax_csd.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]",
                      fontsize=12, fontweight='bold')
    ax_csd.set_ylabel("|S_τu(f)| [signal product / Hz]", fontsize=12, fontweight='bold')
    ax_csd.set_title("Cross-Spectrum Magnitude: τ' vs u' (All Probes)",
                     fontsize=13, fontweight='bold')
    ax_csd.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax_csd.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax_csd.legend(fontsize=10, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    if OUTPUT_DIR is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig_csd.savefig(os.path.join(OUTPUT_DIR, "spectral_crossspectrum_overlay.png"),
                       dpi=150, bbox_inches='tight')
        print(f"  Saved: spectral_crossspectrum_overlay.png")
    plt.show()

    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================

    print(f"\n" + "="*70)
    print("SPECTRAL ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nSampling frequency: {fs:.2f} Hz")
    print(f"Total time span: {ref_time[-1] - ref_time[0]:.2f} s")
    print(f"Number of samples: {len(ref_time)}")
    print(f"Welch parameters: nperseg={NPERSEG}, noverlap={NOVERLAP}, window={WINDOW}")
    print(f"Frequency resolution: {fs/NPERSEG:.6e} Hz")
    print(f"Nondimensionalization: f* = f × {c} / {u_infty}")
    print(f"\nGenerated 3 figures:")
    print(f"  1. PSD overlay (all 5 signals)")
    print(f"  2. Coherence overlay (τ' vs each probe)")
    print(f"  3. Cross-spectrum magnitude overlay (τ' vs each probe)")
    print("="*70)

else:
    print("\n⚠ Insufficient spectral data for analysis")


print("DONE")
print("=" * 70)
