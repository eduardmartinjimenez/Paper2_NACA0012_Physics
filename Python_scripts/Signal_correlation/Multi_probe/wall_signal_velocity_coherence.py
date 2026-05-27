"""
Wall Signal - Streamwise Velocity Coherence Maps
=================================================

Generates 2D coherence maps for both wall shear stress (τ_w) and wall
pressure (p_w) with streamwise velocity fluctuation u_s at multiple
wall-normal probe locations.

Methodology:
1. Extract τ_w'(t,z) and p_w'(t,z) and u_s'(t,z,n_j) across full spanwise domain
2. Compute Welch spectra for each z-plane
3. Average spectra over spanwise direction
4. Compute coherence γ²(St_c, n/c) from z-averaged spectra
5. Assemble 2D coherence matrices (probe height vs frequency) for both signals
6. Generate filled contour maps with log-scale frequency axis

Output:
- coherence_tau_w_us_AOA..._xc_....png/pdf
- coherence_p_w_us_AOA..._xc_....png/pdf
- coherence_tau_w_us_yplus_AOA..._xc_....png/pdf
- coherence_p_w_us_yplus_AOA..._xc_....png/pdf
"""

import os
import sys
import re
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

# LaTeX style
plt.rc('text', usetex=True)
plt.rc('font', size=16, family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

# ============================================================================
# CONFIGURATION
# ============================================================================

# AoA 12º
AOA_deg = 12.0           # Angle of attack [degrees]

GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"

CACHE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Coherence/"

# Define wall-normal probe targets
n_probe_targets = np.linspace(0, 0.2, 1000)

# # AoA 5º
# AOA_deg = 5.0           # Angle of attack [degrees]

# GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
# GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
# GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_9/"
# MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
# MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_9/"

# CACHE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Coherence/"

# # Define wall-normal probe targets
# n_probe_targets = np.linspace(0, 0.1, 1000)

# Physical parameters
rho_ref = 1.0           # Reference density [kg/m³]
u_infty = 1.0           # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord [m]
Re_c = 50000            # Reynolds number
AOA_rad = np.radians(AOA_deg)

dt_iteration = 2.0e-06  # Physical time per iteration [s]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity
nu_ref = mu_ref / rho_ref              # Kinematic viscosity

# Spectral analysis parameters (Welch's method)
NPERSEG = 4096
NOVERLAP = NPERSEG // 2
WINDOW = 'hann'
DETREND_TYPE = 'linear'

# Output directory
OUTPUT_DIR = None

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
    """Compute wall shear stress for all z positions at once."""
    tangent = np.array([normal_at_point[1], -normal_at_point[0], 0.0])
    tangent = tangent / np.linalg.norm(tangent)

    u_vals = u_data[:, y_idx, 0]
    v_vals = v_data[:, y_idx, 0]
    w_vals = w_data[:, y_idx, 0]

    u_t_vals = u_vals * tangent[0] + v_vals * tangent[1] + w_vals * tangent[2]
    tau_w = mu_ref * u_t_vals / distance_at_point

    return tau_w


def compute_pressure_all_z(p_data, y_idx):
    """Extract pressure at surface for all z positions."""
    return p_data[:, y_idx, 0]


def preprocess_signal_for_welch(x, detrend_type='linear'):
    """Preprocess signal for Welch/CSD computations."""
    x_arr = np.asarray(x, dtype=float)
    valid_idx = ~np.isnan(x_arr)
    x_clean = x_arr[valid_idx]

    if detrend_type == 'linear':
        x_preprocessed = signal.detrend(x_clean, type='linear')
    elif detrend_type == 'constant':
        x_preprocessed = x_clean - np.mean(x_clean)
    elif detrend_type is None:
        x_preprocessed = x_clean
    else:
        raise ValueError(f"Unsupported detrend_type: {detrend_type}")

    return x_preprocessed


def compute_spectra_welch(signal1, signal2, fs, window='hann', nperseg=None,
                         noverlap=None, detrend='linear'):
    """Compute autospectra and complex cross-spectrum using Welch's method."""
    signal1_arr = np.asarray(signal1, dtype=float)
    signal2_arr = np.asarray(signal2, dtype=float)

    valid_mask = ~(np.isnan(signal1_arr) | np.isnan(signal2_arr))
    signal1_masked = signal1_arr[valid_mask]
    signal2_masked = signal2_arr[valid_mask]

    signal1_preprocessed = preprocess_signal_for_welch(signal1_masked, detrend_type=detrend)
    signal2_preprocessed = preprocess_signal_for_welch(signal2_masked, detrend_type=detrend)

    f, S_11 = signal.welch(signal1_preprocessed, fs=fs, window=window,
                           nperseg=nperseg, noverlap=noverlap, scaling='density')

    _, S_22 = signal.welch(signal2_preprocessed, fs=fs, window=window,
                           nperseg=nperseg, noverlap=noverlap, scaling='density')

    _, S_12 = signal.csd(signal1_preprocessed, signal2_preprocessed, fs=fs,
                        window=window, nperseg=nperseg, noverlap=noverlap,
                        scaling='density')

    return f, S_11, S_22, S_12


def nondimensionalize_frequency(frequency_array, U_inf, c):
    """Convert dimensional frequency to nondimensional: f* = f * c / U_inf"""
    return frequency_array * c / U_inf


# ============================================================================
# LOAD GEOMETRY AND MESH
# ============================================================================

print("="*70)
print("LOAD GEOMETRY AND MESH")
print("="*70)

if not os.path.exists(GEO_FILE):
    raise FileNotFoundError(f"Geometric data file not found: {GEO_FILE}")
if not os.path.exists(MESH_SLICE_FILE):
    raise FileNotFoundError(f"Mesh slice file not found: {MESH_SLICE_FILE}")

with h5py.File(GEO_FILE, 'r') as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

suction_side_points = interface_points[interface_points[:, 1] >= 0]
suction_side_indices = np.where(interface_points[:, 1] >= 0)[0]

loader = CompressedSnapshotLoader(MESH_SLICE_FILE)
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print(f"✓ Mesh loaded: shape (nz, ny, nx) = {x_data.shape}")
nz, ny, nx = x_data.shape

y_unique = np.unique(y_data[:, :, 0][0, :])
z_unique = np.unique(z_data[:, 0, 0])
print(f"  Y-grid: {len(y_unique)} points")
print(f"  Z-grid: {len(z_unique)} points (nz={nz})")

# Determine reference x/c position from slice location
XC_REF_TARGET = np.mean(x_data)
print(f"✓ Slice x/c position: {XC_REF_TARGET:.6f}")


print(f"✓ Probe distribution: {len(n_probe_targets)} probes, logarithmically spaced")

# ============================================================================
# FIND SURFACE REFERENCE POINT AND WALL-NORMAL PROBES
# ============================================================================

print("\n" + "="*70)
print("FIND SURFACE REFERENCE POINT AND PROBE LOCATIONS")
print("="*70)

x_distances = np.abs(suction_side_points[:, 0] - XC_REF_TARGET)
closest_idx = np.argmin(x_distances)
closest_surface_point = suction_side_points[closest_idx]
closest_interface_idx = suction_side_indices[closest_idx]

xc_ref_actual = closest_surface_point[0]
y_wall = closest_surface_point[1]

surface_normal = proj_normals[closest_interface_idx]
surface_distance = proj_distances[closest_interface_idx]

print(f"Selected x/c: {xc_ref_actual:.6f}")
print(f"Surface point: y/c = {y_wall:.6f}")
print(f"Wall distance: {surface_distance:.6e}")

# Find closest mesh y-index to surface point
y_wall_idx = np.argmin(np.abs(y_unique - y_wall))
y_wall_actual = y_unique[y_wall_idx]
print(f"Surface y (actual mesh): {y_wall_actual:.6f}")

# ============================================================================
# CREATE WALL-NORMAL PROBE DISTRIBUTION
# ============================================================================

print("\n" + "="*70)
print("CREATE WALL-NORMAL PROBE DISTRIBUTION")
print("="*70)

print(f"Requested wall-normal distances: {len(n_probe_targets)} probes")
print(f"  Range: {n_probe_targets[0]:.6f} to {n_probe_targets[-1]:.6f}")

# Calculate target y-coordinates
y_probe_targets = y_wall_actual + n_probe_targets

# Find closest mesh indices, remove duplicates
probe_y_indices_raw = np.array([np.argmin(np.abs(y_unique - y_t)) for y_t in y_probe_targets])
unique_indices, unique_inverse = np.unique(probe_y_indices_raw, return_inverse=True)

# Build probe info
probe_info = []
for u_idx in unique_indices:
    y_actual = y_unique[u_idx]
    n_actual = y_actual - y_wall_actual
    probe_info.append({
        'probe_id': len(probe_info),
        'y_target': y_probe_targets[probe_y_indices_raw == u_idx][0],
        'y_actual': y_actual,
        'y_idx': u_idx,
        'n_target': n_probe_targets[probe_y_indices_raw == u_idx][0],
        'n_actual': n_actual
    })

probe_info = sorted(probe_info, key=lambda x: x['n_actual'])

print(f"Unique mesh probes retained: {len(probe_info)}")
print(f"Wall-normal distance range: {min(p['n_actual'] for p in probe_info):.6f} to {max(p['n_actual'] for p in probe_info):.6f}")

for i, p in enumerate(probe_info[:5]):
    print(f"  Probe {i}: n/c={p['n_actual']:.6f}, y/c={p['y_actual']:.6f}")
if len(probe_info) > 5:
    print(f"  ... ({len(probe_info)-5} more probes)")

# ============================================================================
# VISUALIZATION: PROBE LOCATIONS AND AIRFOIL SURFACE
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION: PROBE LAYOUT WITH AIRFOIL GEOMETRY")
print("="*70)

fig, ax = plt.subplots(figsize=(14, 8))

# Plot airfoil surface (suction side)
suction_x = suction_side_points[:, 0]
suction_y = suction_side_points[:, 1]
ax.scatter(suction_x, suction_y, c='k', s=20, linewidth=2.0, label='Airfoil surface (suction)',
        zorder=5)

# Mark reference vertical line
ax.axvline(x=xc_ref_actual, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
          label=f'Reference x/c={xc_ref_actual:.4f}')

# Mark surface reference point
ax.scatter(xc_ref_actual, y_wall_actual, c='orange', s=400, marker='*',
          edgecolors='black', linewidths=2.5, zorder=15,
          label=f'Surface ref (y/c={y_wall_actual:.5f})')

# Plot probe locations with gradient coloring by height
probe_colors = plt.cm.cool(np.linspace(0, 1, len(probe_info)))

for i, probe in enumerate(probe_info):
    color = probe_colors[i]
    ax.scatter(xc_ref_actual, probe['y_actual'], c=[color], s=150, marker='o',
              edgecolors='black', linewidths=1.0, zorder=10)

# Add probe info legend (sample)
sample_indices = [0, len(probe_info)//4, len(probe_info)//2, 3*len(probe_info)//4, -1]
sample_indices = [i for i in sample_indices if 0 <= i < len(probe_info)]

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orange', edgecolor='black', label='Surface ref'),
    Patch(facecolor='k', label='Airfoil surface'),
]

for idx in sample_indices:
    probe = probe_info[idx]
    color = probe_colors[idx]
    legend_elements.append(
        Patch(facecolor=color, edgecolor='black',
              label=f'P{idx}: n/c={probe["n_actual"]:.5f}')
    )

ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.01, max(probe_info[-1]['y_actual'], y_wall_actual) * 1.15)
ax.set_xlabel('x/c', fontsize=13, fontweight='bold')
ax.set_ylabel('y/c', fontsize=13, fontweight='bold')
ax.set_title(
    rf'Probe Layout: {len(probe_info)} probes, $x/c={xc_ref_actual:.3f}$, AoA={AOA_deg:.0f}$^\circ$',
    fontsize=13, fontweight='bold'
)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.2, which='both')

plt.tight_layout()
plt.show()

print(f"✓ Probe layout visualization complete")
print(f"  Surface reference: x/c={xc_ref_actual:.6f}, y/c={y_wall_actual:.6f}")
print(f"  Probe wall-normal range: n/c=[{min(p['n_actual'] for p in probe_info):.6f}, {max(p['n_actual'] for p in probe_info):.6f}]")

# ============================================================================
# GET DATA FILES
# ============================================================================

print("\n" + "="*70)
print("GET DATA FILES")
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
# PHASE 1: DEFINE CACHE
# ============================================================================

print("\n" + "="*70)
print("DEFINE CACHE")
print("="*70)

cache_dir = CACHE_PATH if CACHE_PATH is not None else os.path.dirname(GEO_FILE)
os.makedirs(cache_dir, exist_ok=True)

cache_file = os.path.join(cache_dir, f"timeseries_both_xc_{XC_REF_TARGET:.3f}.h5")
print(f"Cache file: {cache_file}")

def load_cache_metadata(cache_file):
    """Load metadata from HDF5 cache."""
    try:
        with h5py.File(cache_file, 'r') as f:
            if '_metadata' not in f:
                return None
            meta_group = f['_metadata']
            metadata = {}
            for key in meta_group.attrs:
                metadata[key] = meta_group.attrs[key]
            for key in meta_group.keys():
                metadata[key] = meta_group[key][...]
            return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def validate_cache(cache_metadata, Nt, nz, probe_y_actual, xc_actual):
    """Validate cache metadata."""
    if cache_metadata is None:
        return False

    messages = []
    is_valid = True

    if 'Nt' in cache_metadata and cache_metadata['Nt'] != Nt:
        messages.append(f"❌ Nt mismatch: {cache_metadata['Nt']} vs {Nt}")
        is_valid = False

    if 'nz' in cache_metadata and cache_metadata['nz'] != nz:
        messages.append(f"❌ nz mismatch: {cache_metadata['nz']} vs {nz}")
        is_valid = False

    if 'num_probes' in cache_metadata and cache_metadata['num_probes'] != len(probe_y_actual):
        messages.append(f"❌ num_probes mismatch")
        is_valid = False

    if 'probe_y_actual' in cache_metadata:
        if not np.allclose(cache_metadata['probe_y_actual'], probe_y_actual, rtol=1e-5):
            messages.append(f"❌ probe_y_actual mismatch")
            is_valid = False

    if 'xc_actual' in cache_metadata:
        if not np.isclose(cache_metadata['xc_actual'], xc_actual, rtol=1e-5):
            messages.append(f"❌ xc_actual mismatch")
            is_valid = False

    for msg in messages:
        print(f"  {msg}")

    return is_valid


# ============================================================================
# PHASE 2: EXTRACT TIME SERIES OR LOAD FROM CACHE
# ============================================================================

print("\n" + "="*70)
print("EXTRACT TIME SERIES")
print("="*70)

timeseries_data = None
cache_metadata = load_cache_metadata(cache_file) if os.path.exists(cache_file) else None

if cache_metadata is not None:
    probe_y_actual = np.array([p['y_actual'] for p in probe_info])
    is_valid = validate_cache(cache_metadata, len(data_files), nz, probe_y_actual, xc_ref_actual)
    if is_valid:
        print("Cache validation passed. Loading time series...")
        try:
            with h5py.File(cache_file, 'r') as f:
                timeseries_data = {
                    'iterations': f['iterations'][...],
                    'time': f['time'][...],
                    'tau_w_z': f['tau_w_z'][...],
                    'tau_w_prime': f['tau_w_prime'][...],
                    'pressure_z': f['pressure_z'][...],
                    'pressure_prime': f['pressure_prime'][...],
                }
                for i in range(len(probe_info)):
                    timeseries_data[f'u_s_z_{i}'] = f[f'u_s_z_{i}'][...]
                    timeseries_data[f'u_s_prime_{i}'] = f[f'u_s_prime_{i}'][...]
            print(f"✓ Loaded time series from cache")
        except Exception as e:
            print(f"Error loading cache: {e}")
            timeseries_data = None

if timeseries_data is None:
    print(f"Extracting time series from {len(data_files)} snapshots...")

    timeseries_data = {
        'iterations': [],
        'tau_w_z': [],
        'pressure_z': [],
    }

    for i in range(len(probe_info)):
        timeseries_data[f'u_s_z_{i}'] = []

    cos_aoa = np.cos(AOA_rad)
    sin_aoa = np.sin(AOA_rad)

    for file_idx, data_file in enumerate(data_files):
        if (file_idx + 1) % max(1, len(data_files) // 5) == 0 or file_idx == 0:
            print(f"  Progress: {file_idx + 1}/{len(data_files)}")

        match = re.search(r'_(\d+)-COMP-DATA', data_file)
        iteration = int(match.group(1)) if match else file_idx

        try:
            snapshot = loader.load_snapshot(data_file)
            u_data = loader.reconstruct_field(snapshot["u"])[1:-1, :, :]
            v_data = loader.reconstruct_field(snapshot["v"])[1:-1, :, :]
            w_data = loader.reconstruct_field(snapshot["w"])[1:-1, :, :]
            p_data = loader.reconstruct_field(snapshot["p"])[1:-1, :, :]

            # Compute both wall signals
            tau_w_vals = compute_tau_w_all_z(u_data, v_data, w_data,
                                            y_wall_idx, mu_ref,
                                            surface_normal, surface_distance)
            pressure_vals = compute_pressure_all_z(p_data, y_wall_idx)

            timeseries_data['iterations'].append(iteration)
            timeseries_data['tau_w_z'].append(tau_w_vals)
            timeseries_data['pressure_z'].append(pressure_vals)

            # Streamwise velocity at each probe
            for i, probe in enumerate(probe_info):
                probe_y_idx = probe['y_idx']
                u_vals = u_data[:, probe_y_idx, 0]
                v_vals = v_data[:, probe_y_idx, 0]
                u_s_vals = u_vals * cos_aoa + v_vals * sin_aoa
                timeseries_data[f'u_s_z_{i}'].append(u_s_vals)

        except Exception as e:
            print(f"  ⚠ Error loading snapshot {file_idx}: {e}")
            continue

    # Convert to arrays
    print("Converting to arrays and computing fluctuations...")
    iterations = np.array(timeseries_data['iterations'])
    timeseries_data['time'] = iterations * dt_iteration
    timeseries_data['tau_w_z'] = np.array(timeseries_data['tau_w_z'])
    timeseries_data['pressure_z'] = np.array(timeseries_data['pressure_z'])

    for i in range(len(probe_info)):
        timeseries_data[f'u_s_z_{i}'] = np.array(timeseries_data[f'u_s_z_{i}'])

    Nt = len(iterations)
    print(f"  Time series shape: (Nt={Nt}, Nz={nz})")

    # Compute fluctuations (remove mean at each z)
    tau_w_mean = np.mean(timeseries_data['tau_w_z'], axis=0)
    timeseries_data['tau_w_prime'] = timeseries_data['tau_w_z'] - tau_w_mean

    pressure_mean = np.mean(timeseries_data['pressure_z'], axis=0)
    timeseries_data['pressure_prime'] = timeseries_data['pressure_z'] - pressure_mean

    for i in range(len(probe_info)):
        u_mean = np.mean(timeseries_data[f'u_s_z_{i}'], axis=0)
        timeseries_data[f'u_s_prime_{i}'] = timeseries_data[f'u_s_z_{i}'] - u_mean

    # Save cache with metadata
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('iterations', data=timeseries_data['iterations'])
        f.create_dataset('time', data=timeseries_data['time'])
        f.create_dataset('tau_w_z', data=timeseries_data['tau_w_z'])
        f.create_dataset('tau_w_prime', data=timeseries_data['tau_w_prime'])
        f.create_dataset('pressure_z', data=timeseries_data['pressure_z'])
        f.create_dataset('pressure_prime', data=timeseries_data['pressure_prime'])

        for i in range(len(probe_info)):
            f.create_dataset(f'u_s_z_{i}', data=timeseries_data[f'u_s_z_{i}'])
            f.create_dataset(f'u_s_prime_{i}', data=timeseries_data[f'u_s_prime_{i}'])

        meta_group = f.create_group('_metadata')
        meta_group.attrs['Nt'] = Nt
        meta_group.attrs['nz'] = nz
        meta_group.attrs['num_probes'] = len(probe_info)
        meta_group.attrs['AOA_deg'] = AOA_deg
        meta_group.attrs['xc_actual'] = xc_ref_actual
        meta_group.attrs['y_wall_actual'] = y_wall_actual
        meta_group.attrs['dt_iteration'] = dt_iteration
        meta_group.attrs['Re_c'] = Re_c

        probe_y_actual = np.array([p['y_actual'] for p in probe_info])
        probe_n_actual = np.array([p['n_actual'] for p in probe_info])
        meta_group.create_dataset('probe_y_actual', data=probe_y_actual)
        meta_group.create_dataset('probe_n_actual', data=probe_n_actual)

    print(f"✓ Time series cached to: {cache_file}")

# ============================================================================
# COMPUTE SAMPLING FREQUENCY
# ============================================================================

print("\n" + "="*70)
print("COMPUTE SAMPLING FREQUENCY")
print("="*70)

time = timeseries_data['time']
if isinstance(time, list):
    time = np.array(time)
dt_mean = np.mean(np.diff(time))
fs = 1.0 / dt_mean
Nt = len(time)

print(f"Sampling frequency: {fs:.2f} Hz")
print(f"Total time span: {time[-1] - time[0]:.2f} s")
print(f"Number of samples: {Nt}")

# Adjust Welch parameters if necessary
nperseg = NPERSEG
noverlap = NOVERLAP
if nperseg > Nt:
    nperseg = Nt // 2
    noverlap = nperseg // 2
    print(f"⚠ nperseg adjusted to {nperseg}")

print(f"Welch: nperseg={nperseg}, noverlap={noverlap}, window={WINDOW}")
print(f"Frequency resolution: {fs / nperseg:.6e} Hz")
print(f"Number of Welch segments: {(Nt - nperseg) // (nperseg - noverlap) + 1}")

# ============================================================================
# COMPUTE WELCH SPECTRA AND COHERENCE
# ============================================================================

print("\n" + "="*70)
print("COMPUTE WELCH SPECTRA AND COHERENCE FOR BOTH WALL SIGNALS")
print("="*70)

# Dictionary to store results for each wall signal
results = {}

for wall_signal_type in ["tau_w", "pressure"]:
    print(f"\nProcessing wall signal: {wall_signal_type}")
    print(f"Processing {len(probe_info)} probes...")

    coherence_2d = np.zeros((len(probe_info), len(np.fft.rfftfreq(nperseg, 1/fs))))
    phase_2d = np.zeros_like(coherence_2d)
    frequency = None

    for i, probe in enumerate(probe_info):
        if (i + 1) % max(1, len(probe_info) // 5) == 0 or i == 0:
            print(f"  Probe {i + 1}/{len(probe_info)}")

        # Collect spectra for all z-planes
        S_qq_all_z = []
        S_uu_all_z = []
        S_qu_all_z = []

        for iz in range(nz):
            # Select wall signal based on type
            if wall_signal_type == "tau_w":
                q_z = timeseries_data['tau_w_prime'][:, iz]
            elif wall_signal_type == "pressure":
                q_z = timeseries_data['pressure_prime'][:, iz]
            else:
                raise ValueError(f"Unknown wall signal: {wall_signal_type}")

            u_z = timeseries_data[f'u_s_prime_{i}'][:, iz]

            f, S_qq_z, S_uu_z, S_qu_z = compute_spectra_welch(
                q_z, u_z, fs, window=WINDOW, nperseg=nperseg,
                noverlap=noverlap, detrend=DETREND_TYPE
            )

            S_qq_all_z.append(S_qq_z)
            S_uu_all_z.append(S_uu_z)
            S_qu_all_z.append(S_qu_z)

            if frequency is None:
                frequency = f

        # Average spectra over z
        S_qq_all_z = np.array(S_qq_all_z)
        S_uu_all_z = np.array(S_uu_all_z)
        S_qu_all_z = np.array(S_qu_all_z)

        S_qq_mean = np.mean(S_qq_all_z, axis=0)
        S_uu_mean = np.mean(S_uu_all_z, axis=0)
        S_qu_mean = np.mean(S_qu_all_z, axis=0)

        # Compute coherence from averaged spectra
        coherence_i = np.abs(S_qu_mean)**2 / (S_qq_mean * S_uu_mean + 1e-30)
        coherence_i = np.clip(coherence_i, 0.0, 1.0)

        # Compute phase
        phase_i = np.angle(S_qu_mean)

        # Store in 2D arrays
        coherence_2d[i, :] = coherence_i
        phase_2d[i, :] = phase_i

    print(f"✓ Coherence computed for {len(probe_info)} probes")
    print(f"  Frequency bins: {len(frequency)}")
    print(f"  Coherence range: [{np.min(coherence_2d):.4f}, {np.max(coherence_2d):.4f}]")

    # Store results
    results[wall_signal_type] = {
        'coherence_2d': coherence_2d,
        'phase_2d': phase_2d,
        'frequency': frequency
    }

# ============================================================================
# CONVERT TO NONDIMENSIONAL QUANTITIES
# ============================================================================

print("\n" + "="*70)
print("CONVERT TO NONDIMENSIONAL QUANTITIES")
print("="*70)

St_c = nondimensionalize_frequency(frequency, u_infty, c)
n_over_c = np.array([p['n_actual'] for p in probe_info])
tau_w_mean_ref = float(np.mean(timeseries_data['tau_w_z']))
u_tau_ref = np.sqrt(np.abs(tau_w_mean_ref) / rho_ref)
y_plus = n_over_c * u_tau_ref / nu_ref

print(f"St_c range: [{St_c[1]:.6e}, {St_c[-1]:.6e}]")
print(f"n/c range: [{n_over_c[0]:.6f}, {n_over_c[-1]:.6f}]")
print(f"Reference wall-shear mean: {tau_w_mean_ref:.6e} Pa")
print(f"Reference friction velocity: {u_tau_ref:.6e} m/s")
print(f"y+ range: [{y_plus[0]:.6f}, {y_plus[-1]:.6f}]")

# ============================================================================
# GENERATE 2D COHERENCE MAP (CONTOUR PLOT)
# ============================================================================

print("\n" + "="*70)
print("GENERATE 2D COHERENCE MAPS")
print("="*70)

if OUTPUT_DIR is None:
    OUTPUT_DIR = os.path.dirname(cache_file)

os.makedirs(OUTPUT_DIR, exist_ok=True)

freq_start = 1  # Remove zero frequency

for wall_signal_type, result_data in results.items():
    coherence_2d = result_data['coherence_2d']
    frequency = result_data['frequency']

    X, Y = np.meshgrid(St_c[freq_start:], n_over_c)
    Z = coherence_2d[:, freq_start:]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Filled contours - use data min/max for better contrast
    vmin_data = np.min(Z)
    vmax_data = np.max(Z)
    levels = np.linspace(vmin_data, vmax_data, 21)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='YlOrRd', vmin=vmin_data, vmax=vmax_data)


    # Black contour lines at key coherence levels
    # contour = ax.contour(X, Y, Z, levels=[0.05, 0.10, 0.20],
    #                      colors='black', linewidths=1.0, alpha=0.5)

    # ax.clabel(contour, inline=True, fontsize=9, fmt='%.2f')

    # Colorbar
    if wall_signal_type == "tau_w":
        colorbar_label = r'$\gamma^2_{\tau_w u_s}$'
    else:
        colorbar_label = r'$\gamma^2_{p_w u_s}$'
    cbar = plt.colorbar(contourf, ax=ax, label=colorbar_label)

    ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$n/c$', fontsize=12, fontweight='bold')
    ax.set_xscale('log')

    if wall_signal_type == "tau_w":
        title_signal = r"$\tau_w^\prime$"
    else:
        title_signal = r"$p_w^\prime$"

    ax.set_title(
        rf'Spanwise-averaged coherence: {title_signal} vs $u_s^\prime$, '
        rf'$x/c={xc_ref_actual:.3f}$, AoA={AOA_deg:.0f}$^\circ$',
        fontsize=13, fontweight='bold'
    )

    # Info text box
    n_segments = (Nt - nperseg) // (nperseg - noverlap) + 1
    info_text = (
        f"$N_t$ = {Nt}\n"
        f"$N_z$ = {nz}\n"
        f"$f_s$ = {fs:.1f} Hz\n"
        f"$n_{{seg}}$ = {nperseg}\n"
        f"$n_{{ovlp}}$ = {noverlap}\n"
        f"$N_{{Welch}}$ = {n_segments}\n"
        f"$\Delta f$ = {fs/nperseg:.3e} Hz"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.grid(True, alpha=0.2, which='both')
    plt.tight_layout()

    # Save with appropriate filename
    png_file = os.path.join(OUTPUT_DIR, f"coherence_{wall_signal_type}_us_AOA{AOA_deg:.0f}_xc_{XC_REF_TARGET:.3f}.png")
    eps_file = os.path.join(OUTPUT_DIR, f"coherence_{wall_signal_type}_us_AOA{AOA_deg:.0f}_xc_{XC_REF_TARGET:.3f}.eps")

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(eps_file, bbox_inches='tight')
    print(f"✓ Figure saved for {wall_signal_type}:")
    print(f"  PNG: {png_file}")
    print(f"  EPS: {eps_file}")


# ============================================================================
# GENERATE 2D COHERENCE MAPS IN WALL UNITS
# ============================================================================

print("\n" + "="*70)
print("GENERATE 2D COHERENCE MAPS IN WALL UNITS")
print("="*70)

for wall_signal_type, result_data in results.items():
    coherence_2d = result_data['coherence_2d']

    positive_y_mask = y_plus > 0
    y_plus_plot = y_plus[positive_y_mask]
    Z = coherence_2d[positive_y_mask, freq_start:]
    X, Y = np.meshgrid(St_c[freq_start:], y_plus_plot)

    fig, ax = plt.subplots(figsize=(12, 8))

    vmin_data = np.min(Z)
    vmax_data = np.max(Z)
    levels = np.linspace(vmin_data, vmax_data, 21)
    #contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', vmin=vmin_data, vmax=vmax_data)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='YlOrRd', vmin=vmin_data, vmax=vmax_data)


    # contour = ax.contour(X, Y, Z, levels=[0.05, 0.10, 0.20],
    #                      colors='black', linewidths=1.0, alpha=0.5)
    # ax.clabel(contour, inline=True, fontsize=9, fmt='%.2f')

    if wall_signal_type == "tau_w":
        colorbar_label = r'$\gamma^2_{\tau_w u_s}$'
        title_signal = r"$\tau_w^\prime$"
    else:
        colorbar_label = r'$\gamma^2_{p_w u_s}$'
        title_signal = r"$p_w^\prime$"

    plt.colorbar(contourf, ax=ax, label=colorbar_label)

    ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$y^+$', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    y_plus_to_y_over_c = lambda y_plus_val: y_plus_val * nu_ref / (u_tau_ref * c)
    y_over_c_to_y_plus = lambda y_over_c_val: y_over_c_val * u_tau_ref * c / nu_ref
    secax = ax.secondary_yaxis('right', functions=(y_plus_to_y_over_c, y_over_c_to_y_plus))
    secax.set_ylabel(r'$y/c$', fontsize=12, fontweight='bold')
    secax.set_yscale('log')
    ax.set_title(
        rf'Spanwise-averaged coherence in wall units: {title_signal} vs $u_s^\prime$, '
        rf'$x/c={xc_ref_actual:.3f}$, AoA={AOA_deg:.0f}$^\circ$',
        fontsize=13, fontweight='bold'
    )

    ax.set_ylim(min(y_plus_plot), max(y_plus_plot))

    n_segments = (Nt - nperseg) // (nperseg - noverlap) + 1
    info_text = (
        f"$N_t$ = {Nt}\n"
        f"$N_z$ = {nz}\n"
        f"$f_s$ = {fs:.1f} Hz\n"
        f"$u_\tau$ = {u_tau_ref:.3e} m/s\n"
        f"$y^+$ = [{y_plus_plot[0]:.3f}, {y_plus_plot[-1]:.3f}]\n"
        f"$n_{{seg}}$ = {nperseg}\n"
        f"$n_{{ovlp}}$ = {noverlap}\n"
        f"$N_{{Welch}}$ = {n_segments}\n"
        f"$\Delta f$ = {fs/nperseg:.3e} Hz"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.grid(True, alpha=0.2, which='both')
    plt.tight_layout()

    png_file = os.path.join(OUTPUT_DIR, f"coherence_{wall_signal_type}_us_yplus_AOA{AOA_deg:.0f}_xc_{XC_REF_TARGET:.3f}.png")
    eps_file = os.path.join(OUTPUT_DIR, f"coherence_{wall_signal_type}_us_yplus_AOA{AOA_deg:.0f}_xc_{XC_REF_TARGET:.3f}.eps")

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(eps_file, bbox_inches='tight')
    print(f"✓ Wall-unit figure saved for {wall_signal_type}:")
    print(f"  PNG: {png_file}")
    print(f"  EPS: {eps_file}")

plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nDomain:")
print(f"  Temporal: Nt={Nt}, fs={fs:.2f} Hz")
print(f"  Spanwise: Nz={nz}")
print(f"  Probes: {len(probe_info)} at wall-normal locations")
print(f"\nReference:")
print(f"  x/c = {xc_ref_actual:.6f}")
print(f"  y/c = {y_wall_actual:.6f}")
print(f"  y+ = [{y_plus[0]:.6f}, {y_plus[-1]:.6f}]")
print(f"  AoA = {AOA_deg:.1f}°")
print(f"\nWelch parameters:")
print(f"  nperseg={nperseg}, noverlap={noverlap}, window={WINDOW}")
print(f"\nOutput:")
print(f"  Coherence maps: 4 figures total (2 in n/c, 2 in y+) × {len(probe_info)} probes × {len(frequency)} frequencies")
for signal_type in ["tau_w", "pressure"]:
    print(f"    - {signal_type}")
print("="*70)
