"""
Energy Spectra from Slice Data - Variable Stride Analysis
============================================================

Compute raw periodograms of streamwise and cross-stream velocity fluctuations
from slice snapshot data at physically-defined probe locations, with ability
to control temporal sampling frequency via a stride parameter.

This script processes multiple stride values (e.g., 1, 2, 4, 8) in a single
execution, allowing investigation of how spectral properties change with
reduced temporal data.

Key features:
- Variable stride parameter (integer): use every Nth slice
- Multi-stride analysis: process multiple strides in one run
- Comparison visualization: see how spectra change across strides
- Maintains all features from v2 (numeric sorting, variance validation, etc.)
"""

import os
import sys
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

### AOA 5º
# Slice data paths
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_9/"

# Geometric data (for visualization only)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# Output directory
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Energy_spectra/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Physical parameters
rho_ref = 1.0           # Reference density [kg/m³]
u_infty = 1.0           # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord length [m]
Re_c = 50000            # Reynolds number
AOA_deg = 5.0           # Angle of attack [degrees]
AOA_rad = np.radians(AOA_deg)

# Physical time step [CRITICAL - must match simulation]
dt_iteration = 2.0e-06  # Physical time per iteration [s]

# Probe locations: absolute y-coordinates in domain
Y_LOCATIONS = [0.03, 0.06, 0.09]  # Specify actual y-coordinate values

# STRIDE CONFIGURATION: Analyze multiple stride values
# stride=1 uses every slice, stride=2 uses every 2nd slice, etc.
SLICE_STRIDES = [1, 25]  # Modify this list as needed


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    """Check path exists and print confirmation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"{kind} exists: {path}")


def get_slice_files_sorted(slices_path: str) -> tuple:
    """
    Get snapshot slice files sorted numerically by iteration number (not lexicographic).

    Filenames expected: slice_1_15000000-COMP-DATA.h5, slice_1_15000800-COMP-DATA.h5, etc.
    Extracts iteration numbers, sorts numerically, validates spacing.

    Returns:
        tuple: (sorted_file_list, iterations_array, delta_iteration)

    Raises:
        FileNotFoundError: If no snapshot files found
        ValueError: If iteration number extraction fails or spacing is inconsistent
    """
    # Find all snapshot files
    snapshot_files = []
    for file in Path(slices_path).glob("slice_*-COMP-DATA.h5"):
        if "avg" not in file.name:  # Exclude average files
            snapshot_files.append(str(file))

    if not snapshot_files:
        raise FileNotFoundError(f"No snapshot files found in {slices_path}")

    print(f"\nFound {len(snapshot_files)} snapshot files")

    # Extract iteration numbers via regex
    iterations = []
    files_with_iter = []

    for filepath in snapshot_files:
        filename = os.path.basename(filepath)
        match = re.search(r'_(\d+)-COMP-DATA\.h5', filename)
        if not match:
            raise ValueError(f"Cannot extract iteration number from {filename}. "
                           f"Expected format: slice_*_NNNNNNNN-COMP-DATA.h5")
        iter_num = int(match.group(1))
        iterations.append(iter_num)
        files_with_iter.append((iter_num, filepath))

    # Sort by iteration number
    files_with_iter.sort(key=lambda x: x[0])
    sorted_files = [f[1] for f in files_with_iter]
    iterations = np.array([f[0] for f in files_with_iter])

    # Validate spacing
    if len(iterations) < 2:
        raise ValueError("Need at least 2 snapshot files to compute time step")

    delta_iterations = np.diff(iterations)
    unique_deltas = np.unique(delta_iterations)

    # Print iteration information
    print(f"\nIteration information:")
    print(f"  First 10 iterations: {iterations[:10]}")
    print(f"  Last 10 iterations: {iterations[-10:]}")
    print(f"  Total samples: {len(iterations)}")
    print(f"  Iteration spacing: {unique_deltas}")
    print(f"  Strictly monotonic: {np.all(delta_iterations > 0)}")

    # Check for consistent spacing
    if len(unique_deltas) > 1:
        raise ValueError(
            f"Iteration spacing is NOT constant: {unique_deltas}. "
            f"Cannot reliably compute time step. Aborting."
        )

    delta_iter = int(unique_deltas[0])
    print(f"  Constant delta_iteration: {delta_iter}")

    return sorted_files, iterations, delta_iter


def filter_files_by_stride(sorted_files: list, iterations: np.ndarray, stride: int) -> tuple:
    """
    Filter files and iterations by stride.

    Takes every stride-th file starting from index 0.

    Args:
        sorted_files: List of sorted file paths
        iterations: Array of iteration numbers
        stride: Stride value (1 = all files, 2 = every 2nd, etc.)

    Returns:
        tuple: (filtered_files, filtered_iterations, n_samples_with_stride)
    """
    filtered_files = sorted_files[::stride]
    filtered_iterations = iterations[::stride]
    n_samples_with_stride = len(filtered_files)

    print(f"\n  Stride {stride}:")
    print(f"    Files selected: {n_samples_with_stride} (from {len(sorted_files)} total)")
    print(f"    First 5 iterations: {filtered_iterations[:5]}")
    if n_samples_with_stride > 5:
        print(f"    Last 5 iterations: {filtered_iterations[-5:]}")

    return filtered_files, filtered_iterations, n_samples_with_stride


# ============================================================================
# LOAD GEOMETRY AND MESH (One-time load, shared across all strides)
# ============================================================================

print("="*70)
print("LOAD GEOMETRY AND MESH")
print("="*70)

assert_exists(GEO_FILE, "Geometrical data file")
assert_exists(MESH_SLICE_FILE, "Mesh slice file")

# Load geometrical data
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)

# Extract suction and pressure side surfaces
suction_side_points = interface_points[interface_points[:, 1] >= 0]
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
print(f"Slice structure verified: single x-plane (nx=1), ny={x_data.shape[1]}, nz={x_data.shape[0]}")
print(f"Slice x-coordinate: {slice_x:.6f}")

# Get spanwise parameters
z_unique = np.unique(z_data[:, 0, 0])
nz = z_unique.size
dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
L_z = dz * nz

print(f"Spanwise domain: nz={nz}, dz={dz:.6e} m, Lz={L_z:.6e} m")

# Build fluid region and interface
y_unique = np.unique(y_data[:, :, 0][0, :])
print(f"Total y-grid points: {len(y_unique)}, range: {y_unique[0]:.6e} to {y_unique[-1]:.6e}")

# Find interface y-coordinate (closest to airfoil at this slice x)
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

# Infer slice_id from SLICES_PATH
match = re.search(r'slice_(\d+)', SLICES_PATH)
if match:
    slice_id = f"slice_{match.group(1)}"
else:
    raise ValueError(f"Cannot infer slice_id from path: {SLICES_PATH}")

print(f"Inferred slice_id from path: {slice_id}")

# ============================================================================
# SELECT PROBE LOCATIONS
# ============================================================================

print("\n" + "="*70)
print("SELECT PROBE LOCATIONS")
print("="*70)

y_locations_idx = []
y_locations_val = []
y_locations_target = []

for i, y_target in enumerate(Y_LOCATIONS):
    # Find closest grid point in full domain
    idx_closest = np.argmin(np.abs(y_unique - y_target))
    j_idx = idx_closest
    actual_y = y_unique[j_idx]

    y_locations_idx.append(j_idx)
    y_locations_val.append(actual_y)
    y_locations_target.append(y_target)

    dist_error = np.abs(actual_y - y_target)
    print(f"Probe {i}: y_target={y_target:.6e} -> y_actual={actual_y:.6e}, "
          f"error={dist_error:.6e} (j_idx={j_idx})")

# ============================================================================
# VISUALIZE AIRFOIL AND PROBE LOCATIONS
# ============================================================================

print("\n" + "="*70)
print("VISUALIZE AIRFOIL AND PROBE LOCATIONS")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 7))

# Plot airfoil surfaces as scatter
ax.scatter(suction_side_points[:, 0], suction_side_points[:, 1],
          s=10, c='b', label='Suction side', zorder=3, alpha=0.6)
ax.scatter(pressure_side_points[:, 0], pressure_side_points[:, 1],
          s=10, c='r', label='Pressure side', zorder=3, alpha=0.6)

# Slice plane
ax.axvline(x=slice_x, color='green', linewidth=2.5, linestyle='--',
           label=f'Slice (x={slice_x:.6f})', zorder=2, alpha=0.8)

# Probe markers: separate requested (diamond) and actual (circle)
colors = plt.cm.viridis(np.linspace(0, 1, len(Y_LOCATIONS)))

for i, (y_target, y_actual, color) in enumerate(zip(Y_LOCATIONS, y_locations_val, colors)):
    # Requested location (diamond marker)
    ax.plot(slice_x, y_target, 'D', markersize=8, color=color, zorder=5,
            markeredgecolor='black', markeredgewidth=1.5, alpha=0.7,
            label=f'P{i}_req')

    # Actual grid location (circle marker)
    ax.plot(slice_x, y_actual, 'o', markersize=7, color=color, zorder=5,
            markeredgecolor='black', markeredgewidth=1, alpha=1.0,
            label=f'P{i}_act')

    # Connect if different
    if np.abs(y_target - y_actual) > 1e-6:
        ax.plot([slice_x, slice_x], [y_target, y_actual], '--',
                color=color, linewidth=1, alpha=0.5, zorder=4)

ax.set_xlabel('x (chord)', fontsize=12, fontweight='bold')
ax.set_ylabel('y (chord)', fontsize=12, fontweight='bold')
ax.set_title(f'Airfoil Surface and Probe Locations ({slice_id})\nAOA={AOA_deg}°, slice_x={slice_x:.6f}',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.set_aspect('equal')
ax.margins(0.05)

plt.tight_layout()
plt.show()

# ============================================================================
# MAIN STRIDE LOOP: Process each stride value
# ============================================================================

print("\n" + "="*70)
print("PROCESSING MULTIPLE STRIDES")
print("="*70)

# Storage for comparison across strides: {stride: {j_idx: energy_spectra_dict}}
all_stride_results = {}

for stride in SLICE_STRIDES:
    print(f"\n{'='*70}")
    print(f"STRIDE = {stride}")
    print(f"{'='*70}")

    # Filter files by stride
    stride_files, stride_iterations, n_samples_stride = filter_files_by_stride(
        slice_files, iter_numbers, stride
    )

    # Compute effective physical time step
    dt_save_effective = delta_iter * stride * dt_iteration
    fs_effective = 1.0 / dt_save_effective

    print(f"\nTime step computation for stride {stride}:")
    print(f"  dt_iteration: {dt_iteration:.6e} s/iteration")
    print(f"  delta_iter: {delta_iter} iterations")
    print(f"  stride: {stride}")
    print(f"  dt_save_effective: {dt_save_effective:.6e} s/snapshot ({dt_save_effective*1000:.6f} ms)")
    print(f"  Effective sampling frequency fs: {fs_effective:.6e} Hz")
    print(f"  Expected time steps: {n_samples_stride}")
    print(f"  Total time span: {(n_samples_stride-1)*dt_save_effective:.6e} s")

    # ========================================================================
    # EXTRACT TIME SERIES (ALL Z DATA)
    # ========================================================================

    print(f"\n{'─'*70}")
    print(f"EXTRACT TIME SERIES (stride {stride})")
    print(f"{'─'*70}")

    # Initialize storage: {j_idx: {'u': (n_times, n_z), 'v': (n_times, n_z)}}
    time_series_data = {j_idx: {'u': [], 'v': []} for j_idx in y_locations_idx}

    print(f"Extracting time series from {len(stride_files)} snapshots...")

    failed_files = []
    successful_loads = 0

    for t, slice_file in enumerate(stride_files):
        if not os.path.exists(slice_file):
            failed_files.append(slice_file)
            continue

        try:
            # Load snapshot
            fields = loader.load_snapshot(slice_file)
            u_data = loader.reconstruct_field(fields["u"])[1:-1, :, :]  # (nz, ny, nx)
            v_data = loader.reconstruct_field(fields["v"])[1:-1, :, :]
            w_data = loader.reconstruct_field(fields["w"])[1:-1, :, :]

            # Rotate to streamwise/cross-stream coordinates
            u_stream = u_data * np.cos(AOA_rad) + v_data * np.sin(AOA_rad)
            v_cross = -u_data * np.sin(AOA_rad) + v_data * np.cos(AOA_rad)

            # Extract at probe locations (all z indices)
            for j_idx in y_locations_idx:
                # Extract full z profile: (nz,)
                u_vec = u_stream[:, j_idx, 0]  # (nz,)
                v_vec = v_cross[:, j_idx, 0]

                time_series_data[j_idx]['u'].append(u_vec)
                time_series_data[j_idx]['v'].append(v_vec)

            successful_loads += 1

            if (t + 1) % max(1, len(stride_files) // 10) == 0:
                print(f"  Processed {t + 1}/{len(stride_files)} snapshots (successful: {successful_loads})")

        except Exception as e:
            failed_files.append((slice_file, str(e)))
            print(f"  ERROR at snapshot {t}: {str(e)[:100]}")
            continue

    print(f"\n  Loop completed. Total files processed: {len(stride_files)}")
    print(f"  Successful loads: {successful_loads}")
    print(f"  Failed loads: {len(failed_files)}")

    # Check for failures
    if failed_files:
        print(f"\n  Detailed failures:")
        for i, fail in enumerate(failed_files[:5]):
            if isinstance(fail, tuple):
                print(f"    {i+1}. {fail[0]}: {fail[1][:200]}")
            else:
                print(f"    {i+1}. {fail}")
        raise RuntimeError(f"Failed to load {len(failed_files)} files. Aborting. "
                          f"First error: {failed_files[0]}")

    # Verify all probes have consistent time samples
    n_samples_list = [len(time_series_data[j_idx]['u']) for j_idx in y_locations_idx]
    if len(set(n_samples_list)) > 1:
        raise ValueError(f"Inconsistent sample counts across probes: {set(n_samples_list)}")

    n_samples = n_samples_list[0]
    print(f"\nActual loaded samples: {n_samples} (matched across all {len(y_locations_idx)} probes)")
    print(f"  Sample counts per probe: {n_samples_list}")
    print(f"  Expected samples (stride={stride}): {len(stride_files)}")
    print(f"  Ratio: {n_samples}/{len(stride_files)} = {n_samples/len(stride_files):.4f}")

    # Convert lists to numpy arrays: (n_times, n_z)
    for j_idx in y_locations_idx:
        time_series_data[j_idx]['u'] = np.array(time_series_data[j_idx]['u'])
        time_series_data[j_idx]['v'] = np.array(time_series_data[j_idx]['v'])

        shape = time_series_data[j_idx]['u'].shape
        print(f"Probe at j={j_idx}, y={y_locations_val[y_locations_idx.index(j_idx)]:.6e}: "
              f"u shape = {shape}")

    # ========================================================================
    # COMPUTE ENERGY SPECTRA
    # ========================================================================

    print(f"\n{'─'*70}")
    print(f"COMPUTE ENERGY SPECTRA (stride {stride})")
    print(f"{'─'*70}")

    # One-sided frequency vector using rfft
    frequencies = np.fft.rfftfreq(n_samples, d=dt_save_effective)
    nfreq = len(frequencies)
    df = frequencies[1] - frequencies[0] if nfreq > 1 else 1.0
    f_star = frequencies * c / u_infty  # Nondimensional convective frequency

    print(f"Frequency parameters:")
    print(f"  n_samples: {n_samples}")
    print(f"  nfreq (from rfft): {nfreq} = {n_samples}//2 + 1")
    print(f"  Frequency resolution df: {df:.6e} Hz")
    print(f"  Frequency range: {frequencies[1]:.6e} to {frequencies[-1]:.6e} Hz")

    energy_spectra = {}

    for j_idx, y_val in zip(y_locations_idx, y_locations_val):
        u_data = time_series_data[j_idx]['u']  # (n_times, n_z)
        v_data = time_series_data[j_idx]['v']

        # Initialize storage for z-resolved spectra
        E_uu_z = np.zeros((nfreq, nz))
        E_vv_z = np.zeros((nfreq, nz))
        var_u_time_z = np.zeros(nz)
        var_v_time_z = np.zeros(nz)
        var_u_spectral_z = np.zeros(nz)
        var_v_spectral_z = np.zeros(nz)

        # Compute spectra for each z position independently
        for iz in range(nz):
            u_fluct = u_data[:, iz]  # (n_times,)
            v_fluct = v_data[:, iz]

            # Time-domain variance
            var_u_time_z[iz] = np.var(u_fluct)
            var_v_time_z[iz] = np.var(v_fluct)

            # Remove temporal mean
            u_fluct = u_fluct - np.mean(u_fluct)
            v_fluct = v_fluct - np.mean(v_fluct)

            # RFFT: returns one-sided spectrum
            U_rfft = np.fft.rfft(u_fluct)
            V_rfft = np.fft.rfft(v_fluct)

            # One-sided periodogram normalization
            E_uu_z[:, iz] = (2.0 * dt_save_effective / n_samples) * (np.abs(U_rfft) ** 2)
            E_vv_z[:, iz] = (2.0 * dt_save_effective / n_samples) * (np.abs(V_rfft) ** 2)

            # Correct for DC and Nyquist
            E_uu_z[0, iz] /= 2.0
            E_vv_z[0, iz] /= 2.0

            if n_samples % 2 == 0:
                E_uu_z[-1, iz] /= 2.0
                E_vv_z[-1, iz] /= 2.0

            # Variance recovered from spectrum
            var_u_spectral_z[iz] = np.sum(E_uu_z[:, iz] * df)
            var_v_spectral_z[iz] = np.sum(E_vv_z[:, iz] * df)

        # Average spectra over z
        E_uu = np.mean(E_uu_z, axis=1)  # (nfreq,)
        E_vv = np.mean(E_vv_z, axis=1)

        # Z-averaged variance
        var_u_time = np.mean(var_u_time_z)
        var_v_time = np.mean(var_v_time_z)
        var_u_spectral = np.mean(var_u_spectral_z)
        var_v_spectral = np.mean(var_v_spectral_z)

        # Compute relative errors
        rel_error_u_time_z = np.abs(var_u_spectral_z - var_u_time_z) / (var_u_time_z + 1e-15)
        rel_error_v_time_z = np.abs(var_v_spectral_z - var_v_time_z) / (var_v_time_z + 1e-15)

        rel_error_u = np.abs(var_u_spectral - var_u_time) / (var_u_time + 1e-15)
        rel_error_v = np.abs(var_v_spectral - var_v_time) / (var_v_time + 1e-15)

        max_rel_error_u_z = np.max(rel_error_u_time_z)
        max_rel_error_v_z = np.max(rel_error_v_time_z)

        tolerance = 0.05  # 5% tolerance

        if max_rel_error_u_z > tolerance or max_rel_error_v_z > tolerance:
            print(f"WARNING (Probe at y={y_val:.6e}): Variance z-mismatch exceeds {tolerance*100}%: "
                  f"u_max={max_rel_error_u_z*100:.2f}%, v_max={max_rel_error_v_z*100:.2f}%")

        energy_spectra[j_idx] = {
            'y': y_val,
            'n_samples': n_samples,
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
            'rel_error_u_time_z': rel_error_u_time_z,
            'rel_error_v_time_z': rel_error_v_time_z,
            'u_rms': np.sqrt(var_u_spectral),
            'v_rms': np.sqrt(var_v_spectral)
        }

    # Print variance validation
    print(f"\nVariance validation (time vs. spectral domain):")
    for i, j_idx in enumerate(y_locations_idx[:2]):
        y_val = energy_spectra[j_idx]['y']
        var_time_u = energy_spectra[j_idx]['var_u_time']
        var_spec_u = energy_spectra[j_idx]['var_u_spectral']
        var_time_v = energy_spectra[j_idx]['var_v_time']
        var_spec_v = energy_spectra[j_idx]['var_v_spectral']

        rel_err_u = energy_spectra[j_idx]['rel_error_u']
        rel_err_v = energy_spectra[j_idx]['rel_error_v']

        rms_u = energy_spectra[j_idx]['u_rms']
        rms_v = energy_spectra[j_idx]['v_rms']

        print(f"\nProbe {i} (y={y_val:.6e}):")
        print(f"  u: var_time={var_time_u:.6e}, var_spectral={var_spec_u:.6e}, "
              f"rel_error={rel_err_u*100:.3f}%, rms={rms_u:.6e}")
        print(f"  v: var_time={var_time_v:.6e}, var_spectral={var_spec_v:.6e}, "
              f"rel_error={rel_err_v*100:.3f}%, rms={rms_v:.6e}")

    # ========================================================================
    # CREATE SPECTRAL PLOTS FOR THIS STRIDE
    # ========================================================================

    print(f"\n{'─'*70}")
    print(f"CREATE SPECTRAL PLOTS (stride {stride})")
    print(f"{'─'*70}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # Sort probes by y position for consistent color ordering
    sorted_indices = np.argsort([energy_spectra[j_idx]['y'] for j_idx in y_locations_idx])
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_indices)))

    # Plot E_uu (streamwise)
    ax = axes[0]
    for plot_idx, sorted_idx in enumerate(sorted_indices):
        j_idx = y_locations_idx[sorted_idx]
        y_val = energy_spectra[j_idx]['y']
        f_star_vals = energy_spectra[j_idx]['f_star']
        E_uu = energy_spectra[j_idx]['E_uu']

        # Multiplicative offset
        offset_factor = plot_idx * 0.4
        E_plot = E_uu * (10.0 ** offset_factor)

        ax.loglog(f_star_vals[1:], E_plot[1:], color=colors[plot_idx],
                  label=f"y={y_val:.4e}", linewidth=2, alpha=0.8)

    ax.set_xlabel(r"$f^* = f \cdot c / U_\infty$  (nondimensional)", fontsize=12)
    ax.set_ylabel(r"$E_{uu}$ (offset by 10$^{0.4k}$)", fontsize=12)
    ax.set_title(f"Streamwise Velocity Energy Spectrum ({slice_id}, stride={stride})\nAOA={AOA_deg}°, x={slice_x:.4f}",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Plot E_vv (cross-stream)
    ax = axes[1]
    for plot_idx, sorted_idx in enumerate(sorted_indices):
        j_idx = y_locations_idx[sorted_idx]
        y_val = energy_spectra[j_idx]['y']
        f_star_vals = energy_spectra[j_idx]['f_star']
        E_vv = energy_spectra[j_idx]['E_vv']

        # Multiplicative offset
        offset_factor = plot_idx * 0.4
        E_plot = E_vv * (10.0 ** offset_factor)

        ax.loglog(f_star_vals[1:], E_plot[1:], color=colors[plot_idx],
                  label=f"y={y_val:.4e}", linewidth=2, alpha=0.8)

    ax.set_xlabel(r"$f^* = f \cdot c / U_\infty$  (nondimensional)", fontsize=12)
    ax.set_ylabel(r"$E_{vv}$ (offset by 10$^{0.4k}$)", fontsize=12)
    ax.set_title(f"Cross-stream Velocity Energy Spectrum ({slice_id}, stride={stride})\nAOA={AOA_deg}°, x={slice_x:.4f}",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    spectra_file = os.path.join(SAVE_DIR, f"energy_spectra_uv_{slice_id}_stride_{stride:02d}.png")
    plt.savefig(spectra_file, dpi=150, bbox_inches='tight')
    print(f"Spectra plot saved: {spectra_file}")
    plt.show()

    # ========================================================================
    # SAVE RESULTS TO HDF5 (with stride information)
    # ========================================================================

    print(f"\n{'─'*70}")
    print(f"SAVE RESULTS TO HDF5 (stride {stride})")
    print(f"{'─'*70}")

    output_file = os.path.join(SAVE_DIR, f"energy_spectra_data_{slice_id}_stride_{stride:02d}.h5")

    with h5py.File(output_file, "w") as f:
        # Global attributes
        f.attrs["slice_id"] = slice_id
        f.attrs["slice_x"] = slice_x
        f.attrs["slice_comment"] = f"Spectral analysis at slice={slice_id}, x={slice_x:.6f} chord"

        f.attrs["AOA_deg"] = AOA_deg
        f.attrs["dt_iteration"] = dt_iteration
        f.attrs["delta_iter"] = delta_iter
        f.attrs["stride"] = stride
        f.attrs["dt_save_effective"] = dt_save_effective
        f.attrs["u_infty"] = u_infty
        f.attrs["c"] = c
        f.attrs["Re"] = Re_c
        f.attrs["n_samples"] = n_samples
        f.attrs["n_z"] = nz
        f.attrs["fs"] = fs_effective
        f.attrs["nfreq"] = nfreq

        # Per-probe data
        for probe_idx, j_idx in enumerate(y_locations_idx):
            y_target = Y_LOCATIONS[probe_idx]
            y_actual = y_locations_val[probe_idx]

            grp_name = f"probe_{probe_idx:02d}"
            grp = f.create_group(grp_name)

            # Metadata for this probe
            grp.attrs["probe_name"] = f"probe_{probe_idx:02d}"
            grp.attrs["y_target"] = y_target
            grp.attrs["y_actual"] = y_actual
            grp.attrs["y_distance_error"] = np.abs(y_actual - y_target)
            grp.attrs["j_index"] = j_idx

            # Frequencies (shared across all probes)
            grp.create_dataset("frequencies", data=energy_spectra[j_idx]['frequencies'])
            grp.create_dataset("f_star", data=energy_spectra[j_idx]['f_star'])

            # Z-averaged spectra
            grp.create_dataset("E_uu", data=energy_spectra[j_idx]['E_uu'])
            grp.create_dataset("E_vv", data=energy_spectra[j_idx]['E_vv'])

            # Z-resolved spectra
            grp.create_dataset("E_uu_z", data=energy_spectra[j_idx]['E_uu_z'])
            grp.create_dataset("E_vv_z", data=energy_spectra[j_idx]['E_vv_z'])

            # Variance: time domain
            grp.create_dataset("var_u_time_z", data=energy_spectra[j_idx]['var_u_time_z'])
            grp.create_dataset("var_v_time_z", data=energy_spectra[j_idx]['var_v_time_z'])

            grp.attrs["var_u_time_mean"] = energy_spectra[j_idx]['var_u_time']
            grp.attrs["var_v_time_mean"] = energy_spectra[j_idx]['var_v_time']
            grp.attrs["u_rms_time"] = np.sqrt(energy_spectra[j_idx]['var_u_time'])
            grp.attrs["v_rms_time"] = np.sqrt(energy_spectra[j_idx]['var_v_time'])

            # Variance: spectral domain
            grp.create_dataset("var_u_spectral_z", data=energy_spectra[j_idx]['var_u_spectral_z'])
            grp.create_dataset("var_v_spectral_z", data=energy_spectra[j_idx]['var_v_spectral_z'])

            grp.attrs["var_u_spectral"] = energy_spectra[j_idx]['var_u_spectral']
            grp.attrs["var_v_spectral"] = energy_spectra[j_idx]['var_v_spectral']
            grp.attrs["u_rms_spectral"] = energy_spectra[j_idx]['u_rms']
            grp.attrs["v_rms_spectral"] = energy_spectra[j_idx]['v_rms']

            # Validation error metrics
            grp.create_dataset("rel_error_u_z_percent",
                              data=energy_spectra[j_idx]['rel_error_u_time_z'] * 100)
            grp.create_dataset("rel_error_v_z_percent",
                              data=energy_spectra[j_idx]['rel_error_v_time_z'] * 100)

            grp.attrs["rel_error_u_percent"] = energy_spectra[j_idx]['rel_error_u'] * 100
            grp.attrs["rel_error_v_percent"] = energy_spectra[j_idx]['rel_error_v'] * 100

    print(f"Results saved to: {output_file}")

    # Store results for comparison plot
    all_stride_results[stride] = energy_spectra


# ============================================================================
# CREATE COMPARISON PLOT (All strides side-by-side)
# ============================================================================

print("\n" + "="*70)
print("CREATE COMPARISON PLOT (All Strides)")
print("="*70)

# Select first probe for comparison plot
j_idx_comparison = y_locations_idx[0]
y_val_comparison = y_locations_val[0]

n_strides = len(SLICE_STRIDES)
fig, axes = plt.subplots(n_strides, 2, figsize=(14, 5*n_strides))

# Handle single stride case
if n_strides == 1:
    axes = axes.reshape(1, -1)

for stride_idx, stride in enumerate(SLICE_STRIDES):
    energy_spectra = all_stride_results[stride]

    # Left plot: E_uu
    ax = axes[stride_idx, 0]
    f_star_vals = energy_spectra[j_idx_comparison]['f_star']
    E_uu = energy_spectra[j_idx_comparison]['E_uu']

    ax.loglog(f_star_vals[1:], E_uu[1:], color='blue', linewidth=2.5, alpha=0.8)
    ax.set_xlabel(r"$f^* = f \cdot c / U_\infty$", fontsize=11)
    ax.set_ylabel(r"$E_{uu}$", fontsize=11)
    ax.set_title(f"Streamwise (stride={stride}, n_samples={len(E_uu)})",
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    # Right plot: E_vv
    ax = axes[stride_idx, 1]
    E_vv = energy_spectra[j_idx_comparison]['E_vv']

    ax.loglog(f_star_vals[1:], E_vv[1:], color='red', linewidth=2.5, alpha=0.8)
    ax.set_xlabel(r"$f^* = f \cdot c / U_\infty$", fontsize=11)
    ax.set_ylabel(r"$E_{vv}$", fontsize=11)
    ax.set_title(f"Cross-stream (stride={stride}, n_samples={len(E_vv)})",
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(f"Spectral Comparison Across Strides ({slice_id})\nProbe at y={y_val_comparison:.4e}, AOA={AOA_deg}°",
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

comparison_file = os.path.join(SAVE_DIR, f"energy_spectra_comparison_{slice_id}_all_strides.png")
plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
print(f"Comparison plot saved: {comparison_file}")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY OF ENERGY SPECTRA ANALYSIS (VARIABLE STRIDE)")
print("="*70)
print(f"Slice: {slice_id}")
print(f"AOA: {AOA_deg}°")
print(f"Slice x-location: {slice_x:.6f} chord")
print(f"Strides analyzed: {SLICE_STRIDES}")
print(f"Number of probes: {len(Y_LOCATIONS)}")
print(f"Spanwise samples per probe: {nz}")

print(f"\nBasic file information (all strides):")
for stride in SLICE_STRIDES:
    energy_spectra = all_stride_results[stride]
    n_samples = energy_spectra[j_idx_comparison]['n_samples']  # Get actual time samples
    nfreq = len(energy_spectra[j_idx_comparison]['frequencies'])  # Number of frequency points
    dt_eff = delta_iter * stride * dt_iteration
    fs_eff = 1.0 / dt_eff
    total_time = (n_samples - 1) * dt_eff

    print(f"\n  Stride {stride}:")
    print(f"    Number of snapshots (time samples): {n_samples}")
    print(f"    Number of frequency points (after rfft): {nfreq}")
    print(f"    Effective dt_save: {dt_eff:.6e} s")
    print(f"    Effective fs: {fs_eff:.6e} Hz")
    print(f"    Frequency range: {energy_spectra[j_idx_comparison]['frequencies'][1]:.6e} to "
          f"{energy_spectra[j_idx_comparison]['frequencies'][-1]:.6e} Hz")
    print(f"    Total time span: {total_time:.6e} s")

print(f"\nOutput files:")
for stride in SLICE_STRIDES:
    print(f"  Stride {stride}:")
    print(f"    Spectra plot: energy_spectra_uv_{slice_id}_stride_{stride:02d}.png")
    print(f"    HDF5 data: energy_spectra_data_{slice_id}_stride_{stride:02d}.h5")
print(f"  Comparison plot: {comparison_file}")

print("="*70)
