"""
Energy Spectra of Surface Variables from Slice Data
=====================================================

Compute raw periodograms of wall shear and pressure fluctuations at the surface
from slice snapshot data, following the clean methodology of energy_spectra_u_v_from_slices.py.

Key features:
- Numeric file sorting (not lexicographic)
- Correct physical time step computation
- Raw FFT periodogram (consistent with Euu spectra)
- Z-averaged PSD computation
- Clean normalization with variance validation
- Nondimensional frequency output
- Structured HDF5 output

Surface variables:
- τ' : wall shear stress fluctuation (from compute_tau_w_all_z logic)
- p'_w : wall pressure fluctuation (from compute_pressure_all_z logic)
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

### AOA 12º
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"

# Geometric data
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# Output directory
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Energy_spectra/"
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
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity

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
            raise ValueError(f"Cannot extract iteration number from {filename}. "
                           f"Expected format: slice_*_NNNNNNNN-COMP-DATA.h5")
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
        raise ValueError(
            f"Iteration spacing is NOT constant: {unique_deltas}. "
            f"Cannot reliably compute time step. Aborting."
        )

    delta_iter = int(unique_deltas[0])
    print(f"  Constant delta_iteration: {delta_iter}")

    return sorted_files, iterations, delta_iter


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
    tangent = np.array([normal_at_point[1], -normal_at_point[0], 0.0])
    tangent = tangent / np.linalg.norm(tangent)

    u_vals = u_data[:, y_idx, 0]
    v_vals = v_data[:, y_idx, 0]
    w_vals = w_data[:, y_idx, 0]

    u_t_vals = u_vals * tangent[0] + v_vals * tangent[1] + w_vals * tangent[2]
    tau_w = mu_ref * u_t_vals / distance_at_point

    return tau_w


def compute_pressure_all_z(p_data, y_idx):
    """
    Extract pressure for ALL z positions at once.

    Args:
        p_data: Pressure field with shape (nz, ny, nx)
        y_idx: y-index of the surface point

    Returns:
        p: Array of pressure with shape (nz,)
    """
    p = p_data[:, y_idx, 0]
    return p


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

# ============================================================================
# VERIFY SLICE STRUCTURE
# ============================================================================

x_unique_in_mesh = np.unique(x_data)
if len(x_unique_in_mesh) > 1:
    raise ValueError(f"Slice mesh has {len(x_unique_in_mesh)} unique x values. "
                     f"Expected single x-plane (2D slice).")

x_all = x_data.flatten()
x_std = np.std(x_all[~np.isnan(x_all)])
x_rel_std = x_std / np.abs(x_all[~np.isnan(x_all)].mean() + 1e-15)
if x_rel_std > 1e-6:
    raise ValueError(f"x-coordinate varies too much in slice (rel_std={x_rel_std:.6e}). "
                     f"This slice may not be a valid x-plane.")

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
# LOAD GEOMETRIC DATA FOR WALL SHEAR
# ============================================================================

print("\n" + "="*70)
print("LOAD GEOMETRIC DATA FOR WALL SHEAR COMPUTATION")
print("="*70)

with h5py.File(GEO_FILE, 'r') as f:
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

suction_side_indices = np.where(interface_points[:, 1] >= 0)[0]

x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
closest_interface_idx = suction_side_indices[closest_idx]

surface_normal = proj_normals[closest_interface_idx]
surface_distance = proj_distances[closest_interface_idx]

print(f"Surface reference point: x={slice_x:.6f}")
print(f"Wall distance: {surface_distance:.6e}")
print(f"Surface normal: {surface_normal}")

ref_probe_y = suction_side_points[closest_idx, 1]
ref_probe_idx = np.argmin(np.abs(y_unique - ref_probe_y))

print(f"Surface y-index: {ref_probe_idx}, y-value: {y_unique[ref_probe_idx]:.6e}")

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
# EXTRACT TIME SERIES (ALL Z DATA)
# ============================================================================

print("\n" + "="*70)
print("EXTRACT TIME SERIES (ALL SPANWISE DATA)")
print("="*70)

time_series_data = {
    'tau': [],
    'p': []
}

print(f"Extracting time series from {len(slice_files)} snapshots...")

failed_files = []

for t, slice_file in enumerate(slice_files):
    if not os.path.exists(slice_file):
        failed_files.append(slice_file)
        continue

    try:
        fields = loader.load_snapshot(slice_file)
        u_data = loader.reconstruct_field(fields["u"])[1:-1, :, :]
        v_data = loader.reconstruct_field(fields["v"])[1:-1, :, :]
        w_data = loader.reconstruct_field(fields["w"])[1:-1, :, :]
        p_data = loader.reconstruct_field(fields["p"])[1:-1, :, :]

        # Compute wall shear across all z (in wall-tangent frame from geometry)
        tau_w_vec = compute_tau_w_all_z(u_data, v_data, w_data, ref_probe_idx,
                                        mu_ref, surface_normal, surface_distance)

        # Compute pressure across all z
        p_vec = compute_pressure_all_z(p_data, ref_probe_idx)

        time_series_data['tau'].append(tau_w_vec)
        time_series_data['p'].append(p_vec)

        if (t + 1) % max(1, len(slice_files) // 10) == 0:
            print(f"  Processed {t + 1}/{len(slice_files)} snapshots")

    except Exception as e:
        failed_files.append((slice_file, str(e)))
        continue

if failed_files:
    raise RuntimeError(f"Failed to load {len(failed_files)} files. Aborting. "
                      f"First error: {failed_files[0]}")

n_samples_list = [len(time_series_data['tau']), len(time_series_data['p'])]
if len(set(n_samples_list)) > 1:
    raise ValueError(f"Inconsistent sample counts: {set(n_samples_list)}")

n_samples = n_samples_list[0]
print(f"\nActual loaded samples: {n_samples}")

# Convert to arrays and compute fluctuations
time_series_data['tau'] = np.array(time_series_data['tau'])
time_series_data['p'] = np.array(time_series_data['p'])

tau_mean = np.mean(time_series_data['tau'], axis=0)
p_mean = np.mean(time_series_data['p'], axis=0)

tau_prime = time_series_data['tau'] - tau_mean
p_prime = time_series_data['p'] - p_mean

print(f"τ' shape = {tau_prime.shape}")
print(f"p'_w shape = {p_prime.shape}")

# ============================================================================
# COMPUTE ENERGY SPECTRA (RAW PERIODOGRAM - RFFT)
# ============================================================================

print("\n" + "="*70)
print("COMPUTE ENERGY SPECTRA (RAW PERIODOGRAM - RFFT)")
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

# Compute spectra for wall shear
tau_prime_data = tau_prime
p_prime_data = p_prime

E_tautau_z = np.zeros((nfreq, nz))
E_pp_z = np.zeros((nfreq, nz))
var_tau_time_z = np.zeros(nz)
var_p_time_z = np.zeros(nz)
var_tau_spectral_z = np.zeros(nz)
var_p_spectral_z = np.zeros(nz)

for iz in range(nz):
    tau_fluct = tau_prime_data[:, iz]
    p_fluct = p_prime_data[:, iz]

    var_tau_time_z[iz] = np.var(tau_fluct)
    var_p_time_z[iz] = np.var(p_fluct)

    tau_fluct = tau_fluct - np.mean(tau_fluct)
    p_fluct = p_fluct - np.mean(p_fluct)

    tau_rfft = np.fft.rfft(tau_fluct)
    p_rfft = np.fft.rfft(p_fluct)

    E_tautau_z[:, iz] = (2.0 * dt_save / n_samples) * (np.abs(tau_rfft) ** 2)
    E_pp_z[:, iz] = (2.0 * dt_save / n_samples) * (np.abs(p_rfft) ** 2)

    E_tautau_z[0, iz] /= 2.0
    E_pp_z[0, iz] /= 2.0

    if n_samples % 2 == 0:
        E_tautau_z[-1, iz] /= 2.0
        E_pp_z[-1, iz] /= 2.0

    var_tau_spectral_z[iz] = np.sum(E_tautau_z[:, iz] * df)
    var_p_spectral_z[iz] = np.sum(E_pp_z[:, iz] * df)

# Average spectra over z
E_tautau = np.mean(E_tautau_z, axis=1)
E_pp = np.mean(E_pp_z, axis=1)

var_tau_time = np.mean(var_tau_time_z)
var_p_time = np.mean(var_p_time_z)
var_tau_spectral = np.mean(var_tau_spectral_z)
var_p_spectral = np.mean(var_p_spectral_z)

rel_error_tau_time_z = np.abs(var_tau_spectral_z - var_tau_time_z) / (var_tau_time_z + 1e-15)
rel_error_p_time_z = np.abs(var_p_spectral_z - var_p_time_z) / (var_p_time_z + 1e-15)

rel_error_tau = np.abs(var_tau_spectral - var_tau_time) / (var_tau_time + 1e-15)
rel_error_p = np.abs(var_p_spectral - var_p_time) / (var_p_time + 1e-15)

max_rel_error_tau_z = np.max(rel_error_tau_time_z)
max_rel_error_p_z = np.max(rel_error_p_time_z)

tolerance = 0.05

if max_rel_error_tau_z > tolerance or max_rel_error_p_z > tolerance:
    print(f"WARNING: Variance z-mismatch exceeds {tolerance*100}%: "
          f"tau_max={max_rel_error_tau_z*100:.2f}%, p_max={max_rel_error_p_z*100:.2f}%")

print(f"\nVariance validation (time vs. spectral domain):")
print(f"  τ': var_time={var_tau_time:.6e}, var_spectral={var_tau_spectral:.6e}, "
      f"rel_error={rel_error_tau*100:.3f}%, rms={np.sqrt(var_tau_spectral):.6e}")
print(f"  p'_w: var_time={var_p_time:.6e}, var_spectral={var_p_spectral:.6e}, "
      f"rel_error={rel_error_p*100:.3f}%, rms={np.sqrt(var_p_spectral):.6e}")

# ============================================================================
# CREATE SPECTRAL PLOTS
# ============================================================================

print("\n" + "="*70)
print("CREATE SPECTRAL PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(11, 9))

# Plot E_tautau (wall shear)
ax = axes[0]
ax.loglog(f_star[1:], E_tautau[1:], 'o-', linewidth=2, markersize=4,
         color='#d62728', label='τ\'', alpha=0.8)
ax.set_xlabel(r"$f^* = f \cdot c / U_\infty$  (nondimensional)", fontsize=12)
ax.set_ylabel(r"$E_{\tau\tau}$", fontsize=12)
ax.set_title(f"Wall Shear Energy Spectrum ({slice_id})\nAOA={AOA_deg}°, x={slice_x:.4f}",
            fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# Plot E_pp (pressure)
ax = axes[1]
ax.loglog(f_star[1:], E_pp[1:], 's-', linewidth=2, markersize=4,
         color='#ff7f0e', label='p\'_w', alpha=0.8)
ax.set_xlabel(r"$f^* = f \cdot c / U_\infty$  (nondimensional)", fontsize=12)
ax.set_ylabel(r"$E_{pp}$", fontsize=12)
ax.set_title(f"Wall Pressure Energy Spectrum ({slice_id})\nAOA={AOA_deg}°, x={slice_x:.4f}",
            fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
spectra_file = os.path.join(SAVE_DIR, f"energy_spectra_surface_{slice_id}.png")
plt.savefig(spectra_file, dpi=150, bbox_inches='tight')
print(f"Spectra plot saved: {spectra_file}")
plt.show()

# ============================================================================
# SAVE RESULTS TO HDF5
# ============================================================================

print("\n" + "="*70)
print("SAVE RESULTS TO HDF5")
print("="*70)

output_file = os.path.join(SAVE_DIR, f"energy_spectra_surface_{slice_id}.h5")

with h5py.File(output_file, "w") as f:
    # Global attributes
    f.attrs["slice_id"] = slice_id
    f.attrs["slice_x"] = slice_x
    f.attrs["slice_comment"] = f"Surface variable spectral analysis at {slice_id}, x={slice_x:.6f} chord"

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

    # Wall shear group
    grp_tau = f.create_group("tau_shear")
    grp_tau.attrs["description"] = "Wall shear stress fluctuation"
    grp_tau.attrs["units"] = "Pa"
    grp_tau.attrs["var_time_mean"] = var_tau_time
    grp_tau.attrs["var_spectral"] = var_tau_spectral
    grp_tau.attrs["rms_spectral"] = np.sqrt(var_tau_spectral)
    grp_tau.attrs["rel_error_percent"] = rel_error_tau * 100

    grp_tau.create_dataset("frequencies", data=frequencies)
    grp_tau.create_dataset("f_star", data=f_star)
    grp_tau.create_dataset("E_tautau", data=E_tautau)
    grp_tau.create_dataset("E_tautau_z", data=E_tautau_z)
    grp_tau.create_dataset("var_time_z", data=var_tau_time_z)
    grp_tau.create_dataset("var_spectral_z", data=var_tau_spectral_z)
    grp_tau.create_dataset("rel_error_z_percent", data=rel_error_tau_time_z * 100)

    # Wall pressure group
    grp_p = f.create_group("pressure")
    grp_p.attrs["description"] = "Wall pressure fluctuation"
    grp_p.attrs["units"] = "Pa"
    grp_p.attrs["var_time_mean"] = var_p_time
    grp_p.attrs["var_spectral"] = var_p_spectral
    grp_p.attrs["rms_spectral"] = np.sqrt(var_p_spectral)
    grp_p.attrs["rel_error_percent"] = rel_error_p * 100

    grp_p.create_dataset("frequencies", data=frequencies)
    grp_p.create_dataset("f_star", data=f_star)
    grp_p.create_dataset("E_pp", data=E_pp)
    grp_p.create_dataset("E_pp_z", data=E_pp_z)
    grp_p.create_dataset("var_time_z", data=var_p_time_z)
    grp_p.create_dataset("var_spectral_z", data=var_p_spectral_z)
    grp_p.create_dataset("rel_error_z_percent", data=rel_error_p_time_z * 100)

print(f"Results saved to: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: SURFACE VARIABLE ENERGY SPECTRA ANALYSIS")
print("="*70)
print(f"Slice: {slice_id}")
print(f"AOA: {AOA_deg}°")
print(f"Slice x-location: {slice_x:.6f} chord")
print(f"Number of snapshots: {n_samples}")
print(f"Physical time step: dt_save = {dt_save:.6e} s")
print(f"Sampling frequency: fs = {fs:.6e} Hz")
print(f"Frequency range: {frequencies[1]:.6e} to {frequencies[-1]:.6e} Hz")
print(f"Nondimensional frequency range: f* ∈ [{f_star[1]:.6e}, {f_star[-1]:.6e}]")
print(f"Spanwise samples per variable: {nz}")
print(f"\nOutput files:")
print(f"  Spectra plot: {spectra_file}")
print(f"  HDF5 data: {output_file}")
print("="*70)
