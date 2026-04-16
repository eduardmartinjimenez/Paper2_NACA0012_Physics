"""
Full Spanwise Spectral Analysis - Minimal and Focused
======================================================

Combines signal extraction and spectral analysis for the FULL SPANWISE DOMAIN.
Uses spanwise direction as an ensemble/statistical direction to improve convergence.

This script:
1. Loads geometry and slice mesh
2. Identifies surface point on suction side (fixed in x,y)
3. Identifies fixed velocity probe location by y
4. Loads all slice snapshots and extracts:
   - Wall shear stress tau_w(t, z) for the surface point
   - Streamwise velocity u_flow(t, z) for the velocity probe (after AOA rotation)
5. Computes fluctuations: tau_prime(t, z) and u_prime(t, z)
6. For each z independently computes spectral quantities
7. Averages over z: mean PSD, mean magnitude-squared coherence, mean cross-spectrum magnitude
8. Plots and saves z-averaged spectral results

Key design choices:
- Only essential variables stored (tau_w, u_flow)
- Immediate rotation to streamwise frame during extraction (no intermediate storage)
- Single cross-spectrum magnitude definition: <|S_tau_u(f, z)|>_z
- Spanwise dimension used purely for ensemble averaging to improve statistics
"""

import os
import sys
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

# Reference slice data paths (for mesh/geometry reference)
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"

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
U_inf = 1.0             # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord length [m]
Re_c = 50000            # Reynolds number
AOA_deg = 12.0          # Angle of attack [degrees]
AOA_rad = np.radians(AOA_deg)

# Physical time step [CRITICAL - must match simulation]
dt_iteration = 2.0e-06  # Physical time per iteration [s]

# Compute reference dynamic viscosity
mu_ref = rho_ref * U_inf * c / Re_c

# Fixed probe location for signal correlation analysis
Y_PROBE_FIXED = 0.09

# Welch parameters for spectral analysis
NPERSEG = 4096
NOVERLAP = NPERSEG // 2
WINDOW = 'hann'

# Data loader module
module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    """Check path exists and print confirmation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"✓ {kind} exists: {path}")


def get_data_files_for_slice(slice_path: str) -> list:
    """Get all velocity data files in slice directory, sorted by iteration."""
    data_files = glob.glob(os.path.join(slice_path, "*-COMP-DATA.h5"))

    def get_iteration(filepath):
        match = re.search(r'_(\d+)-COMP-DATA', filepath)
        return int(match.group(1)) if match else 0

    data_files.sort(key=get_iteration)
    return data_files


def compute_tau_w_all_z(u_data: np.ndarray, v_data: np.ndarray, w_data: np.ndarray,
                        y_idx: int, mu_ref: float,
                        normal_at_point: np.ndarray,
                        distance_at_point: float) -> np.ndarray:
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
    # Compute tangent vector from normal (2D normal -> 2D tangent)
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
    signal_centered = signal_data - np.mean(signal_data)
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
    signal1_centered = signal1 - np.mean(signal1)
    signal2_centered = signal2 - np.mean(signal2)
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
    signal1_centered = signal1 - np.mean(signal1)
    signal2_centered = signal2 - np.mean(signal2)
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
# PHASE 1: LOAD GEOMETRY AND MESH
# ============================================================================

print("="*70)
print("PHASE 1: LOAD GEOMETRY AND MESH")
print("="*70)

assert_exists(GEO_FILE, "Geometrical data file")
assert_exists(MESH_SLICE_FILE, "Mesh slice file")

# Load geometrical data
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

# Extract suction side surfaces
suction_side_points = interface_points[interface_points[:, 1] >= 0]
suction_side_indices = np.where(interface_points[:, 1] >= 0)[0]

print(f"Suction side points: {suction_side_points.shape[0]}")

# Load mesh
loader = CompressedSnapshotLoader(MESH_SLICE_FILE)
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print(f"Mesh shape (nz, ny, nx): {x_data.shape}")
nz, ny, nx = x_data.shape

# Verify single x-plane
x_unique = np.unique(x_data)
if len(x_unique) > 1:
    raise ValueError(f"Slice mesh has {len(x_unique)} unique x values. Expected single x-plane.")

slice_x = x_data[0, 0, 0]
print(f"✓ Single x-plane verified: slice_x={slice_x:.6f}")

z_unique = np.unique(z_data[:, 0, 0])
dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
L_z = dz * nz
print(f"  Spanwise domain: nz={nz}, dz={dz:.6e} m, Lz={L_z:.6e} m")

y_unique = np.unique(y_data[:, :, 0][0, :])
print(f"  Total y-grid points: {len(y_unique)}")

# ============================================================================
# PHASE 2: FIND CLOSEST SURFACE POINT ON SUCTION SIDE
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: FIND CLOSEST SURFACE POINT")
print("="*70)

x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
closest_surface_point = suction_side_points[closest_idx]
closest_interface_idx = suction_side_indices[closest_idx]
surface_x = closest_surface_point[0]
surface_y = closest_surface_point[1]

surface_normal = proj_normals[closest_interface_idx]
surface_distance = proj_distances[closest_interface_idx]

print(f"Surface point: x={surface_x:.6e}, y={surface_y:.6e}")
print(f"Wall distance: {surface_distance:.6e}")

# ============================================================================
# PHASE 3: SELECT PROBE LOCATIONS
# ============================================================================

print("\n" + "="*70)
print("PHASE 3: SELECT PROBE LOCATIONS")
print("="*70)

probe_definitions = [
    {'label': 'surface', 'y_target': surface_y},
    {'label': 'probe_fixed_y', 'y_target': Y_PROBE_FIXED}
]

probe_locations = []
for i, probe_def in enumerate(probe_definitions):
    y_target = probe_def['y_target']
    idx_closest = np.argmin(np.abs(y_unique - y_target))
    y_actual = y_unique[idx_closest]

    probe_locations.append({
        'probe_id': i,
        'label': probe_def['label'],
        'y_target': y_target,
        'y_actual': y_actual,
        'y_idx': idx_closest
    })

    print(f"Probe {i} ({probe_def['label']}): y_actual={y_actual:.6e}")

# ============================================================================
# PHASE 4: INFER SLICE ID AND GET DATA FILES
# ============================================================================

match = re.search(r'slice_(\w+)', SLICES_PATH)
slice_id = f"slice_{match.group(1)}" if match else "unknown_slice"

assert_exists(SLICES_PATH, "Configured slice directory")
data_files = get_data_files_for_slice(SLICES_PATH)
print(f"\nFound {len(data_files)} time steps")

if len(data_files) == 0:
    raise FileNotFoundError(f"No data files found in {SLICES_PATH}")

# ============================================================================
# PHASE 5: EXTRACT TIME SERIES FOR ALL Z LOCATIONS
# ============================================================================

print("\n" + "="*70)
print(f"PHASE 5: EXTRACT TIME SERIES (nz={nz})")
print("="*70)

timeseries_raw = {
    'iterations': [],
    'tau_w': [],      # List of arrays (Nt, Nz)
    'u_flow': []      # List of arrays (Nt, Nz)
}

total_timesteps = 0
cos_aoa = np.cos(AOA_rad)
sin_aoa = np.sin(AOA_rad)

print(f"Processing {len(data_files)} snapshots...")

for file_idx, data_file in enumerate(data_files):
    if (file_idx + 1) % max(1, len(data_files) // 10) == 0 or file_idx == 0:
        print(f"  Progress: {file_idx + 1}/{len(data_files)}")

    # Extract iteration number
    match = re.search(r'_(\d+)-COMP-DATA', data_file)
    iteration = int(match.group(1)) if match else file_idx

    try:
        # Load velocity components
        snapshot = loader.load_snapshot(data_file)
        u_data = loader.reconstruct_field(snapshot["u"])[1:-1, :, :]
        v_data = loader.reconstruct_field(snapshot["v"])[1:-1, :, :]
        w_data = loader.reconstruct_field(snapshot["w"])[1:-1, :, :]

        # Get surface point index
        surface_y_idx = probe_locations[0]['y_idx']

        # Compute tau_w for ALL z at surface point
        tau_w_vals = compute_tau_w_all_z(u_data, v_data, w_data,
                                         surface_y_idx, mu_ref,
                                         surface_normal, surface_distance)

        # Get velocity probe index
        probe_y_idx = probe_locations[1]['y_idx']

        # Extract u and v for ALL z at probe point
        u_vals = u_data[:, probe_y_idx, 0]
        v_vals = v_data[:, probe_y_idx, 0]

        # Rotate to flow-aligned frame: streamwise velocity
        u_flow_vals = u_vals * cos_aoa + v_vals * sin_aoa

        # Store
        timeseries_raw['iterations'].append(iteration)
        timeseries_raw['tau_w'].append(tau_w_vals)
        timeseries_raw['u_flow'].append(u_flow_vals)

        total_timesteps += 1

    except Exception as e:
        print(f"  ⚠ Error loading snapshot {file_idx}: {e}")
        continue

print(f"✓ Processed {total_timesteps} valid time steps")

# ============================================================================
# PHASE 6: CONVERT TO ARRAYS AND COMPUTE TEMPORAL MEANS
# ============================================================================

print("\n" + "="*70)
print("PHASE 6: CONVERT TO ARRAYS AND COMPUTE TEMPORAL MEANS")
print("="*70)

iterations = np.array(timeseries_raw['iterations'])
tau_w_array = np.array(timeseries_raw['tau_w'])  # Shape: (Nt, Nz)
u_flow_array = np.array(timeseries_raw['u_flow'])  # Shape: (Nt, Nz)

Nt = len(iterations)

print(f"Time series shapes:")
print(f"  tau_w: {tau_w_array.shape}")
print(f"  u_flow: {u_flow_array.shape}")

# Compute temporal means (averaged over time for each z)
tau_w_mean = np.mean(tau_w_array, axis=0)  # Shape: (Nz,)
u_flow_mean = np.mean(u_flow_array, axis=0)  # Shape: (Nz,)

print(f"Temporal means:")
print(f"  <tau_w>: min={np.min(tau_w_mean):.6e}, max={np.max(tau_w_mean):.6e}")
print(f"  <u_flow>: min={np.min(u_flow_mean):.6e}, max={np.max(u_flow_mean):.6e}")

# ============================================================================
# PHASE 7: VALIDATE TEMPORAL SPACING
# ============================================================================

print("\n" + "="*70)
print("PHASE 7: VALIDATE TEMPORAL SPACING")
print("="*70)

iter_diff = np.diff(iterations)
iter_diff_mean = np.mean(iter_diff)
iter_diff_std = np.std(iter_diff)

dt_mean = iter_diff_mean * dt_iteration
fs = 1.0 / dt_mean
time_steps = iterations * dt_iteration

print(f"Time samples: {Nt}")
print(f"Iteration range: {iterations[0]:,d} to {iterations[-1]:,d}")
print(f"Sampling frequency: {fs:.6f} Hz")
print(f"Total time span: {time_steps[-1] - time_steps[0]:.6f} s")

uniformity_tolerance = 0.01
is_uniform = (iter_diff_std / iter_diff_mean) < uniformity_tolerance
print(f"Temporal spacing uniform: {'✓ YES' if is_uniform else '✗ NO'}")

# ============================================================================
# PHASE 8: COMPUTE FLUCTUATIONS
# ============================================================================

print("\n" + "="*70)
print("PHASE 8: COMPUTE FLUCTUATIONS")
print("="*70)

# Compute fluctuations (broadcast operation)
tau_prime = tau_w_array - tau_w_mean  # Shape: (Nt, Nz)
u_prime = u_flow_array - u_flow_mean  # Shape: (Nt, Nz)

print(f"Fluctuation means (should be ~0):")
print(f"  mean(tau_prime) = {np.mean(tau_prime):.6e}")
print(f"  mean(u_prime) = {np.mean(u_prime):.6e}")

# ============================================================================
# PHASE 9: VALIDATE WELCH PARAMETERS
# ============================================================================

print("\n" + "="*70)
print("PHASE 9: VALIDATE WELCH PARAMETERS")
print("="*70)

nperseg = NPERSEG
noverlap = NOVERLAP

if nperseg > Nt:
    nperseg = Nt // 2
    noverlap = nperseg // 2
    print(f"⚠ nperseg reduced to {nperseg} (original {NPERSEG} > Nt={Nt})")

if nperseg < 4:
    raise ValueError(f"nperseg={nperseg} is too small")

print(f"Using Welch parameters:")
print(f"  nperseg: {nperseg}")
print(f"  noverlap: {noverlap}")
print(f"  window: {WINDOW}")
print(f"  Frequency resolution: {fs / nperseg:.6e} Hz")

# ============================================================================
# PHASE 10: COMPUTE SPECTRAL QUANTITIES FOR EACH Z
# ============================================================================

print("\n" + "="*70)
print("PHASE 10: COMPUTE SPECTRAL QUANTITIES FOR EACH Z")
print("="*70)

psd_tau_all_z = []  # Shape will be (Nz, Nf)
psd_u_all_z = []
csd_mag_all_z = []
coherence_all_z = []
f = None

print(f"Computing spectral quantities for {nz} z-positions...")

for iz in range(nz):
    tau_z = tau_prime[:, iz]
    u_z = u_prime[:, iz]

    # Compute PSD for both signals
    f_psd, psd_tau_z = compute_psd_welch(tau_z, fs, window=WINDOW,
                                         nperseg=nperseg, noverlap=noverlap)
    _, psd_u_z = compute_psd_welch(u_z, fs, window=WINDOW,
                                   nperseg=nperseg, noverlap=noverlap)

    # Compute cross-spectrum magnitude
    _, csd_mag_z = compute_cross_spectrum_welch(tau_z, u_z, fs, window=WINDOW,
                                                nperseg=nperseg, noverlap=noverlap)

    # Compute coherence
    _, coh_z = compute_coherence_welch(tau_z, u_z, fs, window=WINDOW,
                                       nperseg=nperseg, noverlap=noverlap)

    psd_tau_all_z.append(psd_tau_z)
    psd_u_all_z.append(psd_u_z)
    csd_mag_all_z.append(csd_mag_z)
    coherence_all_z.append(coh_z)

    if f is None:
        f = f_psd

    if (iz + 1) % max(1, nz // 5) == 0 or iz == 0:
        print(f"  z-index {iz}/{nz-1}: Spectral quantities computed")

# Convert to arrays
psd_tau_all_z = np.array(psd_tau_all_z)  # Shape: (Nz, Nf)
psd_u_all_z = np.array(psd_u_all_z)
csd_mag_all_z = np.array(csd_mag_all_z)
coherence_all_z = np.array(coherence_all_z)

print(f"\nSpectral array shapes: {psd_tau_all_z.shape}")

# ============================================================================
# PHASE 11: AVERAGE SPECTRAL QUANTITIES OVER Z
# ============================================================================

print("\n" + "="*70)
print("PHASE 11: AVERAGE SPECTRAL QUANTITIES OVER Z")
print("="*70)

# Average over z (axis=0)
psd_tau_mean = np.mean(psd_tau_all_z, axis=0)
psd_u_mean = np.mean(psd_u_all_z, axis=0)
csd_magnitude_mean = np.mean(csd_mag_all_z, axis=0)
coherence_mean = np.mean(coherence_all_z, axis=0)

print(f"Z-averaged spectral results:")
print(f"  PSD tau: min={np.min(psd_tau_mean):.6e}, max={np.max(psd_tau_mean):.6e}")
print(f"  PSD u: min={np.min(psd_u_mean):.6e}, max={np.max(psd_u_mean):.6e}")
print(f"  CSD magnitude: min={np.min(csd_magnitude_mean):.6e}, max={np.max(csd_magnitude_mean):.6e}")
print(f"  Coherence: min={np.min(coherence_mean):.6e}, max={np.max(coherence_mean):.6e}")

# ============================================================================
# PHASE 12: CONVERT TO NONDIMENSIONAL FREQUENCY
# ============================================================================

print("\n" + "="*70)
print("PHASE 12: CONVERT TO NONDIMENSIONAL FREQUENCY")
print("="*70)

f_star = nondimensionalize_frequency(f, U_inf, c)

print(f"Dimensional frequency range: {f[1]:.6e} to {f[-1]:.6f} Hz")
print(f"Nondimensional frequency (f*) range: {f_star[1]:.6e} to {f_star[-1]:.6e}")

# ============================================================================
# PHASE 13: PLOT Z-AVERAGED SPECTRAL RESULTS
# ============================================================================

print("\n" + "="*70)
print("PHASE 13: PLOT Z-AVERAGED SPECTRAL RESULTS")
print("="*70)

freq_idx_start = 1  # Skip zero frequency

# Figure 1: PSDs
fig, ax = plt.subplots(figsize=(12, 8))

ax.loglog(f_star[freq_idx_start:], psd_tau_mean[freq_idx_start:], 'o-', linewidth=1.5, markersize=3,
          label="τ' (z-averaged)", color='#d62728', alpha=0.8)
ax.loglog(f_star[freq_idx_start:], psd_u_mean[freq_idx_start:], 's-', linewidth=1.5, markersize=3,
          label="u' (z-averaged)", color='#1f77b4', alpha=0.8)

ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=12, fontweight='bold')
ax.set_ylabel("PSD [signal²/Hz]", fontsize=12, fontweight='bold')
ax.set_title(f"Power Spectral Density (z-averaged, Nz={nz})", fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

info_text = (
    f"Nt = {Nt}, Nz = {nz}\n"
    f"fs = {fs:.2f} Hz\n"
    f"Δt = {time_steps[-1] - time_steps[0]:.2f} s\n"
    f"nperseg = {nperseg}"
)
ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

plt.tight_layout()
print("Displaying PSD plot...")
plt.show()

# Figure 2: Cross-spectrum and coherence
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Cross-spectrum magnitude
ax1 = axes[0]
ax1.loglog(f_star[freq_idx_start:], csd_magnitude_mean[freq_idx_start:],
           'o-', linewidth=1.5, markersize=3, color='#2ca02c', alpha=0.8)
ax1.set_ylabel("|S_τu| [signal product / Hz]", fontsize=11, fontweight='bold')
ax1.set_title(f"Cross-Spectrum Magnitude: <|S_τu(f, z)|>_z (z-averaged over Nz={nz})",
              fontsize=12, fontweight='bold')
ax1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)

# Subplot 2: Coherence
ax2 = axes[1]
ax2.semilogx(f_star[freq_idx_start:], coherence_mean[freq_idx_start:],
             'o-', linewidth=1.5, markersize=3, color='#ff7f0e', alpha=0.8)
ax2.set_ylim([0, 1.05])
ax2.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=11, fontweight='bold')
ax2.set_ylabel("γ²(f) [-]", fontsize=11, fontweight='bold')
ax2.set_title(f"Coherence (z-averaged over Nz={nz})", fontsize=12, fontweight='bold')
ax2.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)

fig.suptitle("Cross-Spectrum and Coherence (z-averaged)", fontsize=13, fontweight='bold', y=1.00)

plt.tight_layout()
print("Displaying cross-spectrum/coherence plot...")
plt.show()

print("Plot display complete!")

# ============================================================================
# PHASE 14: SAVE RESULTS TO HDF5
# ============================================================================

print("\n" + "="*70)
print("PHASE 14: SAVE RESULTS TO HDF5")
print("="*70)

output_file = os.path.join(SAVE_DIR, f"full_spanwise_spectral_{slice_id}.h5")
print(f"Saving to: {output_file}")

with h5py.File(output_file, 'w') as hf:
    # ==================================================
    # Group 1: Extracted Time Series Arrays
    # ==================================================
    ts_group = hf.create_group('timeseries')

    ts_group.create_dataset('tau_prime', data=tau_prime, compression='gzip')
    ts_group.attrs['tau_prime_units'] = '[Pa]'
    ts_group.attrs['tau_prime_shape'] = '(Nt, Nz)'

    ts_group.create_dataset('u_prime', data=u_prime, compression='gzip')
    ts_group.attrs['u_prime_units'] = '[m/s]'
    ts_group.attrs['u_prime_shape'] = '(Nt, Nz)'

    ts_group.create_dataset('iterations', data=iterations)
    ts_group.create_dataset('time', data=time_steps)

    ts_group.attrs['Nt'] = Nt
    ts_group.attrs['Nz'] = nz
    ts_group.attrs['dt_iteration'] = dt_iteration
    ts_group.attrs['sampling_frequency'] = fs

    # ==================================================
    # Group 2: Frequency Information
    # ==================================================
    freq_group = hf.create_group('frequency')
    freq_group.create_dataset('f', data=f)
    freq_group.create_dataset('f_star', data=f_star)
    freq_group.attrs['U_inf'] = U_inf
    freq_group.attrs['c'] = c
    freq_group.attrs['f_star_formula'] = 'f * c / U_inf'
    freq_group.attrs['Nf'] = len(f)

    # ==================================================
    # Group 3: Z-Averaged Spectral Results
    # ==================================================
    spec_group = hf.create_group('spectral')

    spec_group.create_dataset('psd_tau_mean', data=psd_tau_mean, compression='gzip')
    spec_group.attrs['psd_tau_mean_units'] = '[signal²/Hz]'
    spec_group.attrs['psd_tau_mean_description'] = 'Mean PSD of tau_prime over z'

    spec_group.create_dataset('psd_u_mean', data=psd_u_mean, compression='gzip')
    spec_group.attrs['psd_u_mean_units'] = '[signal²/Hz]'
    spec_group.attrs['psd_u_mean_description'] = 'Mean PSD of u_prime over z'

    spec_group.create_dataset('csd_magnitude_mean', data=csd_magnitude_mean, compression='gzip')
    spec_group.attrs['csd_magnitude_mean_units'] = '[signal product / Hz]'
    spec_group.attrs['csd_magnitude_mean_description'] = (
        'Mean cross-spectrum magnitude over z: <|S_tau_u(f, z)|>_z'
    )

    spec_group.create_dataset('coherence_mean', data=coherence_mean, compression='gzip')
    spec_group.attrs['coherence_mean_units'] = '[-]'
    spec_group.attrs['coherence_mean_range'] = '[0, 1]'
    spec_group.attrs['coherence_mean_description'] = 'Mean magnitude-squared coherence over z'

    spec_group.attrs['nperseg'] = nperseg
    spec_group.attrs['noverlap'] = noverlap
    spec_group.attrs['window'] = WINDOW
    spec_group.attrs['Nz_averaged'] = nz

    # ==================================================
    # Group 4: All-Z Spectral Arrays (for reference)
    # ==================================================
    allz_group = hf.create_group('all_z_spectral')

    psd_group = allz_group.create_group('psd')
    psd_group.create_dataset('tau', data=psd_tau_all_z, compression='gzip')
    psd_group.create_dataset('u', data=psd_u_all_z, compression='gzip')

    csd_group = allz_group.create_group('csd')
    csd_group.create_dataset('magnitude', data=csd_mag_all_z, compression='gzip')

    coh_group = allz_group.create_group('coherence')
    coh_group.create_dataset('values', data=coherence_all_z, compression='gzip')

    # ==================================================
    # Root Level Metadata
    # ==================================================
    hf.attrs['description'] = 'Full-spanwise spectral analysis (minimal formulation)'
    hf.attrs['slice_id'] = slice_id
    hf.attrs['slice_x'] = slice_x
    hf.attrs['Nt'] = Nt
    hf.attrs['Nz'] = nz
    hf.attrs['Nf'] = len(f)
    hf.attrs['dt_iteration'] = dt_iteration
    hf.attrs['sampling_frequency'] = fs
    hf.attrs['AOA_deg'] = AOA_deg
    hf.attrs['AOA_rad'] = AOA_rad
    hf.attrs['Re_c'] = Re_c
    hf.attrs['surface_point_y'] = surface_y
    hf.attrs['probe_location_y'] = Y_PROBE_FIXED
    hf.attrs['temporal_spacing_uniform'] = is_uniform

print(f"✓ Data saved to: {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\nConfiguration:")
print(f"  Slice: {slice_id} (x={slice_x:.6f})")
print(f"  AOA: {AOA_deg}°")
print(f"  Time samples (Nt): {Nt}")
print(f"  Spanwise points (Nz): {nz}")
print(f"  Frequency points (Nf): {len(f)}")
print(f"  Total time span: {time_steps[-1] - time_steps[0]:.6f} s")
print(f"  Sampling frequency: {fs:.6f} Hz")
print(f"  Welch nperseg: {nperseg}")

print(f"\nProbes:")
print(f"  Surface: y={surface_y:.6e}")
print(f"  Velocity: y={probe_locations[1]['y_actual']:.6e}")

print(f"\nZ-Averaged Spectral Results:")
print(f"  PSD tau_prime peak: {np.max(psd_tau_mean):.6e} at f* = {f_star[np.argmax(psd_tau_mean)]:.6e}")
print(f"  PSD u_prime peak: {np.max(psd_u_mean):.6e} at f* = {f_star[np.argmax(psd_u_mean)]:.6e}")
print(f"  CSD magnitude peak: {np.max(csd_magnitude_mean):.6e} at f* = {f_star[np.argmax(csd_magnitude_mean)]:.6e}")
print(f"  Coherence peak: {np.max(coherence_mean):.6f} at f* = {f_star[np.argmax(coherence_mean)]:.6e}")
print(f"  Mean coherence: {np.mean(coherence_mean):.6f}")

print(f"\nOutput:")
print(f"  HDF5: {output_file}")
print(f"  Data structure:")
print(f"    /timeseries/: tau_prime(Nt,Nz), u_prime(Nt,Nz), time, iterations")
print(f"    /frequency/: f, f_star")
print(f"    /spectral/: psd_tau_mean, psd_u_mean, csd_magnitude_mean, coherence_mean")
print(f"    /all_z_spectral/: per-z arrays for detailed analysis")

print("="*70)
