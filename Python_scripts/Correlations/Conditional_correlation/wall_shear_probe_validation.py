import os
import re
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Configuration
# ============================================================================

BASE_SURFACE_DIR  = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/"
BATCH_PATTERN     = "batch_*"

MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/Probe_validation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

u_infty = 1.0
AOA     = 12        # degrees
AOA_rad = np.deg2rad(AOA)
c       = 1.0       # chord length

# --- Surface reference point ---
# Chord location (x/c) on suction side (upper surface)
X_C_REF = 0.5   # <- set the x/c of the reference surface point

# --- Velocity probe point ---
# Physical (x, y) coordinates; nearest grid node will be selected automatically
# Choose a point away from the surface (e.g. shear layer or freestream)
PROBE_X = -0.15   # <- set probe x coordinate
PROBE_Y = 0.35   # <- set probe y coordinate


# ============================================================================
# Load geometrical data
# ============================================================================
print("=" * 70)
print("TWO-POINT TEMPORAL CORRELATION: VALIDATION SCRIPT")
print("=" * 70)

print("\n" + "=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points    = f["interface_points"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

N_surf      = len(interface_points)
x_interface = interface_points[:, 0]
y_interface = interface_points[:, 1]

print(f"  Number of 2D interface points: {N_surf}")

# Split into suction (upper) and pressure (lower) sides
y_mean    = np.mean(y_interface)
upper_mask = y_interface > y_mean

# ============================================================================
# Find surface reference point at X_C_REF on suction side
# ============================================================================
print("\n" + "=" * 70)
print("IDENTIFYING SURFACE REFERENCE POINT")
print("=" * 70)

upper_indices          = np.where(upper_mask)[0]
distances_surf         = np.abs(x_interface[upper_indices] - X_C_REF)
closest_idx_in_upper   = np.argmin(distances_surf)
surf_global_idx        = upper_indices[closest_idx_in_upper]
surf_x_c_actual        = x_interface[surf_global_idx]
surf_y_actual          = y_interface[surf_global_idx]

print(f"  Target x/c       = {X_C_REF:.2f}")
print(f"  Found index      = {surf_global_idx}")
print(f"  Actual x/c       = {surf_x_c_actual:.4f}")
print(f"  Actual y         = {surf_y_actual:.4f}")

# ============================================================================
# Load mesh and find nearest velocity probe grid point
# ============================================================================
print("\n" + "=" * 70)
print("LOADING MESH AND FINDING VELOCITY PROBE POINT")
print("=" * 70)

loader = CompressedSnapshotLoader(MESH_FILE)

# Full 3D coordinates, trimming ghost planes
x_data = loader.x[1:-1, :, :]   # (Nz_phys, Ny, Nx)
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

Nz_phys     = x_data.shape[0]
midplane_idx = Nz_phys // 2

print(f"  Domain shape  : (Nz_phys={Nz_phys}, Ny={x_data.shape[1]}, Nx={x_data.shape[2]})")
print(f"  Midplane index: {midplane_idx}")

# Nearest-neighbour search in the midplane
x_mid  = x_data[midplane_idx, :, :]   # (Ny, Nx)
y_mid  = y_data[midplane_idx, :, :]
dist_2d = np.sqrt((x_mid - PROBE_X) ** 2 + (y_mid - PROBE_Y) ** 2)
flat_idx = np.argmin(dist_2d)
iy_probe, ix_probe = np.unravel_index(flat_idx, dist_2d.shape)

probe_x_actual = float(x_mid[iy_probe, ix_probe])
probe_y_actual = float(y_mid[iy_probe, ix_probe])
probe_dist     = float(dist_2d[iy_probe, ix_probe])

print(f"\n  Target probe   : ({PROBE_X:.4f}, {PROBE_Y:.4f})")
print(f"  Nearest node   : iy={iy_probe}, ix={ix_probe}")
print(f"  Actual coords  : ({probe_x_actual:.4f}, {probe_y_actual:.4f})")
print(f"  Distance       : {probe_dist:.5f}")

# Warn if the probe is too far from the requested location
if probe_dist > 0.05:
    print(f"  [WARNING] Nearest node is {probe_dist:.3f} away - consider adjusting PROBE_X/Y")

# ============================================================================
# Load mean velocity field for fluctuation computation
# ============================================================================
print("\n" + "=" * 70)
print("LOADING MEAN VELOCITY FIELD")
print("=" * 70)

fields_avg   = loader.load_snapshot_avg(LAST_SNAPSHOT_FILE)
avg_u_3d     = loader.reconstruct_field(fields_avg["avg_u"])  # (Nz_total, Ny, Nx)
avg_v_3d     = loader.reconstruct_field(fields_avg["avg_v"])

# Spanwise average -> 2D field, then rotate to streamwise direction
avg_u_2d = np.mean(avg_u_3d, axis=0)   # (Ny, Nx)
avg_v_2d = np.mean(avg_v_3d, axis=0)
avg_u_streamwise = avg_u_2d * np.cos(AOA_rad) + avg_v_2d * np.sin(AOA_rad)

u_mean_probe = float(avg_u_streamwise[iy_probe, ix_probe])
print(f"  Mean streamwise velocity at probe: {u_mean_probe:.4f}")

# ============================================================================
# Discover and match data files by numeric timestamp
# (mirrors mid_2.py: regex extraction + integer sort + explicit cross-match)
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR DATA FILES AND MATCHING BY TIMESTAMP")
print("=" * 70)

def extract_timestamp(filename):
    """Return the numeric iteration string from a filename.
    Handles both surface and snapshot naming conventions:
      surface_3d_NACA0012_..._6350000-COMP-DATA.h5  -> '6350000'
      3d_NACA0012_..._6350000-COMP-DATA.h5           -> '6350000'
    """
    basename = os.path.basename(filename)
    match = re.search(r'_(\d+)-COMP-DATA', basename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract timestamp from: {basename}")

# Collect all surface files across batches
batch_surface_dirs = sorted(glob(os.path.join(BASE_SURFACE_DIR, BATCH_PATTERN)))
all_surface_files  = []
for batch_dir in batch_surface_dirs:
    surface_dir = os.path.join(batch_dir, "Surface_data")
    if os.path.exists(surface_dir):
        all_surface_files.extend(sorted(glob(os.path.join(surface_dir, "surface_*.h5"))))

# Collect all snapshot files across batches
batch_snapshot_dirs = sorted(glob(os.path.join(BASE_SNAPSHOT_DIR, BATCH_PATTERN)))
all_snapshots_files = []
for batch_dir in batch_snapshot_dirs:
    if os.path.exists(batch_dir):
        all_snapshots_files.extend(sorted(glob(os.path.join(batch_dir, "*A.h5"))))

print(f"  Surface files found  : {len(all_surface_files)}")
print(f"  Snapshot files found : {len(all_snapshots_files)}")

# Build timestamp -> file dictionaries (last file wins if duplicates exist)
surface_by_ts  = {}
surface_dups   = []
for f in all_surface_files:
    ts = extract_timestamp(f)
    if ts in surface_by_ts:
        surface_dups.append((ts, f))
    surface_by_ts[ts] = f

snapshot_by_ts = {}
snapshot_dups  = []
for f in all_snapshots_files:
    ts = extract_timestamp(f)
    if ts in snapshot_by_ts:
        snapshot_dups.append((ts, f))
    snapshot_by_ts[ts] = f

if surface_dups:
    print(f"  [WARNING] {len(surface_dups)} duplicate surface timestamp(s) — kept last occurrence")
if snapshot_dups:
    print(f"  [WARNING] {len(snapshot_dups)} duplicate snapshot timestamp(s) — kept last occurrence")

# Intersect and sort by integer timestamp value
common_timestamps      = sorted(
    set(surface_by_ts.keys()) & set(snapshot_by_ts.keys()), key=int
)
matched_surface_files  = [surface_by_ts[ts]  for ts in common_timestamps]
matched_snapshot_files = [snapshot_by_ts[ts] for ts in common_timestamps]
n_snapshots            = len(common_timestamps)

# Report any files without a counterpart
unmatched_surf = set(surface_by_ts.keys())  - set(snapshot_by_ts.keys())
unmatched_snap = set(snapshot_by_ts.keys()) - set(surface_by_ts.keys())
if unmatched_surf:
    print(f"  [WARNING] {len(unmatched_surf)} surface file(s) have no matching snapshot")
if unmatched_snap:
    print(f"  [WARNING] {len(unmatched_snap)} snapshot file(s) have no matching surface file")

print(f"  Matched pairs        : {n_snapshots}  "
      f"(timestamps {common_timestamps[0]} … {common_timestamps[-1]})")

if n_snapshots == 0:
    raise RuntimeError("No matched timestamp pairs found!")


# ============================================================================
# Compute mean tau_w at the surface reference point
# Method: spanwise average first, then time average  (identical to mid_2.py)
#   tau_w_mean = < mean_z( tau_w(z, t) ) >_t
#   tau_w_rms  = sqrt( <mean_z(tau_w^2)>_t  -  tau_w_mean^2 )
# This is the mean that will be subtracted when building tau'_w(t) below,
# so it must be consistent with the reference script.
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING MEAN TAU_W AT SURFACE REFERENCE POINT (SPANWISE + TIME AVG)")
print("=" * 70)

tau_w_2d_sum    = 0.0   # accumulates spanwise-avg tau_w   over time
tau_w_2_2d_sum  = 0.0   # accumulates spanwise-avg tau_w^2 over time
n_valid_surf    = 0

for surface_file in matched_surface_files:
    try:
        with h5py.File(surface_file, "r") as f:
            tau_w_col = f["tau_w"][:, surf_global_idx]  # (Nz_phys,) — all z at this surface point
        tau_w_2d_sum   += np.mean(tau_w_col)        # spanwise avg of tau_w
        tau_w_2_2d_sum += np.mean(tau_w_col ** 2)   # spanwise avg of tau_w^2
        n_valid_surf   += 1
    except Exception as e:
        print(f"  [WARNING] {os.path.basename(surface_file)}: {e}")

tau_w_mean = tau_w_2d_sum   / n_valid_surf
tau_w_rms  = np.sqrt(tau_w_2_2d_sum / n_valid_surf - tau_w_mean ** 2)

print(f"  Loaded {n_valid_surf} surface files")
print(f"  tau_w_mean = {tau_w_mean:.6e}  (spanwise + time avg)")
print(f"  tau_w_rms  = {tau_w_rms:.6e}   (from <tau_w^2>_z,t - mean^2)")


# ============================================================================
# Extract temporal time series: tau'_w(t) and u'_probe(t)
# ============================================================================
print("\n" + "=" * 70)
print("EXTRACTING TIME SERIES")
print("=" * 70)

tau_prime_series = np.full(n_snapshots, np.nan)
u_prime_series   = np.full(n_snapshots, np.nan)

for snap_idx in range(n_snapshots):
    if (snap_idx + 1) % 50 == 0 or snap_idx == 0:
        print(f"  Snapshot {snap_idx + 1}/{n_snapshots}...", flush=True)

    try:
        # --- Surface: tau'_w at midplane ---
        with h5py.File(matched_surface_files[snap_idx], "r") as f:
            tau_w_inst = float(f["tau_w"][midplane_idx, surf_global_idx])
        tau_prime_series[snap_idx] = tau_w_inst - tau_w_mean

        # --- Velocity: u' at probe (midplane) ---
        fields_inst = loader.load_snapshot(matched_snapshot_files[snap_idx])
        u_inst_full = loader.reconstruct_field(fields_inst["u"])  # (Nz_total, Ny, Nx)
        v_inst_full = loader.reconstruct_field(fields_inst["v"])

        # Slice physical planes (trim ghosts), pick midplane and probe node
        u_at_probe = float(u_inst_full[1:-1][midplane_idx, iy_probe, ix_probe])
        v_at_probe = float(v_inst_full[1:-1][midplane_idx, iy_probe, ix_probe])

        # Rotate to streamwise direction
        u_stream_inst = u_at_probe * np.cos(AOA_rad) + v_at_probe * np.sin(AOA_rad)
        u_prime_series[snap_idx] = u_stream_inst - u_mean_probe

    except Exception as e:
        print(f"  [WARNING] Snapshot {snap_idx}: {e}")

# Drop any failed snapshots
valid            = ~(np.isnan(tau_prime_series) | np.isnan(u_prime_series))
tau_prime_series = tau_prime_series[valid]
u_prime_series   = u_prime_series[valid]
N                = len(tau_prime_series)

print(f"\n  Valid snapshots  : {N}/{n_snapshots}")
if N < 10:
    raise RuntimeError("Too few valid snapshots - check data files!")


# ============================================================================
# Compute temporal cross-correlation R(tau)
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING TEMPORAL CROSS-CORRELATION R(tau)")
print("=" * 70)

# Remove residual mean (should be ~0 but enforce it)
tau_prime_series -= np.mean(tau_prime_series)
u_prime_series   -= np.mean(u_prime_series)

sigma_tau = np.std(tau_prime_series)
sigma_u   = np.std(u_prime_series)

print(f"  N              = {N}")
print(f"  sigma_tau      = {sigma_tau:.6e}")
print(f"  sigma_u        = {sigma_u:.6e}")

# Normalize
tau_norm = tau_prime_series / sigma_tau
u_norm   = u_prime_series   / sigma_u

# Linear cross-correlation via zero-padded FFT
# R(tau) = (1/N) * sum_t tau'(t) * u'(t+tau) / (sigma_tau * sigma_u)
n_fft   = 2 * N - 1
Tau_fft = np.fft.rfft(tau_norm, n=n_fft)
U_fft   = np.fft.rfft(u_norm,   n=n_fft)
R_full  = np.fft.irfft(np.conj(Tau_fft) * U_fft, n=n_fft)[:n_fft] / N

# Rearrange from [0, N-1, -(N-1)] order to symmetric [-N+1, ..., 0, ..., N-1]
lags  = np.arange(-(N - 1), N)
R_tau = np.concatenate([R_full[N:], R_full[:N]])

# Key scalars
zero_lag_idx = N - 1
R_zero       = float(R_tau[zero_lag_idx])
peak_idx     = int(np.argmax(np.abs(R_tau)))
peak_lag     = int(lags[peak_idx])
peak_R       = float(R_tau[peak_idx])

print(f"\n  Zero-lag R(0)  = {R_zero:.4f}")
print(f"  Peak |R|       = {abs(peak_R):.4f}  at lag = {peak_lag}")

# Running zero-lag correlation for convergence check
running_R0 = np.cumsum(tau_norm * u_norm) / np.arange(1, N + 1)


# ============================================================================
# Save results to HDF5
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

output_name = (
    f"probe_validation"
    f"_xc{X_C_REF:.2f}"
    f"_probe_x{PROBE_X:.3f}_y{PROBE_Y:.3f}.h5"
)
output_path = os.path.join(OUTPUT_DIR, output_name)

with h5py.File(output_path, "w") as f:
    # --- Metadata ---
    f.attrs['X_C_REF']          = X_C_REF
    f.attrs['surf_x_c_actual']  = surf_x_c_actual
    f.attrs['surf_y_actual']    = surf_y_actual
    f.attrs['surf_global_idx']  = surf_global_idx
    f.attrs['probe_x_target']   = PROBE_X
    f.attrs['probe_y_target']   = PROBE_Y
    f.attrs['probe_x_actual']   = probe_x_actual
    f.attrs['probe_y_actual']   = probe_y_actual
    f.attrs['iy_probe']         = iy_probe
    f.attrs['ix_probe']         = ix_probe
    f.attrs['midplane_idx']     = midplane_idx
    f.attrs['N_snapshots']      = N
    f.attrs['tau_w_mean']       = tau_w_mean
    f.attrs['tau_w_rms']        = tau_w_rms
    f.attrs['sigma_tau']        = sigma_tau
    f.attrs['sigma_u']          = sigma_u
    f.attrs['R_zero_lag']       = R_zero
    f.attrs['peak_lag']         = peak_lag
    f.attrs['peak_R']           = peak_R

    # --- Time series ---
    f.create_dataset('tau_prime',  data=tau_prime_series, compression='gzip')
    f.create_dataset('u_prime',    data=u_prime_series,   compression='gzip')

    # --- Cross-correlation ---
    f.create_dataset('R_tau',      data=R_tau,            compression='gzip')
    f.create_dataset('lags',       data=lags,             compression='gzip')

    # --- Convergence ---
    f.create_dataset('running_R0', data=running_R0,       compression='gzip')

print(f"  Saved to: {output_path}")


# ============================================================================
# Plots
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

snapshot_idx = np.arange(N)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle(
    f"Two-point temporal correlation  |  "
    f"surface x/c={surf_x_c_actual:.3f}  "
    f"probe ({probe_x_actual:.3f}, {probe_y_actual:.3f})  "
    f"midplane z={midplane_idx}",
    fontsize=12, fontweight='bold'
)

# --- Panel 1: normalized time series on twin axes ---
ax   = axes[0]
ax_r = ax.twinx()
ax.plot(snapshot_idx, tau_norm,
        color='steelblue', lw=0.7, label=r"$\tau'_w / \sigma_\tau$  (surface)")
ax_r.plot(snapshot_idx, u_norm,
          color='tomato', lw=0.7, alpha=0.75, label=r"$u' / \sigma_u$  (probe)")
ax.set_xlabel('Snapshot index', fontsize=11)
ax.set_ylabel(r"$\tau'_w / \sigma_\tau$",  color='steelblue', fontsize=11)
ax_r.set_ylabel(r"$u' / \sigma_u$",        color='tomato',    fontsize=11)
ax.set_title('Normalized time series', fontsize=12)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# --- Panel 2: R(tau), showing only central lags for readability ---
ax         = axes[1]
lag_show   = min(N // 4, 200)        # show at most ±200 lags for clarity
lag_mask   = np.abs(lags) <= lag_show
ax.plot(lags[lag_mask], R_tau[lag_mask], color='navy', lw=1.2)
ax.axvline(peak_lag, color='red',   linestyle='--', lw=1.2,
           label=f'Peak lag = {peak_lag}  (R = {peak_R:.3f})')
ax.axvline(0,        color='grey',  linestyle=':',  lw=1.0)
ax.axhline(0,        color='k',     lw=0.5)
ax.set_xlabel('Lag (snapshot index)', fontsize=11)
ax.set_ylabel(r'$R(\tau)$',           fontsize=11)
ax.set_title(r'Temporal cross-correlation $R(\tau)$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel 3: running R(0) convergence ---
ax = axes[2]
ax.plot(snapshot_idx, running_R0, color='darkgreen', lw=1.2)
ax.axhline(running_R0[-1], color='darkgreen', linestyle='--', lw=1.0, alpha=0.6,
           label=f'Final R(0) = {running_R0[-1]:.4f}')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Snapshot index', fontsize=11)
ax.set_ylabel(r'Running $R(0)$', fontsize=11)
ax.set_title(r'Convergence of zero-lag correlation', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, output_name.replace('.h5', '_timeseries.png'))
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Time-series plot saved to: {plot_path}")


# --- Domain map: show where the two points are ---
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.scatter(x_interface[upper_mask],  y_interface[upper_mask],
            c='steelblue', s=2, alpha=0.5, label='Suction side')
ax2.scatter(x_interface[~upper_mask], y_interface[~upper_mask],
            c='tomato',    s=2, alpha=0.5, label='Pressure side')

ax2.scatter(surf_x_c_actual, surf_y_actual,
            c='green', s=220, marker='*', edgecolors='black', lw=1.5, zorder=5,
            label=f'Surface ref  x/c={surf_x_c_actual:.3f}')
ax2.scatter(probe_x_actual, probe_y_actual,
            c='orange', s=180, marker='^', edgecolors='black', lw=1.5, zorder=5,
            label=f'Velocity probe  ({probe_x_actual:.3f}, {probe_y_actual:.3f})')

ax2.plot([surf_x_c_actual, probe_x_actual],
         [surf_y_actual,   probe_y_actual],
         'k--', lw=1.0, alpha=0.4)

ax2.set_xlabel('x/c', fontsize=13)
ax2.set_ylabel('y/c', fontsize=13)
ax2.set_title('Reference surface point and velocity probe location', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

plt.tight_layout()
map_path = os.path.join(OUTPUT_DIR, output_name.replace('.h5', '_domain.png'))
plt.savefig(map_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Domain map saved to: {map_path}")


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print(f"  R(0)      = {R_zero:.4f}")
print(f"  Peak |R|  = {abs(peak_R):.4f}  at lag {peak_lag}")
print(f"  Output    : {OUTPUT_DIR}")
