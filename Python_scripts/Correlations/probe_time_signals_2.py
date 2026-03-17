"""
probe_time_signals_2.py
=======================
Extract time signals at user-defined probe locations and compute temporal
cross-correlations between surface tau'_w and domain u'.

PROBES
------
  Surface probes: list of x/c values on the suction (upper) side.
                  -> tau'_w(t) extracted at z_mid.

  Domain probes:  list of (x, y) physical coordinates.
                  -> u'_streamwise(t) extracted at the nearest grid node at z_mid.

CORRELATION METHOD
------------------
  Direct accumulation method (transparent computation):
  For each lag τ:
      R(τ) = sum(a[t] * b[t+τ]) / (N * σ_a * σ_b)
  
  Uses the same normalization as FFT method for identical results.
  This replaces the FFT-based method for more transparent computation.

SNAPSHOT ORDERING
-----------------
  Files are matched by numeric timestamp extracted with the regex
      _(\d+)-COMP-DATA
  and then sorted by int(timestamp) to guarantee chronological order.
  Duplicate timestamps trigger a warning; the last file encountered is kept.

OUTPUT
------
  An HDF5 file containing:
    /time_series/surf_<i>/tau_prime      - tau'_w time series for surface probe i
    /time_series/domain_<j>/u_prime      - u' time series for domain probe j
    /correlations/surf<i>_dom<j>/R_tau   - temporal cross-correlation
    /correlations/surf<i>_dom<j>/lags    - corresponding lag indices
    /correlations/surf<i>_dom<j>/running_R0
  plus metadata attributes for all probes and convergence scalars.
"""

import os
import re
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import local data loader
# ---------------------------------------------------------------------------
module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)
from data_loader_functions import CompressedSnapshotLoader


# ============================================================================
# CONFIGURATION  —  edit this section
# ============================================================================

# --- Data directories -------------------------------------------------------
BASE_SURFACE_DIR  = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/"
BATCH_PATTERN     = "batch_*"

# --- Geometry / mesh files --------------------------------------------------
MESH_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
    "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
)
LAST_SNAPSHOT_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
    "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
)
GEO_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
    "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
)

# --- Output -----------------------------------------------------------------
OUTPUT_DIR = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Probe_time_signals/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "probe_time_signals_2.h5")

# --- Physical parameters ----------------------------------------------------
u_infty = 1.0
AOA     = 12         # degrees
AOA_rad = np.deg2rad(AOA)
c       = 1.0        # chord length

# --- Probe definitions ------------------------------------------------------
# Surface probes: x/c values on the SUCTION (upper) side.
# The nearest surface grid point will be selected automatically.
SURFACE_PROBES = [
    0.5,   # x/c = 0.5
]

# Domain (velocity) probes: (x, y) physical coordinates.
# The nearest grid node in the midplane will be selected automatically.
DOMAIN_PROBES = [
    (-0.125, 0.7),   
    ( 0.37, 0.25),   
    ( 0.64, 0.35),
    ( 1.12, 0.5),
    ( 0.65, 0.16),
    ( 0.22, 0.11)   
]
# DOMAIN_PROBES = [
#     (-0.125, 0.7),
# ]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestamp(filename):
    """Return the numeric iteration string from a filename.

    Works for both surface and snapshot naming conventions:
      surface_3d_NACA0012_..._6350000-COMP-DATA.h5  ->  '6350000'
      3d_NACA0012_..._6350000-COMP-DATA.h5           ->  '6350000'
    """
    basename = os.path.basename(filename)
    match = re.search(r'_(\d+)-COMP-DATA', basename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract timestamp from: {basename}")


def match_files_by_timestamp(surface_dirs, snapshot_dirs, batch_pattern):
    """Discover, match and sort surface + snapshot files by integer timestamp."""
    # Collect surface files
    all_surface_files = []
    for bd in sorted(glob(os.path.join(surface_dirs, batch_pattern))):
        sd = os.path.join(bd, "Surface_data")
        if os.path.exists(sd):
            all_surface_files.extend(sorted(glob(os.path.join(sd, "surface_*.h5"))))

    # Collect snapshot files
    all_snapshot_files = []
    for bd in sorted(glob(os.path.join(snapshot_dirs, batch_pattern))):
        if os.path.exists(bd):
            all_snapshot_files.extend(sorted(glob(os.path.join(bd, "*A.h5"))))

    print(f"  Surface files found  : {len(all_surface_files)}")
    print(f"  Snapshot files found : {len(all_snapshot_files)}")

    # Build timestamp -> file dicts (last file wins on duplicate)
    surface_by_ts  = {}
    surface_dups   = 0
    for f in all_surface_files:
        ts = extract_timestamp(f)
        if ts in surface_by_ts:
            surface_dups += 1
        surface_by_ts[ts] = f

    snapshot_by_ts = {}
    snapshot_dups  = 0
    for f in all_snapshot_files:
        ts = extract_timestamp(f)
        if ts in snapshot_by_ts:
            snapshot_dups += 1
        snapshot_by_ts[ts] = f

    if surface_dups:
        print(f"  [WARNING] {surface_dups} duplicate surface timestamp(s) — kept last occurrence")
    if snapshot_dups:
        print(f"  [WARNING] {snapshot_dups} duplicate snapshot timestamp(s) — kept last occurrence")

    # Intersect timestamps and sort chronologically
    common_ts = sorted(
        set(surface_by_ts.keys()) & set(snapshot_by_ts.keys()), key=int
    )

    unmatched_surf = set(surface_by_ts.keys()) - set(snapshot_by_ts.keys())
    unmatched_snap = set(snapshot_by_ts.keys()) - set(surface_by_ts.keys())
    if unmatched_surf:
        print(f"  [WARNING] {len(unmatched_surf)} surface file(s) without matching snapshot")
    if unmatched_snap:
        print(f"  [WARNING] {len(unmatched_snap)} snapshot file(s) without matching surface file")

    matched_surf = [surface_by_ts[ts]  for ts in common_ts]
    matched_snap = [snapshot_by_ts[ts] for ts in common_ts]

    print(f"  Matched pairs        : {len(common_ts)}"
          f"  (timestamps {common_ts[0]} … {common_ts[-1]})")

    return matched_surf, matched_snap, common_ts


# ============================================================================
# STEP 1 — Load geometry and mesh
# ============================================================================
print("=" * 70)
print("PROBE TIME-SIGNAL EXTRACTION AND CORRELATION (Direct Method)")
print("=" * 70)

print("\n" + "=" * 70)
print("STEP 1  —  LOADING GEOMETRY AND MESH")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points    = f["interface_points"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

x_interface = interface_points[:, 0]
y_interface = interface_points[:, 1]
y_mean      = np.mean(y_interface)
upper_mask  = y_interface > y_mean
upper_indices = np.where(upper_mask)[0]

print(f"  Interface points : {len(interface_points)}")
print(f"  Upper surface    : {len(upper_indices)} points")

# Load mesh
loader = CompressedSnapshotLoader(MESH_FILE)
x_data = loader.x[1:-1, :, :]   # (Nz_phys, Ny, Nx)
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]
Nz_phys     = x_data.shape[0]
midplane_idx = Nz_phys // 2
print(f"  Domain shape     : (Nz_phys={Nz_phys}, Ny={x_data.shape[1]}, Nx={x_data.shape[2]})")
print(f"  Midplane index   : {midplane_idx}")


# ============================================================================
# STEP 2 — Resolve probe locations
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2  —  RESOLVING PROBE LOCATIONS")
print("=" * 70)

# --- Surface probes ---------------------------------------------------------
surf_probe_info = []   # list of dicts
print("\n  Surface probes (suction side):")
for xc_target in SURFACE_PROBES:
    dist        = np.abs(x_interface[upper_indices] - xc_target)
    idx_in_up   = np.argmin(dist)
    global_idx  = upper_indices[idx_in_up]
    xc_actual   = float(x_interface[global_idx])
    y_actual    = float(y_interface[global_idx])
    surf_probe_info.append({
        'xc_target'  : xc_target,
        'xc_actual'  : xc_actual,
        'y_actual'   : y_actual,
        'global_idx' : int(global_idx),
    })
    print(f"    x/c={xc_target:.2f}  ->  index={global_idx}, "
          f"actual x/c={xc_actual:.4f}, y={y_actual:.4f}")

# --- Domain (velocity) probes -----------------------------------------------
x_mid = x_data[midplane_idx, :, :]   # (Ny, Nx)
y_mid = y_data[midplane_idx, :, :]

domain_probe_info = []
print("\n  Domain probes (nearest grid node at midplane):")
for (px, py) in DOMAIN_PROBES:
    dist_2d  = np.sqrt((x_mid - px) ** 2 + (y_mid - py) ** 2)
    flat_idx = np.argmin(dist_2d)
    iy, ix   = np.unravel_index(flat_idx, dist_2d.shape)
    xa       = float(x_mid[iy, ix])
    ya       = float(y_mid[iy, ix])
    d        = float(dist_2d[iy, ix])
    if d > 0.05:
        print(f"    [WARNING] probe ({px:.3f}, {py:.3f}): nearest node is {d:.4f} away "
              f"— consider adjusting coordinates")
    domain_probe_info.append({
        'x_target' : px,
        'y_target' : py,
        'x_actual' : xa,
        'y_actual' : ya,
        'iy'       : int(iy),
        'ix'       : int(ix),
        'dist'     : d,
    })
    print(f"    ({px:.3f}, {py:.3f})  ->  iy={iy}, ix={ix}, "
          f"actual ({xa:.4f}, {ya:.4f}), dist={d:.5f}")

# --- Visualization of probe locations ---------------------------------------
print("\n  Creating probe location visualization...")
fig_probes, ax_probes = plt.subplots(figsize=(12, 8))

# Plot interface points
ax_probes.scatter(x_interface[upper_mask],  y_interface[upper_mask],
                 c='steelblue', s=3, alpha=0.5, label='Suction side (upper)')
ax_probes.scatter(x_interface[~upper_mask], y_interface[~upper_mask],
                 c='tomato', s=3, alpha=0.5, label='Pressure side (lower)')

# Plot surface probes
for k, info in enumerate(surf_probe_info):
    ax_probes.scatter(info['xc_actual'], info['y_actual'],
                     c='green', s=250, marker='*', edgecolors='black', lw=2,
                     zorder=10, label=f"Surface probe {k}: x/c={info['xc_actual']:.3f}")
    # Add text annotation
    ax_probes.annotate(f"S{k}", 
                      xy=(info['xc_actual'], info['y_actual']),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=9, fontweight='bold', color='darkgreen')

# Plot domain probes
for j, info in enumerate(domain_probe_info):
    ax_probes.scatter(info['x_actual'], info['y_actual'],
                     c='orange', s=180, marker='^', edgecolors='black', lw=2,
                     zorder=10, label=f"Domain probe {j}: ({info['x_actual']:.3f}, {info['y_actual']:.3f})")
    # Add text annotation
    ax_probes.annotate(f"D{j}", 
                      xy=(info['x_actual'], info['y_actual']),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=9, fontweight='bold', color='darkorange')

ax_probes.set_xlabel('x/c', fontsize=13)
ax_probes.set_ylabel('y/c', fontsize=13)
ax_probes.set_title('Resolved Probe Locations on Airfoil', fontsize=14, fontweight='bold')
ax_probes.legend(fontsize=9, loc='best', framealpha=0.9)
ax_probes.grid(True, alpha=0.3)
ax_probes.set_aspect('equal', adjustable='box')
plt.tight_layout()

plt.show()


# ============================================================================
# STEP 3 — Load mean velocity field
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3  —  LOADING MEAN VELOCITY FIELD")
print("=" * 70)

fields_avg  = loader.load_snapshot_avg(LAST_SNAPSHOT_FILE)
avg_u_3d    = loader.reconstruct_field(fields_avg["avg_u"])   # (Nz_total, Ny, Nx)
avg_v_3d    = loader.reconstruct_field(fields_avg["avg_v"])

# Spanwise average -> 2D, then rotate to streamwise direction
avg_u_2d          = np.mean(avg_u_3d, axis=0)
avg_v_2d          = np.mean(avg_v_3d, axis=0)
avg_u_streamwise  = avg_u_2d * np.cos(AOA_rad) + avg_v_2d * np.sin(AOA_rad)

for info in domain_probe_info:
    info['u_mean'] = float(avg_u_streamwise[info['iy'], info['ix']])
    print(f"  Mean streamwise u at ({info['x_target']:.3f}, {info['y_target']:.3f})"
          f"  =  {info['u_mean']:.4f}")


# ============================================================================
# STEP 4 — File discovery and timestamp matching
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4  —  FILE DISCOVERY AND TIMESTAMP MATCHING")
print("=" * 70)

matched_surf_files, matched_snap_files, common_ts = match_files_by_timestamp(
    BASE_SURFACE_DIR, BASE_SNAPSHOT_DIR, BATCH_PATTERN
)
n_snapshots = len(common_ts)

if n_snapshots == 0:
    raise RuntimeError("No matched file pairs found — check directories and BATCH_PATTERN.")


# ============================================================================
# STEP 5 — First pass: compute mean tau_w at each surface probe
#           Method: spanwise-average tau_w per snapshot, then time-average.
#           This is identical to the approach in wall_shear_correlations_mid_2.py.
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5  —  COMPUTING MEAN AND RMS OF TAU_W AT SURFACE PROBES")
print("=" * 70)

n_surf_probes   = len(surf_probe_info)
tau_w_sum       = np.zeros(n_surf_probes)   # sum of spanwise-avg tau_w
tau_w_sq_sum    = np.zeros(n_surf_probes)   # sum of spanwise-avg tau_w^2
n_valid_surf    = 0

print(f"  Loading {n_snapshots} surface files...")
for idx, sf in enumerate(matched_surf_files):
    if (idx + 1) % 100 == 0 or idx == 0:
        print(f"    {idx + 1}/{n_snapshots}...", flush=True)
    try:
        with h5py.File(sf, "r") as f:
            # tau_w : (Nz_phys, N_surf)
            for k, info in enumerate(surf_probe_info):
                col = f["tau_w"][:, info['global_idx']]   # (Nz_phys,)
                tau_w_sum[k]    += np.mean(col)
                tau_w_sq_sum[k] += np.mean(col ** 2)
        n_valid_surf += 1
    except Exception as e:
        print(f"  [WARNING] {os.path.basename(sf)}: {e}")

if n_valid_surf == 0:
    raise RuntimeError("Could not read any surface files.")

tau_w_mean_arr = tau_w_sum    / n_valid_surf           # (n_surf_probes,)
tau_w_rms_arr  = np.sqrt(
    np.maximum(tau_w_sq_sum / n_valid_surf - tau_w_mean_arr ** 2, 0.0)
)

for k, info in enumerate(surf_probe_info):
    info['tau_w_mean'] = float(tau_w_mean_arr[k])
    info['tau_w_rms']  = float(tau_w_rms_arr[k])
    print(f"  x/c={info['xc_actual']:.4f}  "
          f"tau_w_mean={info['tau_w_mean']:.4e}  "
          f"tau_w_rms={info['tau_w_rms']:.4e}")


# ============================================================================
# STEP 6 — Second pass: extract time series at z_mid
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6  —  EXTRACTING TIME SERIES (MIDPLANE Z_MID)")
print("=" * 70)

# Pre-allocate arrays: NaN so we can detect failed snapshots
n_dom_probes = len(domain_probe_info)
tau_prime_ts = np.full((n_surf_probes, n_snapshots), np.nan)   # tau'_w
u_prime_ts   = np.full((n_dom_probes,  n_snapshots), np.nan)   # u'

print(f"  Extracting signals for {n_snapshots} snapshots...")
print(f"    Surface probes : {n_surf_probes}")
print(f"    Domain probes  : {n_dom_probes}")

for snap_idx in range(n_snapshots):
    if (snap_idx + 1) % 50 == 0 or snap_idx == 0:
        print(f"    Snapshot {snap_idx + 1}/{n_snapshots}...", flush=True)

    # --- Surface: tau'_w at midplane ---
    try:
        with h5py.File(matched_surf_files[snap_idx], "r") as f:
            for k, info in enumerate(surf_probe_info):
                tau_inst = float(f["tau_w"][midplane_idx, info['global_idx']])
                tau_prime_ts[k, snap_idx] = tau_inst - info['tau_w_mean']
    except Exception as e:
        print(f"  [WARNING] Surface (snap {snap_idx}): {e}")

    # --- Velocity: u' at midplane, nearest grid node ---
    try:
        fields_inst = loader.load_snapshot(matched_snap_files[snap_idx])
        u_full      = loader.reconstruct_field(fields_inst["u"])   # (Nz_total, Ny, Nx)
        v_full      = loader.reconstruct_field(fields_inst["v"])

        # Trim ghost planes, select midplane
        u_phys = u_full[1:-1]   # (Nz_phys, Ny, Nx)
        v_phys = v_full[1:-1]

        for j, info in enumerate(domain_probe_info):
            u_inst = float(u_phys[midplane_idx, info['iy'], info['ix']])
            v_inst = float(v_phys[midplane_idx, info['iy'], info['ix']])
            u_stream = u_inst * np.cos(AOA_rad) + v_inst * np.sin(AOA_rad)
            u_prime_ts[j, snap_idx] = u_stream - info['u_mean']

    except Exception as e:
        print(f"  [WARNING] Velocity (snap {snap_idx}): {e}")

# Count valid (non-NaN) entries along time axis
valid_mask = ~(np.any(np.isnan(tau_prime_ts), axis=0) |
               np.any(np.isnan(u_prime_ts),   axis=0))
n_valid    = int(np.sum(valid_mask))
print(f"\n  Valid snapshots: {n_valid}/{n_snapshots}")

if n_valid < 10:
    raise RuntimeError("Too few valid snapshots — check data files.")

tau_prime_ts = tau_prime_ts[:, valid_mask]
u_prime_ts   = u_prime_ts[:,   valid_mask]
valid_ts     = [ts for ts, v in zip(common_ts, valid_mask) if v]


# ============================================================================
# STEP 7 — Compute zero-lag correlations (Direct Accumulation Method)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7  —  COMPUTING ZERO-LAG CORRELATIONS (Direct Accumulation)")
print("=" * 70)

# Normalize time series (remove mean and divide by RMS)
tau_prime_normalized = np.zeros_like(tau_prime_ts)
u_prime_normalized   = np.zeros_like(u_prime_ts)

for k in range(n_surf_probes):
    tau_mean = np.mean(tau_prime_ts[k])
    tau_std = np.std(tau_prime_ts[k])
    if tau_std > 1e-14:
        tau_prime_normalized[k] = (tau_prime_ts[k] - tau_mean) / tau_std
    else:
        tau_prime_normalized[k] = 0.0

for j in range(n_dom_probes):
    u_mean = np.mean(u_prime_ts[j])
    u_std = np.std(u_prime_ts[j])
    if u_std > 1e-14:
        u_prime_normalized[j] = (u_prime_ts[j] - u_mean) / u_std
    else:
        u_prime_normalized[j] = 0.0

# Compute correlations by accumulation (similar to wall_shear_correlations_mid_2.py)
corr_results = {}

for k in range(n_surf_probes):
    for j in range(n_dom_probes):
        # Accumulate product
        numerator = np.sum(tau_prime_normalized[k] * u_prime_normalized[j])
        
        # Normalize by N to get correlation coefficient
        R0 = numerator / n_valid
        
        # Compute running correlation for convergence check
        running_R0 = np.cumsum(tau_prime_normalized[k] * u_prime_normalized[j]) / np.arange(1, n_valid + 1)
        
        corr_results[(k, j)] = {
            'R0': float(R0),
            'running_R0': running_R0
        }
        
        print(f"  surf[{k}] (x/c={surf_probe_info[k]['xc_actual']:.3f})  x  "
              f"dom[{j}] ({domain_probe_info[j]['x_actual']:.3f}, {domain_probe_info[j]['y_actual']:.3f})"
              f"  ->  R(0)={R0:.4f}")


# ============================================================================
# STEP 8 — Save results to HDF5
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8  —  SAVING TO HDF5")
print("=" * 70)

with h5py.File(OUTPUT_FILE, "w") as hf:

    # Global metadata
    hf.attrs['AOA_deg']        = AOA
    hf.attrs['midplane_idx']   = midplane_idx
    hf.attrs['n_snapshots']    = n_valid
    hf.attrs['n_surf_probes']  = n_surf_probes
    hf.attrs['n_domain_probes']= n_dom_probes
    hf.attrs['first_timestamp']= valid_ts[0]
    hf.attrs['last_timestamp'] = valid_ts[-1]
    hf.attrs['correlation_method'] = 'direct_accumulation'

    # Time series group
    ts_grp = hf.create_group("time_series")

    for k, info in enumerate(surf_probe_info):
        sg = ts_grp.create_group(f"surf_{k}")
        sg.attrs['xc_target']   = info['xc_target']
        sg.attrs['xc_actual']   = info['xc_actual']
        sg.attrs['y_actual']    = info['y_actual']
        sg.attrs['global_idx']  = info['global_idx']
        sg.attrs['tau_w_mean']  = info['tau_w_mean']
        sg.attrs['tau_w_rms']   = info['tau_w_rms']
        sg.create_dataset('tau_prime', data=tau_prime_ts[k], compression='gzip')

    for j, info in enumerate(domain_probe_info):
        dg = ts_grp.create_group(f"domain_{j}")
        dg.attrs['x_target']  = info['x_target']
        dg.attrs['y_target']  = info['y_target']
        dg.attrs['x_actual']  = info['x_actual']
        dg.attrs['y_actual']  = info['y_actual']
        dg.attrs['iy']        = info['iy']
        dg.attrs['ix']        = info['ix']
        dg.attrs['u_mean']    = info['u_mean']
        dg.create_dataset('u_prime', data=u_prime_ts[j], compression='gzip')

    # Cross-correlations group
    cc_grp = hf.create_group("correlations")

    for (k, j), res in corr_results.items():
        key = f"surf{k}_dom{j}"
        cg  = cc_grp.create_group(key)
        cg.attrs['surf_idx']   = k
        cg.attrs['dom_idx']    = j
        cg.attrs['R0']         = res['R0']
        cg.create_dataset('running_R0',  data=res['running_R0'],  compression='gzip')

print(f"  Saved: {OUTPUT_FILE}")


# ============================================================================
# STEP 9 — Summary plots
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9  —  GENERATING SUMMARY PLOTS")
print("=" * 70)

snap_idx_arr = np.arange(n_valid)

# ----- Plot A: Convergence of R(0) for all probe pairs ---------------------
n_pairs  = n_surf_probes * n_dom_probes
fig, axes = plt.subplots(n_pairs, 1,
                         figsize=(14, 3 * n_pairs),
                         squeeze=False)

pair_row = 0
for k, sinfo in enumerate(surf_probe_info):
    for j, dinfo in enumerate(domain_probe_info):
        res   = corr_results[(k, j)]
        ax    = axes[pair_row, 0]
        running_R0 = res['running_R0']

        ax.plot(snap_idx_arr, running_R0, color='navy', lw=1.2, label='Running R(0)')
        ax.axhline(res['R0'], color='red', linestyle='--', lw=1.2,
                   label=f"Final R(0)={res['R0']:.4f}")
        ax.axhline(0, color='k',    lw=0.5)
        ax.set_ylabel(r'$R(0)$', fontsize=10)
        ax.set_title(
            rf"surf[{k}] x/c={sinfo['xc_actual']:.3f}  $\times$  "
            rf"dom[{j}] ({dinfo['x_actual']:.3f}, {dinfo['y_actual']:.3f})",
            fontsize=10
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        pair_row += 1

axes[-1, 0].set_xlabel('Snapshot index', fontsize=11)
fig.suptitle('Zero-lag correlation convergence (Direct Accumulation Method)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ----- Plot B: Time series of one selected pair (k=0, j=0) -----------------
k0, j0   = 0, 0
tau_sig  = tau_prime_ts[k0] / surf_probe_info[k0]['tau_w_rms']
u_sig    = u_prime_ts[j0]   / np.std(u_prime_ts[j0])

fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2r = ax2.twinx()
ax2.plot(snap_idx_arr, tau_sig,
         color='steelblue', lw=0.7,
         label=rf"$\tau'_w / \tau_{{rms}}$  surf[{k0}] x/c={surf_probe_info[k0]['xc_actual']:.3f}")
ax2r.plot(snap_idx_arr, u_sig,
          color='tomato', lw=0.7, alpha=0.8,
          label=rf"$u'/\sigma_u$  dom[{j0}] ({domain_probe_info[j0]['x_actual']:.3f}, "
                rf"{domain_probe_info[j0]['y_actual']:.3f})")
ax2.set_xlabel('Snapshot index', fontsize=11)
ax2.set_ylabel(r"$\tau'_w / \tau_{rms}$", color='steelblue', fontsize=11)
ax2r.set_ylabel(r"$u'/\sigma_u$",          color='tomato',    fontsize=11)
ax2.set_title('Normalised time series — probe pair [0, 0]', fontsize=12)
lines1, lab1 = ax2.get_legend_handles_labels()
lines2, lab2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, lab1 + lab2, loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()

# ----- Plot C: domain map — all probe locations -------------------------
fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.scatter(x_interface[upper_mask],  y_interface[upper_mask],
            c='steelblue', s=2, alpha=0.4, label='Suction side')
ax3.scatter(x_interface[~upper_mask], y_interface[~upper_mask],
            c='tomato',    s=2, alpha=0.4, label='Pressure side')

for k, info in enumerate(surf_probe_info):
    ax3.scatter(info['xc_actual'], info['y_actual'],
                c='green', s=200, marker='*', edgecolors='black', lw=1.5,
                zorder=5, label=f"surf[{k}] x/c={info['xc_actual']:.3f}" if k == 0
                                else f"surf[{k}] x/c={info['xc_actual']:.3f}")

for j, info in enumerate(domain_probe_info):
    ax3.scatter(info['x_actual'], info['y_actual'],
                c='orange', s=160, marker='^', edgecolors='black', lw=1.5,
                zorder=5, label=f"dom[{j}] ({info['x_actual']:.3f}, {info['y_actual']:.3f})"
                                if j == 0 else
                                f"dom[{j}] ({info['x_actual']:.3f}, {info['y_actual']:.3f})")

ax3.set_xlabel('x/c', fontsize=13)
ax3.set_ylabel('y/c', fontsize=13)
ax3.set_title('Probe locations', fontsize=13)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"  HDF5 output : {OUTPUT_FILE}")
print(f"  Plots       : {OUTPUT_DIR}")
