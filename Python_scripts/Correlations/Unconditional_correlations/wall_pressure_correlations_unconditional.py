import os
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Configuration
# ============================================================================

# Data directories
BASE_SURFACE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/"

# Pattern to match batch directories
BATCH_PATTERN = "batch_*"

# Mesh data file
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Averaged snapshot file (for mean fields)
LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_pressure_correlations/test_4/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Analysis Parameters
# ============================================================================

u_infty = 1.0
AOA = 12  # degrees
AOA_rad = np.deg2rad(AOA)
c = 1.0  # chord length

# Chord locations for correlation analysis (x/c values)
# X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
X_C_LOCATIONS = [0.3, 0.5, 0.7, 0.9]
# X_C_LOCATIONS = [0.5]


# ============================================================================
# Load Geometrical Data
# ============================================================================
print("=" * 70)
print("UNCONDITIONAL WALL PRESSURE CORRELATION ANALYSIS")
print("=" * 70)
print(f"\nAnalysis configuration:")
print(f"  Chord locations (x/c): {X_C_LOCATIONS}")
print(f"  Suction side (upper surface): closest point selected")
print(f"  Using total pressure: Ptotal = p_surface + p_bulk")
print(f"  Analysis type: Unconditional (all samples, no event classification)")

print("\n" + "=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

N_surf = len(interface_points)
print(f"  Number of 2D interface points: {N_surf}")

# Extract coordinates
x_interface = interface_points[:, 0]
y_interface = interface_points[:, 1]
x_over_c = x_interface

# Separate upper and lower surfaces
y_mean = np.mean(y_interface)
upper_mask = y_interface > y_mean
lower_mask = ~upper_mask

# ============================================================================
# Find Points at Specified Chord Locations
# ============================================================================
print("\n" + "=" * 70)
print("IDENTIFYING CLOSEST POINT ON SUCTION SIDE")
print("=" * 70)

point_indices = {}

for x_c_target in X_C_LOCATIONS:
    # Find points on upper surface (suction side)
    upper_indices = np.where(upper_mask)[0]

    if len(upper_indices) == 0:
        print(f"  x/c = {x_c_target:.2f}: No points on upper surface!")
        continue

    # Find the closest point to target x/c
    distances = np.abs(x_over_c[upper_indices] - x_c_target)
    closest_idx_in_upper = np.argmin(distances)
    closest_global_idx = upper_indices[closest_idx_in_upper]

    actual_x_c = x_over_c[closest_global_idx]
    actual_y = y_interface[closest_global_idx]

    point_indices[x_c_target] = {
        'indices': np.array([closest_global_idx]),
        'x_c_actual': actual_x_c,
        'y': actual_y,
    }

    print(f"  x/c = {x_c_target:.2f}: index {closest_global_idx} at actual x/c = {actual_x_c:.4f}")

if len(point_indices) == 0:
    raise RuntimeError("No points found at any specified chord locations!")

# ============================================================================
# Find All Surface Data Files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR SURFACE DATA FILES")
print("=" * 70)

batch_surface_dirs = sorted(glob(os.path.join(BASE_SURFACE_DIR, BATCH_PATTERN)))
print(f"  Found {len(batch_surface_dirs)} batch directories")

all_surface_files = []
for batch_dir in batch_surface_dirs:
    surface_dir = os.path.join(batch_dir, "Surface_data")
    if not os.path.exists(surface_dir):
        continue

    surface_files = sorted(glob(os.path.join(surface_dir, "surface_*.h5")))
    all_surface_files.extend(surface_files)

N_total_surf = len(all_surface_files)
print(f"  Total surface data files: {N_total_surf}")

if N_total_surf == 0:
    raise RuntimeError("No surface data files found!")

# ============================================================================
# Find All Snapshots Data Files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR SNAPSHOTS DATA FILES")
print("=" * 70)

batch_snapshot_dirs = sorted(glob(os.path.join(BASE_SNAPSHOT_DIR, BATCH_PATTERN)))
print(f"  Found {len(batch_snapshot_dirs)} batch directories")

all_snapshots_files = []
for batch_dir in batch_snapshot_dirs:
    if not os.path.exists(batch_dir):
        continue

    snapshot_files = sorted(glob(os.path.join(batch_dir, "*A.h5")))
    all_snapshots_files.extend(snapshot_files)

N_total_snapshots = len(all_snapshots_files)
print(f"  Total snapshots data files: {N_total_snapshots}")

if N_total_snapshots == 0:
    raise RuntimeError("No snapshots data files found!")

print(f"  NOTE: Both surface and snapshot lists must correspond to the same times")

# ============================================================================
# Compute Mean Total Pressure from All Surface Data Snapshots
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING <PTOTAL> FROM ALL SURFACE DATA SNAPSHOTS")
print("=" * 70)

ptotal_2d_sum = None
ptotal_2_2d_sum = None
n_snapshots = 0

print(f"Loading {N_total_surf} surface snapshots to compute mean...")

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Loading surface snapshot {idx+1}/{N_total_surf}...", flush=True)

    try:
        with h5py.File(surface_file, "r") as f:
            p_w = f["p_w"][:]
            p_bulk = f.attrs["p_bulk"]

            # Compute total pressure: Ptotal = p_wall + p_bulk
            ptotal = p_w + p_bulk  # (Nz_phys, N_surf)
            ptotal_2 = ptotal * ptotal

            # Spanwise average for each snapshot
            ptotal_2d = np.mean(ptotal, axis=0)       # (N_surf,)
            ptotal_2_2d = np.mean(ptotal_2, axis=0)   # (N_surf,)

            if ptotal_2d_sum is None:
                ptotal_2d_sum = ptotal_2d.copy()
                ptotal_2_2d_sum = ptotal_2_2d.copy()
                Nz_phys = ptotal.shape[0]
            else:
                ptotal_2d_sum += ptotal_2d
                ptotal_2_2d_sum += ptotal_2_2d

            n_snapshots += 1

    except Exception as e:
        print(f"  [WARNING] Error loading {surface_file}: {e}")
        continue

if n_snapshots == 0:
    raise RuntimeError("No valid snapshots loaded; check surface files and datasets.")

# Ensure we don't exceed available snapshot files
n_snapshots = min(n_snapshots, N_total_snapshots)

# Compute 2D time-averaged means
ptotal_mean = ptotal_2d_sum / n_snapshots  # (N_surf,)
ptotal_2_mean = ptotal_2_2d_sum / n_snapshots  # (N_surf,)

print(f"  Successfully loaded {n_snapshots} surface snapshots")
print(f"  2D mean shape: (N_surf={len(ptotal_mean)})")
print(f"  Spanwise planes in each snapshot: Nz={Nz_phys}")

# ============================================================================
# Collect Surface Data at Each Chord Location
# ============================================================================
print("\n" + "=" * 70)
print("COLLECTING SURFACE DATA")
print("=" * 70)

surface_signals = {}

for x_c_target in point_indices.keys():
    surface_signals[x_c_target] = {
        'p_total_prime': []
    }

print(f"Processing {n_snapshots} snapshots for surface data...")

for idx, surface_file in enumerate(all_surface_files[:n_snapshots]):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Processing snapshot {idx+1}/{n_snapshots}...", flush=True)

    try:
        with h5py.File(surface_file, "r") as f:
            p_w = f["p_w"][:]
            p_bulk = f.attrs["p_bulk"]

        # Compute total pressure
        ptotal = p_w + p_bulk  # (Nz_phys, N_surf)

        # Extract at each chord location
        for x_c_target, point_info in point_indices.items():
            idx_point = point_info['indices'][0]

            # Extract surface values at all z-locations for this point
            ptotal_at_xc = ptotal[:, idx_point]  # (Nz_phys,)

            # Compute fluctuations
            ptotal_at_xc_fluct = ptotal_at_xc - ptotal_mean[idx_point]
            surface_signals[x_c_target]['p_total_prime'].extend(ptotal_at_xc_fluct)

    except Exception as e:
        print(f"  [WARNING] Error processing {surface_file}: {e}")
        continue

# ============================================================================
# Load Mean Velocity Field
# ============================================================================
print("\n" + "=" * 70)
print("LOADING MEAN VELOCITY FIELD")
print("=" * 70)

loader = CompressedSnapshotLoader(MESH_FILE)
fields = loader.load_snapshot_avg(LAST_SNAPSHOT_FILE)

# Coordinates
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print("x_data shape:", x_data.shape)
print("y_data shape:", y_data.shape)
print("z_data shape:", z_data.shape)

# Mean velocities
avg_u_data = loader.reconstruct_field(fields["avg_u"])
avg_v_data = loader.reconstruct_field(fields["avg_v"])
avg_w_data = loader.reconstruct_field(fields["avg_w"])

print("reconstructed avg_u shape:", avg_u_data.shape)

# Average in spanwise direction (axis=0) to get 2D mean field
avg_u_data = np.mean(avg_u_data, axis=0)
avg_v_data = np.mean(avg_v_data, axis=0)

# Compute streamwise velocity (aligned with angle of attack)
# V_streamwise = u*cos(AOA) + v*sin(AOA)
avg_u_data = avg_u_data * np.cos(AOA_rad) + avg_v_data * np.sin(AOA_rad)

print("avg_u_data shape (streamwise):", avg_u_data.shape)

# ============================================================================
# Compute p_total_rms at Reference Points
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING P_TOTAL_RMS AT REFERENCE POINTS")
print("=" * 70)

p_total_rms_dict = {}
for x_c_target, point_info in point_indices.items():
    idx_point = point_info['indices'][0]

    # p_total_rms = sqrt( <p_total^2> - <p_total>^2 )
    p_total_rms = np.sqrt(ptotal_2_mean[idx_point] - ptotal_mean[idx_point]**2)
    p_total_rms_dict[x_c_target] = p_total_rms

    ptotal_mean_val = ptotal_mean[idx_point]

    print(f"  x/c = {x_c_target:.2f}:")
    print(f"    p_total_mean = {ptotal_mean_val:.6e}")
    print(f"    p_total_rms  = {p_total_rms:.6e}")

# ============================================================================
# Define Correlation Spatial Window
# ============================================================================
print("\n" + "=" * 70)
print("DEFINING CORRELATION SPATIAL WINDOW FOR EACH X/C LOCATION")
print("=" * 70)

# Determine domain extents
x_min_domain, x_max_domain = np.min(x_data), np.max(x_data)
y_min_domain, y_max_domain = np.min(y_data), np.max(y_data)

print(f"  Full domain: x=[{x_min_domain:.3f}, {x_max_domain:.3f}], y=[{y_min_domain:.3f}, {y_max_domain:.3f}]")

# Define window parameters (relative to reference point)
dx_upstream = 0.75
dx_downstream = 0.75
dy_down = 0.05
dy_up = 0.5

# Get 2D grid for index finding
x_2d = x_data[0, :, :]
y_2d = y_data[0, :, :]
x_1d = x_2d[0, :]
y_1d = y_2d[:, 0]

# Store cropped window info for each x/c location
crop_windows = {}

for x_c_target, point_info in point_indices.items():
    x_ref = point_info['x_c_actual']
    y_ref = point_info['y']

    print(f"\n  x/c = {x_c_target:.2f} (actual: {x_ref:.4f}, y = {y_ref:.4f}):")

    # Define window bounds
    x_min_crop = x_ref - dx_upstream
    x_max_crop = x_ref + dx_downstream
    y_min_crop = max(y_ref - dy_down, y_min_domain)
    y_max_crop = y_ref + dy_up

    # Find indices for cropped region
    ix_min = np.argmin(np.abs(x_1d - x_min_crop))
    ix_max = np.argmin(np.abs(x_1d - x_max_crop))
    iy_min = np.argmin(np.abs(y_1d - y_min_crop))
    iy_max = np.argmin(np.abs(y_1d - y_max_crop))

    # Store for this location
    crop_windows[x_c_target] = {
        'ix_min': ix_min,
        'ix_max': ix_max,
        'iy_min': iy_min,
        'iy_max': iy_max,
        'Nx_crop': ix_max - ix_min,
        'Ny_crop': iy_max - iy_min,
        'Nz_crop': Nz_phys,
    }

    print(f"    Window: x=[{x_min_crop:.3f}, {x_max_crop:.3f}], y=[{y_min_crop:.3f}, {y_max_crop:.3f}]")
    print(f"    Indices: ix=[{ix_min}:{ix_max}], iy=[{iy_min}:{iy_max}]")
    print(f"    Shape: (Nz={Nz_phys}, Ny={crop_windows[x_c_target]['Ny_crop']}, Nx={crop_windows[x_c_target]['Nx_crop']})")

# Visualization of the domain and reference point
print("\n  Creating visualization of correlation domain...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot interface points (airfoil surface)
ax.scatter(x_interface[upper_mask], y_interface[upper_mask],
           c='blue', s=2, alpha=0.5, label='Upper surface')
ax.scatter(x_interface[lower_mask], y_interface[lower_mask],
           c='red', s=2, alpha=0.5, label='Lower surface')

# Highlight reference point (use last x_c_target for visualization)
ax.scatter(x_ref, y_ref, c='green', s=200, marker='*',
           edgecolors='black', linewidths=2, label=f'Reference point (x/c={x_ref:.3f})', zorder=5)

# Draw correlation window as rectangle
rect = patches.Rectangle((x_min_crop, y_min_crop),
                          x_max_crop - x_min_crop,
                          y_max_crop - y_min_crop,
                          linewidth=3, edgecolor='green', facecolor='green',
                          alpha=0.2, label='Correlation window')
ax.add_patch(rect)

# Draw reference lines
ax.axvline(x_ref, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y_ref, color='green', linestyle='--', linewidth=1, alpha=0.5)

# Labels and formatting
ax.set_xlabel('x/c', fontsize=14)
ax.set_ylabel('y/c', fontsize=14)
ax.set_title(f'Correlation Domain for x/c = {x_c_target:.2f}', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(min(x_min_crop - 0.1, x_min_domain), max(x_max_crop + 0.1, x_max_domain))
ax.set_ylim(min(y_min_crop - 0.05, y_min_domain), max(y_max_crop + 0.05, y_max_domain))

# plot_path = os.path.join(OUTPUT_DIR, f'correlation_domain_xc_{x_c_target:.3f}.png')
plt.tight_layout()
# plt.savefig(plot_path)
plt.show()

# ============================================================================
# Initialize Accumulation Arrays for Unconditional Correlation
# ============================================================================
print("\n" + "=" * 70)
print("INITIALIZING CORRELATION ACCUMULATION ARRAYS")
print("=" * 70)

correlation_data = {}

for x_c_target in point_indices.keys():
    crop_info = crop_windows[x_c_target]
    Nz_crop = crop_info['Nz_crop']
    Ny_crop = crop_info['Ny_crop']
    Nx_crop = crop_info['Nx_crop']

    correlation_data[x_c_target] = {
        'numerator': np.zeros((Nz_crop, Ny_crop, Nx_crop), dtype=np.float64),
        'u_prime_sq': np.zeros((Nz_crop, Ny_crop, Nx_crop), dtype=np.float64),
        'N_samples': 0,
        'p_total_prime_sq_sum': 0.0,
    }

    print(f"  x/c = {x_c_target:.2f}: shape (Nz={Nz_crop}, Ny={Ny_crop}, Nx={Nx_crop})")

print(f"\nInitialized accumulation arrays for {len(point_indices)} chord locations")

# ============================================================================
# Load Instantaneous Velocity Fields and Compute Correlation
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING UNCONDITIONAL CORRELATION")
print("=" * 70)

# Unconditional correlation: normalized correlation between total wall-pressure
# fluctuation at the reference point and streamwise velocity fluctuation in the
# cropped 3D domain. Uses all snapshots and all spanwise reference planes.
# FFT-based circular correlation exploits the periodic z direction.

print(f"Processing {n_snapshots} snapshots...")
print(f"  Snapshot files and surface files must align in order!")
print(f"  Exploiting spanwise periodicity: using all {Nz_phys} z-planes as reference points")
print(f"  -> z-axis of the result represents relative separation Dz (Dz=0 at index 0)")

for snap_idx in range(n_snapshots):
    if (snap_idx + 1) % 10 == 0 or snap_idx == 0:
        print(f"\n  Snapshot {snap_idx+1}/{n_snapshots}...", flush=True)

    snapshot_file = all_snapshots_files[snap_idx]

    try:
        # Load instantaneous velocity fields
        fields_inst = loader.load_snapshot(snapshot_file)
        u_inst_full = loader.reconstruct_field(fields_inst["u"])  # (Nz, Ny, Nx)
        v_inst_full = loader.reconstruct_field(fields_inst["v"])  # (Nz, Ny, Nx)

        # Compute streamwise velocity
        u_inst_full = u_inst_full * np.cos(AOA_rad) + v_inst_full * np.sin(AOA_rad)

        # Process each chord location
        for x_c_target, point_info in point_indices.items():
            crop_info = crop_windows[x_c_target]
            ix_min = crop_info['ix_min']
            ix_max = crop_info['ix_max']
            iy_min = crop_info['iy_min']
            iy_max = crop_info['iy_max']

            # Crop to correlation window
            u_inst = u_inst_full[1:-1, iy_min:iy_max, ix_min:ix_max]  # (Nz_phys, Ny_crop, Nx_crop)

            # Compute fluctuation: u' = u - <u>
            avg_u_crop = avg_u_data[iy_min:iy_max, ix_min:ix_max]  # (Ny_crop, Nx_crop)
            u_prime = u_inst - avg_u_crop[np.newaxis, :, :]  # (Nz_phys, Ny_crop, Nx_crop)

            # Create mask for valid points (not NaN - excludes inside airfoil)
            valid_mask = ~np.isnan(u_prime)

            # Replace NaN values with 0
            u_prime = np.where(valid_mask, u_prime, 0.0)

            # Extract p'_total for this snapshot at all z-planes
            p_total_prime_current = np.array(surface_signals[x_c_target]['p_total_prime'][
                snap_idx * Nz_phys : (snap_idx + 1) * Nz_phys
            ])  # (Nz_phys,)

            # ---------------------------------------------------------------
            # FFT-based circular cross-correlation along z (periodic)
            #
            # For each z-plane k:
            #   p'_total[k] * u'[(k + Dz) % Nz, y, x]
            # Summed over k:
            #   C[Dz, y, x] = IFFT( conj(FFT(p_total)) * FFT(u') )  along z
            #
            # For u'^2 normalization, the indicator function selects all planes:
            #   U2[Dz, y, x] = IFFT( conj(FFT(ones)) * FFT(u'^2) )
            # ---------------------------------------------------------------

            u_fft = np.fft.rfft(u_prime, axis=0)         # (Nz//2+1, Ny, Nx)
            u2_fft = np.fft.rfft(u_prime**2, axis=0)     # (Nz//2+1, Ny, Nx)

            # Compute cross-correlation numerator
            p_fft = np.fft.rfft(p_total_prime_current)   # (Nz//2+1,)
            numerator = np.fft.irfft(np.conj(p_fft[:, None, None]) * u_fft,
                                    n=Nz_phys, axis=0)

            # Compute u'^2 contribution with all-ones indicator
            indicator_ones_fft = np.fft.rfft(np.ones(Nz_phys))  # (Nz//2+1,)
            u_prime_sq = np.fft.irfft(np.conj(indicator_ones_fft[:, None, None]) * u2_fft,
                                     n=Nz_phys, axis=0)

            # Accumulate
            correlation_data[x_c_target]['numerator'] += numerator
            correlation_data[x_c_target]['u_prime_sq'] += u_prime_sq
            correlation_data[x_c_target]['N_samples'] += Nz_phys
            correlation_data[x_c_target]['p_total_prime_sq_sum'] += float(np.sum(p_total_prime_current**2))

    except Exception as e:
        print(f"  [WARNING] Error loading snapshot {snapshot_file}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "=" * 70)
print("CORRELATION ACCUMULATION COMPLETE")
print("=" * 70)

# ============================================================================
# Normalize to Compute Correlation Coefficients
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING NORMALIZED CORRELATION COEFFICIENTS")
print("=" * 70)

correlation_results = {}

for x_c_target, corr_data in correlation_data.items():
    print(f"\n  x/c = {x_c_target:.2f}:")

    N_samples = corr_data['N_samples']
    print(f"    Total samples: N_samples = {N_samples}")

    # Compute unconditional RMS of u' at each spatial location
    u_rms = np.sqrt(corr_data['u_prime_sq'] / N_samples)

    # Protect against NaN and zero
    u_rms = np.where(np.isnan(u_rms), 1e-12, u_rms)
    u_rms_safe = np.where(u_rms > 1e-12, u_rms, 1e-12)

    # Compute unconditional p_total_rms
    p_total_rms = np.sqrt(corr_data['p_total_prime_sq_sum'] / N_samples)

    print(f"    p_total_rms = {p_total_rms:.6e}")

    # Replace NaN in numerator with 0
    numerator = np.where(np.isnan(corr_data['numerator']), 0.0, corr_data['numerator'])

    # Compute normalized correlation coefficient
    R = numerator / (N_samples * p_total_rms * u_rms_safe)

    # Replace remaining NaN and Inf with 0
    R = np.where(np.isnan(R) | np.isinf(R), 0.0, R)

    # Store results
    correlation_results[x_c_target] = {
        'R': R,
        'u_rms': u_rms,
        'p_total_rms': p_total_rms,
        'N_samples': N_samples,
    }

    # Report peak correlation
    R_max = np.max(np.abs(R))
    print(f"    Peak |R| = {R_max:.4f}")

# ============================================================================
# Save Results to HDF5
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS TO HDF5")
print("=" * 70)

for x_c_target, results in correlation_results.items():
    point_info = point_indices[x_c_target]
    idx_point = point_info['indices'][0]

    crop_info = crop_windows[x_c_target]
    ix_min = crop_info['ix_min']
    ix_max = crop_info['ix_max']
    iy_min = crop_info['iy_min']
    iy_max = crop_info['iy_max']

    output_filename = f"wall_pressure_correlation_unconditional_xc_{x_c_target:.3f}.h5"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n  Saving x/c = {x_c_target:.2f} to {output_filename}")

    with h5py.File(output_path, "w") as f:
        # Metadata
        f.attrs['x_c_target'] = x_c_target
        f.attrs['x_c_actual'] = point_info['x_c_actual']
        f.attrs['y_actual'] = point_info['y']
        f.attrs['idx_point'] = idx_point
        f.attrs['p_total_mean'] = ptotal_mean[idx_point]
        f.attrs['p_total_rms'] = results['p_total_rms']
        f.attrs['N_snapshots'] = n_snapshots
        f.attrs['N_samples'] = results['N_samples']
        f.attrs['ix_min'] = ix_min
        f.attrs['ix_max'] = ix_max
        f.attrs['iy_min'] = iy_min
        f.attrs['iy_max'] = iy_max

        # Correlation field and RMS
        f.create_dataset('R', data=results['R'], compression='gzip')
        f.create_dataset('u_rms', data=results['u_rms'], compression='gzip')

        # Grid coordinates (cropped for this x/c location, all z-planes)
        x_crop = x_data[:, iy_min:iy_max, ix_min:ix_max]
        y_crop = y_data[:, iy_min:iy_max, ix_min:ix_max]
        z_crop = z_data[:, iy_min:iy_max, ix_min:ix_max]

        f.create_dataset('x', data=x_crop, compression='gzip')
        f.create_dataset('y', data=y_crop, compression='gzip')
        f.create_dataset('z', data=z_crop, compression='gzip')

    print(f"    Saved successfully!")

print("\n" + "=" * 70)
print("UNCONDITIONAL CORRELATION ANALYSIS COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUTPUT_DIR}")
