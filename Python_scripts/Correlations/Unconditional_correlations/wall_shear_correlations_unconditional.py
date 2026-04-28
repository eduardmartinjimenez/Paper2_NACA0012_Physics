import os
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Configuration
# ============================================================================

# Data directories
BASE_SURFACE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Snapshots/"
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Steady_state/"

# Pattern to match batch directories
BATCH_PATTERN = "batch_305*"

# Mesh data file
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Averaged snapshot file (for mean fields)
LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA5_avg_25340000-COMP-DATA.h5"
LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_3/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Analysis Parameters
# ============================================================================

u_infty = 1.0
AOA = 5  # degrees
AOA_rad = np.deg2rad(AOA)
c = 1.0  # chord length

# Chord locations for correlation analysis (x/c values)
X_C_LOCATIONS = [0.3, 0.5, 0.7, 0.9]
# X_C_LOCATIONS = [0.5]



# ============================================================================
# Load Geometrical Data
# ============================================================================

print("=" * 70)
print("UNCONDITIONAL WALL SHEAR STRESS CORRELATION ANALYSIS")
print("=" * 70)
print(f"\nAnalysis configuration:")
print(f"  Chord locations (x/c): {X_C_LOCATIONS}")
print(f"  Suction side (upper surface): closest point selected")

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
# Identify Closest Point on Suction Side
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


# ============================================================================
# Compute Wall-Shear Mean and RMS from All Surface Data Snapshots
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING MEAN TAU_W FROM ALL SURFACE SNAPSHOTS")
print("=" * 70)

tau_w_2d_sum = None
tau_w_2_2d_sum = None
n_snapshots = 0

print(f"Loading {N_total_surf} surface snapshots to compute mean...")

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Loading surface snapshot {idx+1}/{N_total_surf}...", flush=True)

    try:
        with h5py.File(surface_file, "r") as f:
            tau_w = f["tau_w"][:]  # (Nz_phys, N_surf)

            tau_w_2 = tau_w * tau_w

            # Spanwise average for each snapshot
            tau_w_2d = np.mean(tau_w, axis=0)  # (N_surf,)
            tau_w_2_2d = np.mean(tau_w_2, axis=0)  # (N_surf,)

            if tau_w_2d_sum is None:
                tau_w_2d_sum = tau_w_2d.copy()
                tau_w_2_2d_sum = tau_w_2_2d.copy()
                Nz_phys = tau_w.shape[0]
            else:
                tau_w_2d_sum += tau_w_2d
                tau_w_2_2d_sum += tau_w_2_2d

            n_snapshots += 1

    except Exception as e:
        print(f"  [WARNING] Error loading {surface_file}: {e}")
        continue

if n_snapshots == 0:
    raise RuntimeError("No valid snapshots loaded; check surface files and datasets.")

# Ensure we don't exceed available snapshot files
n_snapshots = min(n_snapshots, N_total_snapshots)

# Compute time-averaged means
tau_w_mean = tau_w_2d_sum / n_snapshots  # (N_surf,)
tau_w_2_mean = tau_w_2_2d_sum / n_snapshots  # (N_surf,)

print(f"  Successfully loaded {n_snapshots} surface snapshots")
print(f"  2D mean shape: (N_surf={len(tau_w_mean)})")
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
        'tau_prime': []
    }

print(f"Processing {n_snapshots} snapshots for surface data...")

for idx, surface_file in enumerate(all_surface_files[:n_snapshots]):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Processing snapshot {idx+1}/{n_snapshots}...", flush=True)

    try:
        with h5py.File(surface_file, "r") as f:
            tau_w = f["tau_w"][:]  # (Nz_phys, N_surf)

        # Extract at each chord location
        for x_c_target, point_info in point_indices.items():
            idx_point = point_info['indices'][0]

            # Extract surface values at all z-locations for this point
            tau_at_xc = tau_w[:, idx_point]  # (Nz_phys,)

            # Compute fluctuations
            tau_at_xc_fluct = tau_at_xc - tau_w_mean[idx_point]
            surface_signals[x_c_target]['tau_prime'].extend(tau_at_xc_fluct)

    except Exception as e:
        print(f"  [WARNING] Error processing {surface_file}: {e}")
        continue


# ============================================================================
# Load Mean Fields from Averaged Snapshot
# ============================================================================

print("\n" + "=" * 70)
print("LOADING MEAN FIELDS FROM AVERAGED SNAPSHOT")
print("=" * 70)

# Load mesh
loader = CompressedSnapshotLoader(MESH_FILE)

# Load averaged snapshot fields
fields = loader.load_snapshot_avg(LAST_SNAPSHOT_FILE)

# Coordinates:
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

# Average in spanwise direction (axis=0)
avg_u_data = np.mean(avg_u_data, axis=0)
avg_v_data = np.mean(avg_v_data, axis=0)
avg_w_data = np.mean(avg_w_data, axis=0)

# Compute streamwise velocity (aligned with angle of attack)
avg_u_data = avg_u_data * np.cos(AOA_rad) + avg_v_data * np.sin(AOA_rad)

print("avg_u_data shape (streamwise):", avg_u_data.shape)


# ============================================================================
# Compute tau_w_rms at Reference Points
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING TAU_W_RMS AT REFERENCE POINTS")
print("=" * 70)

tau_rms_dict = {}
for x_c_target, point_info in point_indices.items():
    idx_point = point_info['indices'][0]

    # tau_rms = sqrt( <tau_w^2> - <tau_w>^2 )
    tau_rms = np.sqrt(tau_w_2_mean[idx_point] - tau_w_mean[idx_point]**2)
    tau_rms_dict[x_c_target] = tau_rms

    tau_mean_val = tau_w_mean[idx_point]

    print(f"  x/c = {x_c_target:.2f}:")
    print(f"    tau_w_mean = {tau_mean_val:.6e}")
    print(f"    tau_w_rms  = {tau_rms:.6e}")


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

# Define window parameters (relative to each reference point)
dx_upstream = 0.3
dx_downstream = 0.3
dy_down = 0.05
dy_up = 0.25

# Get 2D grid for index finding
x_2d = x_data[0, :, :]  # (Ny, Nx)
y_2d = y_data[0, :, :]  # (Ny, Nx)
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

    # Find indices
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

# Highlight reference point
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

# Annotations
ax.annotate(f'Δx: [{-dx_upstream:.2f}, +{dx_downstream:.2f}]',
            xy=(x_ref, y_max_crop), xytext=(x_ref, y_max_crop + 0.05),
            ha='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green'))

ax.annotate(f'Δy: [{-dy_down:.2f}, +{dy_up:.2f}]',
            xy=(x_max_crop, y_ref), xytext=(x_max_crop + 0.05, y_ref),
            va='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green'))

# Labels and formatting
ax.set_xlabel('x/c', fontsize=14)
ax.set_ylabel('y/c', fontsize=14)
ax.set_title(f'Correlation Domain for x/c = {x_c_target:.2f}', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Set axis limits
ax.set_xlim(min(x_min_crop - 0.1, x_min_domain), max(x_max_crop + 0.1, x_max_domain))
ax.set_ylim(min(y_min_crop - 0.05, y_min_domain), max(y_max_crop + 0.05, y_max_domain))

plt.tight_layout()
plt.show()


# ============================================================================
# Initialize Accumulation Arrays
# ============================================================================

print("\n" + "=" * 70)
print("INITIALIZING ACCUMULATION ARRAYS")
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
        'tau_prime_sq_sum': 0.0,
    }

    print(f"  x/c = {x_c_target:.2f}: shape (Nz={Nz_crop}, Ny={Ny_crop}, Nx={Nx_crop})")

print(f"\nInitialized accumulation arrays for {len(point_indices)} chord locations")


# ============================================================================
# Load Instantaneous Velocity Fields and Compute Correlation
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING UNCONDITIONAL CORRELATION FROM INSTANTANEOUS FIELDS")
print("=" * 70)
print(f"Processing {n_snapshots} snapshots...")
print(f"  Snapshot files and surface files must align in order!")
print(f"  Exploiting spanwise periodicity: using all {Nz_phys} z-planes as reference points")
print(f"  -> z-axis of result represents relative separation Δz (Δz=0 at index 0)")

# Timing for snapshot loading
total_load_time = 0.0
start_total = time.time()

for snap_idx in range(n_snapshots):
    if (snap_idx + 1) % 10 == 0 or snap_idx == 0:
        print(f"\n  Snapshot {snap_idx+1}/{n_snapshots}...", flush=True)

    snapshot_file = all_snapshots_files[snap_idx]

    try:
        # Load instantaneous velocity fields
        load_start = time.time()
        fields_inst = loader.load_snapshot(snapshot_file)
        u_inst_full = loader.reconstruct_field(fields_inst["u"])  # (Nz, Ny, Nx)
        v_inst_full = loader.reconstruct_field(fields_inst["v"])  # (Nz, Ny, Nx)
        load_end = time.time()
        total_load_time += (load_end - load_start)

        # Compute streamwise velocity (aligned with angle of attack)
        u_inst_full = u_inst_full * np.cos(AOA_rad) + v_inst_full * np.sin(AOA_rad)

        # Process each chord location
        for x_c_target, point_info in point_indices.items():
            # Get crop window for this x/c location
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

            # Create mask for valid points (not NaN)
            valid_mask = ~np.isnan(u_prime)

            # Replace NaN values with 0 for FFT processing
            u_prime = np.where(valid_mask, u_prime, 0.0)

            # Extract tau'_w for this snapshot at all z-planes
            tau_prime_current = np.array(surface_signals[x_c_target]['tau_prime'][
                snap_idx * Nz_phys : (snap_idx + 1) * Nz_phys
            ])  # (Nz_phys,)

            # ---------------------------------------------------------------
            # FFT-based circular cross-correlation along z (exploits periodicity)
            #
            # Computes the unconditional normalized correlation between
            # wall-shear fluctuation at the reference point and streamwise
            # velocity fluctuation in the cropped 3D domain, using all
            # snapshots and all spanwise reference planes.
            #
            # For each relative separation Δz:
            #   C[Δz, y, x] = IFFT( conj(FFT(tau')) * FFT(u') )  along z
            #
            # For u'^2 normalization:
            #   U2[Δz, y, x] = IFFT( conj(FFT(ones)) * FFT(u'^2) )
            # ---------------------------------------------------------------

            u_fft = np.fft.rfft(u_prime, axis=0)  # (Nz//2+1, Ny, Nx)
            u2_fft = np.fft.rfft(u_prime**2, axis=0)  # (Nz//2+1, Ny, Nx)

            # Unconditional FFT-based correlation
            tau_fft = np.fft.rfft(tau_prime_current)  # (Nz//2+1,)
            num = np.fft.irfft(np.conj(tau_fft[:, None, None]) * u_fft,
                                n=Nz_phys, axis=0)  # (Nz_phys, Ny_crop, Nx_crop)

            # u'^2 contribution with all-ones indicator
            indicator_ones_fft = np.fft.rfft(np.ones(Nz_phys))  # (Nz//2+1,)
            u2 = np.fft.irfft(np.conj(indicator_ones_fft[:, None, None]) * u2_fft,
                              n=Nz_phys, axis=0)  # (Nz_phys, Ny_crop, Nx_crop)

            # Accumulate
            correlation_data[x_c_target]['numerator'] += num
            correlation_data[x_c_target]['u_prime_sq'] += u2
            correlation_data[x_c_target]['N_samples'] += Nz_phys
            correlation_data[x_c_target]['tau_prime_sq_sum'] += float(np.sum(tau_prime_current**2))

    except Exception as e:
        print(f"  [WARNING] Error loading snapshot {snapshot_file}: {e}")
        import traceback
        traceback.print_exc()
        continue

end_total = time.time()
total_time = end_total - start_total

print("\n" + "=" * 70)
print("TIMING SUMMARY")
print("=" * 70)
print(f"  Total time for all snapshots: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"  Time spent loading snapshots: {total_load_time:.2f} seconds ({total_load_time/60:.2f} minutes)")
print(f"  Average time per snapshot: {total_time/n_snapshots:.2f} seconds")
print(f"  Average loading time per snapshot: {total_load_time/n_snapshots:.2f} seconds")
print("=" * 70)


# ============================================================================
# Normalization and Correlation Computation
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING NORMALIZED CORRELATION COEFFICIENTS")
print("=" * 70)

correlation_results = {}

for x_c_target, corr_data in correlation_data.items():
    print(f"\n  x/c = {x_c_target:.2f}:")

    N_samples = corr_data['N_samples']
    print(f"    Total samples: N = {N_samples}")

    # Compute unconditional RMS of u' at each spatial location
    u_rms = np.sqrt(corr_data['u_prime_sq'] / N_samples)

    # Replace NaN with small value to avoid division issues
    u_rms = np.where(np.isnan(u_rms), 1e-12, u_rms)

    # Protect against division by zero
    u_rms_safe = np.where(u_rms > 1e-12, u_rms, 1e-12)

    # Compute unconditional tau_rms (from all samples)
    tau_rms = np.sqrt(corr_data['tau_prime_sq_sum'] / N_samples)

    print(f"    tau_w_rms = {tau_rms:.6e}")

    # Replace NaN in numerator with 0
    numerator = np.where(np.isnan(corr_data['numerator']), 0.0, corr_data['numerator'])

    # Compute unconditional correlation field
    r = numerator / (N_samples * tau_rms * u_rms_safe)

    # Replace any remaining NaN or Inf values with 0
    r = np.where(np.isnan(r) | np.isinf(r), 0.0, r)

    # Store results
    correlation_results[x_c_target] = {
        'r': r,
        'u_rms': u_rms,
        'tau_rms': tau_rms,
        'N_samples': N_samples,
    }

    # Report peak correlation
    r_max = np.max(np.abs(r))
    print(f"    Peak |R| = {r_max:.4f}")


# ============================================================================
# Save Results to HDF5
# ============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS TO HDF5")
print("=" * 70)

for x_c_target, results in correlation_results.items():
    point_info = point_indices[x_c_target]
    idx_point = point_info['indices'][0]

    # Get crop window info
    crop_info = crop_windows[x_c_target]
    ix_min = crop_info['ix_min']
    ix_max = crop_info['ix_max']
    iy_min = crop_info['iy_min']
    iy_max = crop_info['iy_max']

    output_filename = f"wall_shear_correlation_unconditional_xc_{x_c_target:.3f}.h5"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n  Saving x/c = {x_c_target:.2f} to {output_filename}")

    with h5py.File(output_path, "w") as f:
        # Metadata
        f.attrs['x_c_target'] = x_c_target
        f.attrs['x_c_actual'] = point_info['x_c_actual']
        f.attrs['y_actual'] = point_info['y']
        f.attrs['idx_point'] = idx_point
        f.attrs['tau_w_mean'] = tau_w_mean[idx_point]
        f.attrs['tau_w_rms'] = tau_rms_dict[x_c_target]
        f.attrs['N_snapshots'] = n_snapshots
        f.attrs['N_samples'] = results['N_samples']

        # Correlation field
        f.create_dataset('R', data=results['r'], compression='gzip')
        f.create_dataset('u_rms', data=results['u_rms'], compression='gzip')

        # Grid coordinates (cropped for this x/c location, all z-planes)
        x_crop = x_data[:, iy_min:iy_max, ix_min:ix_max]
        y_crop = y_data[:, iy_min:iy_max, ix_min:ix_max]
        z_crop = z_data[:, iy_min:iy_max, ix_min:ix_max]

        f.create_dataset('x', data=x_crop, compression='gzip')
        f.create_dataset('y', data=y_crop, compression='gzip')
        f.create_dataset('z', data=z_crop, compression='gzip')

        # Index ranges for reference
        f.attrs['ix_min'] = ix_min
        f.attrs['ix_max'] = ix_max
        f.attrs['iy_min'] = iy_min
        f.attrs['iy_max'] = iy_max

    print(f"    Saved successfully!")

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUTPUT_DIR}")
