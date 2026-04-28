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
# Base directory containing all SURFACE DATA batch folders
BASE_SURFACE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"
# BASE_SURFACE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
# Base directory containing all SNAPSHOTS DATA batch folders
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/"
# BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/Test/Steady_state/"

# Pattern to match batch directories
BATCH_PATTERN = "batch_30*"

# Mesh data file
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Averaged last snapshot file (for mean fields)
LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
# Reference parameters
u_infty = 1.0
AOA = 12  # degrees
AOA_rad = np.deg2rad(AOA)
c = 1.0  # chord length

# Chord locations for correlation analysis (x/c values)
X_C_LOCATIONS = [0.5]

# ============================================================================
# Define PF/NF classification parameters
# ============================================================================
print("\n" + "=" * 70)
print("PF/NF CLASSIFICATION SETUP")
print("=" * 70)

# Threshold factor (alpha): set to 0.0 for simple sign-based classification
# Can increase to e.g. 0.3 to filter weak events (as in Cheng2020)
ALPHA = 0.5

print(f"  Classification threshold: alpha = {ALPHA}")
print(f"  PF: tau'_w > {ALPHA}*tau_rms")
print(f"  NF: tau'_w < -{ALPHA}*tau_rms")


# ============================================================================
# Load geometrical data
# ============================================================================
print("=" * 70)
print("CORRELATION ANALYSIS OF WALL SHEAR STRESS FLUCTUATIONS")
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
# Find points at specified chord locations
# ============================================================================
print("\n" + "=" * 70)
print("IDENTIFYING CLOSEST POINT ON SUCTION SIDE")
print("=" * 70)

# Dictionary to store indices for each x/c location
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
        'indices': np.array([closest_global_idx]),  # Global index of closest point
        'x_c_actual': actual_x_c,                   # Actual x/c (might not be exactly x/c_target)
        'y': actual_y,                              # y-coordinate of this point
    }
    
    print(f"  x/c = {x_c_target:.2f}: index {closest_global_idx} at actual x/c = {actual_x_c:.4f}")

if len(point_indices) == 0:
    raise RuntimeError("No points found at any specified chord locations!")


# ============================================================================
# Find all surface data files
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
# Find all Snapshots data files
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
# Load all surface data snapshots and compute meann <tau_w> (for fluctuations)
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING <TAU_W> FROM ALL SURFACE DATA SNAPSHOTS")
print("=" * 70)

tau_w_2d_sum = None
tau_w_2_2d_sum = None
n_snapshots = 0

print(f"Loading {N_total_snapshots} surface snapshots to compute mean...")

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Loading surface snapshot {idx+1}/{N_total_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:

            tau_w = f["tau_w"][:]    # (Nz_phys, N_surf)
            
            tau_w_2 = tau_w * tau_w
            
            # Spanwise average for each snapshot
            tau_w_2d = np.mean(tau_w, axis=0)       # (N_surf,)
            tau_w_2_2d = np.mean(tau_w_2, axis=0)       # (N_surf,)
            

            if tau_w_2d_sum is None:
                tau_w_2d_sum = tau_w_2d.copy()
                tau_w_2_2d_sum = tau_w_2_2d.copy()
                Nz_phys = tau_w.shape[0]  # Store for later
            else:
                tau_w_2d_sum += tau_w_2d
                tau_w_2_2d_sum += tau_w_2_2d
            
            n_snapshots += 1
            
    except Exception as e:
        print(f"  [WARNING] Error loading {surface_file}: {e}")
        continue

if n_snapshots == 0:
    raise RuntimeError("No valid snapshots loaded; check surface files and datasets.")

# Compute 2D time-averaged means
tau_w_mean = tau_w_2d_sum / n_snapshots  # (N_surf,)
tau_w_2_mean = tau_w_2_2d_sum / n_snapshots  # (N_surf,)

print(f"  Successfully loaded {n_snapshots} surface snapshots")
print(f"  2D mean shape: (N_surf={len(tau_w_mean)})")
print(f"  Spanwise planes in each snapshot: Nz={Nz_phys}")


# ============================================================================
# Collect surface data at each chord location and compute fluctuations
# ============================================================================
print("\n" + "=" * 70)
print("COLLECTING SURFACE DATA")
print("=" * 70)

surface_data = {}

for x_c_target in point_indices.keys():
    surface_data[x_c_target] = {
        'tau_prime_w': []
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
            idx_point = point_info['indices'][0]  # Single point
            
            # Extract surface values at all z-locations for this point
            tau_at_xc = tau_w[:, idx_point]  # (Nz_phys,)

            # compute fluctuations
            tau_at_xc_fluct = tau_at_xc - tau_w_mean[idx_point]
            surface_data[x_c_target]['tau_prime_w'].extend(tau_at_xc_fluct) # contains n_snapshots * Nz_phys scalar values for each x/c location
    except Exception as e:
        print(f"  [WARNING] Error processing {surface_file}: {e}")
        continue


# ============================================================================
# Load last snapshot for mean fields
# ============================================================================
print("\n" + "=" * 70)
print("LOADING LAST SNAPSHOT FOR MEAN FIELDS")
print("=" * 70)

# Load mesh
loader = CompressedSnapshotLoader(MESH_FILE)

# Load last snapshot fields
fields = loader.load_snapshot_avg(LAST_SNAPSHOT_FILE)

# Coordinates:
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print("x_data shape:", x_data.shape)
print("y_data shape:", y_data.shape)
print("z_data shape:", z_data.shape)

# Mean velocities: 
avg_u_data = loader.reconstruct_field(fields["avg_u"])
avg_v_data = loader.reconstruct_field(fields["avg_v"])
avg_w_data = loader.reconstruct_field(fields["avg_w"])

print("reconstructed avg_u shape:", avg_u_data.shape)

# Average in spanwise direction (axis=0)
avg_u_data = np.mean(avg_u_data, axis=0)
avg_v_data = np.mean(avg_v_data, axis=0)
avg_w_data = np.mean(avg_w_data, axis=0)

# Compute streamwise velocity (aligned with angle of attack)
# V_streamwise = u*cos(AOA) + v*sin(AOA)
avg_u_data = avg_u_data * np.cos(AOA_rad) + avg_v_data * np.sin(AOA_rad)

print("avg_u_data shape (streamwise):", avg_u_data.shape)
print("avg_v_data shape:", avg_v_data.shape)
print("avg_w_data shape:", avg_w_data.shape)


# ============================================================================
# Compute tau_w_rms at chosen points
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
# Define correlation spatial window (relative to reference point)
# ============================================================================
print("\n" + "=" * 70)
print("DEFINING CORRELATION SPATIAL WINDOW")
print("=" * 70)

# Determine domain extents
x_min_domain, x_max_domain = np.min(x_data), np.max(x_data)
y_min_domain, y_max_domain = np.min(y_data), np.max(y_data)

# Get reference point location (assuming single x/c location)
x_c_ref = list(point_indices.keys())[0]
point_info_ref = point_indices[x_c_ref]
x_ref = point_info_ref['x_c_actual']
y_ref = point_info_ref['y']

print(f"  Reference point: x/c = {x_ref:.4f}, y = {y_ref:.4f}")

# Define asymmetric window relative to reference point
# x: [x_ref - 0.1, x_ref + 0.3] (extends more downstream)
# y: [y_ref - 0.1, y_ref + 0.3] (extends more upward from wall)
dx_upstream = 0.25
dx_downstream = 0.25
dy_down = 0.05
dy_up = 0.2

x_min_crop = x_ref - dx_upstream
x_max_crop = x_ref + dx_downstream
y_min_crop = max(y_ref - dy_down, y_min_domain)  # Don't go below domain
y_max_crop = y_ref + dy_up

print(f"  Full domain: x=[{x_min_domain:.3f}, {x_max_domain:.3f}], y=[{y_min_domain:.3f}, {y_max_domain:.3f}]")
print(f"  Cropped window: x=[{x_min_crop:.3f}, {x_max_crop:.3f}], y=[{y_min_crop:.3f}, {y_max_crop:.3f}]")

# Find indices for cropped region (using 2D spanwise-averaged coordinates)
x_2d = x_data[0, :, :]  # (Ny, Nx)
y_2d = y_data[0, :, :]  # (Ny, Nx)

# Find x index range
x_1d = x_2d[0, :]  # Extract 1D x coordinates
ix_min = np.argmin(np.abs(x_1d - x_min_crop))
ix_max = np.argmin(np.abs(x_1d - x_max_crop))

# Find y index range
y_1d = y_2d[:, 0]  # Extract 1D y coordinates
iy_min = np.argmin(np.abs(y_1d - y_min_crop))
iy_max = np.argmin(np.abs(y_1d - y_max_crop))

print(f"  Index ranges: ix=[{ix_min}:{ix_max}], iy=[{iy_min}:{iy_max}]")

# Cropped grid sizes
Ny_crop = iy_max - iy_min
Nx_crop = ix_max - ix_min
Nz_crop = Nz_phys  # Keep all z-planes for 3D correlation

print(f"  Cropped array shape: (Nz={Nz_crop}, Ny={Ny_crop}, Nx={Nx_crop})")

# # Visualization of the domain and reference point
# print("\n  Creating visualization of correlation domain...")

# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot interface points (airfoil surface)
# ax.scatter(x_interface[upper_mask], y_interface[upper_mask],
#            c='blue', s=2, alpha=0.5, label='Upper surface')
# ax.scatter(x_interface[lower_mask], y_interface[lower_mask],
#            c='red', s=2, alpha=0.5, label='Lower surface')

# # Highlight reference point
# ax.scatter(x_ref, y_ref, c='green', s=200, marker='*',
#            edgecolors='black', linewidths=2, label=f'Reference point (x/c={x_ref:.3f})', zorder=5)

# # Draw correlation window as rectangle
# rect = patches.Rectangle((x_min_crop, y_min_crop),
#                           x_max_crop - x_min_crop,
#                           y_max_crop - y_min_crop,
#                           linewidth=3, edgecolor='green', facecolor='green',
#                           alpha=0.2, label='Correlation window')
# ax.add_patch(rect)

# # Draw reference lines
# ax.axvline(x_ref, color='green', linestyle='--', linewidth=1, alpha=0.5)
# ax.axhline(y_ref, color='green', linestyle='--', linewidth=1, alpha=0.5)

# # Annotations
# ax.annotate(f'Δx: [{-dx_upstream:.2f}, +{dx_downstream:.2f}]',
#             xy=(x_ref, y_max_crop), xytext=(x_ref, y_max_crop + 0.05),
#             ha='center', fontsize=10, color='green',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green'))

# ax.annotate(f'Δy: [{-dy_down:.2f}, +{dy_up:.2f}]',
#             xy=(x_max_crop, y_ref), xytext=(x_max_crop + 0.05, y_ref),
#             va='center', fontsize=10, color='green',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green'))

# # Labels and formatting
# ax.set_xlabel('x/c', fontsize=14)
# ax.set_ylabel('y/c', fontsize=14)
# ax.set_title(f'Correlation Domain for x/c = {x_c_ref:.2f}', fontsize=16, fontweight='bold')
# ax.legend(loc='upper right', fontsize=11)
# ax.grid(True, alpha=0.3)
# ax.set_aspect('equal', adjustable='box')

# # Set axis limits to show context
# ax.set_xlim(min(x_min_crop - 0.1, x_min_domain), max(x_max_crop + 0.1, x_max_domain))
# ax.set_ylim(min(y_min_crop - 0.05, y_min_domain), max(y_max_crop + 0.05, y_max_domain))

# # Save figure
# plot_path = os.path.join(OUTPUT_DIR, f'correlation_domain_xc_{x_c_ref:.3f}.png')
# plt.tight_layout()
# plt.show()


# ============================================================================
# Initialize accumulation arrays for correlation
# ============================================================================
print("\n" + "=" * 70)
print("INITIALIZING CORRELATION ACCUMULATION ARRAYS")
print("=" * 70)

# For each x/c location, we accumulate:
# - Numerator_PF, Numerator_NF, Numerator_all
# - u_prime_squared (for RMS computation)
# - Counters: N_PF, N_NF, N_all

correlation_data = {}

for x_c_target in point_indices.keys():
    correlation_data[x_c_target] = {
        'Numerator_PF': np.zeros((Nz_crop, Ny_crop, Nx_crop), dtype=np.float64),
        'Numerator_NF': np.zeros((Nz_crop, Ny_crop, Nx_crop), dtype=np.float64),
        'Numerator_all': np.zeros((Nz_crop, Ny_crop, Nx_crop), dtype=np.float64),
        'u_prime_sq': np.zeros((Nz_crop, Ny_crop, Nx_crop), dtype=np.float64),
        'N_PF': 0,
        'N_NF': 0,
        'N_all': 0,
        'tau_prime_PF_sq_sum': 0.0,  # For computing tau_rms_PF
        'tau_prime_NF_sq_sum': 0.0,  # For computing tau_rms_NF
        'u_prime_sq_uncond': np.zeros((Nz_crop, Ny_crop, Nx_crop), dtype=np.float64),
        'N_total': 0,
    }

print(f"  Initialized accumulation arrays for {len(point_indices)} chord locations")


# ============================================================================
# Load instantaneous velocity fields from snapshots for correlation analysis
# ============================================================================
print("\n" + "=" * 70)
print("LOADING INSTANTANEOUS VELOCITY FIELDS FROM SNAPSHOTS FOR CORRELATION ANALYSIS")
print("=" * 70)

# We'll process snapshots and accumulate correlations
print(f"Processing {n_snapshots} snapshots...")
print(f"  Snapshot files and surface files must align in order!")

# Get middle z-plane index
z_mid = Nz_phys // 2
print(f"\nUsing middle z-plane: z_mid = {z_mid} (out of {Nz_phys})")

for snap_idx in range(n_snapshots):
    if (snap_idx + 1) % 10 == 0 or snap_idx == 0:
        print(f"\n  Snapshot {snap_idx+1}/{n_snapshots}...", flush=True)

    snapshot_file = all_snapshots_files[snap_idx]

    try:
        # Load instantaneous velocity fields
        fields_inst = loader.load_snapshot(snapshot_file)
        u_inst_full = loader.reconstruct_field(fields_inst["u"])  # (Nz, Ny, Nx)
        v_inst_full = loader.reconstruct_field(fields_inst["v"])  # (Nz, Ny, Nx)

        # Compute streamwise velocity (aligned with angle of attack)
        # V_streamwise = u*cos(AOA) + v*sin(AOA)
        u_inst_full = u_inst_full * np.cos(AOA_rad) + v_inst_full * np.sin(AOA_rad)

        # Crop to correlation window (keep all z-planes)
        u_inst = u_inst_full[1:-1, iy_min:iy_max, ix_min:ix_max]  # (Nz_crop, Ny_crop, Nx_crop)

        # Compute fluctuation: u' = u - <u>
        # avg_u_data is 2D (Ny, Nx), broadcast to match u_inst
        avg_u_crop = avg_u_data[iy_min:iy_max, ix_min:ix_max]  # (Ny_crop, Nx_crop)
        u_prime = u_inst - avg_u_crop[np.newaxis, :, :]  # (Nz_crop, Ny_crop, Nx_crop)

        # Create mask for valid points (not NaN - excludes inside airfoil)
        valid_mask = ~np.isnan(u_prime)
        
        # Replace NaN values with 0 to prevent propagation in accumulation
        u_prime = np.where(valid_mask, u_prime, 0.0)

        # Now process each chord location
        for x_c_target, point_info in point_indices.items():
            idx_point = point_info['indices'][0]
            tau_rms = tau_rms_dict[x_c_target]
            threshold = ALPHA * tau_rms

            # Extract tau'_w for this snapshot at all z-planes
            # surface_data[x_c_target]['tau_prime_w'] is a list we've been appending to
            # Each snapshot contributed Nz_phys samples
            tau_prime_current = surface_data[x_c_target]['tau_prime_w'][
                snap_idx * Nz_phys : (snap_idx + 1) * Nz_phys
            ]
            tau_prime_current = np.array(tau_prime_current)  # (Nz_phys,)

            # Extract tau'_w at middle z-plane (single scalar value)
            tau_prime_val = tau_prime_current[z_mid]

            # Accumulate unconditional u'^2 (ALL snapshots, for Cheng2020-style normalization)
            correlation_data[x_c_target]['u_prime_sq_uncond'] += u_prime**2
            correlation_data[x_c_target]['N_total'] += 1

            # Classify as PF, NF, or neither
            is_PF = tau_prime_val > threshold
            is_NF = tau_prime_val < -threshold

            if not (is_PF or is_NF):
                continue  # Skip samples in intermediate range

            # Correlate scalar tau'_w(mid z-point) with entire 3D cropped velocity field
            # u_prime shape: (Nz_crop, Ny_crop, Nx_crop)
            # product shape: (Nz_crop, Ny_crop, Nx_crop)
            product = tau_prime_val * u_prime

            # Accumulate correlation numerator
            correlation_data[x_c_target]['Numerator_all'] += product
            correlation_data[x_c_target]['N_all'] += 1

            # Accumulate u'^2 for RMS
            correlation_data[x_c_target]['u_prime_sq'] += u_prime**2

            if is_PF:
                correlation_data[x_c_target]['Numerator_PF'] += product
                correlation_data[x_c_target]['N_PF'] += 1
                correlation_data[x_c_target]['tau_prime_PF_sq_sum'] += tau_prime_val**2

            elif is_NF:
                correlation_data[x_c_target]['Numerator_NF'] += product
                correlation_data[x_c_target]['N_NF'] += 1
                correlation_data[x_c_target]['tau_prime_NF_sq_sum'] += tau_prime_val**2

        # Periodic progress tracking
        if (snap_idx + 1) % 10 == 0:
            for x_c_target in point_indices.keys():
                N_PF = correlation_data[x_c_target]['N_PF']
                N_NF = correlation_data[x_c_target]['N_NF']
                print(f"    x/c={x_c_target:.2f}: N_PF={N_PF}, N_NF={N_NF}")

    except Exception as e:
        print(f"  [WARNING] Error loading snapshot {snapshot_file}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "=" * 70)
print("CORRELATION ACCUMULATION COMPLETE")
print("=" * 70)


# ============================================================================
# Normalize to compute correlation coefficients
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING NORMALIZED CORRELATION COEFFICIENTS")
print("=" * 70)

correlation_results = {}

for x_c_target, corr_data in correlation_data.items():
    print(f"\n  x/c = {x_c_target:.2f}:")

    N_PF = corr_data['N_PF']
    N_NF = corr_data['N_NF']
    N_all = corr_data['N_all']

    print(f"    Total samples: N_all = {N_all}")
    print(f"    PF samples:    N_PF  = {N_PF}")
    print(f"    NF samples:    N_NF  = {N_NF}")

    # Sanity check
    if ALPHA == 0.0:
        expected_total = N_PF + N_NF
        print(f"    Sanity check: N_PF + N_NF = {expected_total} (should equal {N_all})")

    # Compute UNCONDITIONAL RMS of u' at each spatial location (Cheng2020-style)
    N_total = corr_data['N_total']
    print(f"    Total snapshots (unconditional): N_total = {N_total}")
    u_rms = np.sqrt(corr_data['u_prime_sq_uncond'] / N_total)

    # Replace NaN in u_rms with small value to avoid division issues
    u_rms = np.where(np.isnan(u_rms), 1e-12, u_rms)

    # Avoid division by zero in u_rms
    u_rms_safe = np.where(u_rms > 1e-12, u_rms, 1e-12)

    # Compute conditional tau_rms for PF and NF
    if N_PF > 0:
        tau_rms_PF = np.sqrt(corr_data['tau_prime_PF_sq_sum'] / N_PF)
    else:
        tau_rms_PF = 1.0  # Avoid division by zero
        print(f"    [WARNING] No PF samples found!")

    if N_NF > 0:
        tau_rms_NF = np.sqrt(corr_data['tau_prime_NF_sq_sum'] / N_NF)
    else:
        tau_rms_NF = 1.0
        print(f"    [WARNING] No NF samples found!")

    print(f"    tau_rms_PF = {tau_rms_PF:.6e}")
    print(f"    tau_rms_NF = {tau_rms_NF:.6e}")

    # Replace NaN in numerators with 0
    Numerator_all = np.where(np.isnan(corr_data['Numerator_all']), 0.0, corr_data['Numerator_all'])
    Numerator_PF = np.where(np.isnan(corr_data['Numerator_PF']), 0.0, corr_data['Numerator_PF'])
    Numerator_NF = np.where(np.isnan(corr_data['Numerator_NF']), 0.0, corr_data['Numerator_NF'])

    # Unconditioned correlation
    R_all = Numerator_all / (N_all * tau_rms_dict[x_c_target] * u_rms_safe)

    # Conditional correlations (Cheng2020-style: unconditional sigma in denominator)
    tau_rms_uncond = tau_rms_dict[x_c_target]
    if N_PF > 0:
        R_PF = Numerator_PF / (N_PF * tau_rms_uncond * u_rms_safe)
    else:
        R_PF = np.zeros_like(R_all)

    if N_NF > 0:
        R_NF = Numerator_NF / (N_NF * tau_rms_uncond * u_rms_safe)
    else:
        R_NF = np.zeros_like(R_all)

    # Replace any remaining NaN or Inf values with 0
    R_all = np.where(np.isnan(R_all) | np.isinf(R_all), 0.0, R_all)
    R_PF = np.where(np.isnan(R_PF) | np.isinf(R_PF), 0.0, R_PF)
    R_NF = np.where(np.isnan(R_NF) | np.isinf(R_NF), 0.0, R_NF)

    # Store results
    correlation_results[x_c_target] = {
        'R_all': R_all,
        'R_PF': R_PF,
        'R_NF': R_NF,
        'u_rms': u_rms,
        'tau_rms_PF': tau_rms_PF,
        'tau_rms_NF': tau_rms_NF,
        'N_PF': N_PF,
        'N_NF': N_NF,
        'N_all': N_all,
    }

    # Report peak correlations
    R_PF_max = np.max(np.abs(R_PF))
    R_NF_max = np.max(np.abs(R_NF))
    R_all_max = np.max(np.abs(R_all))

    print(f"    Peak |R_PF|  = {R_PF_max:.4f}")
    print(f"    Peak |R_NF|  = {R_NF_max:.4f}")
    print(f"    Peak |R_all| = {R_all_max:.4f}")

    # Additional sanity check: Numerator_all ≈ Numerator_PF + Numerator_NF
    if ALPHA == 0.0:
        Num_sum = Numerator_PF + Numerator_NF
        Num_diff = np.nanmax(np.abs(Numerator_all - Num_sum))
        print(f"    Numerator check: max|Num_all - (Num_PF+Num_NF)| = {Num_diff:.3e}")


# ============================================================================
# Save results to HDF5
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS TO HDF5")
print("=" * 70)

for x_c_target, results in correlation_results.items():
    point_info = point_indices[x_c_target]
    idx_point = point_info['indices'][0]

    output_filename = f"wall_shear_correlation_xc_{x_c_target:.3f}_alpha_{ALPHA:.1f}_200_unconditional.h5"
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
        f.attrs['tau_rms_PF'] = results['tau_rms_PF']
        f.attrs['tau_rms_NF'] = results['tau_rms_NF']
        f.attrs['alpha'] = ALPHA
        f.attrs['N_snapshots'] = n_snapshots
        f.attrs['N_PF'] = results['N_PF']
        f.attrs['N_NF'] = results['N_NF']
        f.attrs['N_all'] = results['N_all']

        # Correlation fields
        f.create_dataset('R_PF', data=results['R_PF'], compression='gzip')
        f.create_dataset('R_NF', data=results['R_NF'], compression='gzip')
        f.create_dataset('R_all', data=results['R_all'], compression='gzip')
        f.create_dataset('u_rms', data=results['u_rms'], compression='gzip')

        # Grid coordinates (cropped, all z-planes)
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

