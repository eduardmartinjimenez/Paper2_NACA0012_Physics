import os
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

# Base directory containing all SNAPSHOTS DATA batch folders
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/"

# Mesh data file
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Averaged last snapshot file (for mean fields)
LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

# Correlation output directory (for reading correlation files)
CORR_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_pressure_correlations/"

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/TKE_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
u_infty = 1.0
AOA = 12  # degrees
AOA_rad = np.deg2rad(AOA)
c = 1.0  # chord length

X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA = 1.0
N_SNAPSHOTS = None  # Set to None to use all available, or specify a number

# ============================================================================
# Load mesh and mean fields
# ============================================================================
print("=" * 70)
print("LOADING MESH AND MEAN FIELDS")
print("=" * 70)

loader = CompressedSnapshotLoader(MESH_FILE)
fields = loader.load_snapshot_avg(LAST_SNAPSHOT_FILE)

# Coordinates:
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

print(f"Grid shape: {x_data.shape}")

# Mean velocities:
avg_u_data = loader.reconstruct_field(fields["avg_u"])
avg_v_data = loader.reconstruct_field(fields["avg_v"])
avg_w_data = loader.reconstruct_field(fields["avg_w"])

# Average in spanwise direction and compute streamwise component
avg_u_data = np.mean(avg_u_data, axis=0)
avg_v_data = np.mean(avg_v_data, axis=0)
avg_w_data = np.mean(avg_w_data, axis=0)

# Streamwise velocity
avg_u_streamwise = avg_u_data * np.cos(AOA_rad) + avg_v_data * np.sin(AOA_rad)

print("Mean fields loaded and processed")

# ============================================================================
# Find snapshot files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR SNAPSHOT FILES")
print("=" * 70)

batch_snapshot_dirs = sorted(glob(os.path.join(BASE_SNAPSHOT_DIR, "batch_*")))
all_snapshots_files = []
for batch_dir in batch_snapshot_dirs:
    if not os.path.exists(batch_dir):
        continue
    snapshot_files = sorted(glob(os.path.join(batch_dir, "*A.h5")))
    all_snapshots_files.extend(snapshot_files)

N_total_snapshots = len(all_snapshots_files)
print(f"Found {N_total_snapshots} snapshot files")

if N_total_snapshots == 0:
    raise RuntimeError("No snapshot files found!")

# Limit to specified number
if N_SNAPSHOTS is not None:
    all_snapshots_files = all_snapshots_files[:N_SNAPSHOTS]
    N_snapshots = len(all_snapshots_files)
else:
    N_snapshots = N_total_snapshots

print(f"Processing {N_snapshots} snapshots")

# ============================================================================
# Load correlation results to get crop windows
# ============================================================================
print("\n" + "=" * 70)
print("LOADING CORRELATION RESULTS FOR CROP WINDOWS")
print("=" * 70)

correlation_files = {}
crop_windows_info = {}

for x_c in X_C_LOCATIONS:
    corr_filename = f"wall_pressure_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5"
    corr_filepath = os.path.join(CORR_DIR, corr_filename)

    if os.path.exists(corr_filepath):
        correlation_files[x_c] = corr_filepath
        with h5py.File(corr_filepath, "r") as f:
            crop_windows_info[x_c] = {
                'ix_min': int(f.attrs['ix_min']),
                'ix_max': int(f.attrs['ix_max']),
                'iy_min': int(f.attrs['iy_min']),
                'iy_max': int(f.attrs['iy_max']),
                'Nx': int(f.attrs['ix_max']) - int(f.attrs['ix_min']),
                'Ny': int(f.attrs['iy_max']) - int(f.attrs['iy_min']),
            }
        print(f"  x/c = {x_c:.2f}: Found correlation results")
    else:
        print(f"  x/c = {x_c:.2f}: Correlation file NOT found - skipping")

if len(correlation_files) == 0:
    raise RuntimeError("No correlation files found!")

# ============================================================================
# Compute TKE in correlation domain
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING TKE IN CORRELATION DOMAINS")
print("=" * 70)

tke_results = {}

for x_c in correlation_files.keys():
    print(f"\n  Processing x/c = {x_c:.2f}...")

    crop_info = crop_windows_info[x_c]
    ix_min = crop_info['ix_min']
    ix_max = crop_info['ix_max']
    iy_min = crop_info['iy_min']
    iy_max = crop_info['iy_max']
    Nx_crop = crop_info['Nx']
    Ny_crop = crop_info['Ny']

    # Initialize accumulators
    tke_sum = None
    n_valid = 0

    # Process snapshots
    for snap_idx in range(N_snapshots):
        if (snap_idx + 1) % max(1, N_snapshots // 10) == 0:
            print(f"    Snapshot {snap_idx+1}/{N_snapshots}...", flush=True)

        snapshot_file = all_snapshots_files[snap_idx]

        try:
            # Load instantaneous velocity fields
            fields_inst = loader.load_snapshot(snapshot_file)
            u_inst = loader.reconstruct_field(fields_inst["u"])  # (Nz, Ny, Nx)
            v_inst = loader.reconstruct_field(fields_inst["v"])  # (Nz, Ny, Nx)
            w_inst = loader.reconstruct_field(fields_inst["w"])  # (Nz, Ny, Nx)

            # Remove boundary layers (same as in correlation analysis)
            u_inst = u_inst[1:-1, :, :]  # (Nz_phys, Ny, Nx)
            v_inst = v_inst[1:-1, :, :]
            w_inst = w_inst[1:-1, :, :]

            # Crop to correlation window
            u_crop = u_inst[:, iy_min:iy_max, ix_min:ix_max]  # (Nz, Ny_crop, Nx_crop)
            v_crop = v_inst[:, iy_min:iy_max, ix_min:ix_max]
            w_crop = w_inst[:, iy_min:iy_max, ix_min:ix_max]

            # Get mean velocities in cropped region
            avg_u_crop = avg_u_streamwise[iy_min:iy_max, ix_min:ix_max]
            avg_v_crop = avg_v_data[iy_min:iy_max, ix_min:ix_max]
            avg_w_crop = avg_w_data[iy_min:iy_max, ix_min:ix_max]

            # Compute fluctuations (broadcast mean to all z-planes)
            u_prime = u_crop - avg_u_crop[np.newaxis, :, :]
            v_prime = v_crop - avg_v_crop[np.newaxis, :, :]
            w_prime = w_crop - avg_w_crop[np.newaxis, :, :]

            # Replace NaN (inside airfoil) with 0
            valid_mask = ~(np.isnan(u_prime) | np.isnan(v_prime) | np.isnan(w_prime))
            u_prime = np.where(valid_mask, u_prime, 0.0)
            v_prime = np.where(valid_mask, v_prime, 0.0)
            w_prime = np.where(valid_mask, w_prime, 0.0)

            # Compute TKE = 0.5 * (u'^2 + v'^2 + w'^2)
            # Average first over z (spanwise), then accumulate
            tke_inst = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)  # (Nz, Ny_crop, Nx_crop)
            tke_2d = np.mean(tke_inst, axis=0)  # (Ny_crop, Nx_crop) - average over spanwise

            if tke_sum is None:
                tke_sum = tke_2d.copy()
            else:
                tke_sum += tke_2d

            n_valid += 1

        except Exception as e:
            print(f"    [WARNING] Error loading snapshot {snapshot_file}: {e}")
            continue

    if n_valid == 0:
        print(f"    [ERROR] No valid snapshots processed!")
        continue

    # Compute time-averaged TKE
    tke_mean = tke_sum / n_valid

    tke_results[x_c] = {
        'tke': tke_mean,
        'n_snapshots': n_valid,
        'shape': tke_mean.shape,
    }

    print(f"    TKE computed from {n_valid} snapshots")
    print(f"    TKE range: [{np.min(tke_mean):.3e}, {np.max(tke_mean):.3e}]")
    print(f"    TKE mean: {np.nanmean(tke_mean):.3e}")

print("\n" + "=" * 70)
print("TKE COMPUTATION COMPLETE")
print("=" * 70)

# ============================================================================
# Save TKE results
# ============================================================================
print("\n" + "=" * 70)
print("SAVING TKE RESULTS")
print("=" * 70)

for x_c in tke_results.keys():
    output_filename = f"tke_xc_{x_c:.3f}.h5"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"  Saving x/c = {x_c:.2f}: {output_filename}")

    with h5py.File(output_path, "w") as f:
        f.attrs['x_c'] = x_c
        f.attrs['n_snapshots'] = tke_results[x_c]['n_snapshots']
        f.create_dataset('tke', data=tke_results[x_c]['tke'], compression='gzip')

print("TKE results saved!")

# ============================================================================
# Create comparison visualizations
# ============================================================================
print("\n" + "=" * 70)
print("CREATING COMPARISON VISUALIZATIONS")
print("=" * 70)

for x_c in tke_results.keys():
    if x_c not in correlation_files:
        continue

    print(f"  Creating visualization for x/c = {x_c:.2f}...")

    # Load correlation data
    corr_file = correlation_files[x_c]
    with h5py.File(corr_file, "r") as f:
        R_PF = f["R_PF"][:]  # (Nz, Ny_crop, Nx_crop)
        R_NF = f["R_NF"][:]

        # Average over z for comparison
        R_PF_2d = np.mean(np.abs(R_PF), axis=0)
        R_NF_2d = np.mean(np.abs(R_NF), axis=0)

    tke_2d = tke_results[x_c]['tke']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: TKE
    im1 = axes[0].contourf(tke_2d, levels=20, cmap='viridis')
    axes[0].set_title(f'Turbulent Kinetic Energy\nx/c = {x_c:.2f}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x-index')
    axes[0].set_ylabel('y-index')
    plt.colorbar(im1, ax=axes[0], label='TKE')

    # Plot 2: Correlation PF
    im2 = axes[1].contourf(R_PF_2d, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f'|Correlation| (PF)\nx/c = {x_c:.2f}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x-index')
    axes[1].set_ylabel('y-index')
    plt.colorbar(im2, ax=axes[1], label='|R_PF|')

    # Plot 3: Correlation NF
    im3 = axes[2].contourf(R_NF_2d, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title(f'|Correlation| (NF)\nx/c = {x_c:.2f}', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('x-index')
    axes[2].set_ylabel('y-index')
    plt.colorbar(im3, ax=axes[2], label='|R_NF|')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'tke_vs_correlation_xc_{x_c:.3f}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {plot_path}")
    plt.close()

    # Create overlaid contours (TKE + correlation contours)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Background: TKE
    im = ax.contourf(tke_2d, levels=20, cmap='viridis', alpha=0.8)

    # Overlay: Correlation contours for PF
    levels_corr = np.linspace(0, np.max(R_PF_2d), 8)
    contours = ax.contour(R_PF_2d, levels=levels_corr, colors='white', linewidths=1, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8)

    ax.set_title(f'TKE with PF Correlation Contours\nx/c = {x_c:.2f}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x-index')
    ax.set_ylabel('y-index')
    plt.colorbar(im, ax=ax, label='TKE')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'tke_with_corr_overlay_xc_{x_c:.3f}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {plot_path}")
    plt.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUTPUT_DIR}")
