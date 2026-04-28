import os
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

# ============================================================================
# Configuration
# ============================================================================

# Data directories
BASE_SURFACE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"
BATCH_PATTERN = "batch_*"

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Shear_fluctuations_visualization/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
u_infty = 1.0
AOA = 12  # degrees
c = 1.0   # chord length
Re_c = 50000

# Number of snapshots to load (set to None to load all)
MAX_SNAPSHOTS = 100

# Visualization options
VISUALIZE_RMS = True              # RMS distribution along chord
VISUALIZE_INSTANTANEOUS = True    # Instantaneous fluctuations on surface
VISUALIZE_SPACETIME = True        # Space-time diagrams at specific locations
VISUALIZE_STATISTICS = True       # Statistical distributions

# For space-time diagrams, specify chord locations
X_C_LOCATIONS = [0.2, 0.4, 0.6, 0.8]

print("=" * 70)
print("WALL SHEAR STRESS FLUCTUATION VISUALIZATION")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  AoA: {AOA}°")
print(f"  Re_c: {Re_c:,}")
print(f"  Max snapshots: {MAX_SNAPSHOTS if MAX_SNAPSHOTS else 'All'}")
print(f"  Output directory: {OUTPUT_DIR}")

# ============================================================================
# Load geometrical data
# ============================================================================
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

print(f"  Upper surface points: {np.sum(upper_mask)}")
print(f"  Lower surface points: {np.sum(lower_mask)}")

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

N_total = len(all_surface_files)
print(f"  Total surface data files found: {N_total}")

if N_total == 0:
    raise RuntimeError("No surface data files found!")

# Limit number of snapshots if requested
if MAX_SNAPSHOTS is not None:
    all_surface_files = all_surface_files[:MAX_SNAPSHOTS]
    N_snapshots = len(all_surface_files)
else:
    N_snapshots = N_total

print(f"  Will load: {N_snapshots} snapshots")

# ============================================================================
# Load data and compute statistics
# ============================================================================
print("\n" + "=" * 70)
print("LOADING SURFACE DATA AND COMPUTING STATISTICS")
print("=" * 70)

tau_w_2d_sum = None
tau_w_2_2d_sum = None
tau_w_snapshots = []  # Store for space-time visualization

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Loading snapshot {idx+1}/{N_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:
            tau_w = f["tau_w"][:]  # (Nz_phys, N_surf)
            
            # Spanwise average
            tau_w_2d = np.mean(tau_w, axis=0)  # (N_surf,)
            tau_w_2_2d = np.mean(tau_w**2, axis=0)
            
            # Store for later analysis
            tau_w_snapshots.append(tau_w_2d)
            
            if tau_w_2d_sum is None:
                tau_w_2d_sum = tau_w_2d.copy()
                tau_w_2_2d_sum = tau_w_2_2d.copy()
                Nz_phys = tau_w.shape[0]
            else:
                tau_w_2d_sum += tau_w_2d
                tau_w_2_2d_sum += tau_w_2_2d
            
    except Exception as e:
        print(f"  [WARNING] Error loading {surface_file}: {e}")
        continue

# Compute statistics
tau_w_mean = tau_w_2d_sum / N_snapshots
tau_w_2_mean = tau_w_2_2d_sum / N_snapshots
tau_w_rms = np.sqrt(tau_w_2_mean - tau_w_mean**2)

print(f"\n  Successfully loaded {N_snapshots} snapshots")
print(f"  tau_w mean range: [{np.min(tau_w_mean):.6e}, {np.max(tau_w_mean):.6e}]")
print(f"  tau_w RMS range:  [{np.min(tau_w_rms):.6e}, {np.max(tau_w_rms):.6e}]")

# Convert to array for easier manipulation
tau_w_snapshots = np.array(tau_w_snapshots)  # (N_snapshots, N_surf)
print(f"  Snapshot array shape: {tau_w_snapshots.shape}")

# Compute fluctuations
tau_w_prime = tau_w_snapshots - tau_w_mean[np.newaxis, :]  # (N_snapshots, N_surf)

# ============================================================================
# VISUALIZATION 1: RMS Distribution along chord
# ============================================================================
if VISUALIZE_RMS:
    print("\n" + "=" * 70)
    print("CREATING RMS DISTRIBUTION PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full airfoil view
    ax = axes[0]
    
    # Separate upper and lower
    x_upper = x_over_c[upper_mask]
    x_lower = x_over_c[lower_mask]
    tau_rms_upper = tau_w_rms[upper_mask]
    tau_rms_lower = tau_w_rms[lower_mask]
    
    # Sort by x coordinate
    sort_idx_upper = np.argsort(x_upper)
    sort_idx_lower = np.argsort(x_lower)
    
    ax.plot(x_upper[sort_idx_upper], tau_rms_upper[sort_idx_upper], 
            'r-', linewidth=2, label='Suction side (upper)', marker='o', markersize=2)
    ax.plot(x_lower[sort_idx_lower], tau_rms_lower[sort_idx_lower], 
            'b-', linewidth=2, label='Pressure side (lower)', marker='o', markersize=2)
    
    ax.set_xlabel('x/c', fontsize=14)
    ax.set_ylabel("$\\tau'_{w,rms}$", fontsize=14)
    ax.set_title(f'Wall Shear Stress RMS - NACA 0012, AoA={AOA}°, Re={Re_c:,}\n' +
                 f'{N_snapshots} snapshots', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 2D surface scatter
    ax = axes[1]
    scatter = ax.scatter(x_interface, y_interface, c=tau_w_rms, 
                        cmap='hot', s=20, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("$\\tau'_{w,rms}$", fontsize=14)
    
    ax.set_xlabel('x/c', fontsize=14)
    ax.set_ylabel('y/c', fontsize=14)
    ax.set_title('RMS Distribution on Airfoil Surface', fontsize=13, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # rms_path = os.path.join(OUTPUT_DIR, f'tau_w_rms_distribution_AoA{AOA}.png')
    # plt.savefig(rms_path, dpi=300, bbox_inches='tight')
    # print(f"  Saved: {rms_path}")
    # plt.close()
    plt.show()

# ============================================================================
# VISUALIZATION 2: Instantaneous fluctuations
# ============================================================================
if VISUALIZE_INSTANTANEOUS:
    print("\n" + "=" * 70)
    print("CREATING INSTANTANEOUS FLUCTUATION PLOTS")
    print("=" * 70)
    
    # Select a few representative snapshots
    snapshot_indices = [0, N_snapshots//4, N_snapshots//2, 3*N_snapshots//4, N_snapshots-1]
    
    fig, axes = plt.subplots(len(snapshot_indices), 1, figsize=(14, 4*len(snapshot_indices)))
    if len(snapshot_indices) == 1:
        axes = [axes]
    
    # Use symmetric colormap centered at zero
    vmax = np.percentile(np.abs(tau_w_prime), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    for idx, snap_idx in enumerate(snapshot_indices):
        ax = axes[idx]
        
        tau_prime_snap = tau_w_prime[snap_idx, :]
        
        scatter = ax.scatter(x_interface, y_interface, c=tau_prime_snap, 
                           cmap='RdBu_r', norm=norm, s=30, edgecolors='none')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("$\\tau'_w$", fontsize=12)
        
        ax.set_xlabel('x/c', fontsize=12)
        ax.set_ylabel('y/c', fontsize=12)
        ax.set_title(f'Instantaneous fluctuation - Snapshot {snap_idx+1}/{N_snapshots}', 
                    fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Wall Shear Stress Fluctuations - NACA 0012, AoA={AOA}°, Re={Re_c:,}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # inst_path = os.path.join(OUTPUT_DIR, f'tau_w_instantaneous_fluctuations_AoA{AOA}.png')
    # plt.savefig(inst_path, dpi=300, bbox_inches='tight')
    # print(f"  Saved: {inst_path}")
    # plt.close()
    plt.show()

# ============================================================================
# VISUALIZATION 3: Space-time diagrams
# ============================================================================
if VISUALIZE_SPACETIME:
    print("\n" + "=" * 70)
    print("CREATING SPACE-TIME DIAGRAMS")
    print("=" * 70)
    
    # Find points at specified chord locations on upper surface
    print("  Finding reference points on suction side...")
    point_indices = {}
    
    for x_c_target in X_C_LOCATIONS:
        upper_indices = np.where(upper_mask)[0]
        distances = np.abs(x_over_c[upper_indices] - x_c_target)
        closest_idx = upper_indices[np.argmin(distances)]
        point_indices[x_c_target] = closest_idx
        print(f"    x/c = {x_c_target:.2f} -> index {closest_idx}, actual x/c = {x_over_c[closest_idx]:.4f}")
    
    # Create space-time plot
    fig, axes = plt.subplots(len(X_C_LOCATIONS), 1, figsize=(14, 3*len(X_C_LOCATIONS)))
    if len(X_C_LOCATIONS) == 1:
        axes = [axes]
    
    time_indices = np.arange(N_snapshots)
    
    for idx, (x_c_target, surf_idx) in enumerate(point_indices.items()):
        ax = axes[idx]
        
        # Extract time series at this location
        tau_prime_series = tau_w_prime[:, surf_idx]
        tau_rms_loc = tau_w_rms[surf_idx]
        
        # Normalize by local RMS
        tau_prime_norm = tau_prime_series / tau_rms_loc if tau_rms_loc > 1e-10 else tau_prime_series
        
        ax.plot(time_indices, tau_prime_norm, 'k-', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=1, color='r', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=-1, color='b', linestyle=':', linewidth=1, alpha=0.5)
        
        # Fill positive/negative regions
        ax.fill_between(time_indices, 0, tau_prime_norm, where=(tau_prime_norm > 0), 
                       color='red', alpha=0.2, label='Positive')
        ax.fill_between(time_indices, 0, tau_prime_norm, where=(tau_prime_norm < 0), 
                       color='blue', alpha=0.2, label='Negative')
        
        ax.set_ylabel("$\\tau'_w / \\tau'_{w,rms}$", fontsize=12)
        ax.set_title(f'x/c = {x_c_target:.2f} (suction side), $\\tau\'_{{w,rms}}$ = {tau_rms_loc:.6f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        if idx == len(X_C_LOCATIONS) - 1:
            ax.set_xlabel('Snapshot index', fontsize=12)
    
    plt.suptitle(f'Wall Shear Stress Time Evolution - NACA 0012, AoA={AOA}°, Re={Re_c:,}', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # spacetime_path = os.path.join(OUTPUT_DIR, f'tau_w_spacetime_AoA{AOA}.png')
    # plt.savefig(spacetime_path, dpi=300, bbox_inches='tight')
    # print(f"  Saved: {spacetime_path}")
    # plt.close()
    plt.show()

# ============================================================================
# VISUALIZATION 4: Statistical distributions
# ============================================================================
if VISUALIZE_STATISTICS:
    print("\n" + "=" * 70)
    print("CREATING STATISTICAL DISTRIBUTION PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: PDF of normalized fluctuations at different locations
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(X_C_LOCATIONS)))
    
    for idx, (x_c_target, surf_idx) in enumerate(point_indices.items()):
        tau_prime_series = tau_w_prime[:, surf_idx]
        tau_rms_loc = tau_w_rms[surf_idx]
        tau_prime_norm = tau_prime_series / tau_rms_loc if tau_rms_loc > 1e-10 else tau_prime_series
        
        counts, bins = np.histogram(tau_prime_norm, bins=50, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        ax.plot(bin_centers, counts, '-', color=colors[idx], linewidth=2, 
               label=f'x/c = {x_c_target:.2f}', marker='o', markersize=4)
    
    ax.set_xlabel("$\\tau'_w / \\tau'_{w,rms}$", fontsize=14)
    ax.set_ylabel('PDF', fontsize=14)
    ax.set_title('Probability Density Function', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean vs RMS
    ax = axes[0, 1]
    ax.scatter(tau_w_mean[upper_mask], tau_w_rms[upper_mask], 
              c='red', s=20, alpha=0.6, label='Suction side')
    ax.scatter(tau_w_mean[lower_mask], tau_w_rms[lower_mask], 
              c='blue', s=20, alpha=0.6, label='Pressure side')
    
    ax.set_xlabel('$\\langle \\tau_w \\rangle$', fontsize=14)
    ax.set_ylabel("$\\tau'_{w,rms}$", fontsize=14)
    ax.set_title('Mean vs RMS Shear Stress', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Skewness along chord
    ax = axes[1, 0]
    
    # Compute skewness at each surface point
    skewness = np.zeros(N_surf)
    for i in range(N_surf):
        tau_prime_i = tau_w_prime[:, i]
        tau_rms_i = tau_w_rms[i]
        if tau_rms_i > 1e-10:
            skewness[i] = np.mean((tau_prime_i / tau_rms_i)**3)
    
    x_upper = x_over_c[upper_mask]
    x_lower = x_over_c[lower_mask]
    skew_upper = skewness[upper_mask]
    skew_lower = skewness[lower_mask]
    
    sort_idx_upper = np.argsort(x_upper)
    sort_idx_lower = np.argsort(x_lower)
    
    ax.plot(x_upper[sort_idx_upper], skew_upper[sort_idx_upper], 
           'r-', linewidth=2, label='Suction side', marker='o', markersize=2)
    ax.plot(x_lower[sort_idx_lower], skew_lower[sort_idx_lower], 
           'b-', linewidth=2, label='Pressure side', marker='o', markersize=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlabel('x/c', fontsize=14)
    ax.set_ylabel('Skewness', fontsize=14)
    ax.set_title('Fluctuation Skewness Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Kurtosis along chord
    ax = axes[1, 1]
    
    # Compute kurtosis at each surface point
    kurtosis = np.zeros(N_surf)
    for i in range(N_surf):
        tau_prime_i = tau_w_prime[:, i]
        tau_rms_i = tau_w_rms[i]
        if tau_rms_i > 1e-10:
            kurtosis[i] = np.mean((tau_prime_i / tau_rms_i)**4)
    
    kurt_upper = kurtosis[upper_mask]
    kurt_lower = kurtosis[lower_mask]
    
    ax.plot(x_upper[sort_idx_upper], kurt_upper[sort_idx_upper], 
           'r-', linewidth=2, label='Suction side', marker='o', markersize=2)
    ax.plot(x_lower[sort_idx_lower], kurt_lower[sort_idx_lower], 
           'b-', linewidth=2, label='Pressure side', marker='o', markersize=2)
    ax.axhline(y=3, color='gray', linestyle='--', linewidth=1, label='Gaussian')
    
    ax.set_xlabel('x/c', fontsize=14)
    ax.set_ylabel('Kurtosis', fontsize=14)
    ax.set_title('Fluctuation Kurtosis Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Statistical Analysis - NACA 0012, AoA={AOA}°, Re={Re_c:,}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # stats_path = os.path.join(OUTPUT_DIR, f'tau_w_statistics_AoA{AOA}.png')
    # plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    # print(f"  Saved: {stats_path}")
    # plt.close()
    plt.show()

# ============================================================================
# Save processed data
# ============================================================================
print("\n" + "=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)

output_data_file = os.path.join(OUTPUT_DIR, f'shear_fluctuations_data_AoA{AOA}.h5')

with h5py.File(output_data_file, 'w') as f:
    # Geometrical data
    f.create_dataset('x_interface', data=x_interface)
    f.create_dataset('y_interface', data=y_interface)
    f.create_dataset('upper_mask', data=upper_mask)
    f.create_dataset('lower_mask', data=lower_mask)
    
    # Statistical data
    f.create_dataset('tau_w_mean', data=tau_w_mean)
    f.create_dataset('tau_w_rms', data=tau_w_rms)
    f.create_dataset('tau_w_snapshots', data=tau_w_snapshots)
    f.create_dataset('tau_w_prime', data=tau_w_prime)
    
    # Metadata
    f.attrs['N_snapshots'] = N_snapshots
    f.attrs['N_surf'] = N_surf
    f.attrs['AOA'] = AOA
    f.attrs['Re_c'] = Re_c
    f.attrs['c'] = c
    f.attrs['u_infty'] = u_infty

print(f"  Data saved: {output_data_file}")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nAll plots saved to: {OUTPUT_DIR}")
print("\nGenerated visualizations:")
if VISUALIZE_RMS:
    print("  1. RMS distribution along chord")
if VISUALIZE_INSTANTANEOUS:
    print("  2. Instantaneous fluctuation snapshots")
if VISUALIZE_SPACETIME:
    print("  3. Space-time evolution diagrams")
if VISUALIZE_STATISTICS:
    print("  4. Statistical distributions (PDF, skewness, kurtosis)")
print("\n" + "=" * 70)
