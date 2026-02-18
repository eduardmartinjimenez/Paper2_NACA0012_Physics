import os
import sys
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Configuration
# ============================================================================

# Base directory containing all batch folders
BASE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"

# Pattern to match batch directories (e.g., batch_34327986, batch_34327987, etc.)
BATCH_PATTERN = "batch_*"

# Geometrical data file (to get interface points and chord positions)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# # Output directory
# OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
# OUTPUT_NAME_CP = "Cp_chord_from_surface_data.h5"
# OUTPUT_NAME_CF = "Cf_chord_from_surface_data.h5"
# OUTPUT_PATH_CP = os.path.join(OUTPUT_DIR, OUTPUT_NAME_CP)
# OUTPUT_PATH_CF = os.path.join(OUTPUT_DIR, OUTPUT_NAME_CF)

# Reference parameters
rho_ref = 1.0   # Reference density [kg/m3]
u_infty = 1.0   # Free-stream velocity [m/s]
c = 1.0         # Airfoil chord length [m]
Re_c = 50000    # Reynolds number [-]
q_inf = 0.5 * rho_ref * u_infty**2  # Dynamic pressure [Pa]

AOA = 12  # Angle of attack [degrees]

# ============================================================================
# Utilities
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"  [OK] {kind}: {path}")

# ============================================================================
# Load geometrical data
# ============================================================================
print("=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

assert_exists(GEO_FILE, "Geometrical data")

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]    # (N_surf, 3)
    proj_points      = f["proj_points"][:]         # (N_surf, 3)
    proj_normals     = f["proj_normals"][:]        # (N_surf, 3)
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

N_surf = len(interface_points)
print(f"  Number of 2D interface points: {N_surf}")

# Extract x, y coordinates for chord-wise analysis
x_interface = interface_points[:, 0]  # (N_surf,)
y_interface = interface_points[:, 1]  # (N_surf,)

# Compute x/c for each point
x_over_c = x_interface / c

# ============================================================================
# Find all batch directories and surface data files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR BATCH DIRECTORIES AND SURFACE DATA")
print("=" * 70)

# Find all batch directories
batch_dirs = sorted(glob(os.path.join(BASE_DIR, BATCH_PATTERN)))
print(f"  Found {len(batch_dirs)} batch directories:")
for batch_dir in batch_dirs:
    print(f"    - {os.path.basename(batch_dir)}")

if len(batch_dirs) == 0:
    raise RuntimeError(f"No batch directories found matching pattern: {os.path.join(BASE_DIR, BATCH_PATTERN)}")

# Collect all surface data files from all batches
all_surface_files = []
for batch_dir in batch_dirs:
    surface_dir = os.path.join(batch_dir, "Surface_data")
    if not os.path.exists(surface_dir):
        print(f"  [WARNING] Surface_data directory not found in {os.path.basename(batch_dir)}, skipping...")
        continue
    
    surface_files = sorted(glob(os.path.join(surface_dir, "surface_*.h5")))
    all_surface_files.extend(surface_files)
    print(f"    {os.path.basename(batch_dir)}/Surface_data: {len(surface_files)} files")

N_total_snapshots = len(all_surface_files)
print(f"\n  Total surface data files found: {N_total_snapshots}")

if N_total_snapshots == 0:
    raise RuntimeError("No surface data files found!")

# ============================================================================
# Load and average all surface data (BOTH p_w AND tau_w)
# ============================================================================
print("\n" + "=" * 70)
print("LOADING AND AVERAGING SURFACE DATA")
print("=" * 70)

# Initialize accumulators
p_w_sum = None
tau_w_sum = None  # ADD THIS
n_snapshots_loaded = 0

print(f"Processing {N_total_snapshots} surface data files...")

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"  Loading snapshot {idx+1}/{N_total_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:
            p_w = f["p_w"][:]      # (Nz_phys, N_surf)
            tau_w = f["tau_w"][:]  # (Nz_phys, N_surf) - ADD THIS
            
            if p_w_sum is None:
                p_w_sum = p_w.copy()
                tau_w_sum = tau_w.copy()  # ADD THIS
            else:
                p_w_sum += p_w
                tau_w_sum += tau_w  # ADD THIS
            
            n_snapshots_loaded += 1
            
    except Exception as e:
        print(f"  [WARNING] Error loading {os.path.basename(surface_file)}: {e}")
        continue

if n_snapshots_loaded == 0:
    raise RuntimeError("No surface data files were successfully loaded!")

print(f"  Successfully loaded {n_snapshots_loaded} snapshots")

# Compute time-averaged surface pressure and wall shear stress
p_w_avg = p_w_sum / n_snapshots_loaded      # (Nz_phys, N_surf)
tau_w_avg = tau_w_sum / n_snapshots_loaded  # (Nz_phys, N_surf) - ADD THIS

print(f"  Time-averaged surface pressure shape: {p_w_avg.shape}")
print(f"  Time-averaged wall shear stress shape: {tau_w_avg.shape}")  # ADD THIS

# ============================================================================
# Average over spanwise direction and compute Cp and Cf
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING Cp AND Cf ALONG CHORD")
print("=" * 70)

# Average over z-direction (spanwise)
p_w_2d = np.mean(p_w_avg, axis=0)      # (N_surf,)
tau_w_2d = np.mean(tau_w_avg, axis=0)  # (N_surf,) - ADD THIS

print(f"  2D surface pressure shape: {p_w_2d.shape}")
print(f"  2D wall shear stress shape: {tau_w_2d.shape}")  # ADD THIS
print(f"  p_w range: [{np.min(p_w_2d):.6e}, {np.max(p_w_2d):.6e}]")
print(f"  tau_w range: [{np.min(tau_w_2d):.6e}, {np.max(tau_w_2d):.6e}]")  # ADD THIS

# Compute pressure coefficient
# Since p_w = (p_surface - p_bulk), we have:
# Cp = (p_surface - p_bulk) / q_inf = p_w / q_inf
Cp_values = p_w_2d / q_inf

# Compute skin friction coefficient
# Cf = tau_w / q_inf
Cf_values = tau_w_2d / q_inf  # ADD THIS

print(f"  Cp range: [{np.min(Cp_values):.6f}, {np.max(Cp_values):.6f}]")
print(f"  Cf range: [{np.min(Cf_values):.6f}, {np.max(Cf_values):.6f}]")  # ADD THIS

# ============================================================================
# Organize by chord position and separate upper/lower surfaces
# ============================================================================
print("\n" + "=" * 70)
print("ORGANIZING DATA BY CHORD POSITION")
print("=" * 70)

# Separate upper and lower surfaces
# Use y-coordinate: upper surface has larger y, lower has smaller y
y_mean = np.mean(y_interface)
upper_mask = y_interface > y_mean
lower_mask = ~upper_mask

# Upper surface
x_c_upper = x_over_c[upper_mask]
Cp_upper = Cp_values[upper_mask]
Cf_upper = Cf_values[upper_mask]  # ADD THIS
y_upper = y_interface[upper_mask]

# Sort by x/c
sort_idx_upper = np.argsort(x_c_upper)
x_c_upper = x_c_upper[sort_idx_upper]
Cp_upper = Cp_upper[sort_idx_upper]
Cf_upper = Cf_upper[sort_idx_upper]  # ADD THIS
y_upper = y_upper[sort_idx_upper]

# Lower surface
x_c_lower = x_over_c[lower_mask]
Cp_lower = Cp_values[lower_mask]
Cf_lower = Cf_values[lower_mask]  # ADD THIS
y_lower = y_interface[lower_mask]

# Sort by x/c
sort_idx_lower = np.argsort(x_c_lower)
x_c_lower = x_c_lower[sort_idx_lower]
Cp_lower = Cp_lower[sort_idx_lower]
Cf_lower = Cf_lower[sort_idx_lower]  # ADD THIS
y_lower = y_lower[sort_idx_lower]

print(f"  Upper surface points: {len(x_c_upper)}")
print(f"  Lower surface points: {len(x_c_lower)}")
print(f"  Upper Cp range: [{np.min(Cp_upper):.6f}, {np.max(Cp_upper):.6f}]")
print(f"  Lower Cp range: [{np.min(Cp_lower):.6f}, {np.max(Cp_lower):.6f}]")
print(f"  Upper Cf range: [{np.min(Cf_upper):.6f}, {np.max(Cf_upper):.6f}]")  # ADD THIS
print(f"  Lower Cf range: [{np.min(Cf_lower):.6f}, {np.max(Cf_lower):.6f}]")  # ADD THIS

# ============================================================================
# Save results
# ============================================================================
# print("\n" + "=" * 70)
# print("SAVING RESULTS")
# print("=" * 70)

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Save Cp data
# with h5py.File(OUTPUT_PATH_CP, "w") as f:
#     # All surface data
#     f.create_dataset("x_over_c_all", data=x_over_c)
#     f.create_dataset("Cp_all", data=Cp_values)
#     f.create_dataset("p_w_all", data=p_w_2d)
#     f.create_dataset("interface_points", data=interface_points)
    
#     # Upper surface (sorted by x/c)
#     f.create_dataset("x_over_c_upper", data=x_c_upper)
#     f.create_dataset("Cp_upper", data=Cp_upper)
#     f.create_dataset("y_upper", data=y_upper)
    
#     # Lower surface (sorted by x/c)
#     f.create_dataset("x_over_c_lower", data=x_c_lower)
#     f.create_dataset("Cp_lower", data=Cp_lower)
#     f.create_dataset("y_lower", data=y_lower)
    
#     # 3D spanwise-resolved data (for further analysis if needed)
#     grp_3d = f.create_group("spanwise_resolved")
#     grp_3d.create_dataset("p_w_3d", data=p_w_avg, compression="gzip")
#     grp_3d.create_dataset("Cp_3d", data=p_w_avg / q_inf, compression="gzip")
    
#     # Metadata
#     f.attrs["rho_ref"] = rho_ref
#     f.attrs["u_infty"] = u_infty
#     f.attrs["q_inf"] = q_inf
#     f.attrs["chord"] = c
#     f.attrs["Re"] = Re_c
#     f.attrs["AOA"] = AOA
#     f.attrs["n_snapshots"] = n_snapshots_loaded
#     f.attrs["n_batches"] = len(batch_dirs)

# print(f"  Cp results saved to: {OUTPUT_PATH_CP}")

# # Save Cf data - ADD THIS ENTIRE BLOCK
# with h5py.File(OUTPUT_PATH_CF, "w") as f:
#     # All surface data
#     f.create_dataset("x_over_c_all", data=x_over_c)
#     f.create_dataset("Cf_all", data=Cf_values)
#     f.create_dataset("tau_w_all", data=tau_w_2d)
#     f.create_dataset("interface_points", data=interface_points)
    
#     # Upper surface (sorted by x/c)
#     f.create_dataset("x_over_c_upper", data=x_c_upper)
#     f.create_dataset("Cf_upper", data=Cf_upper)
#     f.create_dataset("y_upper", data=y_upper)
    
#     # Lower surface (sorted by x/c)
#     f.create_dataset("x_over_c_lower", data=x_c_lower)
#     f.create_dataset("Cf_lower", data=Cf_lower)
#     f.create_dataset("y_lower", data=y_lower)
    
#     # 3D spanwise-resolved data (for further analysis if needed)
#     grp_3d = f.create_group("spanwise_resolved")
#     grp_3d.create_dataset("tau_w_3d", data=tau_w_avg, compression="gzip")
#     grp_3d.create_dataset("Cf_3d", data=tau_w_avg / q_inf, compression="gzip")
    
#     # Metadata
#     f.attrs["rho_ref"] = rho_ref
#     f.attrs["u_infty"] = u_infty
#     f.attrs["q_inf"] = q_inf
#     f.attrs["chord"] = c
#     f.attrs["Re"] = Re_c
#     f.attrs["AOA"] = AOA
#     f.attrs["n_snapshots"] = n_snapshots_loaded
#     f.attrs["n_batches"] = len(batch_dirs)

# print(f"  Cf results saved to: {OUTPUT_PATH_CF}")

# ============================================================================
# Create plots
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Plot 1: Cp distribution along chord
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_c_upper, Cp_upper, 'b-o', label='Upper surface', markersize=4, linewidth=1.5)
ax.plot(x_c_lower, Cp_lower, 'r-s', label='Lower surface', markersize=4, linewidth=1.5)

ax.set_xlabel('x/c', fontsize=14)
ax.set_ylabel('$C_p$', fontsize=14)
ax.set_title(f'Pressure Coefficient Distribution - NACA 0012\n' + 
             f'AoA = {AOA}°, Re = {Re_c:,}, {n_snapshots_loaded} snapshots averaged',
             fontsize=15)
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.invert_yaxis()  # Convention: negative Cp at top

plt.tight_layout()

# plot1_path = os.path.join(OUTPUT_DIR, "Cp_chord_distribution.png")
# plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
# print(f"  Plot saved: {plot1_path}")

plt.show()

# Plot 2: Cf distribution along chord - ADD THIS ENTIRE BLOCK
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_c_upper, Cf_upper, 'b-o', label='Upper surface', markersize=4, linewidth=1.5)
ax.plot(x_c_lower, Cf_lower, 'r-s', label='Lower surface', markersize=4, linewidth=1.5)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xlabel('x/c', fontsize=14)
ax.set_ylabel('$C_f$', fontsize=14)
ax.set_title(f'Skin Friction Coefficient Distribution - NACA 0012\n' + 
             f'AoA = {AOA}°, Re = {Re_c:,}, {n_snapshots_loaded} snapshots averaged',
             fontsize=15)
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# plot2_path = os.path.join(OUTPUT_DIR, "Cf_chord_distribution.png")
# plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
# print(f"  Plot saved: {plot2_path}")

plt.show()

# Plot 3: Combined Cp and Cf - ADD THIS ENTIRE BLOCK
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Top: Cp distribution
ax1.plot(x_c_upper, Cp_upper, 'b-o', label='Upper surface', markersize=3, linewidth=1.5)
ax1.plot(x_c_lower, Cp_lower, 'r-s', label='Lower surface', markersize=3, linewidth=1.5)
ax1.set_ylabel('$C_p$', fontsize=14)
ax1.set_title(f'Pressure and Skin Friction Coefficients - NACA 0012, AoA = {AOA}°, Re = {Re_c:,}',
              fontsize=15)
ax1.legend(loc='best', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.invert_yaxis()

# Bottom: Cf distribution
ax2.plot(x_c_upper, Cf_upper, 'b-o', label='Upper surface', markersize=3, linewidth=1.5)
ax2.plot(x_c_lower, Cf_lower, 'r-s', label='Lower surface', markersize=3, linewidth=1.5)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('x/c', fontsize=14)
ax2.set_ylabel('$C_f$', fontsize=14)
ax2.legend(loc='best', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim([0, 1])

plt.tight_layout()

# plot3_path = os.path.join(OUTPUT_DIR, "Cp_Cf_combined.png")
# plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
# print(f"  Plot saved: {plot3_path}")

plt.show()

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"  Number of batches processed: {len(batch_dirs)}")
print(f"  Total snapshots averaged: {n_snapshots_loaded}")
print(f"  Number of surface points: {N_surf}")
print(f"    - Upper surface: {len(x_c_upper)}")
print(f"    - Lower surface: {len(x_c_lower)}")

print(f"\n  Cp statistics:")
print(f"    Overall:       min = {np.min(Cp_values):.6f}, max = {np.max(Cp_values):.6f}")
print(f"    Upper surface: min = {np.min(Cp_upper):.6f}, max = {np.max(Cp_upper):.6f}")
print(f"    Lower surface: min = {np.min(Cp_lower):.6f}, max = {np.max(Cp_lower):.6f}")

# ADD THIS ENTIRE BLOCK
print(f"\n  Cf statistics:")
print(f"    Overall:       min = {np.min(Cf_values):.6f}, max = {np.max(Cf_values):.6f}")
print(f"    Upper surface: min = {np.min(Cf_upper):.6f}, max = {np.max(Cf_upper):.6f}")
print(f"    Lower surface: min = {np.min(Cf_lower):.6f}, max = {np.max(Cf_lower):.6f}")

# Check for flow separation (negative Cf indicates reversed flow)
sep_upper = np.sum(Cf_upper < 0)
sep_lower = np.sum(Cf_lower < 0)
if sep_upper > 0:
    x_sep_upper = x_c_upper[Cf_upper < 0]
    print(f"\n  [WARNING] Possible separation on upper surface at {sep_upper} points")
    print(f"            x/c range: [{np.min(x_sep_upper):.4f}, {np.max(x_sep_upper):.4f}]")
if sep_lower > 0:
    x_sep_lower = x_c_lower[Cf_lower < 0]
    print(f"  [WARNING] Possible separation on lower surface at {sep_lower} points")
    print(f"            x/c range: [{np.min(x_sep_lower):.4f}, {np.max(x_sep_lower):.4f}]")

# Compute integrated forces (optional)
def trapz_integrate(x, y):
    """Trapezoidal integration"""
    return np.trapz(y, x)

# Pressure force coefficient
Cp_upper_integral = trapz_integrate(x_c_upper, Cp_upper)
Cp_lower_integral = trapz_integrate(x_c_lower, Cp_lower)

# Friction force coefficient (integrated Cf gives friction drag)
Cf_upper_integral = trapz_integrate(x_c_upper, Cf_upper)
Cf_lower_integral = trapz_integrate(x_c_lower, Cf_lower)

print(f"\n  Integrated Cp (approx):")
print(f"    Upper surface: {Cp_upper_integral:.6f}")
print(f"    Lower surface: {Cp_lower_integral:.6f}")
print(f"    Difference (lift-related): {Cp_lower_integral - Cp_upper_integral:.6f}")

# ADD THIS BLOCK
print(f"\n  Integrated Cf (approx friction drag):")
print(f"    Upper surface: {Cf_upper_integral:.6f}")
print(f"    Lower surface: {Cf_lower_integral:.6f}")
print(f"    Total (friction drag): {Cf_upper_integral + Cf_lower_integral:.6f}")

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)

# ============================================================================
# Export to CSV for external plotting (optional)
# ============================================================================
# csv_path_cp = os.path.join(OUTPUT_DIR, "Cp_data.csv")
# with open(csv_path_cp, 'w') as f:
#     f.write("# Pressure coefficient distribution along chord\n")
#     f.write(f"# NACA 0012, AoA = {AOA} deg, Re = {Re_c}\n")
#     f.write(f"# Averaged over {n_snapshots_loaded} snapshots\n")
#     f.write("#\n")
#     f.write("# Upper surface:\n")
#     f.write("x/c,Cp,y/c\n")
#     for i in range(len(x_c_upper)):
#         f.write(f"{x_c_upper[i]:.8f},{Cp_upper[i]:.8f},{y_upper[i]:.8f}\n")
#     f.write("#\n")
#     f.write("# Lower surface:\n")
#     f.write("x/c,Cp,y/c\n")
#     for i in range(len(x_c_lower)):
#         f.write(f"{x_c_lower[i]:.8f},{Cp_lower[i]:.8f},{y_lower[i]:.8f}\n")

# print(f"\nCp CSV data exported to: {csv_path_cp}")

# # ADD THIS ENTIRE BLOCK
# csv_path_cf = os.path.join(OUTPUT_DIR, "Cf_data.csv")
# with open(csv_path_cf, 'w') as f:
#     f.write("# Skin friction coefficient distribution along chord\n")
#     f.write(f"# NACA 0012, AoA = {AOA} deg, Re = {Re_c}\n")
#     f.write(f"# Averaged over {n_snapshots_loaded} snapshots\n")
#     f.write("#\n")
#     f.write("# Upper surface:\n")
#     f.write("x/c,Cf,y/c\n")
#     for i in range(len(x_c_upper)):
#         f.write(f"{x_c_upper[i]:.8f},{Cf_upper[i]:.8f},{y_upper[i]:.8f}\n")
#     f.write("#\n")
#     f.write("# Lower surface:\n")
#     f.write("x/c,Cf,y/c\n")
#     for i in range(len(x_c_lower)):
#         f.write(f"{x_c_lower[i]:.8f},{Cf_lower[i]:.8f},{y_lower[i]:.8f}\n")

# print(f"Cf CSV data exported to: {csv_path_cf}")