import os
import h5py
import numpy as np
from glob import glob

# ============================================================================
# Configuration
# ============================================================================

# Base directory containing all batch folders
BASE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"

# Pattern to match batch directories
BATCH_PATTERN = "batch_*"

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Mean_Shear_Stress/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Load geometrical data
# ============================================================================
print("\n" + "=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]
    
N_surf = len(interface_points)
print(f"  Number of 2D interface points: {N_surf}")

# ============================================================================
# Find all surface data files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR SURFACE DATA FILES")
print("=" * 70)

batch_dirs = sorted(glob(os.path.join(BASE_DIR, BATCH_PATTERN)))
print(f"  Found {len(batch_dirs)} batch directories")

all_surface_files = []
for batch_dir in batch_dirs:
    surface_dir = os.path.join(batch_dir, "Surface_data")
    if not os.path.exists(surface_dir):
        continue
    
    surface_files = sorted(glob(os.path.join(surface_dir, "surface_*.h5")))
    all_surface_files.extend(surface_files)

N_total_snapshots = len(all_surface_files)
print(f"  Total surface data files: {N_total_snapshots}")

if N_total_snapshots == 0:
    raise RuntimeError("No surface data files found!")



# ============================================================================
# Load all snapshots and compute mean (for fluctuations)
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING TIME-AVERAGED 2D FIELDS")
print("=" * 70)

tau_w_2d_sum = None
n_snapshots = 0

print(f"Loading {N_total_snapshots} snapshots to compute mean...")

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Loading snapshot {idx+1}/{N_total_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:

            tau_w = f["tau_w"][:]    # (Nz_phys, N_surf)
            
            # Spanwise average for each snapshot
            tau_w_2d = np.mean(tau_w, axis=0)       # (N_surf,)

            if tau_w_2d_sum is None:
                tau_w_2d_sum = tau_w_2d.copy()
                Nz_phys = tau_w.shape[0]  # Store for later
            else:
                tau_w_2d_sum += tau_w_2d

            n_snapshots += 1
            
    except Exception as e:
        print(f"  [WARNING] Error loading {surface_file}: {e}")
        continue

if n_snapshots == 0:
    raise RuntimeError("No valid snapshots loaded; check surface files and datasets.")

# Compute 2D time-averaged mean
tau_w_mean = tau_w_2d_sum / n_snapshots  # (N_surf,)

print(f"  Successfully loaded {n_snapshots} snapshots")
print(f"  2D mean shape: (N_surf={len(tau_w_mean)})")
print(f"  Spanwise planes in each snapshot: Nz={Nz_phys}")


# ============================================================================
# Save results to HDF5 file
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

output_file = os.path.join(OUTPUT_DIR, "mean_wall_shear_stress.h5")

with h5py.File(output_file, "w") as f:
    # Save mean wall shear stress
    f.create_dataset("tau_w_mean", data=tau_w_mean, compression="gzip", compression_opts=4)

print(f"  Results saved to: {output_file}")

