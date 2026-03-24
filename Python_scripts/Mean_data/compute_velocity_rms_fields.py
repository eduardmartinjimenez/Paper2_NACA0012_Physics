import os
import sys
import h5py
import numpy as np
from glob import glob
import gc
import time

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Configuration
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING u'_rms OF STREAMWISE VELOCITY FLUCTUATIONS")
print("=" * 70)

# Base directories
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/"
LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"

MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = "u_rms_streamwise_velocity_fluctuations.h5"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Reference parameters
u_infty = 1.0
AOA = 12  # degrees
AOA_rad = np.deg2rad(AOA)

# ============================================================================
# Load mean velocity from averaged snapshot
# ============================================================================
print("\n" + "=" * 70)
print("LOADING MEAN VELOCITY FIELD")
print("=" * 70)

if not os.path.exists(LAST_SNAPSHOT_FILE):
    raise FileNotFoundError(f"Mean snapshot file not found: {LAST_SNAPSHOT_FILE}")

if not os.path.exists(MESH_FILE):
    raise FileNotFoundError(f"Mesh file not found: {MESH_FILE}")

# Initialize loader and load mesh
loader = CompressedSnapshotLoader(MESH_FILE)

# Load mean fields
print(f"Loading mean snapshot from: {LAST_SNAPSHOT_FILE}")
fields_mean = loader.load_snapshot_avg(LAST_SNAPSHOT_FILE)

# Reconstruct 3D mean velocity fields
u_mean_3d = loader.reconstruct_field(fields_mean["avg_u"])  # (Nz, Ny, Nx)
v_mean_3d = loader.reconstruct_field(fields_mean["avg_v"])  # (Nz, Ny, Nx)

# Compute streamwise mean velocity (aligned with angle of attack)
# V_streamwise = u*cos(AOA) + v*sin(AOA)
u_mean_3d = u_mean_3d * np.cos(AOA_rad) + v_mean_3d * np.sin(AOA_rad)

print(f"Mean velocity field shape: {u_mean_3d.shape}")

# ============================================================================
# Find all snapshot files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR SNAPSHOT FILES")
print("=" * 70)

batch_dirs = sorted(glob(os.path.join(BASE_SNAPSHOT_DIR, "batch_*")))
print(f"Found {len(batch_dirs)} batch directories")

all_snapshot_files = []
for batch_dir in batch_dirs:
    if not os.path.exists(batch_dir):
        continue

    snapshot_files = sorted(glob(os.path.join(batch_dir, "*A.h5")))
    all_snapshot_files.extend(snapshot_files)

N_total_snapshots = len(all_snapshot_files)
print(f"Total snapshot files: {N_total_snapshots}")

if N_total_snapshots == 0:
    raise RuntimeError("No snapshot files found!")

# ============================================================================
# Initialize accumulation arrays for u'^2
# ============================================================================
print("\n" + "=" * 70)
print("INITIALIZING ACCUMULATION ARRAYS")
print("=" * 70)

u_prime_sq_accum = None
n_snapshots_processed = 0

print("Computing u'_rms...")
start_time = time.perf_counter()

# ============================================================================
# Loop through all snapshots and accumulate u'^2
# ============================================================================
print("\n" + "=" * 70)
print("PROCESSING SNAPSHOTS")
print("=" * 70)

for idx, snapshot_file in enumerate(all_snapshot_files):
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"Processing snapshot {idx+1}/{N_total_snapshots}: {os.path.basename(snapshot_file)}", flush=True)

    try:
        # Load instantaneous velocity fields
        fields_inst = loader.load_snapshot(snapshot_file)
        u_inst_3d = loader.reconstruct_field(fields_inst["u"])  # (Nz, Ny, Nx)
        v_inst_3d = loader.reconstruct_field(fields_inst["v"])  # (Nz, Ny, Nx)

        # Compute streamwise instantaneous velocity
        u_inst_3d = u_inst_3d * np.cos(AOA_rad) + v_inst_3d * np.sin(AOA_rad)

        # Initialize accumulator on first snapshot
        if u_prime_sq_accum is None:
            u_prime_sq_accum = np.zeros_like(u_inst_3d, dtype=np.float64)
            print(f"Initialized accumulator with shape: {u_prime_sq_accum.shape}")

        # Compute fluctuation: u' = u_inst - u_mean
        u_prime_3d = u_inst_3d - u_mean_3d

        # Accumulate squared fluctuations
        u_prime_sq_accum += u_prime_3d ** 2

        n_snapshots_processed += 1

        # Clean up
        del fields_inst, u_inst_3d, v_inst_3d, u_prime_3d
        gc.collect()

    except Exception as e:
        print(f"[WARNING] Error processing {snapshot_file}: {e}")
        continue

print(f"\nSuccessfully processed {n_snapshots_processed} snapshots")

# ============================================================================
# Compute u'_rms from accumulated values
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING u'_rms")
print("=" * 70)

if n_snapshots_processed > 0:
    # Compute RMS: u'_rms = sqrt(<u'^2>)
    u_rms_3d = np.sqrt(u_prime_sq_accum / n_snapshots_processed)

    print(f"u'_rms 3D shape before averaging: {u_rms_3d.shape}")

    # Average in spanwise direction (z-axis, axis=0) to exploit periodicity
    u_rms_2d = np.mean(u_rms_3d, axis=0)

    print(f"u'_rms 2D shape (averaged in spanwise): {u_rms_2d.shape}")
    print(f"u'_rms min: {np.nanmin(u_rms_2d):.6e}")
    print(f"u'_rms max: {np.nanmax(u_rms_2d):.6e}")
    print(f"u'_rms mean: {np.nanmean(u_rms_2d):.6e}")

    # ========================================================================
    # Save results to HDF5
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS TO HDF5")
    print("=" * 70)

    print(f"Saving to: {OUTPUT_FILE}")

    with h5py.File(OUTPUT_FILE, "w") as f:
        # Metadata
        f.attrs["u_infty"] = u_infty
        f.attrs["AOA"] = AOA
        f.attrs["AOA_rad"] = AOA_rad
        f.attrs["n_snapshots"] = n_snapshots_processed
        f.attrs["description"] = "RMS of streamwise velocity fluctuations (2D, averaged in spanwise direction)"
        f.attrs["formula"] = "u'_rms = sqrt(<u'^2>) where u' = u_inst - <u>"

        # Save RMS field (2D)
        f.create_dataset("u_rms", data=u_rms_2d, compression="gzip")

        # Save 2D coordinates for reference (average in spanwise direction)
        x_data_2d = np.mean(loader.x[1:-1, :, :], axis=0)
        y_data_2d = np.mean(loader.y[1:-1, :, :], axis=0)

        f.create_dataset("x", data=x_data_2d, compression="gzip")
        f.create_dataset("y", data=y_data_2d, compression="gzip")

    print("✓ Data saved successfully!")

    # Verification
    with h5py.File(OUTPUT_FILE, "r") as f:
        print(f"\nVerification:")
        print(f"  Datasets in file: {list(f.keys())}")
        print(f"  u_rms shape: {f['u_rms'].shape}")
        print(f"  Metadata: {dict(f.attrs)}")

else:
    print("ERROR: No snapshots processed!")

elapsed_total = time.perf_counter() - start_time
print("\n" + "=" * 70)
print("COMPUTATION COMPLETE")
print("=" * 70)
print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
print(f"Snapshots processed: {n_snapshots_processed}")
if n_snapshots_processed > 0:
    print(f"Time per snapshot: {elapsed_total/n_snapshots_processed:.2f}s")
print(f"Output file: {OUTPUT_FILE}")
