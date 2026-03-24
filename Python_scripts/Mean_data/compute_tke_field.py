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
print("COMPUTING TURBULENT KINETIC ENERGY (TKE)")
print("=" * 70)

# Base directories
BASE_SNAPSHOT_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Steady_state/"
LAST_SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/last_snapshot/"
LAST_SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA5_avg_25340000-COMP-DATA.h5"

MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

LAST_SNAPSHOT_FILE = os.path.join(LAST_SNAPSHOT_PATH, LAST_SNAPSHOT_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = "tke_turbulent_kinetic_energy.h5"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Reference parameters
u_infty = 1.0
AOA = 5  # degrees
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
w_mean_3d = loader.reconstruct_field(fields_mean["avg_w"])  # (Nz, Ny, Nx)

# Rotate u, v to flow-aligned frame (streamwise and normal-to-streamwise)
# u_rot = u*cos(AOA) + v*sin(AOA)
# v_rot = -u*sin(AOA) + v*cos(AOA)
u_mean_rot = u_mean_3d * np.cos(AOA_rad) + v_mean_3d * np.sin(AOA_rad)
v_mean_rot = -u_mean_3d * np.sin(AOA_rad) + v_mean_3d * np.cos(AOA_rad)
w_mean_rot = w_mean_3d  # Spanwise direction unchanged

print(f"Mean velocity field shape: u={u_mean_rot.shape}, v={v_mean_rot.shape}, w={w_mean_rot.shape}")

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
# Initialize accumulation arrays for velocity variance components
# ============================================================================
print("\n" + "=" * 70)
print("INITIALIZING ACCUMULATION ARRAYS")
print("=" * 70)

u_prime_sq_accum = None
v_prime_sq_accum = None
w_prime_sq_accum = None
n_snapshots_processed = 0

print("Computing TKE = 0.5 * (<u'^2> + <v'^2> + <w'^2>)...")
start_time = time.perf_counter()

# ============================================================================
# Loop through all snapshots and accumulate velocity variances
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
        w_inst_3d = loader.reconstruct_field(fields_inst["w"])  # (Nz, Ny, Nx)

        # Rotate to flow-aligned frame
        u_inst_rot = u_inst_3d * np.cos(AOA_rad) + v_inst_3d * np.sin(AOA_rad)
        v_inst_rot = -u_inst_3d * np.sin(AOA_rad) + v_inst_3d * np.cos(AOA_rad)
        w_inst_rot = w_inst_3d  # Spanwise direction unchanged

        # Initialize accumulators on first snapshot
        if u_prime_sq_accum is None:
            u_prime_sq_accum = np.zeros_like(u_inst_rot, dtype=np.float64)
            v_prime_sq_accum = np.zeros_like(v_inst_rot, dtype=np.float64)
            w_prime_sq_accum = np.zeros_like(w_inst_rot, dtype=np.float64)
            print(f"Initialized accumulators with shape: {u_prime_sq_accum.shape}")

        # Compute fluctuations in flow-aligned frame
        u_prime_3d = u_inst_rot - u_mean_rot
        v_prime_3d = v_inst_rot - v_mean_rot
        w_prime_3d = w_inst_rot - w_mean_rot

        # Accumulate squared fluctuations
        u_prime_sq_accum += u_prime_3d ** 2
        v_prime_sq_accum += v_prime_3d ** 2
        w_prime_sq_accum += w_prime_3d ** 2

        n_snapshots_processed += 1

        # Clean up
        del fields_inst, u_inst_3d, v_inst_3d, w_inst_3d
        del u_inst_rot, v_inst_rot, w_inst_rot
        del u_prime_3d, v_prime_3d, w_prime_3d
        gc.collect()

    except Exception as e:
        print(f"[WARNING] Error processing {snapshot_file}: {e}")
        continue

print(f"\nSuccessfully processed {n_snapshots_processed} snapshots")

# ============================================================================
# Compute TKE from accumulated values
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING TKE")
print("=" * 70)

if n_snapshots_processed > 0:
    # Compute RMS (standard deviations) of each component
    u_prime_rms_3d = np.sqrt(u_prime_sq_accum / n_snapshots_processed)
    v_prime_rms_3d = np.sqrt(v_prime_sq_accum / n_snapshots_processed)
    w_prime_rms_3d = np.sqrt(w_prime_sq_accum / n_snapshots_processed)

    print(f"u'_rms 3D shape: {u_prime_rms_3d.shape}")
    print(f"v'_rms 3D shape: {v_prime_rms_3d.shape}")
    print(f"w'_rms 3D shape: {w_prime_rms_3d.shape}")

    # Compute TKE = 0.5 * (u'^2 + v'^2 + w'^2)
    # Using variances: TKE = 0.5 * (<u'²> + <v'²> + <w'²>)
    tke_3d = 0.5 * (u_prime_sq_accum + v_prime_sq_accum + w_prime_sq_accum) / n_snapshots_processed

    print(f"TKE 3D shape before averaging: {tke_3d.shape}")

    # Average in spanwise direction (z-axis, axis=0) to exploit periodicity
    tke_2d = np.mean(tke_3d, axis=0)
    u_prime_rms_2d = np.mean(u_prime_rms_3d, axis=0)
    v_prime_rms_2d = np.mean(v_prime_rms_3d, axis=0)
    w_prime_rms_2d = np.mean(w_prime_rms_3d, axis=0)

    print(f"TKE 2D shape (averaged in spanwise): {tke_2d.shape}")
    print(f"TKE min: {np.nanmin(tke_2d):.6e}")
    print(f"TKE max: {np.nanmax(tke_2d):.6e}")
    print(f"TKE mean: {np.nanmean(tke_2d):.6e}")

    print(f"\nu'_rms 2D min: {np.nanmin(u_prime_rms_2d):.6e}, max: {np.nanmax(u_prime_rms_2d):.6e}")
    print(f"v'_rms 2D min: {np.nanmin(v_prime_rms_2d):.6e}, max: {np.nanmax(v_prime_rms_2d):.6e}")
    print(f"w'_rms 2D min: {np.nanmin(w_prime_rms_2d):.6e}, max: {np.nanmax(w_prime_rms_2d):.6e}")

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
        f.attrs["description"] = "Turbulent kinetic energy (2D, averaged in spanwise direction)"
        f.attrs["formula"] = "TKE = 0.5 * (<u'^2> + <v'^2> + <w'^2>) in flow-aligned frame"

        # Save TKE field (2D)
        f.create_dataset("tke", data=tke_2d, compression="gzip")

        # Save individual RMS components (2D)
        f.create_dataset("u_prime_rms", data=u_prime_rms_2d, compression="gzip")
        f.create_dataset("v_prime_rms", data=v_prime_rms_2d, compression="gzip")
        f.create_dataset("w_prime_rms", data=w_prime_rms_2d, compression="gzip")

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
        print(f"  TKE shape: {f['tke'].shape}")
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
