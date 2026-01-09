import os
import sys
import h5py
import numpy as np
import glob
import gc
import time


module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Start timer
start_time = time.time()
start_time = time.perf_counter()

### SAVE RESULTS
# Save varibales: Sij_Sij_temporal_avg
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data"
SAVE_NAME = "3d_NACA0012_Re50000_AoA12_strain_rate_tensor_batch_30658504.h5"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# Snapshot directory
SNAPSHOT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/batch_30658504/"
h5_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*-COMP-DATA.h5")))
print(f"Found {len(h5_files)} snapshot files in {SNAPSHOT_DIR}")

# Load Mesh data
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME) 

# Load Average data
SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME_AVG = "3d_NACA0012_Re50000_AoA12_avg_15220000-COMP-DATA.h5"
SNAPSHOT_FILE_AVG = os.path.join(SNAPSHOT_PATH_AVG, SNAPSHOT_NAME_AVG)

# Check if the mesh file exists
if os.path.exists(MESH_FILE):
    print(f"Mesh file exists! {MESH_FILE}")
else:
    print(f"Mesh file does not exist: {MESH_FILE}")

# Check if the avg data file exists
if os.path.exists(SNAPSHOT_FILE_AVG):
    print(f"Average data file exists! {SNAPSHOT_FILE_AVG}")
else:
    print(f"Average Data file does not exist: {SNAPSHOT_FILE_AVG}")

# Import data files 
# Load mesh
loader = CompressedSnapshotLoader(MESH_FILE)

# Coordinates 
x = loader.x
y = loader.y
z = loader.z
print(f"Compressed mesh shape: {x.shape}")

# Load average fields
fields_avg = loader.load_snapshot_avg(SNAPSHOT_FILE_AVG)

u_mean = loader.reconstruct_field(fields_avg["avg_u"])
v_mean = loader.reconstruct_field(fields_avg["avg_v"])
w_mean = loader.reconstruct_field(fields_avg["avg_w"])
print("Loaded mean velocity fields from snapshot")

### COMPUTE EPSILON PRIME SNAPSHOT BY SNAPSHOT
Sij_Sij_accumulator = None
snapshot_count = 0

for file in h5_files:
    with h5py.File(file, "r") as data_file:
        print(f"Processing file: {file}")
        # Load instantaneous snapshot fields
        fields = loader.load_snapshot(file)

        # Read instantaneous and mean velocity fields
        u = loader.reconstruct_field(fields["u"])
        v = loader.reconstruct_field(fields["v"])
        w = loader.reconstruct_field(fields["w"])
        print("Loaded instantaneous velocity fields for snapshot.")

        # Mask invalid data (NaNs)
        valid_mask = ~np.isnan(u)

        # Compute fluctuating velocity components
        u_prime = np.where(valid_mask, u - u_mean, 0.0)
        v_prime = np.where(valid_mask, v - v_mean, 0.0)
        w_prime = np.where(valid_mask, w - w_mean, 0.0)
        print("Computed fluctuating velocity components.")

        # Compute velocity gradients
        du_dx = (u_prime[1:-1, 1:-1, 2:] - u_prime[1:-1, 1:-1, :-2]) / (
            x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]
        )

        dv_dx = (v_prime[1:-1, 1:-1, 2:] - v_prime[1:-1, 1:-1, :-2]) / (
            x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]
        )
        dw_dx = (w_prime[1:-1, 1:-1, 2:] - w_prime[1:-1, 1:-1, :-2]) / (
            x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]
        )

        du_dy = (u_prime[1:-1, 2:, 1:-1] - u_prime[1:-1, :-2, 1:-1]) / (
            y[1:-1, 2:, 1:-1] - y[1:-1, :-2, 1:-1]
        )
        dv_dy = (v_prime[1:-1, 2:, 1:-1] - v_prime[1:-1, :-2, 1:-1]) / (
            y[1:-1, 2:, 1:-1] - y[1:-1, :-2, 1:-1]
        )
        dw_dy = (w_prime[1:-1, 2:, 1:-1] - w_prime[1:-1, :-2, 1:-1]) / (
            y[1:-1, 2:, 1:-1] - y[1:-1, :-2, 1:-1]
        )

        du_dz = (u_prime[2:, 1:-1, 1:-1] - u_prime[:-2, 1:-1, 1:-1]) / (
            z[2:, 1:-1, 1:-1] - z[:-2, 1:-1, 1:-1]
        )
        dv_dz = (v_prime[2:, 1:-1, 1:-1] - v_prime[:-2, 1:-1, 1:-1]) / (
            z[2:, 1:-1, 1:-1] - z[:-2, 1:-1, 1:-1]
        )
        dw_dz = (w_prime[2:, 1:-1, 1:-1] - w_prime[:-2, 1:-1, 1:-1]) / (
            z[2:, 1:-1, 1:-1] - z[:-2, 1:-1, 1:-1]
        )
        print("Computed velocity gradients.")

        del u_prime, v_prime, w_prime
        gc.collect()

        # Compute Strain-rate tensor
        S11_prime = du_dx
        S22_prime = dv_dy
        S33_prime = dw_dz
        S12_prime = 0.5 * (du_dy + dv_dx)
        S13_prime = 0.5 * (du_dz + dw_dx)
        S23_prime = 0.5 * (dv_dz + dw_dy)
    

        del du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz
        gc.collect()

        Sij_Sij = (
            S11_prime**2
            + S22_prime**2
            + S33_prime**2
            + 2 * (S12_prime**2 + S13_prime**2 + S23_prime**2)
        )
        print("Computed Sij_Sij for current snapshot.")

        del (
            S11_prime,
            S22_prime,
            S33_prime,
            S12_prime,
            S13_prime,
            S23_prime,
        )
        gc.collect()

        # Accumulate Sij_Sij
        if Sij_Sij_accumulator is None:
            Sij_Sij_accumulator = Sij_Sij
        else:
            Sij_Sij_accumulator += Sij_Sij

        del Sij_Sij
        gc.collect()

        snapshot_count += 1

        print(f"Finished processing snapshot {snapshot_count}.")

# Mean strain-rate tensor
Sij_Sij_temporal_avg = Sij_Sij_accumulator / snapshot_count
print(f"Computed mean Sij_Sij from {snapshot_count} snapshots.")

# Save Sij_Sij_temporal_avg
with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("Sij_Sij_temporal_avg", data=Sij_Sij_temporal_avg)
print(f"Saved Sij_Sij_temporal_avg to {SAVE_PATH}")

# Load Sij_Sij_temporal_avg from saved file for verification
with h5py.File(SAVE_PATH, "r") as f:
    Sij_Sij_loaded = f["Sij_Sij_temporal_avg"][:]
    print("Loaded Sij_Sij_temporal_avg from saved file for verification.")

# End timer
elapsed_time = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")