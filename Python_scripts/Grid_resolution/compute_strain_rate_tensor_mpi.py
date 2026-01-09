import os
import re
import glob
import h5py
import numpy as np
import gc
import sys
import time

from mpi4py import MPI


module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Start timer
start_time = time.perf_counter()

# Initialize MPI
comm = MPI.COMM_WORLD # Get the global communicator
rank = comm.Get_rank() # Get the rank of the current process
size = comm.Get_size() # Get the total number of processes

# Print MPI info
if rank == 0:
    print("=" * 60)
    print(f"MPI Configuration:")
    print(f"  Total processes: {size}")
    print("=" * 60)

### SET PATHS AND FILE NAMES
# Save varibales: Sij_Sij_temporal_avg
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data"
SAVE_NAME = "3d_NACA0012_Re50000_AoA12_strain_rate_tensor_batch_33056839.h5"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)
# SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/data/"
# SAVE_NAME = "3d_NACA0012_Re50000_AoA5_strain_rate_tensor_test_mpi1.h5"
# SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# Snapshot directory
SNAPSHOT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/batch_33056839/"
# SNAPSHOT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/compressed_snapshots/"


# Load Mesh data
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME) 
# MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
# MESH_NAME = "3d_NACA0012_Re10000_AoA5-CROP-MESH.h5"
# MESH_FILE = os.path.join(MESH_PATH, MESH_NAME) 

# Load Average data
SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME_AVG = "3d_NACA0012_Re50000_AoA12_avg_19680000-COMP-DATA.h5"
SNAPSHOT_FILE_AVG = os.path.join(SNAPSHOT_PATH_AVG, SNAPSHOT_NAME_AVG)
# SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
# SNAPSHOT_NAME_AVG = "3d_NACA0012_Re10000_AoA5_avg_1620000-COMP-DATA.h5"
# SNAPSHOT_FILE_AVG = os.path.join(SNAPSHOT_PATH_AVG, SNAPSHOT_NAME_AVG)

# Load mesh once (all ranks)
loader = CompressedSnapshotLoader(MESH_FILE)
x = loader.x
y = loader.y
z = loader.z

# Load mean fields (all ranks)
fields_avg = loader.load_snapshot_avg(SNAPSHOT_FILE_AVG)
u_mean = loader.reconstruct_field(fields_avg["avg_u"])
v_mean = loader.reconstruct_field(fields_avg["avg_v"])
w_mean = loader.reconstruct_field(fields_avg["avg_w"])

if rank == 0:
    print("Loaded mesh and mean velocity fields")

comm.Barrier()

def get_files_by_extension(folder_path, extensions):
    """
    Returns a list of file paths in the given folder matching the specified extensions.
    """
    if isinstance(extensions, str):
        extensions = (extensions,)
    else:
        extensions = tuple(extensions)

    file_paths = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path) and file.lower().endswith(
            tuple(f".{ext.lower()}" for ext in extensions)
        ):
            file_paths.append(full_path)

    return file_paths

def distribute_file_list(folder_path, extensions, comm):
    """
    Root rank scans 'folder_path' for files with given extensions, then distributes
    the list in (almost) equal chunks to all ranks.

    Returns:
        local_files (list of str): List of file paths assigned to this rank.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: # Root rank
        file_list = get_files_by_extension(folder_path, extensions) # Get all files
        # Sort for deterministic ordering
        file_list.sort()
        N = len(file_list)
        print(f"[Rank 0] Found {N} files in {folder_path}")

        # Base number of files per process
        base = N // size
        # remainder distributed one by one to the first ranks
        rem = N % size

        # Compute chunk sizes
        counts = [base + (1 if r < rem else 0) for r in range(size)]

        # Compute displacements (start indices)
        displs = [sum(counts[:r]) for r in range(size)]
    else:
        file_list = None
        counts = None
        displs = None

    # Broadcast counts and displacements to all ranks
    counts = comm.bcast(counts, root=0)
    displs = comm.bcast(displs, root=0)

    # Now actually send the file-list slices
    if rank == 0:
        local_files = file_list[displs[0] : displs[0] + counts[0]]
        for r in range(1, size):
            start = displs[r]
            end = start + counts[r]
            comm.send(file_list[start:end], dest=r, tag=77)
    else:
        local_files = comm.recv(source=0, tag=77)
    
    # Extract snapshot numbers from filenames
    snapshot_numbers = []
    for f in local_files:
        basename = os.path.basename(f)
        # Split by "-COMP" and get the number before it
        num_str = basename.split('-COMP')[0].split('_')[-1]
        try:
            snapshot_numbers.append(int(num_str))
        except ValueError:
            print(f"[Rank {rank}] Warning: Could not parse snapshot number from {basename}")
    
    print(f"[Rank {rank}] Received {len(local_files)} files, Snapshot numbers: {snapshot_numbers}")
    return local_files

# Distribute snapshot files among ranks
local_files = distribute_file_list(SNAPSHOT_DIR, "h5", comm)

print(f"[Rank {rank}] Processing {len(local_files)} snapshots")

# Each rank computes strain tensor for its snapshots
local_Sij_Sij_sum = None
local_count = 0

for snapshot_file in local_files:
    print(f"[Rank {rank}] Processing {os.path.basename(snapshot_file)}")
    # Load instantaneous snapshot fields
    fields = loader.load_snapshot(snapshot_file)
    
    # Read instantaneous and mean velocity fields
    u = loader.reconstruct_field(fields["u"])
    v = loader.reconstruct_field(fields["v"])
    w = loader.reconstruct_field(fields["w"])

    # Mask invalid data (NaNs)
    valid_mask = ~np.isnan(u)

    # Compute fluctuating velocity components
    u_prime = np.where(valid_mask, u - u_mean, 0.0)
    v_prime = np.where(valid_mask, v - v_mean, 0.0)
    w_prime = np.where(valid_mask, w - w_mean, 0.0)

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
    if local_Sij_Sij_sum is None:
        local_Sij_Sij_sum = Sij_Sij
    else:
        local_Sij_Sij_sum += Sij_Sij

    del Sij_Sij
    gc.collect()

    local_count += 1

print(f"[Rank {rank}] Computed {local_count} snapshots locally")

# Synchronize all ranks
comm.Barrier()

# Gather results from all ranks to rank 0
global_Sij_Sij_sum = comm.reduce(local_Sij_Sij_sum, op=MPI.SUM, root=0)
global_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Rank 0 computes final average and saves
if rank == 0:
    print(f"=" * 60)
    print(f"Total snapshots processed: {global_count}")

    if global_count > 0:
        Sij_Sij_temporal_avg = global_Sij_Sij_sum / global_count
        
        print(f"Saving temporal average to: {SAVE_PATH}")
        with h5py.File(SAVE_PATH, "w") as f:
            f.create_dataset("Sij_Sij_temporal_avg", data=Sij_Sij_temporal_avg, dtype="float32")
        
        print("✓ Completed successfully!")
    else:
        print("ERROR: No snapshots processed!")
    print("=" * 60)

comm.Barrier()

# Verification
if rank == 0:
    with h5py.File(SAVE_PATH, "r") as f:
        Sij_Sij_loaded = f["Sij_Sij_temporal_avg"][:]
        print(f"Verified: Shape={Sij_Sij_loaded.shape}, dtype={Sij_Sij_loaded.dtype}")

elapsed_time = time.perf_counter() - start_time
if rank == 0:
    print(f"Total execution time: {elapsed_time:.2f} seconds")

