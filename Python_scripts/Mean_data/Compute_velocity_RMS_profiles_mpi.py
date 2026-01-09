import os
import sys
import h5py
import numpy as np
import glob
import gc
import time
from scipy.spatial import cKDTree
from mpi4py import MPI

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Start timer
start_time = time.perf_counter()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Print MPI info
if rank == 0:
    print("=" * 70)
    print(f"MPI Configuration:")
    print(f"  Total processes: {size}")
    print("=" * 70)

# Output data path
OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
OUTPUT_DATA_NAME = "AoA5_Re50000_velocity_RMS_profiles_data_mpi.h5"
#OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Mean_data/"
#OUTPUT_DATA_NAME = "AoA5_Re10000_velocity_RMS_profiles_data_mpi.h5"
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DATA_PATH, OUTPUT_DATA_NAME)

# Paths
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
#GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Geometrical_data"
#GEO_NAME = "3d_NACA0012_Test_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
#MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
#MESH_NAME = "3d_NACA0012_Re10000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME_AVG = "3d_NACA0012_Re50000_AoA5_avg_24280000-COMP-DATA.h5"
#SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
#SNAPSHOT_NAME_AVG = "3d_NACA0012_Re10000_AoA5_avg_1620000-COMP-DATA.h5"
SNAPSHOT_FILE_AVG = os.path.join(SNAPSHOT_PATH_AVG, SNAPSHOT_NAME_AVG)

SNAPSHOTS_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Snapshots/"
#SNAPSHOTS_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/compressed_snapshots/"

# Reference parameters
u_infty = 1.0
AOA = 5  # degrees
alpha = np.deg2rad(AOA)
C = 1.0  # chord length

# x/c locations: dense sampling (same as velocity profiles)
x_c_locations_dense = np.arange(0.10, 1.01, 0.01)  # Every 0.01

# Parameters for wall-normal extraction
wall_normal_length = 0.2
n_points = 250

# Utilities
def assert_exists(path: str, kind: str = "File") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    if rank == 0:
        print(f"{kind} exists: {path}")

def rotate_coordinates(x, y, angle_rad):
    """Rotate (x,y) into flow-aligned frame (x',y') by AoA=angle_rad."""
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * ca + y * sa
    y_rot = -x * sa + y * ca
    return x_rot, y_rot


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

    if rank == 0:  # Root rank
        # Find all batch directories
        batch_dirs = sorted(glob.glob(os.path.join(folder_path, "batch_*")))
        
        # Collect all snapshot files from all batches
        file_list = []
        for batch_dir in batch_dirs:
            batch_files = sorted(glob.glob(os.path.join(batch_dir, "*-COMP-DATA.h5")))
            file_list.extend(batch_files)
        
        N = len(file_list)
        print(f"[Rank 0] Found {N} files across {len(batch_dirs)} batches")

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
    
    print(f"[Rank {rank}] Received {len(local_files)} files")
    return local_files


if rank == 0:
    print("=" * 70)
    print("COMPUTING VELOCITY RMS FLUCTUATIONS FOR PROFILES (MPI)")
    print("=" * 70)

# Load geometrical data (all ranks)
assert_exists(GEO_FILE, "Geometrical data file")

with h5py.File(GEO_FILE, "r") as f:
    interface_indices_i = f["interface_indices_i"][...].astype(np.int32)
    interface_indices_j = f["interface_indices_j"][...].astype(np.int32)

if rank == 0:
    print("interface_indices_i shape:", interface_indices_i.shape)

# Load mesh (all ranks)
assert_exists(MESH_FILE, "Mesh file")
loader = CompressedSnapshotLoader(MESH_FILE)

# Coordinates
x_data = loader.x[1, :, :]
y_data = loader.y[1, :, :]

if rank == 0:
    print("x_data shape:", x_data.shape)

# Rotate into flow-aligned frame
x_rot, y_rot = rotate_coordinates(x_data, y_data, alpha)

# Interface points
x_int = x_data[interface_indices_j, interface_indices_i]
y_int = y_data[interface_indices_j, interface_indices_i]
x_int_rot = x_rot[interface_indices_j, interface_indices_i]
y_int_rot = y_rot[interface_indices_j, interface_indices_i]

# Suction side selection
suction_mask = y_int > 0
i_suction = interface_indices_i[suction_mask]
j_suction = interface_indices_j[suction_mask]
x_suction_rot = x_int_rot[suction_mask]
y_suction_rot = y_int_rot[suction_mask]

if rank == 0:
    print("Total suction-side interface points:", x_suction_rot.size)

# Load mean velocity field (all ranks)
assert_exists(SNAPSHOT_FILE_AVG, "Average snapshot file")
fields_avg = loader.load_snapshot_avg(SNAPSHOT_FILE_AVG)

u_mean = loader.reconstruct_field(fields_avg["avg_u"])
v_mean = loader.reconstruct_field(fields_avg["avg_v"])
w_mean = loader.reconstruct_field(fields_avg["avg_w"])

# Average in spanwise direction
u_mean_2d = np.mean(u_mean, axis=0)
v_mean_2d = np.mean(v_mean, axis=0)
w_mean_2d = np.mean(w_mean, axis=0)

# Rotate mean velocities
u_mean_rot, v_mean_rot = rotate_coordinates(u_mean_2d, v_mean_2d, alpha)

if rank == 0:
    print("Loaded and rotated mean velocity fields")

# Build KDTree for profile extraction (all ranks)
x_rot_flat = x_rot.ravel()
y_rot_flat = y_rot.ravel()
mesh_coords = np.column_stack((x_rot_flat, y_rot_flat))
tree = cKDTree(mesh_coords)

# Define profile locations (same as dense velocity profiles) - all ranks
profile_locations = []

for x_c in x_c_locations_dense:
    x_target = x_c * C
    
    # Find closest suction-side point
    dx = np.abs(x_suction_rot - x_target)
    k = int(np.argmin(dx))
    
    x_start = x_suction_rot[k]
    y_start = y_suction_rot[k]
    y_end = y_start + wall_normal_length
    
    # Generate query points along wall-normal
    y_query = np.linspace(y_start, y_end, n_points)
    x_query = np.full_like(y_query, x_start)
    query_points = np.column_stack((x_query, y_query))
    
    # Find nearest mesh points
    distances, indices = tree.query(query_points)
    ny, nx = x_rot.shape
    j_indices = indices // nx
    i_indices = indices % nx
    
    # Remove duplicates
    ij_pairs = np.column_stack((i_indices, j_indices))
    _, unique_idx = np.unique(ij_pairs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    
    profile_loc = {
        "x_c": float(x_c),
        "x_start": float(x_start),
        "y_start": float(y_start),
        "y_end": float(y_end),
        "i_indices": i_indices[unique_idx],
        "j_indices": j_indices[unique_idx],
        "x_prime": x_rot[j_indices[unique_idx], i_indices[unique_idx]],
        "y_prime": y_rot[j_indices[unique_idx], i_indices[unique_idx]],
    }
    
    profile_locations.append(profile_loc)

if rank == 0:
    print(f"Defined {len(profile_locations)} profile locations")

comm.Barrier()

# Distribute snapshot files among ranks
local_files = distribute_file_list(SNAPSHOTS_DIR, "h5", comm)

print(f"[Rank {rank}] Processing {len(local_files)} snapshots")

# Initialize local accumulators for RMS computation
n_profiles = len(profile_locations)
local_u_prime_sq_sum = [np.zeros(len(prof["i_indices"])) for prof in profile_locations]
local_v_prime_sq_sum = [np.zeros(len(prof["i_indices"])) for prof in profile_locations]
local_w_prime_sq_sum = [np.zeros(len(prof["i_indices"])) for prof in profile_locations]

local_snapshot_count = 0

# Process each snapshot assigned to this rank
for idx, file in enumerate(local_files):
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"[Rank {rank}] Processing snapshot {idx+1}/{len(local_files)}: {os.path.basename(file)}")
    
    # Load snapshot
    fields = loader.load_snapshot(file)
    
    # Reconstruct velocity fields
    u = loader.reconstruct_field(fields["u"])
    v = loader.reconstruct_field(fields["v"])
    w = loader.reconstruct_field(fields["w"])
    
    # Average in spanwise direction
    u_2d = np.mean(u, axis=0)
    v_2d = np.mean(v, axis=0)
    w_2d = np.mean(w, axis=0)
    
    # Rotate velocities
    u_rot, v_rot = rotate_coordinates(u_2d, v_2d, alpha)
    w_rot = w_2d  # w component doesn't change with rotation in xy-plane
    
    # Compute fluctuations
    u_prime = u_rot - u_mean_rot
    v_prime = v_rot - v_mean_rot
    w_prime = w_rot - w_mean_2d
    
    # Extract fluctuations along each profile and accumulate squares
    for prof_idx, prof_loc in enumerate(profile_locations):
        i_idx = prof_loc["i_indices"]
        j_idx = prof_loc["j_indices"]
        
        u_p = u_prime[j_idx, i_idx]
        v_p = v_prime[j_idx, i_idx]
        w_p = w_prime[j_idx, i_idx]
        
        local_u_prime_sq_sum[prof_idx] += u_p**2
        local_v_prime_sq_sum[prof_idx] += v_p**2
        local_w_prime_sq_sum[prof_idx] += w_p**2
    
    local_snapshot_count += 1
    
    # Clean up
    del fields, u, v, w, u_2d, v_2d, w_2d, u_rot, v_rot, w_rot
    del u_prime, v_prime, w_prime
    gc.collect()

print(f"[Rank {rank}] Completed processing {local_snapshot_count} snapshots")

# Synchronize all ranks
comm.Barrier()

if rank == 0:
    print("\n" + "=" * 70)
    print("GATHERING RESULTS FROM ALL RANKS")
    print("=" * 70)

# Gather results from all ranks to rank 0
# We need to gather each profile's accumulated squares separately
global_u_prime_sq_sum = []
global_v_prime_sq_sum = []
global_w_prime_sq_sum = []

for prof_idx in range(n_profiles):
    # Reduce each profile's accumulator
    u_sq_global = comm.reduce(local_u_prime_sq_sum[prof_idx], op=MPI.SUM, root=0)
    v_sq_global = comm.reduce(local_v_prime_sq_sum[prof_idx], op=MPI.SUM, root=0)
    w_sq_global = comm.reduce(local_w_prime_sq_sum[prof_idx], op=MPI.SUM, root=0)
    
    if rank == 0:
        global_u_prime_sq_sum.append(u_sq_global)
        global_v_prime_sq_sum.append(v_sq_global)
        global_w_prime_sq_sum.append(w_sq_global)

# Reduce snapshot count
global_snapshot_count = comm.reduce(local_snapshot_count, op=MPI.SUM, root=0)

# Rank 0 computes RMS and saves results
if rank == 0:
    print(f"\n{'='*70}")
    print(f"COMPUTING RMS AND SAVING RESULTS")
    print(f"{'='*70}")
    print(f"Total snapshots processed: {global_snapshot_count}")
    
    if global_snapshot_count > 0:
        # Compute RMS (sqrt of mean of squares)
        rms_profiles = []
        
        for prof_idx, prof_loc in enumerate(profile_locations):
            u_rms = np.sqrt(global_u_prime_sq_sum[prof_idx] / global_snapshot_count)
            v_rms = np.sqrt(global_v_prime_sq_sum[prof_idx] / global_snapshot_count)
            w_rms = np.sqrt(global_w_prime_sq_sum[prof_idx] / global_snapshot_count)
            
            rms_profile = {
                "x_c": prof_loc["x_c"],
                "x_start": prof_loc["x_start"],
                "y_start": prof_loc["y_start"],
                "y_end": prof_loc["y_end"],
                "x_prime": prof_loc["x_prime"],
                "y_prime": prof_loc["y_prime"],
                "u_rms": u_rms,
                "v_rms": v_rms,
                "w_rms": w_rms,
            }
            
            rms_profiles.append(rms_profile)
        
        # Save results to HDF5
        print(f"\nSaving RMS profiles to {OUTPUT_DATA_FILE}")
        os.makedirs(os.path.dirname(OUTPUT_DATA_FILE), exist_ok=True)
        
        with h5py.File(OUTPUT_DATA_FILE, "w") as f:
            # Metadata
            f.attrs["u_infty"] = u_infty
            f.attrs["alpha"] = alpha
            f.attrs["C"] = C
            f.attrs["snapshot_count"] = global_snapshot_count
            f.create_dataset("x_c_locations_dense", data=x_c_locations_dense)
            
            # RMS profiles
            profiles_group = f.create_group("rms_profiles")
            for idx, rms_prof in enumerate(rms_profiles):
                prof_group = profiles_group.create_group(f"profile_{idx}")
                prof_group.attrs["x_c"] = rms_prof["x_c"]
                prof_group.attrs["x_start"] = rms_prof["x_start"]
                prof_group.attrs["y_start"] = rms_prof["y_start"]
                prof_group.attrs["y_end"] = rms_prof["y_end"]
                prof_group.create_dataset("x_prime", data=rms_prof["x_prime"])
                prof_group.create_dataset("y_prime", data=rms_prof["y_prime"])
                prof_group.create_dataset("u_rms", data=rms_prof["u_rms"])
                prof_group.create_dataset("v_rms", data=rms_prof["v_rms"])
                prof_group.create_dataset("w_rms", data=rms_prof["w_rms"])
        
        print("✓ Data saved successfully!")
        
        # Verification
        with h5py.File(OUTPUT_DATA_FILE, "r") as f:
            n_saved_profiles = len(f["rms_profiles"])
            print(f"Verified: {n_saved_profiles} profiles saved")
    else:
        print("ERROR: No snapshots processed!")
    
    print("=" * 70)

comm.Barrier()

elapsed_total = time.perf_counter() - start_time
if rank == 0:
    print(f"\n{'='*70}")
    print(f"COMPUTATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"Snapshots processed: {global_snapshot_count}")
    if global_snapshot_count > 0:
        print(f"Time per snapshot: {elapsed_total/global_snapshot_count:.2f}s")
    print(f"\nOutput file: {OUTPUT_DATA_FILE}")
