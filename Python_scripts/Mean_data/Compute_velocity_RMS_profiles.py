import os
import sys
import h5py
import numpy as np
import glob
import gc
import time
from scipy.spatial import cKDTree

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Start timer
start_time = time.perf_counter()

# Output data path
OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
OUTPUT_DATA_NAME = "AoA5_Re50000_velocity_RMS_profiles_data.h5"
#OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Mean_data/"
#OUTPUT_DATA_NAME = "AoA5_Re10000_velocity_RMS_profiles_data.h5"
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
    print(f"{kind} exists: {path}")

def rotate_coordinates(x, y, angle_rad):
    """Rotate (x,y) into flow-aligned frame (x',y') by AoA=angle_rad."""
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * ca + y * sa
    y_rot = -x * sa + y * ca
    return x_rot, y_rot


print("="*70)
print("COMPUTING VELOCITY RMS FLUCTUATIONS FOR PROFILES")
print("="*70)

# Load geometrical data
assert_exists(GEO_FILE, "Geometrical data file")

with h5py.File(GEO_FILE, "r") as f:
    interface_indices_i = f["interface_indices_i"][...].astype(np.int32)
    interface_indices_j = f["interface_indices_j"][...].astype(np.int32)

print("interface_indices_i shape:", interface_indices_i.shape)

# Load mesh
assert_exists(MESH_FILE, "Mesh file")
loader = CompressedSnapshotLoader(MESH_FILE)

# Coordinates
x_data = loader.x[1, :, :]
y_data = loader.y[1, :, :]
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

print("Total suction-side interface points:", x_suction_rot.size)

# Load mean velocity field
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

print("Loaded and rotated mean velocity fields")

# Build KDTree for profile extraction
x_rot_flat = x_rot.ravel()
y_rot_flat = y_rot.ravel()
mesh_coords = np.column_stack((x_rot_flat, y_rot_flat))
tree = cKDTree(mesh_coords)

# Define profile locations (same as dense velocity profiles)
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

print(f"Defined {len(profile_locations)} profile locations")

# Initialize accumulators for RMS computation
# Store fluctuation squares for each profile
n_profiles = len(profile_locations)
max_profile_length = max(len(prof["i_indices"]) for prof in profile_locations)

u_prime_sq_sum = [np.zeros(len(prof["i_indices"])) for prof in profile_locations]
v_prime_sq_sum = [np.zeros(len(prof["i_indices"])) for prof in profile_locations]
w_prime_sq_sum = [np.zeros(len(prof["i_indices"])) for prof in profile_locations]

snapshot_count = 0

# Find all batch directories
batch_dirs = sorted(glob.glob(os.path.join(SNAPSHOTS_DIR, "batch_*")))
print(f"\nFound {len(batch_dirs)} batch directories")

# Process each batch
for batch_dir in batch_dirs:
    batch_name = os.path.basename(batch_dir)
    print(f"\n{'='*70}")
    print(f"Processing {batch_name}")
    print(f"{'='*70}")
    
    # Find all snapshot files in this batch
    h5_files = sorted(glob.glob(os.path.join(batch_dir, "*-COMP-DATA.h5")))
    print(f"Found {len(h5_files)} snapshot files")
    
    # Process each snapshot in this batch
    for idx, file in enumerate(h5_files):
        print(f"\nProcessing snapshot {idx+1}/{len(h5_files)}: {os.path.basename(file)}")
        
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
            
            u_prime_sq_sum[prof_idx] += u_p**2
            v_prime_sq_sum[prof_idx] += v_p**2
            w_prime_sq_sum[prof_idx] += w_p**2
        
        snapshot_count += 1
        
        # Clean up
        del fields, u, v, w, u_2d, v_2d, w_2d, u_rot, v_rot, w_rot
        del u_prime, v_prime, w_prime
        gc.collect()
        
        if (idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx+1} snapshots in {elapsed:.1f}s")

print(f"\n{'='*70}")
print(f"Finished processing all snapshots")
print(f"Total snapshots processed: {snapshot_count}")
print(f"{'='*70}")

# Compute RMS (sqrt of mean of squares)
print("\nComputing RMS values...")
rms_profiles = []

for prof_idx, prof_loc in enumerate(profile_locations):
    u_rms = np.sqrt(u_prime_sq_sum[prof_idx] / snapshot_count)
    v_rms = np.sqrt(v_prime_sq_sum[prof_idx] / snapshot_count)
    w_rms = np.sqrt(w_prime_sq_sum[prof_idx] / snapshot_count)
    
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
    f.attrs["snapshot_count"] = snapshot_count
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

print("Data saved successfully!")

elapsed_total = time.perf_counter() - start_time
print(f"\n{'='*70}")
print(f"COMPUTATION COMPLETE")
print(f"{'='*70}")
print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
print(f"Snapshots processed: {snapshot_count}")
print(f"Time per snapshot: {elapsed_total/snapshot_count:.2f}s")
print(f"\nOutput file: {OUTPUT_DATA_FILE}")
