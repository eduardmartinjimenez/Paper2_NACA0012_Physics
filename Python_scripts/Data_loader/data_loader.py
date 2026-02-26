import numpy as np
import h5py
import os
import time
from data_loader_functions import CompressedSnapshotLoader


print("\n" + "="*80)
print("TIMING TEST: ORIGINAL LOADER (NO REGION FILTERING)")
print("="*80 + "\n")

# Define path
# Mesh data
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Snapshot files
SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
SNAPSHOT_NAME_AVG = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
SNAPSHOT_FILE_AVG = os.path.join(SNAPSHOT_PATH_AVG, SNAPSHOT_NAME_AVG)

SNAPSHOT_PATH_PRI = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/batch_30658504/"
SNAPSHOT_NAME_PRI = "3d_NACA0012_Re50000_AoA12_6350000-COMP-DATA.h5"
SNAPSHOT_FILE_PRI = os.path.join(SNAPSHOT_PATH_PRI, SNAPSHOT_NAME_PRI)


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

# Check if the primitive data file exists
if os.path.exists(SNAPSHOT_FILE_PRI):
    print(f"Primitive data file exists! {SNAPSHOT_FILE_PRI}")
else:
    print(f"Primitive Data file does not exist: {SNAPSHOT_FILE_PRI}")

# ============================================================================
# TIMING: Initialize loader and load mesh
# ============================================================================
print("\n" + "-"*80)
print("TIMER 1: Initialize loader (load mesh)")
print("-"*80)
t_start = time.time()
loader = CompressedSnapshotLoader(MESH_FILE)
t_mesh = time.time() - t_start
print(f"✓ Time to load mesh: {t_mesh:.4f} seconds")

# Coordinates
x_data = loader.x
y_data = loader.y
z_data = loader.z
tag_ibm_data = loader.tag_ibm

# ============================================================================
# TIMING: Load primitive snapshot
# ============================================================================
print("\n" + "-"*80)
print("TIMER 2: Load primitive snapshot")
print("-"*80)
t_start = time.time()
fields = loader.load_snapshot(SNAPSHOT_FILE_PRI)
t_snapshot_pri = time.time() - t_start
print(f"✓ Time to load primitive snapshot: {t_snapshot_pri:.4f} seconds")

# ============================================================================
# TIMING: Load averaged snapshot
# ============================================================================
print("\n" + "-"*80)
print("TIMER 3: Load averaged snapshot")
print("-"*80)
t_start = time.time()
fields_avg = loader.load_snapshot_avg(SNAPSHOT_FILE_AVG)
t_snapshot_avg = time.time() - t_start
print(f"✓ Time to load averaged snapshot: {t_snapshot_avg:.4f} seconds")

# ============================================================================
# TIMING: Reconstruct velocity field
# ============================================================================
print("\n" + "-"*80)
print("TIMER 4: Reconstruct full 3D velocity field (u)")
print("-"*80)
t_start = time.time()
u_data_pri = loader.reconstruct_field(fields["u"])
t_reconstruct = time.time() - t_start
print(f"✓ Time to reconstruct u field: {t_reconstruct:.4f} seconds")

# Reconstruct other fields (not timed individually)
u_data = loader.reconstruct_field(fields_avg["u"])
u_avg_data = loader.reconstruct_field(fields_avg["avg_u"])

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TIMING SUMMARY - ORIGINAL LOADER")
print("="*80)
print(f"  1. Mesh loading:              {t_mesh:.4f} seconds")
print(f"  2. Primitive snapshot:        {t_snapshot_pri:.4f} seconds")
print(f"  3. Averaged snapshot:         {t_snapshot_avg:.4f} seconds")
print(f"  4. Field reconstruction (u):  {t_reconstruct:.4f} seconds")
print(f"  ---")
print(f"  TOTAL TIME:                   {t_mesh + t_snapshot_pri + t_snapshot_avg + t_reconstruct:.4f} seconds")
print("="*80 + "\n")

# Verify data shapes and contents
print("--- Data Verification ---")
print(f"Mesh coordinates shape: x={x_data.shape}, y={y_data.shape}, z={z_data.shape}")
print(f"IBM tag shape: {tag_ibm_data.shape}")
print(f"Primitive fields - u: {u_data_pri.shape}")
print(f"Average fields - u: {u_data.shape}")
print(f"Average velocity - u_avg: {u_avg_data.shape}")
print(f"Value ranges - u_avg: [{np.nanmin(u_avg_data):.6f}, {np.nanmax(u_avg_data):.6f}]")
print(f"NaN count - u_avg: {np.isnan(u_avg_data).sum()}")
print("--- Verification Complete ---\n")

#-------------------------------------------------------------------
### Import data from a Snapshot

# # Define file path and name
# FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
# FILE_NAME = "3d_ibm_stl_naca0012_1916_1988_128_aoa5_Re50000_22640000.h5"
# FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# # Check if the data file exists
# if os.path.exists(FULL_PATH):
#     print(f"Data file exists! {FULL_PATH}")
# else:
#     print(f"Data file does not exist: {FULL_PATH}")

# # Import data file
# data_file = h5py.File(FULL_PATH, "r")

# ### Import 3D Data
# x_data = data_file["x"][:, :, :]
# y_data = data_file["y"][:, :, :]
# z_data = data_file["z"][:, :, :]
# tag_ibm_data = data_file["tag_IBM"][:, :, :]

# u_data = data_file["avg_u"][:, :, :]
# v_data = data_file["avg_v"][:, :, :]
# w_data = data_file["avg_w"][:, :, :]
# p_data = data_file["avg_P"][:, :, :]

# u_data = data_file["u"][:, :, :]
# v_data = data_file["v"][:, :, :]
# w_data = data_file["w"][:, :, :]
# p_data = data_file["P"][:, :, :]