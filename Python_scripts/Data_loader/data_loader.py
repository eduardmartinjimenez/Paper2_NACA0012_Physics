import numpy as np
import h5py
import os
from data_loader_functions import CompressedSnapshotLoader


# Define path
# Load Mesh data
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME) 

# Load Snapshot data
SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME_AVG = "3d_NACA0012_Re50000_AoA5_avg_22640000-COMP-DATA.h5"
SNAPSHOT_FILE_AVG = os.path.join(SNAPSHOT_PATH_AVG, SNAPSHOT_NAME_AVG)


SNAPSHOT_PATH_PRI = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Snapshots/batch_30296325/"
SNAPSHOT_NAME_PRI = "3d_NACA0012_Re50000_AoA5_7020000-COMP-DATA.h5"
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

# Initialize loader
loader = CompressedSnapshotLoader(MESH_FILE)

# Coordinates
x_data = loader.x
y_data = loader.y
z_data = loader.z
tag_ibm_data = loader.tag_ibm

# Load field from primitive snapshot
fields = loader.load_snapshot(SNAPSHOT_FILE_PRI)

# Load field from average snapshot
fields_avg = loader.load_snapshot_avg(SNAPSHOT_FILE_AVG)

# Reconstruct full 3D fields from primitive snapshot
u_data_pri = loader.reconstruct_field(fields["u"])
v_data_pri = loader.reconstruct_field(fields["v"])
w_data_pri = loader.reconstruct_field(fields["w"])
p_data_pri = loader.reconstruct_field(fields["p"])

# Reconstruct full 3D fields from average snapshot
u_data = loader.reconstruct_field(fields_avg["u"])
v_data = loader.reconstruct_field(fields_avg["v"])
w_data = loader.reconstruct_field(fields_avg["w"])
p_data = loader.reconstruct_field(fields_avg["p"])

u_avg_data = loader.reconstruct_field(fields_avg["avg_u"])
v_avg_data = loader.reconstruct_field(fields_avg["avg_v"])
w_avg_data = loader.reconstruct_field(fields_avg["avg_w"])
p_avg_data = loader.reconstruct_field(fields_avg["avg_p"])

# Verify data shapes and contents
print("\n--- Data Verification ---")
print(f"Mesh coordinates shape: x={x_data.shape}, y={y_data.shape}, z={z_data.shape}")
print(f"IBM tag shape: {tag_ibm_data.shape}")
print(f"Primitive fields - u: {u_data_pri.shape}, v: {v_data_pri.shape}, w: {w_data_pri.shape}, p: {p_data_pri.shape}")
print(f"Average fields - u: {u_data.shape}, v: {v_data.shape}, w: {w_data.shape}, p: {p_data.shape}")
print(f"Average velocity - u_avg: {u_avg_data.shape}, v_avg: {v_avg_data.shape}, w_avg: {w_avg_data.shape}")
print(f"Average pressure - p_avg: {p_avg_data.shape}")
print(f"Value ranges - u_avg: [{np.nanmin(u_avg_data):.6f}, {np.nanmax(u_avg_data):.6f}]")
print(f"Value ranges - p_avg: [{np.nanmin(p_avg_data):.6f}, {np.nanmax(p_avg_data):.6f}]")
print(f"NaN count - u_avg: {np.isnan(u_avg_data).sum()}, p_avg: {np.isnan(p_avg_data).sum()}")
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
