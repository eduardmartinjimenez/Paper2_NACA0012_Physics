import os
import h5py
from stl import mesh
import numpy as np
import os
import sys
import h5py
import numpy as np
from scipy.spatial import cKDTree
import pickle

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Define file path and name
FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/last_slice/"
FILE_NAME = "slice_1_24299200-COMP-DATA.h5"
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# Check if the data file exists
if os.path.exists(FULL_PATH):
    print(f"Data file exists! {FULL_PATH}")
else:
    print(f"Data file does not exist: {FULL_PATH}")

MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/last_slice/"
MESH_NAME = "slice_1-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Check if the mesh file exists
if os.path.exists(MESH_FILE):
    print(f"Mesh file exists! {MESH_FILE}")
else:
    print(f"Mesh file does not exist: {MESH_FILE}")

# Load compressed data using
loader = CompressedSnapshotLoader(MESH_FILE)
fields = loader.load_snapshot_avg(FULL_PATH)

# Load data
x_data = loader.x
y_data = loader.y
z_data = loader.z

u_data = loader.reconstruct_field(fields["u"])
v_data = loader.reconstruct_field(fields["v"])
w_data = loader.reconstruct_field(fields["w"])
p_data = loader.reconstruct_field(fields["p"])

avg_u_data = loader.reconstruct_field(fields["avg_u"])
avg_v_data = loader.reconstruct_field(fields["avg_v"])
avg_w_data = loader.reconstruct_field(fields["avg_w"])
avg_p_data = loader.reconstruct_field(fields["avg_p"])

# Compare u and avg_u (not nan values)
# Compare u and avg_u (not nan values)
mask = ~np.isnan(u_data) & ~np.isnan(avg_u_data)
print(f"Max u: {np.nanmax(u_data)}, Max avg_u: {np.nanmax(avg_u_data)}")
print(f"Min u: {np.nanmin(u_data)}, Min avg_u: {np.nanmin(avg_u_data)}")

print(f"Max w: {np.nanmax(w_data)}, Max avg_w: {np.nanmax(avg_w_data)}")
print(f"Min w: {np.nanmin(w_data)}, Min avg_w: {np.nanmin(avg_w_data)}")

# Cheack if the arrays have the same shape
print(f"Shape of u_data: {u_data.shape}, Shape of avg_u_data: {avg_u_data.shape}")

# Check difference between u and avg_u
difference = np.abs(u_data - avg_u_data)
print(f"Max difference between u and avg_u: {difference.max()}")
print(f"Mean difference between u and avg_u: {difference.mean()}")



