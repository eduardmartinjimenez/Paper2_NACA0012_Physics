import os
import sys
import h5py
import numpy as np
import glob
import gc


module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

### SAVE RESULTS
# Save varibales: Cp
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
SAVE_NAME = "3d_NACA0012_Re50000_AoA12_Cp_19680000.h5"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Geometrical data
### GEOMETRICAL STUFF
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Check if the Geometrical data file exists
if os.path.exists(GEO_FILE):
    print(f"Data file exists! {GEO_FILE}")
else:
    print(f"Data file does not exist: {GEO_FILE}")

# Load geometrical data
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]
    proj_points = f["proj_points"][:]
    proj_normals = f["proj_normals"][:]
    proj_distances = f["proj_distances"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

# Load Mesh data
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Check if the mesh file exists
if os.path.exists(MESH_FILE):
    print(f"Mesh file exists! {MESH_FILE}")
else:
    print(f"Mesh file does not exist: {MESH_FILE}")

# Load Snapshot data
SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_19680000-COMP-DATA.h5"
SNAPSHOT_FILE = os.path.join(SNAPSHOT_PATH, SNAPSHOT_NAME)

# Check if the data file exists
if os.path.exists(SNAPSHOT_FILE):
    print(f"Data file exists! {SNAPSHOT_FILE}")
else:
    print(f"Data file does not exist: {SNAPSHOT_FILE}")

# Initialize loader
loader = CompressedSnapshotLoader(MESH_FILE)

# Load snapshot fields
fields_avg = loader.load_snapshot_avg(SNAPSHOT_FILE)

### Import 3D Data
# Coordinates
x_data = loader.x
y_data = loader.y
z_data = loader.z
tag_ibm_data = loader.tag_ibm

# Averagd velocity fields
p_data = loader.reconstruct_field(fields_avg["avg_p"])

# Reference parameters
rho_ref = 1.0  # Reference density [kg/m3]
u_infty = 1.0  # Free-stream velocity [m/s]

# Compute bulk pressure from all fluid (tag == 0) cells
# Exclude ghost layers from all 3D fields
tag_ibm_interior = tag_ibm_data[1:-1, 1:-1, 1:-1]
p_interior = p_data[1:-1, 1:-1, 1:-1]

# Mask for fluid cells (tag == 0)
fluid_mask = tag_ibm_interior == 0

# Compute central differences along each axis with proper alignment
# dx: varies along x, keep dimensions for (z, y, x)
dx = 0.5 * (x_data[1:-1, 1:-1, 2:] - x_data[1:-1, 1:-1, :-2])
dy = 0.5 * (y_data[1:-1, 2:, 1:-1] - y_data[1:-1, :-2, 1:-1])
dz = 0.5 * (z_data[2:, 1:-1, 1:-1] - z_data[:-2, 1:-1, 1:-1])
cell_volume = dx * dy * dz

# Compute total pressure × volume and total volume for fluid cells only
bulk_pressure_sum = np.sum(p_interior[fluid_mask] * cell_volume[fluid_mask])
volume_sum = np.sum(cell_volume[fluid_mask])

# Final bulk pressure
p_bulk = bulk_pressure_sum / volume_sum
print("Bulk pressure:", p_bulk)

# COMPUTE PRESSURE COEFFICIENT Cp
# Take the mean pressure over the z-direction
p_data = np.mean(p_data, axis=0)

# Get valid dimensions
ny, nx = p_data.shape

# Convert and clip indices
j_indices = np.array(interface_indices_j)
i_indices = np.array(interface_indices_i)

# Extract surface pressure values from the 2D-averaged pressure field
p_surface = p_data[j_indices, i_indices]  # shape: (N_points,)

# Compute dynamic pressure
q_inf = 0.5 * rho_ref * u_infty**2

# Compute Cp values for each surface point
Cp_values = (p_surface - p_bulk) / q_inf
print("Pressure coefficient computed for all interface points.")

# SAVE Cp RESUKTS TO HDF5 FILE
with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("Cp_values", data=Cp_values)
print(f"Cp values saved to {SAVE_PATH}")

# LOAD AND CHECK SAVED Cp VALUES
with h5py.File(SAVE_PATH, "r") as f:
    Cp_loaded = f["Cp_values"][:]
print("Loaded Cp values from file. Number of points:", Cp_loaded.shape[0])