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
# Save variables: Cf
# SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
# SAVE_NAME = "3d_NACA0012_Re50000_AoA12_Cf_19680000_scalar.h5"
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
SAVE_NAME = "3d_NACA0012_Re50000_AoA5_Cf_24280000_scalar.h5"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Geometrical data
### GEOMETRICAL STUFF
# GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data"
# GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
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
# MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
# MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Check if the mesh file exists
if os.path.exists(MESH_FILE):
    print(f"Mesh file exists! {MESH_FILE}")
else:
    print(f"Mesh file does not exist: {MESH_FILE}")

# Load Snapshot data
# SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/temporal_last_snapshot/"
# SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_19680000-COMP-DATA.h5"
SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA5_avg_24280000-COMP-DATA.h5"
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

# Averaged velocity fields - take mean over z-direction for 2D analysis
u_data = np.mean(loader.reconstruct_field(fields_avg["avg_u"]), axis=0)
v_data = np.mean(loader.reconstruct_field(fields_avg["avg_v"]), axis=0)
w_data = np.mean(loader.reconstruct_field(fields_avg["avg_w"]), axis=0)

# Reference parameters
rho_ref = 1.0  # Reference density [kg/m3]
u_infty = 1.0  # Free-stream velocity [m/s]
c = 1.0  # Airfoil chord length [m]
Re_c = 50000  # Reynolds number [-]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity [Pa s]
nu_ref = mu_ref / rho_ref  # Kinetic viscosity [m2/s]

print(f"Reference parameters:")
print(f"  rho_ref = {rho_ref} [kg/m3]")
print(f"  u_infty = {u_infty} [m/s]")
print(f"  Re_c = {Re_c}")
print(f"  mu_ref = {mu_ref} [Pa s]")
print(f"  nu_ref = {nu_ref} [m2/s]")

# Number of interface points
n_interface = len(interface_indices_i)
print(f"Number of interface points: {n_interface}")


# COMPUTE WALL SHEAR STRESS FOR EACH INTERFACE POINT
# Using scalar projection method (same as in premultiplied_spectra.py):
# 1. Compute tangent vector from normal: t = (n_y, -n_x, 0) / ||...||
# 2. Compute tangential velocity: u_t = U · t (scalar projection)
# 3. Compute wall shear stress: tau_w = mu * u_t / distance_to_wall

# Compute tangent vector directly from the surface normal vector
# For a normal vector n = (nx, ny, nz), a perpendicular tangent vector in 2D is (ny, -nx, 0)
tangent_vectors = np.stack(
    [proj_normals[:, 1], -proj_normals[:, 0], np.zeros_like(proj_normals[:, 0])], axis=1
)
tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1, keepdims=True)  # Normalize

# Build velocity array for interface points
u_interface = u_data[interface_indices_j, interface_indices_i]
v_interface = v_data[interface_indices_j, interface_indices_i]
w_interface = w_data[interface_indices_j, interface_indices_i]
U = np.stack((u_interface, v_interface, w_interface), axis=1)  # (N_points, 3)

# Compute tangential velocity component using scalar projection: u_t = U · t
# This is simpler and more direct than decomposing the velocity vector
u_t_scalar = np.sum(U * tangent_vectors, axis=1)  # (N,)

print(f"Tangential velocity (scalar projection) computed for all {n_interface} interface points.")

# Compute wall shear stress: tau_w = mu * (dU_t/dn) ≈ mu * U_t / delta_n
# where delta_n is the distance from the wall to the first fluid point
tau_wall = mu_ref * u_t_scalar / proj_distances  # (N,)

print(f"Wall shear stress computed for all {n_interface} interface points.")

# Compute dynamic pressure
q_inf = 0.5 * rho_ref * u_infty**2

# COMPUTE SKIN FRICTION COEFFICIENT Cf
# Cf = tau_wall / q_inf
Cf_values = tau_wall / q_inf
print("Skin friction coefficient computed for all interface points.")

# Print statistics before saving
print("\n" + "="*60)
print("SKIN FRICTION COEFFICIENT STATISTICS (SCALAR PROJECTION METHOD)")
print("="*60)
print(f"Cf min:  {np.min(Cf_values):.6e}")
print(f"Cf max:  {np.max(Cf_values):.6e}")
print(f"Cf mean: {np.mean(Cf_values):.6e}")
print(f"Cf std:  {np.std(Cf_values):.6e}")
print(f"\nWall shear stress statistics:")
print(f"tau_w min:  {np.min(tau_wall):.6e} [Pa]")
print(f"tau_w max:  {np.max(tau_wall):.6e} [Pa]")
print(f"tau_w mean: {np.mean(tau_wall):.6e} [Pa]")
print(f"\nTangential velocity (scalar projection) statistics:")
print(f"u_t min:  {np.min(u_t_scalar):.6e} [m/s]")
print(f"u_t max:  {np.max(u_t_scalar):.6e} [m/s]")
print(f"u_t mean: {np.mean(u_t_scalar):.6e} [m/s]")
print("="*60 + "\n")

# SAVE Cf RESULTS TO HDF5 FILE
with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("Cf_values", data=Cf_values)
    f.create_dataset("tau_wall", data=tau_wall)
    f.create_dataset("interface_indices_i", data=interface_indices_i)
    f.create_dataset("interface_indices_j", data=interface_indices_j)
    f.create_dataset("proj_distances", data=proj_distances)
    f.create_dataset("u_t_scalar", data=u_t_scalar)
print(f"Cf values saved to {SAVE_PATH}")

# LOAD AND CHECK SAVED Cf VALUES
with h5py.File(SAVE_PATH, "r") as f:
    Cf_loaded = f["Cf_values"][:]
print("Loaded Cf values from file. Number of points:", Cf_loaded.shape[0])
print(f"Verification - Cf min: {np.min(Cf_loaded):.6e}, Cf max: {np.max(Cf_loaded):.6e}, Cf mean: {np.mean(Cf_loaded):.6e}")
