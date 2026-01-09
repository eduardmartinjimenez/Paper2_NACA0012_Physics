import os
import sys
import h5py
from stl import mesh
import numpy as np

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

### SAVE RESULTS
# Save varibales: x_over_c, delta_y_plus, delta_x_plus, delta_z_plus, tau_w
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
SAVE_NAME = "3d_NACA0012_Re50000_AoA12_Mean_Data_19680000.h5"
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
fields = loader.load_snapshot_avg(SNAPSHOT_FILE)

### Import 3D Data
# Coordinates
x_data = loader.x
y_data = loader.y
z_data = loader.z
tag_ibm_data = loader.tag_ibm

# Averagd velocity fields
u_data = np.mean(loader.reconstruct_field(fields["avg_u"]), axis=0)
v_data = np.mean(loader.reconstruct_field(fields["avg_v"]), axis=0)
w_data = np.mean(loader.reconstruct_field(fields["avg_w"]), axis=0)

# Reference parameters
rho_ref = 1.0  # Reference density [kg/m3]
u_infty = 1.0  # Free-stream velocity [m/s]
c = 1.0  # Airfoil chord length [m]
Re_c = 50000  # Reynolds number [-]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity [Pa s]
nu_ref = mu_ref / rho_ref  # Kinetic viscosity [m2/s]


### COMPUTE DELTA_X FOR EACH INTERFACE POINT
# Extract x and y coordinates of projection points
x_proj = proj_points[:, 0]
y_proj = proj_points[:, 1]

# Compute centroid of all projected points
centroid_x = np.mean(x_proj)
centroid_y = np.mean(y_proj)

# Compute polar angles relative to the centroid
angles = np.arctan2(y_proj - centroid_y, x_proj - centroid_x)

# Sort indices by angle to form a closed loop around the airfoil
sorted_indices = np.argsort(angles)
sorted_x = x_proj[sorted_indices]
sorted_y = y_proj[sorted_indices]

# Stack into array of shape (N, 2)
sorted_coords = np.stack((sorted_x, sorted_y), axis=1)

# Compute distances to previous and next point in the sorted loop
prev_coords = np.roll(sorted_coords, shift=1, axis=0)
next_coords = np.roll(sorted_coords, shift=-1, axis=0)

# Compute distances
dist_prev = np.linalg.norm(sorted_coords - prev_coords, axis=1)
dist_next = np.linalg.norm(sorted_coords - next_coords, axis=1)

# Compute average distance (delta_x) for each point
delta_x_sorted = 0.5 * (dist_prev + dist_next)

# Print total perimeter based on delta_x values
print(f"Total perimeter (sorted order): {np.sum(delta_x_sorted)}")

# Map delta_x back to original interface point order
delta_x = np.zeros_like(delta_x_sorted)
delta_x[sorted_indices] = delta_x_sorted

# Print summary info
print(f"Total perimeter (original order): {np.sum(delta_x)}")
print(f"Total delta_x not found: {np.sum(np.isnan(delta_x))}")
print(f"Shape delta_x: {delta_x.shape}")

### COMPUTE DELTA_Z AND DELTA_S FOR EACH INTERFACE POINT

# Compute delta_z: constant, mesh is uniform in z
delta_z = z_data[1, 0, 0] - z_data[0, 0, 0]
print(f"Delta_z = {delta_z}")


### COMPUTE WALL SHEAR STRESS FOR EACH INTERFACE POINT

# Build velocity array for valid interface points
u_interface = u_data[interface_indices_j, interface_indices_i]
v_interface = v_data[interface_indices_j, interface_indices_i]
w_interface = w_data[interface_indices_j, interface_indices_i]
U = np.stack((u_interface, v_interface, w_interface), axis=1)  # (N_points, 3)

# Coordinates and normals for all interface points
x_proj = proj_points[:, 0]
y_proj = proj_points[:, 1]
x_over_c = x_proj / c

# Compute tangent vector directly from the surface normal vector
# For a normal vector n = (nx, ny), a perpendicular tangent vector in 2D is (ny, -nx).

tangent_vectors = np.stack(
    [proj_normals[:, 1], -proj_normals[:, 0], np.zeros_like(proj_normals[:, 0])], axis=1
)
tangent_vectors /= np.linalg.norm(
    tangent_vectors, axis=1, keepdims=True
)  # Normalize tangent vectors

# --- Decompose velocity into normal and tangential components ---
U_n = np.sum(U * proj_normals, axis=1, keepdims=True)  # Normal velocity (N, 1)
U_t = U - U_n * proj_normals  # Tangential velocity (N, 3)
U_t_norm = np.linalg.norm(U_t, axis=1)  # Magnitude of tangential velocity (N,)


# Determine sign based on projection onto tangent vectors
sign_factor = np.sign(np.sum(U_t * tangent_vectors, axis=1))

# Apply sign to tangential velocity magnitude
U_t_signed = U_t_norm * sign_factor  # (N,)

# Compute signed wall shear stress
tau_w = mu_ref * U_t_signed / proj_distances  # (N,)

# Friction velocity
u_tau = np.sqrt(np.abs(tau_w) / rho_ref)

# Dimensionless distances
delta_y_plus = proj_distances * u_tau / nu_ref
delta_x_plus = delta_x * u_tau / nu_ref
delta_z_plus = delta_z * u_tau / nu_ref


### FILTER UPPER SURFACE AND SORT BY x/c

# Normalize coords
x_proj = proj_points[:, 0]
y_proj = proj_points[:, 1]
x_over_c = x_proj / c

# Filter upper surface
upper_mask = y_proj >= 0
x_upper = x_over_c[upper_mask]
dy_plus_upper = delta_y_plus[upper_mask]
dx_plus_upper = delta_x_plus[upper_mask]
dz_plus_upper = delta_z_plus[upper_mask]
tau_w_upper = tau_w[upper_mask]

# Sort by x/C
sorted_idx = np.argsort(x_upper)
x_upper = x_upper[sorted_idx]
dy_plus_upper = dy_plus_upper[sorted_idx]
dx_plus_upper = dx_plus_upper[sorted_idx]
dz_plus_upper = dz_plus_upper[sorted_idx]
tau_w_upper = tau_w_upper[sorted_idx]


print(f"Mean upper delta_y+: {np.mean(dy_plus_upper):.2f}")
print(f"Mean upper delta_x+: {np.mean(dx_plus_upper):.2f}")
print(f"Mean upper delta_z+: {np.mean(dz_plus_upper):.2f}")

### SAVE RESULTS TO FILE
with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("x_over_c", data=x_upper)
    f.create_dataset("delta_y_plus", data=dy_plus_upper)
    f.create_dataset("delta_x_plus", data=dx_plus_upper)
    f.create_dataset("delta_z_plus", data=dz_plus_upper)
    f.create_dataset("tau_w", data=tau_w_upper)
print(f"Wall units data saved to {SAVE_NAME}")



