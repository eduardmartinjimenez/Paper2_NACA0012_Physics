import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader
from scipy.spatial import cKDTree

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Mesh path
# MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_1_compr/"
# MESH_NAME = "slice_1-CROP-MESH.h5"
# MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Mesh slice path
MESH_SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/"
MESH_SLICE_NAME = "slice_1-CROP-MESH.h5"
MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)


# Slice path
SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/"
SLICE_NAME = "slice_1_11280000-COMP-DATA.h5"
SLICE_FILE = os.path.join(SLICE_PATH, SLICE_NAME)

# Average  slice path
AVG_SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/last_slice"
AVG_SLICE_NAME = "slice_1_14302400-COMP-DATA.h5"
AVG_SLICE_FILE = os.path.join(AVG_SLICE_PATH, AVG_SLICE_NAME)

# Reference parameters
rho_ref = 1.0  # Reference density [kg/m3]
u_infty = 1.0  # Free-stream velocity [m/s]
c = 1.0  # Airfoil chord length [m]
Re_c = 50000  # Reynolds number [-]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity [Pa s]
nu_ref = mu_ref / rho_ref  # Kinetic viscosity [m2/s]


# Utilities
def assert_exists(path: str, kind: str = "File") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"{kind} exists: {path}")

# Check files existence
assert_exists(GEO_FILE, "Geometrical data file")
assert_exists(MESH_SLICE_FILE, "Mesh slice file")
assert_exists(SLICE_FILE, "Slice data file")
assert_exists(AVG_SLICE_FILE, "Average slice data file")

# Load geometrical data
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

print("interface_points shape:", interface_points.shape)
print("interface_points sample:", interface_points[:5])
print("proj_normals shape:", proj_normals.shape)
print("proj_normals sample:", proj_normals[:5])
print("proj_distances shape:", proj_distances.shape)
print("proj_distances sample:", proj_distances[:5])

# Plot interface points
plt.figure(figsize=(10, 6))
plt.scatter(interface_points[:, 0], interface_points[:, 1], c='red', s=20, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('NACA 0012 Interface Points')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Filter suction side interface points (y >= 0)
suction_side_points = interface_points[interface_points[:, 1] >= 0]
print("Suction side points shape:", suction_side_points.shape)

# Plot suction side interface points
plt.figure(figsize=(10, 6))
plt.scatter(suction_side_points[:, 0], suction_side_points[:, 1], c='blue', s=20, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('NACA 0012 Suction Side Interface Points')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Load compressed slice data
loader = CompressedSnapshotLoader(MESH_SLICE_FILE)
fields = loader.load_snapshot(SLICE_FILE)

# Load mesh slice data
x_data = loader.x[:, :, :]
y_data = loader.y[:, :, :]
z_data = loader.z[:, :, :]

# Print shapes
print("x_data shape:", x_data.shape)
print("y_data shape:", y_data.shape)
print("z_data shape:", z_data.shape)

# Reconstruct fields
u_data = loader.reconstruct_field(fields["u"])
v_data = loader.reconstruct_field(fields["v"])
w_data = loader.reconstruct_field(fields["w"])
p_data = loader.reconstruct_field(fields["p"])

# Print shapes
print("u_data shape:", u_data.shape)
print("v_data shape:", v_data.shape)
print("w_data shape:", w_data.shape)
print("p_data shape:", p_data.shape)

# Load compressed average slice data
avg_fields = loader.load_snapshot_avg(AVG_SLICE_FILE)

# Reconstruct average fields
avg_u_data = loader.reconstruct_field(avg_fields["avg_u"])
avg_v_data = loader.reconstruct_field(avg_fields["avg_v"])
avg_w_data = loader.reconstruct_field(avg_fields["avg_w"])
avg_p_data = loader.reconstruct_field(avg_fields["avg_p"])

# Print shapes
print("avg_u_data shape:", avg_u_data.shape)
print("avg_v_data shape:", avg_v_data.shape)
print("avg_w_data shape:", avg_w_data.shape)
print("avg_p_data shape:", avg_p_data.shape)

# Find the closest interface point in the suction side matching the slice's x coordinate
# Get the slice's x coordinate (assuming constant x along the slice)
slice_x = x_data[0, 0, 0]  # All x values should be the same in the slice
print(f"Slice x coordinate: {slice_x:.6f}")

# Find the interface point on suction side closest to this x
x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
closest_interface_point = suction_side_points[closest_idx]
interface_y = closest_interface_point[1]  # y coordinate of the interface point

print(f"Closest interface point: ({closest_interface_point[0]:.6f}, {interface_y:.6f})")

# Plot the closest interface point
plt.figure(figsize=(10, 6))
plt.scatter(suction_side_points[:, 0], suction_side_points[:, 1], c='blue', s=10, alpha=0.5, label='Suction Side Interface Points')
plt.scatter(closest_interface_point[0], closest_interface_point[1], c='red', s=10, label='Closest Interface Point')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Closest Interface Point to Slice x Coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Find the slice y grid index closest to the interface y coordinate
slice_y_unique = np.unique(y_data[0, :, 0])  # Get unique y values in the slice
y_distances = np.abs(slice_y_unique - interface_y)
j_closest = np.argmin(y_distances)
slice_y_at_interface = slice_y_unique[j_closest]

print(f"Slice y grid index closest to interface: {j_closest}")
print(f"Slice y value at interface: {slice_y_at_interface:.6f}")
print(f"Distance to interface y: {y_distances[j_closest]:.6e}")

# Plot the slice points, the interface points, and highlight the closest interface point
plt.figure(figsize=(10, 6))
plt.scatter(x_data.flatten(), y_data.flatten(), c='blue', s=5, alpha=0.5, label='Slice Points')
plt.scatter(closest_interface_point[0], closest_interface_point[1], c='red', s=20, label='Closest Interface Point')
plt.scatter(suction_side_points[:, 0], suction_side_points[:, 1], c='blue', s=10, alpha=0.5, label='Suction Side Interface Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Slice Points and Closest Interface Point')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()  

# Extract the normal direction at the closest interface point
# Use the index we already found (closest_idx from interface_points)
# But we need to find the index in the full interface_points array (before filtering)
tree_full = cKDTree(interface_points[:, :2])
_, idx_full = tree_full.query(closest_interface_point[:2])
normal_at_closest_point = proj_normals[idx_full]
distance_at_closest_point = proj_distances[idx_full]

print(f"Normal at closest interface point: {normal_at_closest_point}")
print(f"Wall distance at closest interface point: {distance_at_closest_point:.6e}")

# Plot the normal vector at the closest interface point
# plt.figure(figsize=(10, 6))
# plt.scatter(interface_points[:, 0], interface_points[:, 1], c='lightgray', s=10, alpha=0.5, label='Interface Points')
# plt.scatter(x_data.flatten(), y_data.flatten(), c='blue', s=5, alpha=0.5, label='Slice Points')
# plt.quiver(
#     closest_interface_point[0], closest_interface_point[1],
#     normal_at_closest_point[0], normal_at_closest_point[1],
#     color='red', scale=10, width=0.005, label='Normal'
# )
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Normal Vector at Closest Interface Point with Slice')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()  

# Compute tangential direction at the closest interface point
tangent_at_closest_point = np.array([normal_at_closest_point[1], -normal_at_closest_point[0], 0.0])
tangent_at_closest_point /= np.linalg.norm(tangent_at_closest_point)
print("Tangent at closest interface point:", tangent_at_closest_point)

# Verify orthogonality
# dot_product = np.dot(normal_at_closest_point, tangent_at_closest_point)
# print("Dot product (should be close to 0):", dot_product)

# Plot the normal vector at the closest interface point with the slice
# plt.figure(figsize=(10, 6))
# plt.scatter(interface_points[:, 0], interface_points[:, 1], c='lightgray', s=10, alpha=0.5, label='Interface Points')
# plt.scatter(x_data.flatten(), y_data.flatten(), c='blue', s=5, alpha=0.5, label='Slice Points')
# plt.quiver(
#     closest_interface_point[0], closest_interface_point[1],
#     normal_at_closest_point[0], normal_at_closest_point[1],
#     color='red', scale=10, width=0.005, label='Normal'
# )
# plt.quiver(
#     closest_interface_point[0], closest_interface_point[1],
#     tangent_at_closest_point[0], tangent_at_closest_point[1],
#     color='green', scale=10, width=0.005, label='Tangent'
# )
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Normal Vector at Closest Interface Point with Slice')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()  

# Decompose velocity into normal and tangential components
u_n = (u_data * normal_at_closest_point[0] +
       v_data * normal_at_closest_point[1] +
       w_data * normal_at_closest_point[2]) # Normal velocity component

u_t = (u_data * tangent_at_closest_point[0] +
       v_data * tangent_at_closest_point[1] +
       w_data * tangent_at_closest_point[2]) # Tangential velocity component

print("u_n shape:", u_n.shape)
print("u_t shape:", u_t.shape)

# Decompose average velocity into normal and tangential components
avg_u_n = (avg_u_data * normal_at_closest_point[0] +
           avg_v_data * normal_at_closest_point[1] +
           avg_w_data * normal_at_closest_point[2]) # Normal velocity component

avg_u_t = (avg_u_data * tangent_at_closest_point[0] +
           avg_v_data * tangent_at_closest_point[1] +
           avg_w_data * tangent_at_closest_point[2]) # Tangential velocity component    

print("avg_u_n shape:", avg_u_n.shape)
print("avg_u_t shape:", avg_u_t.shape)

# Compute the tangential velocity fluctuations
u_t_fluct = u_t - avg_u_t
print("u_t_fluct shape:", u_t_fluct.shape)

# Spanwise average of average tangential velocity for wall shear stress
span_avg_avg_u_t = np.mean(avg_u_t, axis=0)
print("span_avg_avg_u_t shape:", span_avg_avg_u_t.shape)

# Compute wall shear stress at the closest interface point
# Use spanwise-averaged tangential velocity at the closest y grid location
u_t_closest_avg = span_avg_avg_u_t[j_closest, 0]
print("Spanwise-averaged tangential velocity at closest point:", u_t_closest_avg)

# Wall shear stress at the closest interface point: τ_w = μ * u_t / distance
tau_w_closest = mu_ref * u_t_closest_avg / distance_at_closest_point

# Friction velocity: u_tau = sqrt(τ_w / ρ)
u_tau = np.sqrt(np.abs(tau_w_closest) / rho_ref)

# Dimensionless wall distance: y+ = u_tau * y / nu
y_plus = u_tau * distance_at_closest_point / nu_ref

print(f"\nWall shear stress at closest interface point:")
print(f"  Interface location: ({closest_interface_point[0]:.6f}, {interface_y:.6f})")
print(f"  Wall shear stress: {tau_w_closest:.6e} Pa")
print(f"  Friction velocity: {u_tau:.6e} m/s")
print(f"  Dimensionless wall distance (y+): {y_plus:.6e}")









