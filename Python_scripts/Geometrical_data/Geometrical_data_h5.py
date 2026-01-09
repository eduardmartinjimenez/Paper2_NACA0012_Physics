import os
import sys
import h5py
from stl import mesh
import numpy as np

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader
from Geometrical_plots import plot_projection_interface_points_and_stl

### GEOMETRICAL STUFF
# Find and Save boundary points and their indices, compute their projections, normals vectors and distances to the airfoil
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data"
SAVE_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
# SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Geometrical_data"
# SAVE_NAME = "3d_NACA0012_Test_Geometrical_Data.h5"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Define Mesh file path
MESH_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"

# Define STL file path
STL_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/naca0012.stl"

# Check if the mesh file exists
if os.path.exists(MESH_FILE):
    print(f"Mesh file exists! {MESH_FILE}")
else:
    print(f"Mesh file does not exist: {MESH_FILE}")

# Check is the STL file exists
if os.path.exists(STL_PATH):
    print(f"STL file exists! {STL_PATH}")
else:
    print(f"STL file does not exist: {STL_PATH}")


# Reference parameters
c = 1.0  # Airfoil chord length [m]

# Import data files 
# Load mesh
loader = CompressedSnapshotLoader(MESH_FILE)

# Coordinates 
x_data = loader.x
y_data = loader.y
z_data = loader.z
tag_ibm_data = loader.tag_ibm

# Import STL file
stl_mesh = mesh.Mesh.from_file(STL_PATH)
triangles = stl_mesh.vectors

# Geometric tolerance
tolerance_isInTriangle = 1.0e-8
tolerance_parallel_ray = 1.0e-8
tolerance_lambda = 1.0e-8
tolerance_normal = 1.0e-8


### DETERMINE IF INTERFACE POINT IS INSIDE OR OUTSIDE THE GEOMETRY

def vectorized_isInTriangle(point, A, B, C):
    """
    Vectorized check if a point is inside a set of triangles.

    Parameters:
        point (np.ndarray): Array of shape (3,) representing the point to check.
        A, B, C (np.ndarray): Arrays of shape (N, 3) representing triangle vertices.

    Returns:
        np.ndarray: Boolean array of shape (N,) indicating whether the point lies inside each triangle.
    """
    # Area of the traingle ABC
    cross_ABC = np.cross(B - A, C - A)
    S_ABC = 0.5 * np.linalg.norm(cross_ABC, axis=1)

    # Area of the sub-triangles PAB, PBC, PCA
    cross_PAB = np.cross(point - A, point - B)
    cross_PBC = np.cross(point - B, point - C)
    cross_PCA = np.cross(point - C, point - A)

    S_PAB = 0.5 * np.linalg.norm(cross_PAB, axis=1)
    S_PBC = 0.5 * np.linalg.norm(cross_PBC, axis=1)
    S_PCA = 0.5 * np.linalg.norm(cross_PCA, axis=1)

    # Check if the sum of sub-triangle areas equals the full triangle area
    return np.abs(S_ABC - (S_PAB + S_PBC + S_PCA)) < tolerance_isInTriangle


def vectorized_plane_line_intersect(point, ray, A, normal_normalized):
    """
    Vectorized intersection of a ray with multiple triangle planes.

    Parameters:
        point (np.ndarray): Starting point of the ray (shape: (3,))
        ray (np.ndarray): Ray direction vector (shape: (3,))
        A (np.ndarray): First triangle vertex for each triangle (shape: (N, 3))
        normal_normalized (np.ndarray): Normal vectors for each triangle (shape: (N, 3))

    Returns:
        intersect (np.ndarray): Intersection points (shape: (N, 3))
        lambda_val (np.ndarray): Scalar parameter along the ray (shape: (N,))
    """

    # Compute lambda value
    numerator = np.sum((A - point) * normal_normalized, axis=1)
    denominator = np.sum(normal_normalized * ray, axis=1)
    lambda_val = numerator / denominator

    # Compute intersection point
    intersect = point + lambda_val[:, None] * ray
    return intersect, lambda_val


def vectorized_isInSTL(point, triangles):
    """
    Vectorized ray-casting algorithm to check if a point is inside an STL-defined geometry.

    Parameters:
        point (np.ndarray): Array of shape (3,) representing the point to test.
        triangles (np.ndarray): Array of shape (N, 3, 3) representing STL triangles.

    Returns:
        bool: True if the point is inside the geometry (odd number of intersections), False otherwise.
    """
    p_inf = np.array([-20, -50, -60])

    # Compute the ray vector from the point to a point in the infinity
    ray = p_inf - point
    ray_norm = np.linalg.norm(ray)  # Ray vector norm
    ray_normalized = ray / ray_norm  # Normalized ray vector

    # Extract the vertices of the triangle
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the normal vector of the triangle
    normals = np.cross(B - A, C - A)
    normal_norms = np.linalg.norm(normals, axis=1)

    # Filter out degenerate triangles with near-zero normal
    valid_mask = normal_norms >= tolerance_normal
    if not np.any(valid_mask):
        return False

    A_valid = A[valid_mask]
    B_valid = B[valid_mask]
    C_valid = C[valid_mask]
    normals_valid = normals[valid_mask]
    normals_normalized = normals_valid / normal_norms[valid_mask][:, None]

    # Check for non-parallel intersections
    dot_products = np.dot(normals_normalized, ray_normalized)
    not_parallel_mask = np.abs(dot_products) > tolerance_parallel_ray
    if not np.any(not_parallel_mask):
        return False

    # Filter for non-parallel triangles
    A_final = A_valid[not_parallel_mask]
    B_final = B_valid[not_parallel_mask]
    C_final = C_valid[not_parallel_mask]
    normals_final = normals_normalized[not_parallel_mask]

    # Compute intersection points and lambda values
    intersects, lambda_vals = vectorized_plane_line_intersect(
        point, ray, A_final, normals_final
    )

    # Determine which intersection points are inside their triangle
    inside = vectorized_isInTriangle(intersects, A_final, B_final, C_final)
    valid_lambda = lambda_vals > tolerance_lambda

    # Count total valid intersections
    intersections_count = np.sum(inside & valid_lambda)

    # Point is inside the geometry if intersection count is odd
    return (intersections_count % 2) == 1


# Get grid dimensions
nz, ny, nx = tag_ibm_data.shape

# Select 2D slice at z=1
tag_plane = tag_ibm_data[1, :, :]  # shape (ny, nx)

# Mask of fluid cells (tag == 0)
fluid_mask = tag_plane == 0

# Mask of mixed cells (0 < tag < 1)
mixed_mask = (tag_plane > 0) & (tag_plane < 1)

# Neighbor offsets (dy, dx)
neighbor_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

# Find all fluid cells adjacent to at least one mixed cell
interface_mask = np.zeros_like(tag_plane, dtype=bool)
for dy, dx in neighbor_offsets:
    shifted_mixed = np.zeros_like(mixed_mask, dtype=bool)
    if dy < 0:
        shifted_mixed[-dy:, :] = mixed_mask[:dy, :]
    elif dy > 0:
        shifted_mixed[:-dy, :] = mixed_mask[dy:, :]
    else:
        shifted_mixed[:, :] = mixed_mask

    if dx < 0:
        shifted_mixed[:, -dx:] = shifted_mixed[:, :dx]
    elif dx > 0:
        shifted_mixed[:, :-dx] = shifted_mixed[:, dx:]
    # No shift for dx == 0

    # Mark as interface if fluid and has at least one mixed neighbor
    interface_mask |= fluid_mask & shifted_mixed

# Extract indices of new interface points (fluid cells next to a mixed cell)
j_indices, i_indices = np.where(interface_mask)

print(f"Total interface (fluid) points: {len(i_indices)}")

# Convert to lists
interface_indices_i = i_indices.tolist()
interface_indices_j = j_indices.tolist()

### FIND THE PROJECTION POINT ON THE STL GEOMETRY OF EACH INTERFACE POINT

# Extract interface points
interface_points = np.array(
    [
        [x_data[1, j, i], y_data[1, j, i], z_data[1, j, i]]
        for i, j in zip(interface_indices_i, interface_indices_j)
    ]
)
num_points = interface_points.shape[0]

#print(interface_points)

# Unpack STL triangles
A = triangles[:, 0, :]  # (T, 3)
B = triangles[:, 1, :]
C = triangles[:, 2, :]
AB = B - A
AC = C - A

# Precompute triangle normals
normals = np.cross(AB, AC)
normal_norms = np.linalg.norm(normals, axis=1)
valid_mask = normal_norms > tolerance_normal

# Filter degenerate triangles
A = A[valid_mask]
B = B[valid_mask]
C = C[valid_mask]
normals = normals[valid_mask]
normal_norms = normal_norms[valid_mask]
normals_normalized = normals / normal_norms[:, None]
num_triangles = A.shape[0]

# Create lists to store the projection information for valid interface points
proj_points = []
proj_normals = []
proj_distances = []

# Create new lists to store valid interface indices
valid_interface_indices_i = []
valid_interface_indices_j = []

# Create a list to store indices for interface points with no valid projection
invalid_interface_indices = []

# Loop over each interface point and project onto all triangles
for idx, point in enumerate(interface_points):
    min_dist = np.inf
    best_proj = None
    best_normal = None

    # Compute projection distance to plane for all triangles
    d_plane = np.einsum("ij,ij->i", point - A, normals_normalized)

    # Project point onto each triangle plane
    projections = point - d_plane[:, None] * normals_normalized  # (T, 3)

    # Check if projection lies inside triangle
    inside_mask = vectorized_isInTriangle(projections, A, B, C)

    # Compute distances to projections (only for valid ones)
    candidate_points = projections[inside_mask]
    candidate_normals = normals_normalized[inside_mask]
    dists = np.linalg.norm(candidate_points - point, axis=1)

    if dists.size > 0:
        best_idx = np.argmin(dists)
        best_proj = candidate_points[best_idx]
        best_normal = candidate_normals[best_idx]
        min_dist = dists[best_idx]

        # Store valid results
        proj_points.append(best_proj)
        proj_normals.append(best_normal)
        proj_distances.append(min_dist)
        valid_interface_indices_i.append(interface_indices_i[idx])
        valid_interface_indices_j.append(interface_indices_j[idx])
    else:
        invalid_interface_indices.append(
            (interface_indices_i[idx], interface_indices_j[idx])
        )
        print(
            f"No valid projection found for interface point at indices i={interface_indices_i[idx]}, j={interface_indices_j[idx]}, x = {point[0]}, y = {point[1]}"
        )

# Convert output to arrays
proj_points = np.array(proj_points)
proj_normals = np.array(proj_normals)
proj_distances = np.array(proj_distances)

# Update interface point indices
interface_indices_i = valid_interface_indices_i
interface_indices_j = valid_interface_indices_j

print(f"Total valid interface points: {len(interface_indices_i)}")

# Filter leading edge points (same y coord)
# Normalize x_proj for filtering
x_proj = proj_points[:, 0]
x_over_c = x_proj / c

# Compute y_coord of interface points (structured grid, has duplicate y values)
y_interface = np.array([y_data[1, j, i] for i, j in zip(interface_indices_i, interface_indices_j)])

# Apply leading edge filter based on x/c
leading_edge_mask = x_over_c <= 0.1
print(f"Leading edge points: {np.sum(leading_edge_mask)}")

# Get indices of leading edge points
le_indices = np.where(leading_edge_mask)[0]

# Group by interface y-coord and keep point with smallest proj_distances
best_indices = {}
for idx in le_indices:
    y_key = round(y_interface[idx], 10)
    if y_key not in best_indices or proj_distances[idx] < proj_distances[best_indices[y_key]]:
        best_indices[y_key] = idx

# Indices to keep
le_keep = list(best_indices.values())
non_le_indices = np.where(x_over_c > 0.1)[0]
final_indices = np.array(le_keep + non_le_indices.tolist())

# Filter all arrays by leading edge deduplication first
proj_points = proj_points[final_indices]
proj_normals = proj_normals[final_indices]
proj_distances = proj_distances[final_indices]
interface_indices_i = [interface_indices_i[i] for i in final_indices]
interface_indices_j = [interface_indices_j[i] for i in final_indices]

print(f"Total interface points after leading edge deduplication: {len(proj_points)}")

# Filter duplicate x-coordinates with nearby y-coordinates using interface grid coordinates
print(f"Total interface points before x-y proximity deduplication: {len(proj_points)}")

Y_PROXIMITY_THRESHOLD = 0.001  # Adjust based on your grid spacing
X_MIN_FILTER = 0.5
X_MAX_FILTER = 0.999

# Separate points: within range and outside range
within_range_indices = []
outside_range_indices = []

for idx in range(len(proj_points)):
    x_interface = x_data[1, interface_indices_j[idx], interface_indices_i[idx]]
    if X_MIN_FILTER <= x_interface <= X_MAX_FILTER:
        within_range_indices.append(idx)
    else:
        outside_range_indices.append(idx)

# Group by x-coordinate only for points WITHIN range
best_indices_xy = {}
for idx in within_range_indices:
    x_interface = x_data[1, interface_indices_j[idx], interface_indices_i[idx]]
    y_interface = y_data[1, interface_indices_j[idx], interface_indices_i[idx]]
    x_key = round(x_interface, 10)
    
    if x_key not in best_indices_xy:
        best_indices_xy[x_key] = []
    best_indices_xy[x_key].append((idx, y_interface))

# For each x-group, cluster by y-proximity and keep closest to surface
xy_keep_indices = []
for x_key, points_list in best_indices_xy.items():
    points_list.sort(key=lambda p: p[1])  # Sort by y
    
    clusters = []
    current_cluster = [points_list[0]]
    
    for i in range(1, len(points_list)):
        if abs(points_list[i][1] - current_cluster[-1][1]) <= Y_PROXIMITY_THRESHOLD:
            current_cluster.append(points_list[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [points_list[i]]
    clusters.append(current_cluster)
    
    # Keep point with smallest proj_distance in each cluster
    for cluster in clusters:
        best_in_cluster = min(cluster, key=lambda p: proj_distances[p[0]])
        xy_keep_indices.append(best_in_cluster[0])

# Combine: filtered points within range + all points outside range
final_keep_indices = np.array(xy_keep_indices + outside_range_indices)

# Filter all arrays
proj_points = proj_points[final_keep_indices]
proj_normals = proj_normals[final_keep_indices]
proj_distances = proj_distances[final_keep_indices]
interface_indices_i = [interface_indices_i[i] for i in final_keep_indices]
interface_indices_j = [interface_indices_j[i] for i in final_keep_indices]

# Rebuild interface_points so it matches the final indices
interface_points = np.array(
    [[x_data[1, j, i], y_data[1, j, i], z_data[1, j, i]]
     for i, j in zip(interface_indices_i, interface_indices_j)]
)

print(f"Total interface points after x-y proximity deduplication: {len(proj_points)}")

# Save files in HDF5 format
print(f"Saving geometrical data to {SAVE_PATH}")
with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("interface_points", data=interface_points, dtype="float32")
    f.create_dataset("proj_points", data=proj_points, dtype="float32")
    f.create_dataset("proj_normals", data=proj_normals, dtype="float32")
    f.create_dataset("proj_distances", data=proj_distances, dtype="float32")
    f.create_dataset("interface_indices_i", data=np.array(interface_indices_i), dtype="int32")
    f.create_dataset("interface_indices_j", data=np.array(interface_indices_j), dtype="int32")

print(f"Geometrical data saved successfully to {SAVE_PATH}")

# Plot projections points, their interface points and the stl
plot_projection_interface_points_and_stl(
    x_data, y_data, interface_indices_i, interface_indices_j, proj_points, triangles
)

