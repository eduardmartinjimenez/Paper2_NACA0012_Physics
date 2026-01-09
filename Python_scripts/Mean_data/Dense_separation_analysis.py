import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader


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

SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA5_avg_24280000-COMP-DATA.h5"
#SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
#SNAPSHOT_NAME = "3d_NACA0012_Re10000_AoA5_avg_1620000-COMP-DATA.h5"
SNAPSHOT_FILE = os.path.join(SNAPSHOT_PATH, SNAPSHOT_NAME)

# Reference parameters
u_infty = 1.0
AOA = 5  # degrees
alpha = np.deg2rad(AOA)
C = 1.0  # chord length


def rotate_coordinates(x, y, angle_rad):
    """Rotate (x,y) into flow-aligned frame (x',y') by AoA=angle_rad."""
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * ca + y * sa
    y_rot = -x * sa + y * ca
    return x_rot, y_rot


def extract_dense_profiles(x_c_locations, x_rot, y_rot, u_rot, v_rot, 
                           interface_indices_i, interface_indices_j, 
                           x_int_rot, y_int_rot, suction_mask, C,
                           wall_normal_length=0.2, n_points=200):
    """Extract wall-normal velocity profiles at dense x/c locations."""
    
    # Get suction side interface points
    i_suction = interface_indices_i[suction_mask]
    j_suction = interface_indices_j[suction_mask]
    x_suction_rot = x_int_rot[suction_mask]
    y_suction_rot = y_int_rot[suction_mask]
    
    # Build KDTree for fast nearest-neighbor lookup
    x_rot_flat = x_rot.ravel()
    y_rot_flat = y_rot.ravel()
    mesh_coords = np.column_stack((x_rot_flat, y_rot_flat))
    tree = cKDTree(mesh_coords)
    
    profiles = []
    
    for x_c in x_c_locations:
        x_target = x_c * C
        
        # Find closest point on suction surface
        dx = np.abs(x_suction_rot - x_target)
        k = int(np.argmin(dx))
        
        x_start = x_suction_rot[k]
        y_start = y_suction_rot[k]
        y_end = y_start + wall_normal_length
        
        # Generate points along wall-normal direction
        y_query = np.linspace(y_start, y_end, n_points)
        x_query = np.full_like(y_query, x_start)
        query_points = np.column_stack((x_query, y_query))
        
        # Find nearest mesh points
        distances, indices = tree.query(query_points)
        
        # Convert flat indices to (j, i) grid indices
        ny, nx = x_rot.shape
        j_indices = indices // nx
        i_indices = indices % nx
        
        # Extract velocities at these points
        profile_x = x_rot[j_indices, i_indices]
        profile_y = y_rot[j_indices, i_indices]
        profile_u = u_rot[j_indices, i_indices]
        profile_v = v_rot[j_indices, i_indices]
        
        # Remove duplicates
        ij_pairs = np.column_stack((i_indices, j_indices))
        _, unique_idx = np.unique(ij_pairs, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        
        profile = {
            "x_c": x_c,
            "x_start": x_start,
            "y_start": y_start,
            "x_prime": profile_x[unique_idx],
            "y_prime": profile_y[unique_idx],
            "u_rot": profile_u[unique_idx],
            "v_rot": profile_v[unique_idx],
        }
        
        profiles.append(profile)
    
    return profiles


def find_zero_velocity_isoline(profiles, u_infty, tolerance=1e-6):
    """Find where u=0 in each profile."""
    
    isoline_x = []
    isoline_y = []
    
    for profile in profiles:
        x_c = profile["x_c"]
        u_norm = profile["u_rot"] / u_infty
        y_prime = profile["y_prime"]
        
        # Sort by y_prime for interpolation
        sort_idx = np.argsort(y_prime)
        y_sorted = y_prime[sort_idx]
        u_sorted = u_norm[sort_idx]
        
        # Check if u crosses zero
        u_min, u_max = u_sorted.min(), u_sorted.max()
        
        if u_min <= 0 <= u_max:
            # Find zero crossing
            try:
                f_interp = interp1d(u_sorted, y_sorted, kind='linear', bounds_error=False)
                y_zero = float(f_interp(0.0))
                
                if not np.isnan(y_zero):
                    isoline_x.append(profile["x_start"])
                    isoline_y.append(y_zero)
            except:
                pass
    
    return np.array(isoline_x), np.array(isoline_y)


# ============================================================================
# Main execution
# ============================================================================

print("Loading geometrical data...")
with h5py.File(GEO_FILE, "r") as f:
    interface_indices_i = f["interface_indices_i"][...].astype(np.int32)
    interface_indices_j = f["interface_indices_j"][...].astype(np.int32)

print("Loading mesh and snapshot data...")
loader = CompressedSnapshotLoader(MESH_FILE)
fields = loader.load_snapshot_avg(SNAPSHOT_FILE)

# Coordinates
x_data = loader.x[1, :, :]
y_data = loader.y[1, :, :]

# Mean velocities
avg_u_data = loader.reconstruct_field(fields["avg_u"])
avg_v_data = loader.reconstruct_field(fields["avg_v"])

# Average in spanwise direction
avg_u_data = np.mean(avg_u_data, axis=0)
avg_v_data = np.mean(avg_v_data, axis=0)

print(f"Velocity field shape: {avg_u_data.shape}")

# Rotate into flow-aligned frame
x_rot, y_rot = rotate_coordinates(x_data, y_data, alpha)
u_rot, v_rot = rotate_coordinates(avg_u_data, avg_v_data, alpha)

# Interface points
x_int_rot = x_rot[interface_indices_j, interface_indices_i]
y_int_rot = y_rot[interface_indices_j, interface_indices_i]

# Suction/pressure side selection
suction_mask = y_data[interface_indices_j, interface_indices_i] > 0
pressure_mask = y_data[interface_indices_j, interface_indices_i] < 0

# Derived variables
x_suction_rot = x_int_rot[suction_mask]
y_suction_rot = y_int_rot[suction_mask]
x_pressure_rot = x_int_rot[pressure_mask]
y_pressure_rot = y_int_rot[pressure_mask]

# Define dense x/c locations for sampling
x_c_dense = np.arange(0.10, 1.01, 0.01)  # Every 0.01 from 0.10 to 1.0
print(f"\nExtracting profiles at {len(x_c_dense)} x/c locations...")


# Extract profiles
profiles = extract_dense_profiles(
    x_c_dense, x_rot, y_rot, u_rot, v_rot,
    interface_indices_i, interface_indices_j,
    x_int_rot, y_int_rot, suction_mask, C
)

print(f"Extracted {len(profiles)} profiles")

# Find zero velocity isoline
print("\nFinding zero velocity isoline...")
isoline_x, isoline_y = find_zero_velocity_isoline(profiles, u_infty)

print(f"Found {len(isoline_x)} points on zero velocity isoline")
if len(isoline_x) > 0:
    print(f"  x/c range: [{isoline_x.min():.3f}, {isoline_x.max():.3f}]")
    print(f"  y/c range: [{isoline_y.min():.4f}, {isoline_y.max():.4f}]")


# ============================================================================
# Plot: Separation bubble
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Plot airfoil surface
ax.scatter(x_suction_rot, y_suction_rot, s=2, c="blue", alpha=0.7, label="Suction surface")
ax.scatter(x_pressure_rot, y_pressure_rot, s=2, c="green", alpha=0.7, label="Pressure surface")

# Plot zero velocity isoline
if len(isoline_x) > 0:
    ax.plot(
        isoline_x,
        isoline_y,
        'r-',
        linewidth=2.5,
        marker='o',
        markersize=4,
        label='Zero streamwise velocity (u=0)',
        zorder=5
    )
    
    # Find separation and reattachment points
    # Separation: first point where isoline leaves surface
    # Reattachment: last point where isoline returns to surface
    
    # Find points closest to surface
    y_surface_interp = np.interp(isoline_x, x_suction_rot, y_suction_rot)
    height_above_surface = isoline_y - y_surface_interp
    
    # Separation point (where height starts increasing)
    if len(height_above_surface) > 1:
        sep_idx = 0  # First point
        reatt_idx = len(height_above_surface) - 1  # Last point
        
        ax.plot(isoline_x[sep_idx], isoline_y[sep_idx], 'go', markersize=12, 
                label=f'Separation (x/c={isoline_x[sep_idx]:.3f})', zorder=6)
        ax.plot(isoline_x[reatt_idx], isoline_y[reatt_idx], 'mo', markersize=12,
                label=f'Reattachment (x/c={isoline_x[reatt_idx]:.3f})', zorder=6)
        
        print(f"\nSeparation point: x/c = {isoline_x[sep_idx]:.4f}, y/c = {isoline_y[sep_idx]:.6f}")
        print(f"Reattachment point: x/c = {isoline_x[reatt_idx]:.4f}, y/c = {isoline_y[reatt_idx]:.6f}")
        print(f"Bubble length: Δx/c = {isoline_x[reatt_idx] - isoline_x[sep_idx]:.4f}")

ax.set_xlabel("x'/c", fontsize=12)
ax.set_ylabel("y'/c", fontsize=12)
ax.set_title("Separation Bubble: Zero Streamwise Velocity Isoline", fontsize=13)
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.25, 0.2)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
