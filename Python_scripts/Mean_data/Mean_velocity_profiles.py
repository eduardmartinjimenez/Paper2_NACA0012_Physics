import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pickle

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader


# Output data path
#OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
#OUTPUT_DATA_NAME = "AoA5_Re50000_velocity_profiles_data.h5"
OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Mean_data/"
OUTPUT_DATA_NAME = "AoA5_Re10000_velocity_profiles_data.h5"
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DATA_PATH, OUTPUT_DATA_NAME)

# Paths
# GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
# GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Geometrical_data"
GEO_NAME = "3d_NACA0012_Test_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
# MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
MESH_NAME = "3d_NACA0012_Re10000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)


# SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
# SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA5_avg_24280000-COMP-DATA.h5"
SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
SNAPSHOT_NAME = "3d_NACA0012_Re10000_AoA5_avg_1620000-COMP-DATA.h5"
SNAPSHOT_FILE = os.path.join(SNAPSHOT_PATH, SNAPSHOT_NAME)

# Reference parameters
u_infty = 1.0
AOA = 5 # degrees
alpha = np.deg2rad(AOA)
C = 1.0  # chord length


# x/c locations
x_c_locations = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

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


def save_velocity_profile_data(filepath, x_rot, y_rot, u_rot, v_rot,
                                 interface_indices_i, interface_indices_j,
                                 x_int_rot, y_int_rot, suction_mask, pressure_mask,
                                 reference_points, wall_normal_profiles_unique,
                                 u_infty, alpha, C, x_c_locations):
    """Save all computed data to HDF5 file."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, "w") as f:
        # Metadata
        f.attrs["u_infty"] = u_infty
        f.attrs["alpha"] = alpha
        f.attrs["C"] = C
        f.create_dataset("x_c_locations", data=x_c_locations)
        
        # Mesh and velocity data
        f.create_dataset("x_rot", data=x_rot, compression="gzip")
        f.create_dataset("y_rot", data=y_rot, compression="gzip")
        f.create_dataset("u_rot", data=u_rot, compression="gzip")
        f.create_dataset("v_rot", data=v_rot, compression="gzip")
        
        # Interface data
        f.create_dataset("interface_indices_i", data=interface_indices_i)
        f.create_dataset("interface_indices_j", data=interface_indices_j)
        f.create_dataset("x_int_rot", data=x_int_rot)
        f.create_dataset("y_int_rot", data=y_int_rot)
        f.create_dataset("suction_mask", data=suction_mask)
        f.create_dataset("pressure_mask", data=pressure_mask)
        
        # Reference points (save as pickle in an attribute since it's a list of dicts)
        f.attrs["reference_points_pickle"] = np.void(pickle.dumps(reference_points))
        
        # Wall-normal profiles - save each profile as a group
        profiles_group = f.create_group("wall_normal_profiles")
        for idx, profile in enumerate(wall_normal_profiles_unique):
            prof_group = profiles_group.create_group(f"profile_{idx}")
            prof_group.attrs["x_c"] = profile["x_c"]
            prof_group.attrs["x_start"] = profile["x_start"]
            prof_group.attrs["y_start"] = profile["y_start"]
            prof_group.attrs["y_end"] = profile["y_end"]
            prof_group.create_dataset("i_indices", data=profile["i_indices"])
            prof_group.create_dataset("j_indices", data=profile["j_indices"])
            prof_group.create_dataset("x_prime", data=profile["x_prime"])
            prof_group.create_dataset("y_prime", data=profile["y_prime"])
            prof_group.create_dataset("u_rot", data=profile["u_rot"])
            prof_group.create_dataset("v_rot", data=profile["v_rot"])
    
    print(f"Data saved to: {filepath}")


def load_velocity_profile_data(filepath):
    """Load all computed data from HDF5 file."""
    
    if not os.path.exists(filepath):
        return None
    
    with h5py.File(filepath, "r") as f:
        # Metadata
        u_infty = f.attrs["u_infty"]
        alpha = f.attrs["alpha"]
        C = f.attrs["C"]
        x_c_locations = f["x_c_locations"][...]
        
        # Mesh and velocity data
        x_rot = f["x_rot"][...]
        y_rot = f["y_rot"][...]
        u_rot = f["u_rot"][...]
        v_rot = f["v_rot"][...]
        
        # Interface data
        interface_indices_i = f["interface_indices_i"][...]
        interface_indices_j = f["interface_indices_j"][...]
        x_int_rot = f["x_int_rot"][...]
        y_int_rot = f["y_int_rot"][...]
        suction_mask = f["suction_mask"][...]
        pressure_mask = f["pressure_mask"][...]
        
        # Reference points
        reference_points = pickle.loads(f.attrs["reference_points_pickle"].tobytes())
        
        # Wall-normal profiles
        wall_normal_profiles_unique = []
        profiles_group = f["wall_normal_profiles"]
        for idx in range(len(profiles_group)):
            prof_group = profiles_group[f"profile_{idx}"]
            profile = {
                "x_c": prof_group.attrs["x_c"],
                "x_start": prof_group.attrs["x_start"],
                "y_start": prof_group.attrs["y_start"],
                "y_end": prof_group.attrs["y_end"],
                "i_indices": prof_group["i_indices"][...],
                "j_indices": prof_group["j_indices"][...],
                "x_prime": prof_group["x_prime"][...],
                "y_prime": prof_group["y_prime"][...],
                "u_rot": prof_group["u_rot"][...],
                "v_rot": prof_group["v_rot"][...],
            }
            wall_normal_profiles_unique.append(profile)
    
    print(f"Data loaded from: {filepath}")
    
    return {
        "x_rot": x_rot,
        "y_rot": y_rot,
        "u_rot": u_rot,
        "v_rot": v_rot,
        "interface_indices_i": interface_indices_i,
        "interface_indices_j": interface_indices_j,
        "x_int_rot": x_int_rot,
        "y_int_rot": y_int_rot,
        "suction_mask": suction_mask,
        "pressure_mask": pressure_mask,
        "reference_points": reference_points,
        "wall_normal_profiles_unique": wall_normal_profiles_unique,
        "u_infty": u_infty,
        "alpha": alpha,
        "C": C,
        "x_c_locations": x_c_locations,
    }


# Try to load cached data first
print("Checking for cached data...")
cached_data = load_velocity_profile_data(OUTPUT_DATA_FILE)

if cached_data is not None:
    print("\n" + "="*60)
    print("USING CACHED DATA - Skipping expensive computations")
    print("="*60 + "\n")
    
    # Extract all variables from cached data
    x_rot = cached_data["x_rot"]
    y_rot = cached_data["y_rot"]
    u_rot = cached_data["u_rot"]
    v_rot = cached_data["v_rot"]
    interface_indices_i = cached_data["interface_indices_i"]
    interface_indices_j = cached_data["interface_indices_j"]
    x_int_rot = cached_data["x_int_rot"]
    y_int_rot = cached_data["y_int_rot"]
    suction_mask = cached_data["suction_mask"]
    pressure_mask = cached_data["pressure_mask"]
    reference_points = cached_data["reference_points"]
    wall_normal_profiles_unique = cached_data["wall_normal_profiles_unique"]
    u_infty = cached_data["u_infty"]
    alpha = cached_data["alpha"]
    C = cached_data["C"]
    x_c_locations = cached_data["x_c_locations"]
    
    # Compute derived variables needed for plotting
    x_data = None  # Not needed for visualization
    y_data = None  # Not needed for visualization
    x_int = None
    y_int = None
    x_suction_rot = x_int_rot[suction_mask]
    y_suction_rot = y_int_rot[suction_mask]
    x_pressure_rot = x_int_rot[pressure_mask]
    y_pressure_rot = y_int_rot[pressure_mask]
    x_rot_flat = x_rot.ravel()
    y_rot_flat = y_rot.ravel()
    
else:
    print("\n" + "="*60)
    print("NO CACHED DATA - Running full computation")
    print("="*60 + "\n")
    
# Load geometrical data
assert_exists(GEO_FILE, "Geometrical data file")

with h5py.File(GEO_FILE, "r") as f:
    interface_indices_i = f["interface_indices_i"][...].astype(np.int32)
    interface_indices_j = f["interface_indices_j"][...].astype(np.int32)

print("interface_indices_i shape:", interface_indices_i.shape)
print("interface_indices_j shape:", interface_indices_j.shape)


# Load mesh + mean snapshot
assert_exists(MESH_FILE, "Mesh file")
assert_exists(SNAPSHOT_FILE, "Snapshot file")

loader = CompressedSnapshotLoader(MESH_FILE)
fields = loader.load_snapshot_avg(SNAPSHOT_FILE)

# Coordinates and IBM tag on that plane: shape (ny, nx)
x_data = loader.x[1, :, :]
y_data = loader.y[1, :, :]

print("x_data shape:", x_data.shape)
print("y_data shape:", y_data.shape)

# Mean velocities: 
avg_u_data = loader.reconstruct_field(fields["avg_u"])
avg_v_data = loader.reconstruct_field(fields["avg_v"])

print("reconstructed avg_u shape:", avg_u_data.shape)

# Average in spanwise direction (axis=0)
avg_u_data = np.mean(avg_u_data, axis=0)
avg_v_data = np.mean(avg_v_data, axis=0)

print("avg_u_data shape:", avg_u_data.shape)
print("avg_v_data shape:", avg_v_data.shape)

# Final shape consistency check
if x_data.shape != avg_u_data.shape:
    raise ValueError(
        "Shape mismatch: x_data and avg_u_data must both be (ny, nx). "
        f"Got x_data={x_data.shape}, avg_u_data={avg_u_data.shape}. "
        "Check which axis is spanwise in reconstruct_field output."
    )


# Rotate into flow-aligned frame
x_rot, y_rot = rotate_coordinates(x_data, y_data, alpha)
u_rot, v_rot = rotate_coordinates(avg_u_data, avg_v_data, alpha)

print("Rotation ranges:")
print("  x_rot:", x_rot.min(), x_rot.max())
print("  y_rot:", y_rot.min(), y_rot.max())
print("  u_rot:", np.nanmin(u_rot), np.nanmax(u_rot))
print("  v_rot:", np.nanmin(v_rot), np.nanmax(v_rot))



# Interface points from indices 
# Global (for reference)
x_int = x_data[interface_indices_j, interface_indices_i]
y_int = y_data[interface_indices_j, interface_indices_i]

# Rotated coordinates
x_int_rot = x_rot[interface_indices_j, interface_indices_i]
y_int_rot = y_rot[interface_indices_j, interface_indices_i]


# Plots (global + rotated)
fig = plt.figure(figsize=(10, 8))
plt.scatter(x_int, y_int, s=1, c="blue")
plt.title("Interface Points from indices (global system)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 8))
plt.scatter(x_int_rot, y_int_rot, s=1, c="red")
plt.title("Interface Points from indices (flow-aligned system)")
plt.xlabel("x'")
plt.ylabel("y'")
plt.axis("equal")
plt.grid(True)
plt.show()

# Suction/pressure side selection
suction_mask = y_int > 0
pressure_mask = y_int < 0


print("Suction-side interface points:", np.sum(suction_mask), "/", suction_mask.size)
print("Pressure-side interface points:", np.sum(pressure_mask), "/", pressure_mask.size)

# Apply the SAME masks to rotated interface coordinates
x_suction_rot = x_int_rot[suction_mask]
y_suction_rot = y_int_rot[suction_mask]

x_pressure_rot = x_int_rot[pressure_mask]
y_pressure_rot = y_int_rot[pressure_mask]

# plot both sides in global frame
plt.figure(figsize=(10, 8))
plt.scatter(x_int[suction_mask], y_int[suction_mask], s=1, label="Suction (global y>0)")
plt.scatter(x_int[pressure_mask], y_int[pressure_mask], s=1, label="Pressure (global y<0)")
plt.title("Interface points classified by side (global frame)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

# plot both sides in rotated frame
plt.figure(figsize=(10, 8))
plt.scatter(x_suction_rot, y_suction_rot, s=1, label="Suction (mask from global)")
plt.scatter(x_pressure_rot, y_pressure_rot, s=1, label="Pressure (mask from global)")
plt.title("Interface points classified by side (flow-aligned frame)")
plt.xlabel("x'")
plt.ylabel("y'")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

# Extract suction-side interface points
i_suction = interface_indices_i[suction_mask]
j_suction = interface_indices_j[suction_mask]

x_suction_rot = x_int_rot[suction_mask]
y_suction_rot = y_int_rot[suction_mask]

print("Total suction-side interface points:", x_suction_rot.size)

# Find closest points to x/c locations
reference_points = [] 

for x_c in x_c_locations:
    x_target = x_c * C  # target x'

    dx = np.abs(x_suction_rot - x_target)
    k = int(np.argmin(dx))

    ref = {
        "x_c": float(x_c),
        "x_target": float(x_target),
        "i": int(i_suction[k]),
        "j": int(j_suction[k]),
        "xprime": float(x_suction_rot[k]),
        "yprime": float(y_suction_rot[k]),
    }

    reference_points.append(ref)

    print(
        f"x'/C={ref['x_c']:.2f} -> (i,j)=({ref['i']},{ref['j']}), "
        f"(x',y')=({ref['xprime']:.6f},{ref['yprime']:.6f}), "
    )

# Plot reference points on suction side
plt.figure(figsize=(10, 8))
plt.scatter(x_suction_rot, y_suction_rot, s=1, label="Suction side points")
plt.scatter(x_pressure_rot, y_pressure_rot, s=1, label="Pressure side points", alpha=0.3)
for ref in reference_points:
    plt.plot(ref["xprime"], ref["yprime"], "ro")
    plt.text(
        ref["xprime"],
        ref["yprime"],
        f"x'/C={ref['x_c']:.2f}",
        color="red",
        fontsize=10,
        ha="left",
        va="bottom",
    )
plt.title("Suction-side interface points with reference locations")
plt.xlabel("x'")
plt.ylabel("y'")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

# Parameters for wall-normal extraction
wall_normal_length = 0.2  # Fixed length in y' direction
n_points = 250  # Number of points to sample along each wall-normal line

# Build a KDTree for fast nearest-neighbor lookup on the rotated mesh
# Flatten the rotated coordinates
x_rot_flat = x_rot.ravel()
y_rot_flat = y_rot.ravel()

# Create coordinate pairs for KDTree
mesh_coords = np.column_stack((x_rot_flat, y_rot_flat))
tree = cKDTree(mesh_coords)

# Store extracted profiles
wall_normal_profiles = []

for ref in reference_points:  
    x_start = ref["xprime"]
    y_start = ref["yprime"]
    
    # Punto final:  y_start + longitud fija
    y_end = y_start + wall_normal_length
    
    # Generate points along the wall-normal direction (constant x', varying y')
    # From the surface (y_start) to y_end (distancia fija de 0.2)
    y_query = np.linspace(y_start, y_end, n_points)
    x_query = np.full_like(y_query, x_start)  # x' stays constant
    
    query_points = np.column_stack((x_query, y_query))
    
    # Find nearest mesh points for each query point
    distances, indices = tree.query(query_points)
    
    # Convert flat indices back to (j, i) grid indices
    ny, nx = x_rot.shape
    j_indices = indices // nx
    i_indices = indices % nx
    
    # Extract the actual coordinates and velocities at these points
    profile_data = {
        "x_c":  ref["x_c"],
        "x_start": x_start,
        "y_start": y_start,
        "y_end": y_end,  # Añadido para referencia
        "i_indices": i_indices,
        "j_indices": j_indices,
        "x_prime":  x_rot[j_indices, i_indices],
        "y_prime": y_rot[j_indices, i_indices],
        "u_rot": u_rot[j_indices, i_indices],
        "v_rot": v_rot[j_indices, i_indices],
        "distances": distances,
    }
    
    wall_normal_profiles. append(profile_data)
    
    print(f"\nx'/C = {ref['x_c']:.2f}:")
    print(f"  y' range: [{y_start:.4f}, {y_end:.4f}] (longitud = {wall_normal_length})")
    print(f"  Points extracted: {len(i_indices)}")
    print(f"  Max distance to mesh: {distances. max():.6f}")


# Remove duplicate points (same grid cell selected multiple times)
wall_normal_profiles_unique = []

for profile in wall_normal_profiles:  
    # Create unique (i, j) pairs
    ij_pairs = np.column_stack((profile["i_indices"], profile["j_indices"]))
    _, unique_idx = np.unique(ij_pairs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # Keep original order
    
    profile_unique = {
        "x_c": profile["x_c"],
        "x_start": profile["x_start"],
        "y_start":  profile["y_start"],
        "y_end": profile["y_end"],
        "i_indices": profile["i_indices"][unique_idx],
        "j_indices": profile["j_indices"][unique_idx],
        "x_prime": profile["x_prime"][unique_idx],
        "y_prime":  profile["y_prime"][unique_idx],
        "u_rot": profile["u_rot"][unique_idx],
        "v_rot": profile["v_rot"][unique_idx],
    }
    
    wall_normal_profiles_unique.append(profile_unique)
    print(f"x'/C = {profile['x_c']:.2f}: {len(unique_idx)} unique mesh points")

    # Save all computed data for future use
    print("\nSaving computed data for future use...")
    save_velocity_profile_data(
        OUTPUT_DATA_FILE,
        x_rot, y_rot, u_rot, v_rot,
        interface_indices_i, interface_indices_j,
        x_int_rot, y_int_rot, suction_mask, pressure_mask,
        reference_points, wall_normal_profiles_unique,
        u_infty, alpha, C, x_c_locations
    )

# Plot extraction lines on the mesh
fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.scatter(x_rot_flat, y_rot_flat, s=0.1, c="gray", alpha=0.3, label="Mesh points")
ax1.scatter(x_suction_rot, y_suction_rot, s=1, c="blue", label="Suction surface")

colors = plt.cm. viridis(np. linspace(0, 1, len(wall_normal_profiles_unique)))
for profile, color in zip(wall_normal_profiles_unique, colors):
    ax1.scatter(
        profile["x_prime"], 
        profile["y_prime"], 
        s=10, 
        c=[color], 
        label=f"x'/C = {profile['x_c']:.1f}"
    )
    # Draw line from surface to y_end
    ax1.plot(
        [profile["x_start"], profile["x_start"]], 
        [profile["y_start"], profile["y_end"]],
        '--', 
        color=color, 
        alpha=0.5
    )

ax1.set_xlabel("x'")
ax1.set_ylabel("y'")
ax1.set_title("Wall-normal extraction lines (longitud fija = 0.2)")
ax1.set_xlim(-0.1, 1.2)
ax1.set_ylim(-0.3, 0.3)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True)
ax1.set_aspect("equal")
plt.tight_layout()
plt.show()


# Plot velocity profiles
fig2, ax2 = plt.subplots(figsize=(8, 6))

for profile, color in zip(wall_normal_profiles_unique, colors):
    # Wall-normal distance from surface
    eta = profile["y_prime"] - profile["y_start"]
    ax2.plot(
        profile["u_rot"] / u_infty, 
        eta, 
        'o-', 
        color=color, 
        markersize=3,
        label=f"x'/C = {profile['x_c']:.1f}"
    )

ax2.set_xlabel("u'/U∞")
ax2.set_ylabel("η (wall-normal distance)")
ax2.set_title("Velocity profiles")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()

# Visualization - Plot 3: Velocity profiles on the airfoil
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Plot the airfoil surface
ax3.scatter(x_suction_rot, y_suction_rot, s=1, c="blue", label="Suction surface")
ax3.scatter(x_pressure_rot, y_pressure_rot, s=1, c="green", label="Pressure surface")

# Scale factor for velocity profiles (adjust to make profiles visible)
scale_factor = 0.05

colors = plt.cm. viridis(np.linspace(0, 1, len(wall_normal_profiles_unique)))

for profile, color in zip(wall_normal_profiles_unique, colors):
    # Wall-normal distance from surface
    eta = profile["y_prime"] - profile["y_start"]
    
    # Normalized velocity
    u_normalized = profile["u_rot"] / u_infty
    
    # Position the velocity profile at the reference point
    # x position:  x_start + scaled velocity (horizontal displacement)
    # y position: y_start + eta (wall-normal distance)
    x_profile = profile["x_start"] + u_normalized * scale_factor
    y_profile = profile["y_start"] + eta
    
    # Plot the velocity profile
    ax3.plot(
        x_profile, 
        y_profile, 
        '-', 
        color=color, 
        linewidth=1.5,
        label=f"x'/C = {profile['x_c']:.1f}"
    )
    
    # Draw the baseline (zero velocity reference line)
    ax3.plot(
        [profile["x_start"], profile["x_start"]], 
        [profile["y_start"], profile["y_start"] + eta. max()],
        '--', 
        color=color, 
        alpha=0.5,
        linewidth=0.8
    )
    
    # Mark the reference point on the surface
    ax3.plot(profile["x_start"], profile["y_start"], 'o', color=color, markersize=6)

ax3.set_xlabel("x'")
ax3.set_ylabel("y'")
ax3.set_title("Velocity profiles on airfoil (suction side)")
ax3.set_xlim(-0.1, 1.2)
ax3.set_ylim(-0.25, 0.2)
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_aspect("equal")
plt.tight_layout()
plt.show()

