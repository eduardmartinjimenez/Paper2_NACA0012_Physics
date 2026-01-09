import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import pickle

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader


# Output data path
#OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
#OUTPUT_DATA_NAME = "AoA5_Re50000_velocity_profiles_dense_data.h5"
OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Mean_data/"
OUTPUT_DATA_NAME = "AoA5_Re10000_velocity_profiles_data_dense.h5"
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DATA_PATH, OUTPUT_DATA_NAME)

# RMS results
# RMS_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
# RMS_DATA_NAME = "AoA5_Re50000_velocity_RMS_profiles_data.h5"
RMS_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Mean_data/"
RMS_DATA_NAME = "AoA5_Re10000_velocity_RMS_profiles_data.h5"
RMS_DATA_FILE = os.path.join(RMS_DATA_PATH, RMS_DATA_NAME)

# Paths
#GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
#GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Geometrical_data"
GEO_NAME = "3d_NACA0012_Test_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

#MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
#MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
MESH_NAME = "3d_NACA0012_Re10000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

#SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/temporal_last_snapshot/"
#SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA5_avg_24280000-COMP-DATA.h5"
SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/"
SNAPSHOT_NAME = "3d_NACA0012_Re10000_AoA5_avg_1620000-COMP-DATA.h5"
SNAPSHOT_FILE = os.path.join(SNAPSHOT_PATH, SNAPSHOT_NAME)

# Reference parameters
u_infty = 1.0
AOA = 5  # degrees
alpha = np.deg2rad(AOA)
C = 1.0  # chord length

# x/c locations: dense sampling
x_c_locations_dense = np.arange(0.10, 1.01, 0.01)  # Every 0.01
# Original locations for special plotting
x_c_locations_original = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

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


def save_velocity_profile_data_dense(filepath, x_rot, y_rot, u_rot, v_rot,
                                       interface_indices_i, interface_indices_j,
                                       x_int_rot, y_int_rot, suction_mask, pressure_mask,
                                       wall_normal_profiles_dense, isoline_data,
                                       u_infty, alpha, C, x_c_locations_dense):
    """Save all computed data to HDF5 file."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, "w") as f:
        # Metadata
        f.attrs["u_infty"] = u_infty
        f.attrs["alpha"] = alpha
        f.attrs["C"] = C
        f.create_dataset("x_c_locations_dense", data=x_c_locations_dense)
        
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
        
        # Dense wall-normal profiles - save each profile as a group
        profiles_group = f.create_group("wall_normal_profiles_dense")
        for idx, profile in enumerate(wall_normal_profiles_dense):
            prof_group = profiles_group.create_group(f"profile_{idx}")
            prof_group.attrs["x_c"] = profile["x_c"]
            prof_group.attrs["x_start"] = profile["x_start"]
            prof_group.attrs["y_start"] = profile["y_start"]
            prof_group.attrs["y_end"] = profile["y_end"]
            prof_group.create_dataset("x_prime", data=profile["x_prime"])
            prof_group.create_dataset("y_prime", data=profile["y_prime"])
            prof_group.create_dataset("u_rot", data=profile["u_rot"])
            prof_group.create_dataset("v_rot", data=profile["v_rot"])
        
        # Zero velocity isoline data
        iso_group = f.create_group("isoline_data")
        iso_group.create_dataset("x_c", data=isoline_data["x_c"])
        iso_group.create_dataset("y_c", data=isoline_data["y_c"])
    
    print(f"Data saved to: {filepath}")


def load_velocity_profile_data_dense(filepath):
    """Load all computed dense data from HDF5 file."""
    
    if not os.path.exists(filepath):
        return None
    
    with h5py.File(filepath, "r") as f:
        # Metadata
        u_infty = f.attrs["u_infty"]
        alpha = f.attrs["alpha"]
        C = f.attrs["C"]
        x_c_locations_dense = f["x_c_locations_dense"][...]
        
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
        
        # Dense wall-normal profiles
        wall_normal_profiles_dense = []
        profiles_group = f["wall_normal_profiles_dense"]
        for idx in range(len(profiles_group)):
            prof_group = profiles_group[f"profile_{idx}"]
            profile = {
                "x_c": prof_group.attrs["x_c"],
                "x_start": prof_group.attrs["x_start"],
                "y_start": prof_group.attrs["y_start"],
                "y_end": prof_group.attrs["y_end"],
                "x_prime": prof_group["x_prime"][...],
                "y_prime": prof_group["y_prime"][...],
                "u_rot": prof_group["u_rot"][...],
                "v_rot": prof_group["v_rot"][...],
            }
            wall_normal_profiles_dense.append(profile)
        
        # Isoline data
        isoline_data = {
            "x_c": f["isoline_data"]["x_c"][...],
            "y_c": f["isoline_data"]["y_c"][...],
        }
    
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
        "wall_normal_profiles_dense": wall_normal_profiles_dense,
        "isoline_data": isoline_data,
        "u_infty": u_infty,
        "alpha": alpha,
        "C": C,
        "x_c_locations_dense": x_c_locations_dense,
    }


def load_velocity_rms_profiles(filepath):
    """Load RMS profile data if available."""

    if not os.path.exists(filepath):
        print(f"RMS data file not found, skipping RMS overlay: {filepath}")
        return None

    with h5py.File(filepath, "r") as f:
        rms_profiles = []
        profiles_group = f["rms_profiles"]
        for idx in range(len(profiles_group)):
            prof_group = profiles_group[f"profile_{idx}"]
            profile = {
                "x_c": prof_group.attrs["x_c"],
                "x_start": prof_group.attrs["x_start"],
                "y_start": prof_group.attrs["y_start"],
                "y_end": prof_group.attrs["y_end"],
                "x_prime": prof_group["x_prime"][...],
                "y_prime": prof_group["y_prime"][...],
                "u_rms": prof_group["u_rms"][...],
                "v_rms": prof_group["v_rms"][...],
                "w_rms": prof_group["w_rms"][...],
            }
            rms_profiles.append(profile)

        snapshot_count = f.attrs.get("snapshot_count", None)

    print(f"RMS data loaded from: {filepath}")

    return {
        "rms_profiles": rms_profiles,
        "snapshot_count": snapshot_count,
    }


# Try to load cached data first
print("Checking for cached data...")
cached_data = load_velocity_profile_data_dense(OUTPUT_DATA_FILE)

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
    wall_normal_profiles_dense = cached_data["wall_normal_profiles_dense"]
    isoline_data = cached_data["isoline_data"]
    u_infty = cached_data["u_infty"]
    alpha = cached_data["alpha"]
    C = cached_data["C"]
    x_c_locations_dense = cached_data["x_c_locations_dense"]
    
    # Compute derived variables
    x_rot_flat = x_rot.ravel()
    y_rot_flat = y_rot.ravel()
    x_suction_rot = x_int_rot[suction_mask]
    y_suction_rot = y_int_rot[suction_mask]
    x_pressure_rot = x_int_rot[pressure_mask]
    y_pressure_rot = y_int_rot[pressure_mask]

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
    
    # Coordinates
    x_data = loader.x[1, :, :]
    y_data = loader.y[1, :, :]
    
    print("x_data shape:", x_data.shape)
    print("y_data shape:", y_data.shape)
    
    # Mean velocities
    avg_u_data = loader.reconstruct_field(fields["avg_u"])
    avg_v_data = loader.reconstruct_field(fields["avg_v"])
    
    # Average in spanwise direction
    avg_u_data = np.mean(avg_u_data, axis=0)
    avg_v_data = np.mean(avg_v_data, axis=0)
    
    print("avg_u_data shape:", avg_u_data.shape)
    print("avg_v_data shape:", avg_v_data.shape)
    
    if x_data.shape != avg_u_data.shape:
        raise ValueError(
            "Shape mismatch: x_data and avg_u_data must both be (ny, nx). "
            f"Got x_data={x_data.shape}, avg_u_data={avg_u_data.shape}."
        )
    
    # Rotate into flow-aligned frame
    x_rot, y_rot = rotate_coordinates(x_data, y_data, alpha)
    u_rot, v_rot = rotate_coordinates(avg_u_data, avg_v_data, alpha)
    
    print("Rotation ranges:")
    print("  x_rot:", x_rot.min(), x_rot.max())
    print("  y_rot:", y_rot.min(), y_rot.max())
    
    # Interface points
    x_int = x_data[interface_indices_j, interface_indices_i]
    y_int = y_data[interface_indices_j, interface_indices_i]
    x_int_rot = x_rot[interface_indices_j, interface_indices_i]
    y_int_rot = y_rot[interface_indices_j, interface_indices_i]
    
    # Suction/pressure side selection
    suction_mask = y_int > 0
    pressure_mask = y_int < 0
    
    print("Suction-side interface points:", np.sum(suction_mask), "/", suction_mask.size)
    print("Pressure-side interface points:", np.sum(pressure_mask), "/", pressure_mask.size)
    
    # Extract suction-side interface points
    i_suction = interface_indices_i[suction_mask]
    j_suction = interface_indices_j[suction_mask]
    x_suction_rot = x_int_rot[suction_mask]
    y_suction_rot = y_int_rot[suction_mask]
    x_pressure_rot = x_int_rot[pressure_mask]
    y_pressure_rot = y_int_rot[pressure_mask]
    
    print("Total suction-side interface points:", x_suction_rot.size)
    
    # Parameters for wall-normal extraction
    wall_normal_length = 0.2
    n_points = 250
    
    # Build KDTree
    x_rot_flat = x_rot.ravel()
    y_rot_flat = y_rot.ravel()
    mesh_coords = np.column_stack((x_rot_flat, y_rot_flat))
    tree = cKDTree(mesh_coords)
    
    # =======================
    # DENSE PROFILE EXTRACTION
    # =======================
    print("\n" + "="*70)
    print("EXTRACTING DENSE VELOCITY PROFILES (every 0.01 in x/c)")
    print("="*70)
    
    wall_normal_profiles_dense = []
    
    for x_c in x_c_locations_dense:
        x_target = x_c * C
        
        # Find closest suction-side point
        dx = np.abs(x_suction_rot - x_target)
        k = int(np.argmin(dx))
        
        x_start = x_suction_rot[k]
        y_start = y_suction_rot[k]
        y_end = y_start + wall_normal_length
        
        # Generate query points along wall-normal
        y_query = np.linspace(y_start, y_end, n_points)
        x_query = np.full_like(y_query, x_start)
        query_points = np.column_stack((x_query, y_query))
        
        # Find nearest mesh points
        distances, indices = tree.query(query_points)
        ny, nx = x_rot.shape
        j_indices = indices // nx
        i_indices = indices % nx
        
        # Remove duplicates
        ij_pairs = np.column_stack((i_indices, j_indices))
        _, unique_idx = np.unique(ij_pairs, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        
        profile = {
            "x_c": float(x_c),
            "x_start": float(x_start),
            "y_start": float(y_start),
            "y_end": float(y_end),
            "x_prime": x_rot[j_indices[unique_idx], i_indices[unique_idx]],
            "y_prime": y_rot[j_indices[unique_idx], i_indices[unique_idx]],
            "u_rot": u_rot[j_indices[unique_idx], i_indices[unique_idx]],
            "v_rot": v_rot[j_indices[unique_idx], i_indices[unique_idx]],
        }
        
        wall_normal_profiles_dense.append(profile)
        
        if int(x_c * 100) % 10 == 0:
            print(f"x'/C = {x_c:.2f}: {len(unique_idx)} unique mesh points")
    
    print(f"\nTotal profiles extracted: {len(wall_normal_profiles_dense)}")
    
    # =======================
    # FIND ZERO VELOCITY ISOLINE
    # =======================
    print("\n" + "="*70)
    print("COMPUTING ZERO VELOCITY ISOLINE (u' = 0)")
    print("="*70)
    
    isoline_x_c = []
    isoline_y_c = []
    
    for profile in wall_normal_profiles_dense:
        # Sort by y' coordinate
        sort_idx = np.argsort(profile["y_prime"])
        y_sorted = profile["y_prime"][sort_idx]
        u_sorted = profile["u_rot"][sort_idx]
        
        # Find where u crosses zero
        sign_changes = np.diff(np.sign(u_sorted))
        zero_crossings = np.where(sign_changes != 0)[0]
        
        if len(zero_crossings) > 0:
            # Use first crossing (separation point)
            idx_cross = zero_crossings[0]
            
            # Interpolate to find exact u=0 location
            y1, y2 = y_sorted[idx_cross], y_sorted[idx_cross + 1]
            u1, u2 = u_sorted[idx_cross], u_sorted[idx_cross + 1]
            
            # Linear interpolation
            y_zero = y1 + (0 - u1) * (y2 - y1) / (u2 - u1)
            
            isoline_x_c.append(profile["x_c"])
            isoline_y_c.append(y_zero)
    
    isoline_x_c = np.array(isoline_x_c)
    isoline_y_c = np.array(isoline_y_c)
    
    print(f"Zero crossings found: {len(isoline_x_c)}")
    if len(isoline_x_c) > 0:
        print(f"Separation point (first u=0): x'/C = {isoline_x_c[0]:.3f}")
        if len(isoline_x_c) > 1:
            print(f"Reattachment point (last u=0): x'/C = {isoline_x_c[-1]:.3f}")
            bubble_length = isoline_x_c[-1] - isoline_x_c[0]
            print(f"Separation bubble length: Δ(x'/C) = {bubble_length:.3f}")
    
    isoline_data = {
        "x_c": isoline_x_c,
        "y_c": isoline_y_c,
    }
    
    # Save all computed data
    print("\nSaving computed dense profile data for future use...")
    save_velocity_profile_data_dense(
        OUTPUT_DATA_FILE,
        x_rot, y_rot, u_rot, v_rot,
        interface_indices_i, interface_indices_j,
        x_int_rot, y_int_rot, suction_mask, pressure_mask,
        wall_normal_profiles_dense, isoline_data,
        u_infty, alpha, C, x_c_locations_dense
    )

# Load RMS data (optional overlay)
rms_data = load_velocity_rms_profiles(RMS_DATA_FILE)
rms_by_xc = {}
if rms_data is not None:
    rms_by_xc = {round(p["x_c"], 6): p for p in rms_data["rms_profiles"]}


# =======================
# VISUALIZATION PLOTS
# =======================
print("\n" + "="*70)
print("GENERATING VISUALIZATION PLOTS")
print("="*70)

# # Plot 1: Velocity profiles at original x/c locations (similar to Jardin 2025)
# fig1, ax1 = plt.subplots(figsize=(12, 8))

# # Find profiles closest to original locations
# colors = plt.cm.viridis(np.linspace(0, 1, len(x_c_locations_original)))

# for x_c_target, color in zip(x_c_locations_original, colors):
#     # Find closest profile
#     diffs = np.abs(np.array([p["x_c"] for p in wall_normal_profiles_dense]) - x_c_target)
#     idx = np.argmin(diffs)
#     profile = wall_normal_profiles_dense[idx]
    
#     # Wall-normal distance from surface
#     eta = profile["y_prime"] - profile["y_start"]
    
#     # Plot normalized velocity
#     ax1.plot(
#         profile["u_rot"] / u_infty,
#         eta,
#         'o-',
#         color=color,
#         markersize=5,
#         linewidth=2,
#         label=f"x'/C = {profile['x_c']:.2f}"
#     )

# ax1.set_xlabel("u'/U∞", fontsize=12)
# ax1.set_ylabel("η (wall-normal distance)", fontsize=12)
# ax1.set_title("Mean Velocity Profiles (Jardin 2025 style)", fontsize=14)
# ax1.legend(fontsize=10, loc='best')
# ax1.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DATA_PATH, "velocity_profiles_original_locations.png"), dpi=150)
# print("Saved: velocity_profiles_original_locations.png")
# plt.show()

# Plot 2: Separation bubble visualization on airfoil with velocity profiles
fig2, ax2 = plt.subplots(figsize=(14, 8))

# Plot airfoil surface as continuous lines (same color for suction/pressure)
suction_order = np.argsort(x_suction_rot)
pressure_order = np.argsort(x_pressure_rot)
ax2.plot(
    x_suction_rot[suction_order],
    y_suction_rot[suction_order],
    '-',
    color="black",
    linewidth=0.8,
    alpha=0.8,
    label="Airfoil surface",
)
ax2.plot(
    x_pressure_rot[pressure_order],
    y_pressure_rot[pressure_order],
    '-',
    color="black",
    linewidth=0.8,
    alpha=0.8,
)

# Plot zero velocity isoline
if len(isoline_data["x_c"]) > 0:
    ax2.plot(
        isoline_data["x_c"],
        isoline_data["y_c"],
        'r-',
        linewidth=1.5,
        label='u\' = 0 (Separation bubble)'
    )
    
    # Mark separation and reattachment points
    ax2.plot(isoline_data["x_c"][0], isoline_data["y_c"][0], 'ro', markersize=5, label='Separation')
    if len(isoline_data["x_c"]) > 1:
        ax2.plot(isoline_data["x_c"][-1], isoline_data["y_c"][-1], 'rs', markersize=5, label='Reattachment')

# Add velocity profiles at original locations (all black)
scale_factor = 0.05  # Scale for profile visualization
rms_profile_label_added = False

for x_c_target in x_c_locations_original:
    # Find closest profile
    diffs = np.abs(np.array([p["x_c"] for p in wall_normal_profiles_dense]) - x_c_target)
    idx = np.argmin(diffs)
    profile = wall_normal_profiles_dense[idx]

    # Wall-normal distance from surface
    eta = profile["y_prime"] - profile["y_start"]

    # Sort by wall-normal distance to keep plots monotonic
    sort_idx = np.argsort(eta)
    eta_sorted = eta[sort_idx]
    u_norm_sorted = (profile["u_rot"] / u_infty)[sort_idx]

    # Vertical dashed guide at each x/c location (from surface to profile top)
    ax2.plot(
        [profile["x_start"], profile["x_start"]],
        [profile["y_start"], profile["y_start"] + eta.max()],
        '--',
        color="black",
        alpha=0.6,
        linewidth=1.0,
    )
    
    # Normalized velocity
    u_normalized = profile["u_rot"] / u_infty
    
    # Position the velocity profile at the reference point
    # x position: x_start + scaled velocity (horizontal displacement)
    # y position: y_start + eta (wall-normal distance)
    x_profile = profile["x_start"] + u_norm_sorted * scale_factor
    y_profile = profile["y_start"] + eta_sorted
    
    # Plot the velocity profile
    ax2.plot(
        x_profile,
        y_profile,
        '-',
        color="black",
        linewidth=1.5,
        alpha=0.7
    )
    
    # Mark the reference point on the surface
    ax2.plot(profile["x_start"], profile["y_start"], 'o', color="black", markersize=4)

    # Overlay RMS profile (dash-dotted) if available for this profile
    rms_profile = rms_by_xc.get(round(profile["x_c"], 6), None)
    if rms_profile is not None and len(rms_profile["u_rms"]) == len(profile["u_rot"]):
        u_rms_sorted = (rms_profile["u_rms"] / u_infty)[sort_idx]

        # Plot u_rms profile as dash-dotted line
        x_profile_rms = profile["x_start"] + u_rms_sorted * scale_factor
        ax2.plot(
            x_profile_rms,
            y_profile,
            '-.',
            color="tab:blue",
            linewidth=1.1,
            alpha=0.9,
            label=("u_rms" if not rms_profile_label_added else None),
        )
        rms_profile_label_added = True

ax2.set_xlabel("x'", fontsize=12)
ax2.set_ylabel("y'", fontsize=12)
ax2.set_title("Separation Bubble with Velocity Profiles", fontsize=14)
ax2.set_xlim(-0.05, 1.1)
ax2.set_ylim(-0.25, 0.25)
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DATA_PATH, "separation_bubble_with_profiles.png"), dpi=150)
print("Saved: separation_bubble_with_profiles.png")
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nData location: {OUTPUT_DATA_FILE}")
print(f"Plots saved in: {OUTPUT_DATA_PATH}")

