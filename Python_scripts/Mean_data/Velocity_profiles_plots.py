import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob


# Path to saved data
OUTPUT_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
OUTPUT_DATA_NAME = "AoA5_Re50000_velocity_profiles_data.h5"
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DATA_PATH, OUTPUT_DATA_NAME)

# Reference data path
REF_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Mean_data/U_mean_profile_data/"


def load_velocity_profile_data(filepath):
    """Load all computed data from HDF5 file."""
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
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


def load_reference_profiles(ref_data_path):
    """Load reference profiles from .dat files (extracted plot coordinates).

    Assumptions:
    - Files named like Re5e4_AOA5_U_mean_0XX_Jardin_2025.dat where 0XX encodes x/c.
    - European decimal commas replaced by dots.
    - Columns represent [x/c, y/c] as extracted directly from the plot axes.
    """

    ref_profiles = {}

    dat_files = sorted(glob.glob(os.path.join(ref_data_path, "Re5e4_AOA5_U_mean_*_Jardin_2025.dat")))
    for dat_file in dat_files:
        base = os.path.basename(dat_file)
        # Extract the 0XX token at position 4 (Re5e4_AOA5_U_mean_0XX_Jardin_2025.dat)
        try:
            x_c_token = base.split("_")[4]
            x_c = float(x_c_token) / 100  # Convert 015 -> 0.15
        except Exception:
            # Skip files that don't follow the naming convention
            continue

        x_vals = []
        y_vals = []
        with open(dat_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                # Reference files store plot axes: first col (x/c), second col (y/c)
                x_val = float(parts[0].replace(",", "."))
                y_val = float(parts[1].replace(",", "."))
                x_vals.append(x_val)
                y_vals.append(y_val)

        if x_vals:
            ref_profiles[x_c] = {
                "x": np.array(x_vals),  # x/c from plot
                "y": np.array(y_vals),  # y/c from plot
                "filename": dat_file,
            }

    return ref_profiles


# Load data
print("Loading velocity profile data...")
data = load_velocity_profile_data(OUTPUT_DATA_FILE)

# Load reference data from plot-extracted files
print("Loading reference velocity profile data...")
ref_profiles = load_reference_profiles(REF_DATA_PATH)
print(f"Reference profiles loaded: {len(ref_profiles)}")

# Extract variables
x_rot = data["x_rot"]
y_rot = data["y_rot"]
u_rot = data["u_rot"]
v_rot = data["v_rot"]
x_int_rot = data["x_int_rot"]
y_int_rot = data["y_int_rot"]
suction_mask = data["suction_mask"]
pressure_mask = data["pressure_mask"]
wall_normal_profiles_unique = data["wall_normal_profiles_unique"]
u_infty = data["u_infty"]

# Compute derived variables
x_suction_rot = x_int_rot[suction_mask]
y_suction_rot = y_int_rot[suction_mask]
x_pressure_rot = x_int_rot[pressure_mask]
y_pressure_rot = y_int_rot[pressure_mask]
x_rot_flat = x_rot.ravel()
y_rot_flat = y_rot.ravel()

print(f"Number of profiles: {len(wall_normal_profiles_unique)}")
print("Generating plots...\n")


# ============================================================================
# Plot 1: Extraction lines on the mesh
# ============================================================================
# fig1, ax1 = plt.subplots(figsize=(10, 8))

# ax1.scatter(x_rot_flat, y_rot_flat, s=0.1, c="gray", alpha=0.3, label="Mesh points")
# ax1.scatter(x_suction_rot, y_suction_rot, s=1, c="blue", label="Suction surface")

colors = plt.cm.viridis(np.linspace(0, 1, len(wall_normal_profiles_unique)))
# for profile, color in zip(wall_normal_profiles_unique, colors):
#     ax1.scatter(
#         profile["x_prime"], 
#         profile["y_prime"], 
#         s=10, 
#         c=[color], 
#         label=f"x'/C = {profile['x_c']:.2f}"
#     )
#     # Draw line from surface to y_end
#     ax1.plot(
#         [profile["x_start"], profile["x_start"]], 
#         [profile["y_start"], profile["y_end"]],
#         '--', 
#         color=color, 
#         alpha=0.5
#     )

# ax1.set_xlabel("x'")
# ax1.set_ylabel("y'")
# ax1.set_title("Wall-normal extraction lines")
# ax1.set_xlim(-0.1, 1.2)
# ax1.set_ylim(-0.3, 0.3)
# ax1.legend(loc="upper left", fontsize=8)
# ax1.grid(True)
# ax1.set_aspect("equal")
# plt.tight_layout()
# plt.show()


# ============================================================================
# Plot 2: Velocity profiles (traditional plot)
# ============================================================================
# fig2, ax2 = plt.subplots(figsize=(8, 6))

# for profile, color in zip(wall_normal_profiles_unique, colors):
#     # Wall-normal distance from surface
#     eta = profile["y_prime"] - profile["y_start"]
#     ax2.plot(
#         profile["u_rot"] / u_infty, 
#         eta, 
#         'o-', 
#         color=color, 
#         markersize=3,
#         label=f"x'/C = {profile['x_c']:.2f}"
#     )

# ax2.set_xlabel("u'/U∞")
# ax2.set_ylabel("η (wall-normal distance)")
# ax2.set_title("Velocity profiles")
# ax2.legend()
# ax2.grid(True)
# plt.tight_layout()
# plt.show()


# ============================================================================
# Plot 3: Velocity profiles on the airfoil
# ============================================================================
# fig3, ax3 = plt.subplots(figsize=(12, 8))

# # Plot the airfoil surface
# ax3.scatter(x_suction_rot, y_suction_rot, s=1, c="blue", label="Suction surface")
# ax3.scatter(x_pressure_rot, y_pressure_rot, s=1, c="green", label="Pressure surface")

# # Scale factor for velocity profiles (adjust to make profiles visible)
# scale_factor = 0.05

# colors = plt.cm.viridis(np.linspace(0, 1, len(wall_normal_profiles_unique)))

# for profile, color in zip(wall_normal_profiles_unique, colors):
#     # Wall-normal distance from surface
#     eta = profile["y_prime"] - profile["y_start"]
    
#     # Normalized velocity
#     u_normalized = profile["u_rot"] / u_infty
    
#     # Position the velocity profile at the reference point
#     # x position: x_start + scaled velocity (horizontal displacement)
#     # y position: y_start + eta (wall-normal distance)
#     x_profile = profile["x_start"] + u_normalized * scale_factor
#     y_profile = profile["y_start"] + eta
    
#     # Plot the velocity profile
#     ax3.plot(
#         x_profile, 
#         y_profile, 
#         '-', 
#         color=color, 
#         linewidth=1.5,
#         label=f"x'/C = {profile['x_c']:.2f}"
#     )
    
#     # Draw the baseline (zero velocity reference line)
#     ax3.plot(
#         [profile["x_start"], profile["x_start"]], 
#         [profile["y_start"], profile["y_start"] + eta.max()],
#         '--', 
#         color=color, 
#         alpha=0.5,
#         linewidth=0.8
#     )
    
#     # Mark the reference point on the surface
#     ax3.plot(profile["x_start"], profile["y_start"], 'o', color=color, markersize=6)

# ax3.set_xlabel("x'")
# ax3.set_ylabel("y'")
# ax3.set_title("Velocity profiles on airfoil (suction side)")
# ax3.set_xlim(-0.1, 1.2)
# ax3.set_ylim(-0.25, 0.2)
# ax3.legend(loc="upper right", fontsize=8)
# ax3.grid(True, alpha=0.3)
# ax3.set_aspect("equal")
# plt.tight_layout()
# plt.show()


# ============================================================================
# Plot 4: Velocity profiles on the airfoil WITH reference overlay
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(12, 8))

# Plot the airfoil surface
ax4.scatter(x_suction_rot, y_suction_rot, s=1, c="blue", label="Suction surface")
ax4.scatter(x_pressure_rot, y_pressure_rot, s=1, c="green", label="Pressure surface")

# Scale factor for velocity profiles (same as Plot 3)
scale_factor = 0.05

colors = plt.cm.viridis(np.linspace(0, 1, len(wall_normal_profiles_unique)))
ref_label_added = False

for profile, color in zip(wall_normal_profiles_unique, colors):
    # Wall-normal distance from surface
    eta = profile["y_prime"] - profile["y_start"]
    
    # Normalized velocity
    u_normalized = profile["u_rot"] / u_infty
    
    # Position the velocity profile at the reference point
    x_profile = profile["x_start"] + u_normalized * scale_factor
    y_profile = profile["y_start"] + eta
    
    # Plot the computed velocity profile
    ax4.plot(
        x_profile, 
        y_profile, 
        '-', 
        color=color, 
        linewidth=1.5,
        label=f"x'/C = {profile['x_c']:.2f}"
    )
    
    # Draw the baseline (zero velocity reference line)
    ax4.plot(
        [profile["x_start"], profile["x_start"]], 
        [profile["y_start"], profile["y_start"] + eta.max()],
        '--', 
        color=color, 
        alpha=0.5,
        linewidth=0.8
    )
    
    # Mark the reference point on the surface
    ax4.plot(profile["x_start"], profile["y_start"], 'o', color=color, markersize=6)
    
    # Overlay reference profile if available
    x_c = profile["x_c"]
    if x_c in ref_profiles:
        ref = ref_profiles[x_c]
        # Plot reference coordinates directly (already in x/c, y/c from the plot)
        # Apply y-offset of 0.02 to align with CFD data
        ax4.plot(
            ref["x"] - 0.003,
            ref["y"] - 0.021,
            'ko-',
            linewidth=1.5,
            markersize=3,
            alpha=0.8,
            label=("Reference (Jardin et al. 2025)" if not ref_label_added else None)
        )
        ref_label_added = True

ax4.set_xlabel("x'")
ax4.set_ylabel("y'")
ax4.set_title("Velocity profiles on airfoil with reference overlay")
ax4.set_xlim(-0.1, 1.2)
ax4.set_ylim(-0.25, 0.2)
ax4.legend(loc="upper right", fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_aspect("equal")
plt.tight_layout()
plt.show()


print("\nAll plots generated successfully!")
