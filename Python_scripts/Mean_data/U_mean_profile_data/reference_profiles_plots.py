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
    """Load reference profiles from .dat files."""
    
    ref_profiles = {}
    
    # Find all reference data files
    dat_files = sorted(glob.glob(os.path.join(ref_data_path, "Re5e4_AOA5_U_mean_*_Jardin_2025.dat")))
    
    for dat_file in dat_files:
        # Extract x/c location from filename
        basename = os.path.basename(dat_file)
        x_c_str = basename.split("_")[4]  # e.g., "015" from "Re5e4_AOA5_U_mean_015_Jardin_2025.dat"
        x_c = float(x_c_str) / 100  # Convert 015 -> 0.15
        
        # Load data from file (space or tab separated, with comma as decimal separator)
        with open(dat_file, 'r') as f:
            lines = f.readlines()
        
        eta_list = []
        u_norm_list = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Replace comma with dot for European decimal format
                u_norm = float(parts[0].replace(',', '.'))
                eta = float(parts[1].replace(',', '.'))
                u_norm_list.append(u_norm)
                eta_list.append(eta)
        
        ref_profiles[x_c] = {
            "eta": np.array(eta_list),
            "u_norm": np.array(u_norm_list),
            "filename": dat_file
        }
        
        print(f"Loaded reference profile at x/c={x_c:.2f}: {len(eta_list)} points")
    
    return ref_profiles


# Load data
print("Loading computed velocity profile data...")
data = load_velocity_profile_data(OUTPUT_DATA_FILE)

print("\nLoading reference velocity profile data...")
ref_profiles = load_reference_profiles(REF_DATA_PATH)

# Extract variables
wall_normal_profiles_unique = data["wall_normal_profiles_unique"]
u_infty = data["u_infty"]

print(f"\nNumber of computed profiles: {len(wall_normal_profiles_unique)}")
print(f"Number of reference profiles: {len(ref_profiles)}")
print("\nGenerating comparison plots...\n")


# ============================================================================
# Create comparison plots for each x/c location
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

colors_comp = plt.cm.Set1(np.linspace(0, 1, 2))

for idx, profile in enumerate(wall_normal_profiles_unique):
    ax = axes[idx]
    x_c = profile["x_c"]
    
    # Computed profile
    eta = profile["y_prime"] - profile["y_start"]
    u_norm = profile["u_rot"] / u_infty
    
    ax.plot(
        u_norm, 
        eta, 
        'o-', 
        color=colors_comp[0], 
        markersize=4,
        linewidth=1.5,
        label="Computed (CFD)"
    )
    
    # Reference profile if available
    if x_c in ref_profiles:
        ref = ref_profiles[x_c]
        ax.plot(
            ref["u_norm"],
            ref["eta"],
            's-',
            color=colors_comp[1],
            markersize=4,
            linewidth=1.5,
            label="Jardin et al. 2025"
        )
    
    ax.set_xlabel("u'/U∞")
    ax.set_ylabel("η (wall-normal distance)")
    ax.set_title(f"x'/C = {x_c:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Remove extra subplots
for idx in range(len(wall_normal_profiles_unique), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()


# ============================================================================
# Create a comprehensive overlay plot
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

colors = plt.cm.viridis(np.linspace(0, 1, len(wall_normal_profiles_unique)))

for profile, color in zip(wall_normal_profiles_unique, colors):
    x_c = profile["x_c"]
    
    # Computed profile
    eta = profile["y_prime"] - profile["y_start"]
    u_norm = profile["u_rot"] / u_infty
    
    ax.plot(
        u_norm, 
        eta, 
        'o-', 
        color=color, 
        markersize=3,
        linewidth=1.5,
        label=f"x'/C={x_c:.2f} (CFD)"
    )
    
    # Reference profile if available
    if x_c in ref_profiles:
        ref = ref_profiles[x_c]
        ax.plot(
            ref["u_norm"],
            ref["eta"],
            's--',
            color=color,
            markersize=3,
            linewidth=1,
            alpha=0.7,
            label=f"x'/C={x_c:.2f} (Ref)"
        )

ax.set_xlabel("u'/U∞", fontsize=12)
ax.set_ylabel("η (wall-normal distance)", fontsize=12)
ax.set_title("Velocity profiles comparison: CFD vs. Reference (Jardin et al. 2025)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

print("\nAll comparison plots generated successfully!")
