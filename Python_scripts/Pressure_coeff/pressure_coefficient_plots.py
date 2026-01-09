import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import uniform_filter1d

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
    proj_points = f["proj_points"][:]
print("Loaded geometrical data.")

# LOAD Cp DATA (AoA = 12)
Cp_12_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
Cp_12_NAME = "3d_NACA0012_Re50000_AoA12_Cp_19680000.h5"
Cp_12_FILE = os.path.join(Cp_12_PATH, Cp_12_NAME)

# Check if the Cp data file exists
if os.path.exists(Cp_12_FILE):
    print(f"Cp data file exists! {Cp_12_FILE}")
else:
    print(f"Cp data file does not exist: {Cp_12_FILE}")

# Load Cp data
with h5py.File(Cp_12_FILE, "r") as f:
    Cp_12_values = f["Cp_values"][:]
print("Loaded Cp (AoA 12) data.")

# LOAD Cp DATA (AoA = 5)
Cp_5_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
Cp_5_NAME = "3d_NACA0012_Re50000_AoA5_Cp_24280000.h5"
Cp_5_FILE = os.path.join(Cp_5_PATH, Cp_5_NAME)

# Check if the Cp data file exists
if os.path.exists(Cp_5_FILE):
    print(f"Cp data file exists! {Cp_5_FILE}")
else:
    print(f"Cp data file does not exist: {Cp_5_FILE}")

# Load Cp data
with h5py.File(Cp_5_FILE, "r") as f:
    Cp_5_values = f["Cp_values"][:]
print("Loaded Cp (AoA 5) data.")


def plot_pressure_coefficient(proj_points, Cp_values, c=1.0):
    """
    Plot the pressure coefficient (Cp) along the airfoil surface,
    separating the upper and lower surfaces.

    Parameters:
        proj_points (np.ndarray): Projected surface points (N x 3).
        Cp_values (np.ndarray): Pressure coefficient at each point (N,).
        c (float): Chord length for normalization (default = 1.0).
    """

    # Extract x and y coordinates of projection points
    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]

    # Normalize x by chord
    x_over_c = x_proj / c

    # Separate upper and lower surfaces using y > 0 as criterion
    upper_mask = y_proj > 0
    lower_mask = ~upper_mask

    x_upper = x_over_c[upper_mask]
    Cp_upper = Cp_values[upper_mask]

    x_lower = x_over_c[lower_mask]
    Cp_lower = Cp_values[lower_mask]

    # Sort by x for clean plotting
    upper_sort = np.argsort(x_upper)
    lower_sort = np.argsort(x_lower)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_upper[upper_sort], Cp_upper[upper_sort], label="Upper surface", marker="o"
    )
    plt.plot(
        x_lower[lower_sort], Cp_lower[lower_sort], label="Lower surface", marker="s"
    )
    plt.gca().invert_yaxis()  # Cp convention: lower values upwards
    plt.xlabel("x/c")
    plt.ylabel("$C_p$")
    plt.title("Pressure Coefficient Distribution on Airfoil Surface")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the Cp distribution
# plot_pressure_coefficient(proj_points, Cp_12_values, c=1.0)
# plot_pressure_coefficient(proj_points, Cp_5_values, c=1.0)


# Plot reference data (AoA = 12)
def plot_cp_ref_data():
    """
    Plot reference Cp data for AoA = 12 degrees from Rodríguez et al.

    """
    base_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Pressure_coeff/Cp_data/"

    comma_to_dot = lambda s: float(s.replace(",", "."))
    
    # Load reference data AoA = 12
    upper_aoa12 = np.genfromtxt(
        os.path.join(base_path, "Rodriguez_2013_Re5e4_aoa12_cp_upper.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    lower_aoa12 = np.genfromtxt(
        os.path.join(base_path, "Rodriguez_2013_Re5e4_aoa12_cp_lower.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    # Load reference data AoA = 5
    upper_aoa5 = np.genfromtxt(
        os.path.join(base_path, "lehmkuhl_2013_Re5e4_aoa5_cp_upper.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    lower_aoa5 = np.genfromtxt(
        os.path.join(base_path, "lehmkuhl_2013_Re5e4_aoa5_cp_lower.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )

    plt.figure(figsize=(10, 5))
    plt.plot(upper_aoa12[:, 0], upper_aoa12[:, 1], "o-", label="Upper surface (AoA=12°)")
    plt.plot(lower_aoa12[:, 0], lower_aoa12[:, 1], "o-", label="Lower surface (AoA=12°)")
    plt.plot(upper_aoa5[:, 0], upper_aoa5[:, 1], "o-", label="Upper surface (AoA=5°)")
    plt.plot(lower_aoa5[:, 0], lower_aoa5[:, 1], "o-", label="Lower surface (AoA=5°)")
    
    plt.gca().invert_yaxis()
    plt.xlabel("x/c")
    plt.ylabel("$C_p$")
    plt.title("Reference Cp Data – Rodríguez et al. & Lehmkuhl et al.")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot reference data
# plot_cp_ref_data()

### Plot Cp comparison

# Set LaTeX style for plots
plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 16, family='serif' )
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb}')

def plot_cp_comparison(proj_points, Cp_5_values, Cp_12_values, c=1.0):
    """
    Plot simulated Cp data for AoA = 5 and AoA = 12 degrees alongside reference data.

    Parameters:
        proj_points (np.ndarray): Projected surface points (N x 3).
        Cp_5_values (np.ndarray): Pressure coefficient at each point for AoA = 5 (N,).
        Cp_12_values (np.ndarray): Pressure coefficient at each point for AoA = 12 (N,).
        c (float): Chord length for normalization (default = 1.0).
    """

    # === Load reference data ===
    base_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Pressure_coeff/Cp_data/"

    comma_to_dot = lambda s: float(s.replace(",", "."))

    upper_aoa12 = np.genfromtxt(
        os.path.join(base_path, "Rodriguez_2013_Re5e4_aoa12_cp_upper.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    lower_aoa12 = np.genfromtxt(
        os.path.join(base_path, "Rodriguez_2013_Re5e4_aoa12_cp_lower.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    upper_aoa5 = np.genfromtxt(
        os.path.join(base_path, "lehmkuhl_2013_Re5e4_aoa5_cp_upper.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    lower_aoa5 = np.genfromtxt(
        os.path.join(base_path, "lehmkuhl_2013_Re5e4_aoa5_cp_lower.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )

    # === Process simulation data for AoA = 5 ===
    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]
    x_over_c = x_proj / c

    upper_mask = y_proj > 0
    lower_mask = ~upper_mask

    x_upper_5 = x_over_c[upper_mask]
    Cp_upper_5 = Cp_5_values[upper_mask]
    x_lower_5 = x_over_c[lower_mask]
    Cp_lower_5 = Cp_5_values[lower_mask]
    upper_sort_5 = np.argsort(x_upper_5)
    lower_sort_5 = np.argsort(x_lower_5)
    
    # Apply uniform filter smoothing (AoA = 5)
    Cp_upper_5_smooth = uniform_filter1d(Cp_upper_5[upper_sort_5], size=5, mode='nearest')
    Cp_lower_5_smooth = uniform_filter1d(Cp_lower_5[lower_sort_5], size=5, mode='nearest')
    
    # === Process simulation data for AoA = 12 ===
    x_upper_12 = x_over_c[upper_mask]
    Cp_upper_12 = Cp_12_values[upper_mask]
    x_lower_12 = x_over_c[lower_mask]
    Cp_lower_12 = Cp_12_values[lower_mask]
    upper_sort_12 = np.argsort(x_upper_12)
    lower_sort_12 = np.argsort(x_lower_12)
    
    # Apply uniform filter smoothing (AoA = 12)
    Cp_upper_12_smooth = uniform_filter1d(Cp_upper_12[upper_sort_12], size=5, mode='nearest')
    Cp_lower_12_smooth = uniform_filter1d(Cp_lower_12[lower_sort_12], size=5, mode='nearest')

    # === Plotting ===
    plt.figure(figsize=(10, 5))
    # Plot simulation curves for AoA = 5
    plt.plot(
        x_upper_5[upper_sort_5], Cp_upper_5_smooth, "-", color="black", label="Sim. (AoA=5°)"
    )
    plt.plot(
        x_lower_5[lower_sort_5], Cp_lower_5_smooth, "-", color="black"
    )
    #plt.plot(
    #    x_upper_5[upper_sort_5], Cp_upper_5, "-", color="black", label="Sim. (AoA=5°)"
    #)
    #plt.plot(
    #    x_lower_5[lower_sort_5], Cp_lower_5, "-", color="black"
    #)

    # Plot reference curves for AoA = 5
    plt.plot(
        upper_aoa5[:, 0], upper_aoa5[:, 1], "o", color="black", markersize=4, label="Ref. (AoA=5°)"
    )
    plt.plot(
        lower_aoa5[:, 0], lower_aoa5[:, 1], "o", color="black", markersize=4
    )
    # Plot simulation curves for AoA = 12
    plt.plot(
        x_upper_12[upper_sort_12], Cp_upper_12_smooth, "--", color="black", label="Sim. (AoA=12°)"
    )
    plt.plot(
        x_lower_12[lower_sort_12], Cp_lower_12_smooth, "--", color="black"
    )
    # Plot reference curves for AoA = 12
    plt.plot(
        upper_aoa12[:, 0], upper_aoa12[:, 1], "s", color="black", markersize=4, label="Ref. (AoA=12°)"
    )
    plt.plot(
        lower_aoa12[:, 0], lower_aoa12[:, 1], "s", color="black", markersize=4
    )
    plt.gca().invert_yaxis()
    plt.xlabel("x/c")
    plt.ylabel("$c_P$")
    #plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot comparison
plot_cp_comparison(proj_points, Cp_5_values, Cp_12_values, c=1.0)

Cp_12_values = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/cp_data_aoa12_Re50000_8700000.npz"


def plot_cp_comparison_2(proj_points, Cp_5_values, Cp_12_values, c=1.0):
    """
    Plot simulated Cp data for AoA = 5 and AoA = 12 degrees alongside reference data.

    Parameters:
        proj_points (np.ndarray): Projected surface points (N x 3).
        Cp_5_values (np.ndarray): Pressure coefficient at each point for AoA = 5 (N,).
        Cp_12_values (np.ndarray): Pressure coefficient at each point for AoA = 12 (N,).
        c (float): Chord length for normalization (default = 1.0).
    """

    # === Load reference data ===
    base_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Pressure_coeff/Cp_data/"

    comma_to_dot = lambda s: float(s.replace(",", "."))

    upper_aoa12 = np.genfromtxt(
        os.path.join(base_path, "Rodriguez_2013_Re5e4_aoa12_cp_upper.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    lower_aoa12 = np.genfromtxt(
        os.path.join(base_path, "Rodriguez_2013_Re5e4_aoa12_cp_lower.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    upper_aoa5 = np.genfromtxt(
        os.path.join(base_path, "lehmkuhl_2013_Re5e4_aoa5_cp_upper.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    lower_aoa5 = np.genfromtxt(
        os.path.join(base_path, "lehmkuhl_2013_Re5e4_aoa5_cp_lower.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )

    # === Process simulation data for AoA = 5 ===
    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]
    x_over_c = x_proj / c

    upper_mask = y_proj > 0
    lower_mask = ~upper_mask

    x_upper_5 = x_over_c[upper_mask]
    Cp_upper_5 = Cp_5_values[upper_mask]
    x_lower_5 = x_over_c[lower_mask]
    Cp_lower_5 = Cp_5_values[lower_mask]
    upper_sort_5 = np.argsort(x_upper_5)
    lower_sort_5 = np.argsort(x_lower_5)
    
    # Apply uniform filter smoothing (AoA = 5)
    Cp_upper_5_smooth = uniform_filter1d(Cp_upper_5[upper_sort_5], size=5, mode='nearest')
    Cp_lower_5_smooth = uniform_filter1d(Cp_lower_5[lower_sort_5], size=5, mode='nearest')

    # === Process simulation data for AoA = 12 ===
    data_12 = np.load(Cp_12_values)
    proj_12_points = data_12["proj_points"]
    Cp_12_values = data_12["cp_values"]

    x_12_proj = proj_12_points[:, 0]
    y_12_proj = proj_12_points[:, 1]
    x_over_c_12 = x_12_proj / c

    upper_mask_12 = y_12_proj > 0
    lower_mask_12 = ~upper_mask_12

    x_upper_12 = x_over_c_12[upper_mask_12]
    Cp_12_upper = Cp_12_values[upper_mask_12]
    x_lower_12 = x_over_c_12[lower_mask_12]
    Cp_12_lower = Cp_12_values[lower_mask_12]

    upper_sort_12 = np.argsort(x_upper_12)
    lower_sort_12 = np.argsort(x_lower_12)

    # Apply uniform filter smoothing (AoA = 12)
    Cp_upper_12_smooth = uniform_filter1d(Cp_12_upper[upper_sort_12], size=5, mode='nearest')
    Cp_lower_12_smooth = uniform_filter1d(Cp_12_lower[lower_sort_12], size=5, mode='nearest')


    # === Plotting ===
    plt.figure(figsize=(6, 4))
    # Plot simulation curves for AoA = 5
    # plt.plot(
    #      x_upper_5[upper_sort_5], Cp_upper_5[upper_sort_5], "-", color="black", label="Sim. (AoA=5°)"
    # )
    # plt.plot(
    #     x_lower_5[lower_sort_5], Cp_lower_5[lower_sort_5], "-", color="black"
    # )
    plt.plot(
        x_upper_5[upper_sort_5], Cp_upper_5_smooth, "-", color="black", linewidth=1, label="Sim. (AoA=5°)"
    )
    plt.plot(
        x_lower_5[lower_sort_5], Cp_lower_5_smooth, "-", color="black", linewidth=1
    )

    # Plot reference curves for AoA = 5
    # plt.plot(
    #     upper_aoa5[::2, 0], upper_aoa5[::2, 1], "o", color="black", markersize=4, linewidth=1, label="Ref. (AoA=5°)"
    # )
    # plt.plot(
    #     lower_aoa5[::2, 0], lower_aoa5[::2, 1], "o", color="black", markersize=4, linewidth=1
    # )
    indices_5 = np.concatenate([np.arange(4), np.arange(4, len(upper_aoa5), 3)])
    indices_5 = indices_5[indices_5 < len(upper_aoa5)]  # Filter out-of-bounds indices
    plt.plot(
        upper_aoa5[indices_5, 0], upper_aoa5[indices_5, 1], "o", color="black", markersize=4, linewidth=1, label="Ref. (AoA=5°)"
    )

    indices_5_lower = np.concatenate([np.arange(4), np.arange(4, len(lower_aoa5), 3)])
    indices_5_lower = indices_5_lower[indices_5_lower < len(lower_aoa5)]
    plt.plot(
        lower_aoa5[indices_5_lower, 0], lower_aoa5[indices_5_lower, 1], "o", color="black", markersize=4, linewidth=1
    )

    # Plot simulation curves for AoA = 12
    plt.plot(
        x_upper_12[upper_sort_12], Cp_upper_12_smooth, "--", color="black", linewidth=1, label="Sim. (AoA=12°)"
    )
    plt.plot(
        x_lower_12[lower_sort_12], Cp_lower_12_smooth, "--", color="black", linewidth=1
    )

    # plt.plot(
    #      x_upper_12[upper_sort_12], Cp_12_upper[upper_sort_12], "--", color="black", label="Sim. (AoA=12°)"
    # )
    # plt.plot(
    #     x_lower_12[lower_sort_12], Cp_12_lower[lower_sort_12], "--", color="black"
    # )

    # Plot reference curves for AoA = 12
    # plt.plot(
    #     upper_aoa12[::2, 0], upper_aoa12[::2, 1], "s", color="black", markersize=4, linewidth=1, label="Ref. (AoA=12°)"
    # )
    # plt.plot(
    #     lower_aoa12[::2, 0], lower_aoa12[::2, 1], "s", color="black", markersize=4, linewidth=1
    # )
    indices_12 = np.concatenate([np.arange(5), np.arange(5, len(upper_aoa12), 3)])
    indices_12 = indices_12[indices_12 < len(upper_aoa12)]
    plt.plot(
        upper_aoa12[indices_12, 0], upper_aoa12[indices_12, 1], "s", color="black", markersize=4, linewidth=1, label="Ref. (AoA=12°)"
    )

    indices_12_lower = np.concatenate([np.arange(5), np.arange(5, len(lower_aoa12), 3)])
    indices_12_lower = indices_12_lower[indices_12_lower < len(lower_aoa12)]
    plt.plot(
        lower_aoa12[indices_12_lower, 0], lower_aoa12[indices_12_lower, 1], "s", color="black", markersize=4, linewidth=1
    )

    plt.gca().invert_yaxis()
    plt.xlabel("x/c")
    plt.ylabel("$c_P$")
    plt.ylim(1.3, -2)  # (ymin, ymax)
    plt.yticks(np.arange(-1.5, 1.5, 0.5))  # Ticks every 0.5
    #plt.legend(frameon=False)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)    
    plt.tight_layout()


    # Save plot
    output_dir = "/home/jofre/Members/Eduard/Paper2/Figures/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "pressure_coefficient_validation.eps"), format="eps", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "pressure_coefficient_validation.png"), format="png", dpi=300, bbox_inches="tight")
    
    print(f"Plots saved to {output_dir}")
    plt.show()

# Plot comparison
plot_cp_comparison_2(proj_points, Cp_5_values, Cp_12_values, c=1.0)

