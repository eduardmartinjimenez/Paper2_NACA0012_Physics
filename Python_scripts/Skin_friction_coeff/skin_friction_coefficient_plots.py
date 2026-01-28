import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import uniform_filter1d
from matplotlib.ticker import FuncFormatter

# Set LaTeX style for plots
plt.rc('text', usetex=True)
plt.rc('font', size=16, family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

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
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]
print("Loaded geometrical data.")

# LOAD Cf DATA (AoA = 12)
Cf_12_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
Cf_12_NAME = "3d_NACA0012_Re50000_AoA12_Cf_19680000.h5"
Cf_12_FILE = os.path.join(Cf_12_PATH, Cf_12_NAME)

# LOAD Cf DATA (AoA = 12)
Cf_5_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
Cf_5_NAME = "3d_NACA0012_Re50000_AoA5_Cf_24280000.h5"
Cf_5_FILE = os.path.join(Cf_5_PATH, Cf_5_NAME)

# Check if the Cf data file exists
if os.path.exists(Cf_12_FILE):
    print(f"Cf data file exists! {Cf_12_FILE}")
else:
    print(f"Cf data file does not exist: {Cf_12_FILE}")

# Check if the Cf data file exists
if os.path.exists(Cf_5_FILE):
    print(f"Cf data file exists! {Cf_5_FILE}")
else:
    print(f"Cf data file does not exist: {Cf_5_FILE}")

# Load Cf data
with h5py.File(Cf_12_FILE, "r") as f:
    Cf_12_values = f["Cf_values"][:]
print("Loaded Cf (AoA 12) data.")

# Load Cf data
with h5py.File(Cf_5_FILE, "r") as f:
    Cf_5_values = f["Cf_values"][:]
print("Loaded Cf (AoA 5) data.")


def plot_skin_friction_coefficient_upper_only(proj_points, Cf_values, c=1.0, title="Skin Friction Coefficient (Upper Surface)"):
    """
    Plot the skin friction coefficient (Cf) along the airfoil upper surface only.

    Parameters:
        proj_points (np.ndarray): Projected surface points (N x 3).
        Cf_values (np.ndarray): Skin friction coefficient at each point (N,).
        c (float): Chord length for normalization (default = 1.0).
        title (str): Title for the plot.
    """

    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]
    x_over_c = x_proj / c

    # Upper surface mask
    upper_mask = y_proj > 0
    x_upper = x_over_c[upper_mask]
    Cf_upper = Cf_values[upper_mask]

    upper_sort = np.argsort(x_upper)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x_upper[upper_sort],
        Cf_upper[upper_sort],
        label="Upper surface",
        marker="o",
        markersize=4,
        linewidth=1.4,
    )
    ax.set_xlabel("x/c", fontsize=12)
    ax.set_ylabel("$C_f$", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_cf_comparison_combined(proj_points, Cf_5_values, Cf_12_values, c=1.0):
    """
    Plot simulated Cf data for AoA = 5 and AoA = 12 degrees (upper surface only).

    Parameters:
        proj_points (np.ndarray): Projected surface points (N x 3).
        Cf_5_values (np.ndarray): Skin friction coefficient at each point for AoA = 5 (N,).
        Cf_12_values (np.ndarray): Skin friction coefficient at each point for AoA = 12 (N,).
        c (float): Chord length for normalization (default = 1.0).
    """

    # === Process simulation data for AoA = 5 ===
    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]
    x_over_c = x_proj / c

    # Upper surface mask
    upper_mask = y_proj > 0

    x_upper_5 = x_over_c[upper_mask]
    Cf_upper_5 = Cf_5_values[upper_mask]
    upper_sort_5 = np.argsort(x_upper_5)
    
    # Apply uniform filter smoothing with different kernels before and after x/c=0.7
    x_sorted_5 = x_upper_5[upper_sort_5]
    Cf_sorted_5 = Cf_upper_5[upper_sort_5]
    
    # Find split point at x/c = 0.7
    split_idx = np.searchsorted(x_sorted_5, 0.7)
    
    # Lighter smoothing before x/c=0.7
    Cf_upper_5_smooth_1 = uniform_filter1d(Cf_sorted_5[:split_idx], size=25, mode='nearest')
    # Heavier smoothing from x/c=0.7 onwards
    Cf_upper_5_smooth_2 = uniform_filter1d(Cf_sorted_5[split_idx:], size=80, mode='nearest')
    
    # Combine both parts
    Cf_upper_5_smooth = np.concatenate([Cf_upper_5_smooth_1, Cf_upper_5_smooth_2])
    
    # === Process simulation data for AoA = 12 ===
    x_upper_12 = x_over_c[upper_mask]
    Cf_upper_12 = Cf_12_values[upper_mask]
    upper_sort_12 = np.argsort(x_upper_12)
    
    # Apply uniform filter smoothing (AoA = 12)
    Cf_upper_12_smooth = uniform_filter1d(Cf_upper_12[upper_sort_12], size=20, mode='nearest')

    # === Plotting ===
    plt.figure(figsize=(6, 4))
    
    # Plot simulation curves for AoA = 5
    plt.plot(
        x_upper_5[upper_sort_5], Cf_upper_5_smooth, "-", color="black", linewidth=1, label="Sim. (AoA=5°)"
    )

    # Plot simulation curves for AoA = 12
    plt.plot(
        x_upper_12[upper_sort_12], Cf_upper_12_smooth, "--", color="black", linewidth=1, label="Sim. (AoA=12°)"
    )

    plt.xlabel("x/c")
    plt.ylabel("$c_f$")
    plt.ylim(-0.007, 0.025)
    plt.yticks(np.arange(-0.005, 0.03, 0.005))  # Ticks every 0.005
    
    # Format tick labels to show 0 without decimals
    def format_x_ticks(x, pos):
        if x == 0:
            return '0'
        return f'{x:.1f}'
    
    def format_y_ticks(y, pos):
        if y == 0:
            return '0'
        return f'{y:.3f}'
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    
    # plt.ylim(0, 0.005)
    # plt.xlim(0.3, 0.9)
    #plt.legend(frameon=False)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)    
    plt.tight_layout()

    # Save plot
    output_dir = "/home/jofre/Members/Eduard/Paper2/Figures/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "skin_friction_coefficient_comparison.eps"), format="eps", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "skin_friction_coefficient_comparison.png"), format="png", dpi=300, bbox_inches="tight")
    
    print(f"Plots saved to {output_dir}")
    plt.show()


def plot_cf_comparison_with_reference(proj_points, Cf_5_values, Cf_12_values, c=1.0):
    """
    Plot simulated Cf data for AoA = 5 and AoA = 12 degrees (upper surface only) with reference data.

    Parameters:
        proj_points (np.ndarray): Projected surface points (N x 3).
        Cf_5_values (np.ndarray): Skin friction coefficient at each point for AoA = 5 (N,).
        Cf_12_values (np.ndarray): Skin friction coefficient at each point for AoA = 12 (N,).
        c (float): Chord length for normalization (default = 1.0).
    """

    # === Load reference data ===
    base_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Skin_friction_coeff/Cf_data/"
    
    comma_to_dot = lambda s: float(s.replace(",", "."))
    
    # Load reference data for AoA = 5
    ref_aoa5 = np.genfromtxt(
        os.path.join(base_path, "Lehmkuhl_2013_Re5e4_aoa5.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )
    
    # Load reference data for AoA = 12
    ref_aoa12 = np.genfromtxt(
        os.path.join(base_path, "Rodriguez_2013_Re5e4_aoa12_cf.dat"),
        converters={0: comma_to_dot, 1: comma_to_dot},
    )

    # === Process simulation data for AoA = 5 ===
    x_proj = proj_points[:, 0]
    y_proj = proj_points[:, 1]
    x_over_c = x_proj / c

    # Upper surface mask
    upper_mask = y_proj > 0

    x_upper_5 = x_over_c[upper_mask]
    Cf_upper_5 = Cf_5_values[upper_mask]
    upper_sort_5 = np.argsort(x_upper_5)
    
    # Apply uniform filter smoothing (AoA = 5)
    Cf_upper_5_smooth = uniform_filter1d(Cf_upper_5[upper_sort_5], size=20, mode='nearest')
    
    # === Process simulation data for AoA = 12 ===
    x_upper_12 = x_over_c[upper_mask]
    Cf_upper_12 = Cf_12_values[upper_mask]
    upper_sort_12 = np.argsort(x_upper_12)
    
    # Apply uniform filter smoothing (AoA = 12)
    Cf_upper_12_smooth = uniform_filter1d(Cf_upper_12[upper_sort_12], size=20, mode='nearest')

    # === Plotting ===
    plt.figure(figsize=(6, 4))
    
    # Plot simulation curves for AoA = 5
    plt.plot(
        x_upper_5[upper_sort_5], Cf_upper_5_smooth, "-", color="black", linewidth=1, label="Sim. (AoA=5°)"
    )

    # Plot reference data for AoA = 5
    indices_5 = np.concatenate([np.arange(4), np.arange(4, len(ref_aoa5), 1)])
    indices_5 = indices_5[indices_5 < len(ref_aoa5)]
    plt.plot(
        ref_aoa5[indices_5, 0], ref_aoa5[indices_5, 1], "o", color="black", markersize=4, linewidth=1, label="Ref. (AoA=5°)"
    )

    # Plot simulation curves for AoA = 12
    plt.plot(
        x_upper_12[upper_sort_12], Cf_upper_12_smooth, "--", color="black", linewidth=1, label="Sim. (AoA=12°)"
    )
    
    # Plot reference data for AoA = 12
    indices_12 = np.concatenate([np.arange(5), np.arange(5, len(ref_aoa12), 1)])
    indices_12 = indices_12[indices_12 < len(ref_aoa12)]
    plt.plot(
        ref_aoa12[indices_12, 0], ref_aoa12[indices_12, 1], "s", color="black", markersize=4, linewidth=1, label="Ref. (AoA=12°)"
    )

    plt.xlabel("x/c")
    plt.ylabel("$c_f$")
    plt.ylim(-0.007, 0.025)
    #plt.legend(frameon=False)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)    
    plt.tight_layout()

    # Save plot
    output_dir = "/home/jofre/Members/Eduard/Paper2/Figures/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "skin_friction_coefficient_validation.eps"), format="eps", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "skin_friction_coefficient_validation.png"), format="png", dpi=300, bbox_inches="tight")
    
    print(f"Plots saved to {output_dir}")
    plt.show()




# Create plots
print("\n" + "="*60)
print("Creating Combined Skin Friction Coefficient Plot")
print("="*60)

output_dir = "/home/jofre/Members/Eduard/Paper2/Figures/"
os.makedirs(output_dir, exist_ok=True)

# Plot: Cf distribution vs x/c (upper surface only)
# fig1, ax1 = plot_skin_friction_coefficient_upper_only(
#     proj_points,
#     Cf_12_values,
#     c=1.0,
#     title="Skin Friction Coefficient on NACA 0012 (Upper Surface, AoA=12°, Re=50000)",
# )
# plt.savefig(os.path.join(output_dir, "Cf_upper_distribution_vs_x_AoA12.png"), dpi=300, bbox_inches='tight')
# print("Saved: Cf_upper_distribution_vs_x_AoA12.png")

# fig2, ax2 = plot_skin_friction_coefficient_upper_only(
#     proj_points,
#     Cf_5_values,
#     c=1.0,
#     title="Skin Friction Coefficient on NACA 0012 (Upper Surface, AoA=5°, Re=50000)",
# )
# plt.savefig(os.path.join(output_dir, "Cf_upper_distribution_vs_x_AoA5.png"), dpi=300, bbox_inches='tight')
# print("Saved: Cf_upper_distribution_vs_x_AoA5.png")


# Plot combined comparison
plot_cf_comparison_combined(proj_points, Cf_5_values, Cf_12_values, c=1.0)

# Plot combined comparison with reference data
#plot_cf_comparison_with_reference(proj_points, Cf_5_values, Cf_12_values, c=1.0)

