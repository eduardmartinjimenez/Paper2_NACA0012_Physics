import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Define paths
FILE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
FILE_NAME = "3d_NACA0012_Re50000_AoA12_Mean_Data_19680000.h5"
FILE_PATH = os.path.join(FILE_DIR, FILE_NAME)

# Load and verify saved data
with h5py.File(FILE_PATH, "r") as f:
    x_over_c_loaded = f["x_over_c"][:]
    delta_y_plus_loaded = f["delta_y_plus"][:]
    delta_x_plus_loaded = f["delta_x_plus"][:]
    delta_z_plus_loaded = f["delta_z_plus"][:]
    tau_w_loaded = f["tau_w"][:]
    print("Loaded data from saved file:")


    # Smooth the data
    window_size = 5
    delta_y_plus_loaded = uniform_filter1d(delta_y_plus_loaded, size=window_size)
    delta_x_plus_loaded = uniform_filter1d(delta_x_plus_loaded, size=window_size)
    delta_z_plus_loaded = uniform_filter1d(delta_z_plus_loaded, size=window_size)

# Plot grid resolution near the wall

# Set LaTeX style for plots
plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 16, family='serif' )
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb}')

def plot_grid_resolution_near_wall(x, dy_plus, dx_plus, dz_plus):
    fig = plt.figure(figsize=(6, 5))
    
    plt.plot(x, dy_plus, linestyle="-", color="black", label=r"$\Delta y^+$")
    plt.plot(x, dz_plus, linestyle="--", color="black", label=r"$\Delta z^+$")
    plt.plot(x, dx_plus, linestyle=":", color="black", label=r"$\Delta x^+$")


    # Add horizontal line at y=1
    #plt.axhline(y=1, color="red", linestyle=":", linewidth=1, label=r"$y^+ = 1$")

    # plt.xlabel(r"$x/C$", fontsize=18)
    # plt.ylabel(r"$\Delta^+$", fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.grid(False, which="both", linestyle="--", linewidth=0.5)
    # plt.legend(fontsize=18, loc="upper right", frameon=False)


    plt.xlabel("x/c")
    #plt.ylabel(r"$\Delta^+$")
    plt.xticks()
    plt.yticks()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", frameon=False)


    plt.ylim(0, 20)
    plt.xlim(0, 1)
    plt.yticks(np.arange(0, 21, 2))
    plt.tight_layout()

    return fig

# Plot grid resolution near the wall
fig = plot_grid_resolution_near_wall(
    x_over_c_loaded,
    delta_y_plus_loaded,
    delta_x_plus_loaded,
    delta_z_plus_loaded,
)

plt.show()

# Save plot as eps
PLOT_DIR = "/home/jofre/Members/Eduard/Paper2/Figures/"
os.makedirs(PLOT_DIR, exist_ok=True)

PLOT_NAME = "grid_resolution_near_wall_NACA0012_AoA12_Re50000.eps"
PLOT_PATH = os.path.join(PLOT_DIR, PLOT_NAME)
fig.savefig(PLOT_PATH, format="eps", dpi=300)


# Save plot as png 
PLOT_NAME_PNG = "grid_resolution_near_wall_NACA0012_AoA12_Re50000.png"
PLOT_PATH_PNG = os.path.join(PLOT_DIR, PLOT_NAME_PNG)
fig.savefig(PLOT_PATH_PNG, format="png", dpi=300)

print(f"Plots saved at: {PLOT_PATH} and {PLOT_PATH_PNG}")



