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
plot_pressure_coefficient(proj_points, Cp_12_values, c=1.0)


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

    plt.figure(figsize=(10, 5))
    plt.plot(upper_aoa12[:, 0], upper_aoa12[:, 1], "o-", label="Upper surface (AoA=12°)")
    plt.plot(lower_aoa12[:, 0], lower_aoa12[:, 1], "o-", label="Lower surface (AoA=12°)")
    
    plt.gca().invert_yaxis()
    plt.xlabel("x/c")
    plt.ylabel("$C_p$")
    plt.title("Reference Cp Data – Rodríguez")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot reference data
plot_cp_ref_data()


# Compute lift coefficient from reference Cp data
def compute_lift_coefficient_from_cp_ref():
    """
    Compute the lift coefficient from reference Cp data.
    
    The lift coefficient is computed by integrating the pressure difference
    between the lower and upper surfaces:
    C_l = ∫₀¹ (Cp_lower - Cp_upper) d(x/c)
    
    Returns:
        float: Lift coefficient
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
    
    # Extract x and Cp values
    x_upper = upper_aoa12[:, 0]
    Cp_upper = upper_aoa12[:, 1]
    x_lower = lower_aoa12[:, 0]
    Cp_lower = lower_aoa12[:, 1]
    
    # Sort by x coordinate for proper integration
    upper_sort = np.argsort(x_upper)
    x_upper = x_upper[upper_sort]
    Cp_upper = Cp_upper[upper_sort]
    
    lower_sort = np.argsort(x_lower)
    x_lower = x_lower[lower_sort]
    Cp_lower = Cp_lower[lower_sort]
    
    # Interpolate Cp values to a common x grid
    # Use a fine grid for accurate integration
    x_common = np.linspace(0, 1, 1000)
    Cp_upper_interp = np.interp(x_common, x_upper, Cp_upper)
    Cp_lower_interp = np.interp(x_common, x_lower, Cp_lower)
    
    # Compute the pressure difference (lower - upper)
    delta_Cp = Cp_lower_interp - Cp_upper_interp
    
    # Integrate using the trapezoidal rule
    Cl = np.trapezoid(delta_Cp, x_common)
    
    print(f"\n{'='*60}")
    print(f"Lift Coefficient from Reference Cp Data (AoA = 12°)")
    print(f"{'='*60}")
    print(f"C_l = {Cl:.6f}")
    print(f"{'='*60}\n")
    
    # Plot the pressure difference distribution
    plt.figure(figsize=(10, 5))
    plt.plot(x_common, delta_Cp, 'b-', linewidth=2, label='$C_{p,lower} - C_{p,upper}$')
    plt.fill_between(x_common, 0, delta_Cp, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel("x/c")
    plt.ylabel(r"$\Delta C_p$")
    plt.title(f"Pressure Difference Distribution (AoA=12°) | $C_l$ = {Cl:.6f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return Cl

# Compute lift coefficient from reference data
Cl_ref = compute_lift_coefficient_from_cp_ref()


# Compute lift coefficient from simulation Cp data
def compute_lift_coefficient_from_simulation(proj_points, Cp_values, c=1.0):
    """
    Compute the lift coefficient from simulation Cp data.
    
    The lift coefficient is computed by integrating the pressure difference
    between the lower and upper surfaces:
    C_l = ∫₀¹ (Cp_lower - Cp_upper) d(x/c)
    
    Parameters:
        proj_points (np.ndarray): Projected surface points (N x 3).
        Cp_values (np.ndarray): Pressure coefficient at each point (N,).
        c (float): Chord length for normalization (default = 1.0).
    
    Returns:
        float: Lift coefficient
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
    
    # Sort by x coordinate for proper integration
    upper_sort = np.argsort(x_upper)
    x_upper = x_upper[upper_sort]
    Cp_upper = Cp_upper[upper_sort]
    
    lower_sort = np.argsort(x_lower)
    x_lower = x_lower[lower_sort]
    Cp_lower = Cp_lower[lower_sort]
    
    # Interpolate Cp values to a common x grid
    # Use a fine grid for accurate integration
    x_common = np.linspace(0, 1, 1000)
    Cp_upper_interp = np.interp(x_common, x_upper, Cp_upper)
    Cp_lower_interp = np.interp(x_common, x_lower, Cp_lower)
    
    # Compute the pressure difference (lower - upper)
    delta_Cp = Cp_lower_interp - Cp_upper_interp
    
    # Integrate using the trapezoidal rule
    Cl = np.trapezoid(delta_Cp, x_common)
    
    print(f"\n{'='*60}")
    print(f"Lift Coefficient from Simulation Cp Data (AoA = 12°)")
    print(f"{'='*60}")
    print(f"C_l = {Cl:.6f}")
    print(f"{'='*60}\n")
    
    # Optional: Plot the pressure difference distribution
    plt.figure(figsize=(10, 5))
    plt.plot(x_common, delta_Cp, 'r-', linewidth=2, label='$C_{p,lower} - C_{p,upper}$ (Simulation)')
    plt.fill_between(x_common, 0, delta_Cp, alpha=0.3, color='red')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel("x/c")
    plt.ylabel(r"$\Delta C_p$")
    plt.title(f"Pressure Difference Distribution - Simulation (AoA=12°) | $C_l$ = {Cl:.6f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return Cl

# Compute lift coefficient from simulation data
Cl_sim = compute_lift_coefficient_from_simulation(proj_points, Cp_12_values, c=1.0)

# Compare results
print(f"\n{'='*60}")
print(f"Lift Coefficient Comparison")
print(f"{'='*60}")
print(f"Reference (Rodríguez et al.): C_l = {Cl_ref:.6f}")
print(f"Simulation:                  C_l = {Cl_sim:.6f}")
print(f"{'='*60}\n")