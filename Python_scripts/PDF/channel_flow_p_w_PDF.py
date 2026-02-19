#################################################################################
####                       POST PROCESSING TOOL                              ####
#################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import glob, re, math
import h5py

rc('text', usetex=True)
rc('font', family='serif')

#############################################
##     Organize Folders and other Stuf     ##
#############################################
base_dir = "/home/jofre/Members/Pablo/Snapshots/3d_turbulent_channel_flow_Retau180/"

#############################################
##          Reference Parameters           ##
#############################################
rho_0  = 1.0                        # Densidad de referencia [kg/m3]
u_tau  = 1.0                        # Velocidad de fricción [m/s]
delta  = 1.0                        # Mitad del canal [m]
Re_tau = 180.0                      # Número de Reynolds de fricción [-]
mu_ref = rho_0 * u_tau * delta / Re_tau   # Viscosidad dinámica [Pa s]
nu_ref = mu_ref / rho_0             # Viscosidad cinemática [m2/s]
tau_w  = rho_0 * u_tau * u_tau      # Tensión de pared de referencia [Pa]

######################################################
##          i) Open Multiple Data Files             ##
###################################################### 
#  3d_turbulent_channel_flow_21350000-DATA.h5
# file_indices = list(range(1000000, 9990000, 10000))
file_indices = list(range(1000000, 2000000, 10000))
# file_indices = list(range(1000000, 2000000, 100000))


data_files = [f"3d_turbulent_channel_flow_{i}-DATA.h5" for i in file_indices]
# data_files = sorted(glob.glob(os.path.join(base_dir, "3d_turbulent_channel_flow_*-DATA.h5")))
print(f"Found {len(data_files)} data files:")
# check first and last snapshot
if len(data_files) > 0:
    print(f"  First file: {data_files[0]}")
    print(f"  Last file: {data_files[-1]}")

with h5py.File(f"{base_dir}/3d_turbulent_channel_flow-MESH.h5", 'r') as data_file:
    x_data = data_file['x'][:,:,:]
    y_data = data_file['y'][:,:,:]
    z_data = data_file['z'][:,:,:]

######################################################
##    Accumulate pressure data across snapshots    ##
######################################################
P_walls_total = np.array([])

for file_name in data_files:
    file_path = os.path.join(base_dir, file_name)
    print(f"Loading data: {file_name}")
    
    with h5py.File(file_path, 'r') as data_file:
        P_data = data_file['p_data'][:,:,:]

        # num_points_x  = P_data[0,0,:].size
        # num_points_y  = P_data[0,:,0].size
        # num_points_z  = P_data[:,0,0].size
        # num_points_xz = num_points_x * num_points_z
    
    ######################################################
    ##    ii) Perform the Domain without Ghost Cells    ##
    ######################################################
    # Operating with the Ghost Cells
    # Delete Ghost Cells in P_data
    # P_data = np.array(P_data)
    # P_data[0, :, :]   = P_data[0, :, :]   + P_data[1, :, :]
    # P_data[-1, :, :]  = P_data[-1, :, :]  + P_data[-2, :, :]
    # P_data[:, 0, :]   = P_data[:, 0, :]   + P_data[:, 1, :]
    # P_data[:, -1, :]  = P_data[:, -1, :]  + P_data[:, -2, :]
    # P_data[:, :, 0]   = P_data[:, :, 0]   + P_data[:, :, 1]
    # P_data[:, :, -1]  = P_data[:, :, -1]  + P_data[:, :, -2]

    ######################################################
    ##           iii) Compute P_thermo                  ##
    ######################################################
    # total_P_volume = 0.0
    # total_volume   = 0.0
    # for i in range(1, num_points_x-1):
    #     for j in range(1, num_points_y-1):
    #         for k in range(1, num_points_z-1):
    #             # Geometrical stuf
    #             delta_x = 0.5 * ( x_data[k, j, i+1] - x_data[k, j, i-1] )
    #             delta_y = 0.5 * ( y_data[k, j+1, i] - y_data[k, j-1, i] )
    #             delta_z = 0.5 * ( z_data[k+1, j, i] - z_data[k-1, j, i] )
    #             volume  = delta_x * delta_y * delta_z
    #             # Update values
    #             total_P_volume += P_data[k, j, i] * volume
    #             total_volume   += volume
    # P_thermo = total_P_volume / total_volume

    ######################################################
    ##            iv) Obtain P_walls                    ##
    ######################################################
    # Extract pressures at bottom (ymin) and top walls (ymax) 
    # P_ymin = P_data[:, 0, :] - P_thermo   
    # P_ymax = P_data[:, -1, :] - P_thermo

    P_ymin = P_data[:, 1, :] 
    P_ymax = P_data[:, -2, :]
    # print(f"  P_ymin: shape {P_ymin.shape}")
    # print(f"  P_ymax: shape {P_ymax.shape}")
    
    P_walls = np.concatenate([P_ymin.flatten(), P_ymax.flatten()])
    # print(f"  P_walls: shape {P_walls.shape}")
    
    # Accumulate pressure data
    P_walls_total = np.concatenate([P_walls_total, P_walls])
    # print(f"  Accumulated wall pressure samples: {len(P_walls_total)}")

######################################################
##    iii) Compute and Plot PDF of P_walls         ##
######################################################
print(f"\nComputing PDF of P_walls...")
print(f"Total number of wall pressure samples: {len(P_walls_total)}")
print(f"Min: {np.min(P_walls_total):.6f}, Max: {np.max(P_walls_total):.6f}")
print(f"Mean: {np.mean(P_walls_total):.6f}, Std: {np.std(P_walls_total):.6f}")

# =========================================================================
# Compute histogram (PDF)
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING PDFs")
print("=" * 70)

N_BINS = 250

p_hist, p_bin_edges = np.histogram(P_walls_total, bins=N_BINS, density=False)
p_bin_centers = 0.5 * (p_bin_edges[:-1] + p_bin_edges[1:])

print(f"\n  Wall Pressure:")
print(f"    Number of samples: {len(P_walls_total)}")
print(f"    Data range:     [{np.min(P_walls_total):.4f}, {np.max(P_walls_total):.4f}]")
print(f"    Max PDF value:  {np.max(p_hist):.4f}")

# =========================================================================
# Plot PDF of raw pressure
# =========================================================================
print("\n" + "=" * 70)
print("PLOTTING PDF (p_w)")
print("=" * 70)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

ax.plot(p_bin_centers, p_hist, linewidth=2.5, color='steelblue')
ax.set_xlabel(r"Wall Pressure $p_w$ [-]", fontsize=13)
ax.set_ylabel(r"PDF $(p_w)$", fontsize=13)
ax.set_title(r"Probability Density Function of Wall Pressure", fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# =========================================================================
# Compute and Plot Normalized Fluctuation PDF
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING NORMALIZED FLUCTUATION PDF")
print("=" * 70)

p_mean = np.mean(P_walls_total)
p_fluct = P_walls_total - p_mean
p_rms = np.sqrt(np.mean(p_fluct**2))

print(f"\n  Wall Pressure:")
print(f"    <p_w> = {p_mean:.6f}")
print(f"    p'_w,rms = {p_rms:.6f}")

if p_rms == 0.0:
    print(f"  [WARNING] Zero RMS for pressure; skipping normalized PDFs")
else:
    p_fluct_norm = p_fluct / p_rms
    
    p_hist_norm, p_bin_edges_norm = np.histogram(p_fluct_norm, bins=N_BINS, density=False)
    p_bin_centers_norm = 0.5 * (p_bin_edges_norm[:-1] + p_bin_edges_norm[1:])
    
    print(f"    Normalized p'_w/p'_w,rms range: [{np.min(p_fluct_norm):.2f}, {np.max(p_fluct_norm):.2f}]")
    print(f"    Mean of normalized: {np.mean(p_fluct_norm):.6f}")
    print(f"    RMS of normalized: {np.sqrt(np.mean((p_fluct_norm)**2)):.6f}")
    
    print("\n" + "=" * 70)
    print("PLOTTING NORMALIZED FLUCTUATION PDF")
    print("=" * 70)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(p_bin_centers_norm, p_hist_norm, linewidth=2.5, color='steelblue')
    ax.set_xlabel(r"Normalized Pressure Fluctuation $p'_w/p'_{w,rms}$ [-]", fontsize=13)
    ax.set_ylabel(r"PDF $(p'_w/p'_{w,rms})$", fontsize=13)
    ax.set_title(r"Normalized Wall Pressure Fluctuation PDF", fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)






