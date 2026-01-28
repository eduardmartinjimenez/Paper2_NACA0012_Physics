import os
import h5py
from stl import mesh
import numpy as np


# Define file path and name
FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_1_compr/last_slice/"
FILE_NAME = "slice_1_output_14302400.h5"
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)



# Check if the data file exists
if os.path.exists(FULL_PATH):
    print(f"Data file exists! {FULL_PATH}")
else:
    print(f"Data file does not exist: {FULL_PATH}")

# Import data file
data_file = h5py.File(FULL_PATH, "r")

### Import 3D Data
# x_data = data_file["x"][:, :, :]
# y_data = data_file["y"][:, :, :]
# z_data = data_file["z"][:, :, :]
# tag_ibm_data = data_file["tag_IBM"][:, :, :]
# u_data = data_file["avg_u"][:, :, :]
# v_data = data_file["avg_v"][:, :, :]
# w_data = data_file["avg_w"][:, :, :]
# p_data = data_file["avg_P"][:, :, :]
# u_data = data_file["u"][:, :, :]
# v_data = data_file["v"][:, :, :]
# w_data = data_file["w"][:, :, :]
# p_data = data_file["P"][:, :, :]


### Import Correlation data
#tau_cor_1 = data_file["avg_corr_1"][:, : ,:]
#tau_cor_2 = data_file["avg_corr_2"][:, : ,:]
#tau_cor_3 = data_file["avg_corr_3"][:, : ,:]
#P_cor_1 = data_file["avg_corr_P_1"][:, :, :]
#P_cor_2 = data_file["avg_corr_P_2"][:, :, :]
#P_cor_3 = data_file["avg_corr_P_3"][:, :, :]

avg_time = data_file.attrs["AveragingTime"]
#avg_time_2nd = data_file.attrs["AveragingTime2nd"]
time = data_file.attrs["Time"]
print(f"Averaging time: {avg_time}")
#print(f"Averaging time with second order statistics: {avg_time_2nd}")
print(f"Simulation time: {time}")
#print(tau_cor_1.max())
#print(P_cor_1.max())

### Print final datasets
print( '\ List of datasets:' )
print( list( data_file.keys() ) )

