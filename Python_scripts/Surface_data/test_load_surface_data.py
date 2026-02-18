import h5py
import numpy as np
import os

SURF_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/batch_30685746/Surface_data/"
SURF_NAME = "surface_3d_NACA0012_Re50000_AoA12_6850000-COMP-DATA.h5"
SURF_FILE = os.path.join(SURF_PATH, SURF_NAME)

with h5py.File(SURF_FILE, "r") as f:
    # print keys
    print("Keys in the HDF5 file:")
    for key in f.keys():
        print(f" - {key}")

    # print attributes
    print("\nAttributes in the HDF5 file:")
    for key in f.attrs.keys():
        print(f" - {key}: {f.attrs[key]}")

    # Load data
    p_w = f["p_w"][:]  # (Nz_phys, N_surf
    tau_w = f["tau_w"][:]  # (Nz_phys, N_surf)
    print(f"\nLoaded p_w shape: {p_w.shape}, tau_w shape: {tau_w.shape}")

    # Loas bulk pressure
    p_bulk = f.attrs["p_bulk"]  # Scalar
    print(f"Loaded p_bulk: {p_bulk}")
    

