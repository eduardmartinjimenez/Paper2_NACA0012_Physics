import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Paths
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_1_compr/"
MESH_NAME = "slice_1-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_1_compr/"
SLICE_NAME = "slice_1_11280000-COMP-DATA.h5"
SLICE_FILE = os.path.join(SLICE_PATH, SLICE_NAME)
    
AVG_SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_1_compr/last_slice/"
AVG_SLICE_NAME = "slice_1_14302400-COMP-DATA.h5"
AVG_SLICE_FILE = os.path.join(AVG_SLICE_PATH, AVG_SLICE_NAME)

