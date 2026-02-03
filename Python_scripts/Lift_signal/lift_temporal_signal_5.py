import os
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lift_temporal_signal_plots import (
    load_lift_data,
    plot_lift_vs_time_aoa5,
    plot_lift_with_mean_aoa5,
    plot_lift_and_accumulated_mean_aoa5,
)

# Setup
# FILE_PATH = "/home/grashof/Documents/Simulations/Allocation/NACA_0012_1148_1042_128_aoa5_Re50000_Ma01/"
FILE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Transient"
#FILE_PATH = "/gpfs/scratch/upc108/EDU/NACA_0012_AOA5_Re50000_1716x1662x128/"
FILE_NAME = "lift_coef_temporal_signal_aoa5_Re50000_merged_9.csv"
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# Load data
t, Cl = load_lift_data(FULL_PATH)

# Plot options
plot_lift_vs_time_aoa5(t, Cl)
plot_lift_with_mean_aoa5(t, Cl, t_threshold=15)
plot_lift_and_accumulated_mean_aoa5(t, Cl, t_start=15)
