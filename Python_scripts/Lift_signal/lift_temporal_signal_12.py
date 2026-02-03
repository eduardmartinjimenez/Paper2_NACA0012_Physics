import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lift_temporal_signal_plots import (
    load_lift_data,
    plot_lift_vs_time_aoa12,
    plot_lift_with_mean_aoa12,
    plot_lift_and_accumulated_mean_aoa12,
)

# Setup
FILE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Transient/"
FILE_NAME = "lift_coef_temporal_signal_aoa12_Re50000_merged_12.csv"
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# Load data
t, Cl = load_lift_data(FULL_PATH)

# Plot options
plot_lift_vs_time_aoa12(t, Cl)
plot_lift_with_mean_aoa12(t, Cl, t_threshold=12)
plot_lift_and_accumulated_mean_aoa12(t, Cl, t_start=12)
