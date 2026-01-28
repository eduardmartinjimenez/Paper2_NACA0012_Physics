#!/usr/bin/env python3
"""
Script to displace x and y values in data files by given offsets.
Modifies files in place, saving them with the same name and location.

Usage:
    1. Set the filename you want to modify
    2. Set the x_offset and y_offset values
    3. Run the script to apply the displacement
"""

import numpy as np
import os

def displace_data(filename, x_offset=0.0, y_offset=0.0):
    """
    Displace x and y values in a data file by given offsets.
    
    Parameters:
    -----------
    filename : str
        Name of the file to modify (relative to U_mean_profile_data directory)
    x_offset : float
        Offset to add to x values (first column)
    y_offset : float
        Offset to add to y values (second column)
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File '{filename}' not found in {script_dir}")
        return
    
    # Read the data (handling comma as decimal separator)
    print(f"Reading file: {filename}")
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse the data, replacing commas with periods for decimal points
    data_list = []
    for line in lines:
        line = line.strip()
        if line:
            # Replace commas with periods and split by whitespace
            values = line.replace(',', '.').split()
            if len(values) >= 2:
                data_list.append([float(values[0]), float(values[1])])
    
    data = np.array(data_list)
    
    # Check if data has at least 2 columns
    if data.ndim < 2 or data.shape[1] < 2:
        print(f"Error: File must have at least 2 columns")
        return
    
    # Store original values for reporting
    original_x_mean = data[:, 0].mean()
    original_y_mean = data[:, 1].mean()
    
    # Apply offsets
    data[:, 0] += x_offset
    data[:, 1] += y_offset
    
    # Report changes
    new_x_mean = data[:, 0].mean()
    new_y_mean = data[:, 1].mean()
    
    print(f"Original x mean: {original_x_mean:.6f}")
    print(f"New x mean:      {new_x_mean:.6f}")
    print(f"x displacement:  {x_offset:+.6f}")
    print(f"\nOriginal y mean: {original_y_mean:.6f}")
    print(f"New y mean:      {new_y_mean:.6f}")
    print(f"y displacement:  {y_offset:+.6f}")
    
    # Save the file (overwriting the original, using comma as decimal separator)
    print(f"\nSaving displaced data to: {filename}")
    with open(filepath, 'w') as f:
        for row in data:
            # Format numbers with commas as decimal separators
            x_str = f"{row[0]:.12f}".replace('.', ',')
            y_str = f"{row[1]:.12f}".replace('.', ',')
            f.write(f"{x_str}  {y_str}\n")
    print("Done!")
    

if __name__ == "__main__":
    # =============================================================================
    # MANUAL CONFIGURATION - MODIFY THESE VALUES
    # =============================================================================
    
    # Set the filename you want to modify
    filename = "Re5e4_AOA5_U_mean_035_Jardin_2025_2.dat"
    
    # Set the displacement offsets
    x_offset = 0  # Offset to add to x values (first column)
    y_offset = -0.0216  # Offset to add to y values (second column)
    
    # =============================================================================
    # END OF MANUAL CONFIGURATION
    # =============================================================================
    
    # Apply the displacement
    displace_data(filename, x_offset, y_offset)
