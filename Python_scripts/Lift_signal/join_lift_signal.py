#!/usr/bin/env python3
"""
Script to merge two CSV files containing instantaneous lift signal data.
The script finds the last iteration in the first file, searches for it in the second file,
and appends the remaining data from the second file.
"""

import pandas as pd
import argparse
import os

def merge_lift_signals(file1_path, file2_path, output_path):
    """
    Merge two lift signal CSV files.
    
    Parameters:
    -----------
    file1_path : str
        Path to the first CSV file (contains the complete signal up to some iteration)
    file2_path : str
        Path to the second CSV file (contains additional signal data)
    output_path : str
        Path where the merged CSV file will be saved
    """
    
    print(f"Reading first file: {file1_path}")
    # Read the first CSV file
    df1 = pd.read_csv(file1_path, skipinitialspace=True)
    
    print(f"Reading second file: {file2_path}")
    # Read the second CSV file
    df2 = pd.read_csv(file2_path, skipinitialspace=True)
    
    # Clean column names (remove leading/trailing spaces)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Get the last iteration from the first file
    last_iteration_file1 = df1['ite'].iloc[-1]
    print(f"Last iteration in first file: {last_iteration_file1}")
    
    # Find the index in the second file where iteration is greater than last_iteration_file1
    idx_start = df2[df2['ite'] > last_iteration_file1].index
    
    if len(idx_start) == 0:
        print("Warning: No new data found in the second file. The second file doesn't contain iterations beyond the first file.")
        print("Saving the first file as output...")
        merged_df = df1
    else:
        start_idx = idx_start[0]
        print(f"Found {len(df2) - start_idx} new entries in second file starting from iteration {df2.loc[start_idx, 'ite']}")
        
        # Append the new data from file2 to file1
        merged_df = pd.concat([df1, df2.iloc[start_idx:]], ignore_index=True)
        print(f"Total entries in merged file: {len(merged_df)}")
    
    # Save the merged data
    print(f"Saving merged file to: {output_path}")
    merged_df.to_csv(output_path, index=False)
    print("Merge completed successfully!")
    
    return merged_df

def main():
    """Main function with argument parsing."""
    
    # Default file paths
    default_file1 = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Transient/lift_coef_temporal_signal_aoa5_Re50000_merged_8.csv"
    default_file2 = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Steady_state/batch_32343687/lift_coef_temporal_signal_aoa5_Re50000.csv"
    default_output = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Transient/lift_coef_temporal_signal_aoa5_Re50000_merged_9.csv"
    
    parser = argparse.ArgumentParser(description='Merge two lift signal CSV files')
    parser.add_argument('--file1', type=str, default=default_file1,
                        help='Path to the first CSV file (complete signal)')
    parser.add_argument('--file2', type=str, default=default_file2,
                        help='Path to the second CSV file (additional signal)')
    parser.add_argument('--output', type=str, default=default_output,
                        help='Path for the output merged CSV file')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.file1):
        print(f"Error: File not found: {args.file1}")
        return
    
    if not os.path.exists(args.file2):
        print(f"Error: File not found: {args.file2}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Merge the files
    merge_lift_signals(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main()
