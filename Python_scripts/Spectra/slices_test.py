import os
import sys
import h5py
import numpy as np
import glob
import gc
import time

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader


class SliceLoader(CompressedSnapshotLoader):
    """
    Extended loader for slice data with support for loading slices from specific directories.
    Inherits mesh and topology loading from CompressedSnapshotLoader.
    """
    
    def list_available_slices(self, slice_directory):
        """
        List all slice files (excluding mesh) in a slice directory.
        
        Parameters
        ----------
        slice_directory : str
            Path to the slice directory (e.g., 'slice_1_compr')
        
        Returns
        -------
        list
            Sorted list of snapshot files in the directory
        """
        if not os.path.exists(slice_directory):
            raise FileNotFoundError(f"Slice directory not found: {slice_directory}")
        
        # Find all COMP-DATA.h5 files (snapshot files)
        slice_files = sorted(glob.glob(os.path.join(slice_directory, "*-COMP-DATA.h5")))
        return slice_files


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1"

# Choose which slice to work with (e.g., 'slice_1', 'slice_3', 'slice_5', etc.)
SLICE_NAME = "slice_1"
SLICE_LOCATION = os.path.join(BASE_DIR, f"{SLICE_NAME}_compr")
MESH_FILE = os.path.join(SLICE_LOCATION, "last_slice", f"{SLICE_NAME}-CROP-MESH.h5")
LAST_SNAPSHOT_DIR = os.path.join(SLICE_LOCATION, "last_slice")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Initialize the slice loader
    print("=" * 70)
    print(f"Loading slice: {SLICE_NAME}")
    print("=" * 70)
    
    loader = SliceLoader(MESH_FILE)
    
    # Display mesh information
    print(f"\nMesh shape (compressed domain): {loader.shape}")
    print(f"Number of fluid points (compressed): {loader.N_points}")
    
    # Get coordinates at fluid points
    x_coords, y_coords, z_coords = loader.get_coordinates()
    print(f"\nCoordinate ranges:")
    print(f"  X: [{x_coords.min():.6f}, {x_coords.max():.6f}]")
    print(f"  Y: [{y_coords.min():.6f}, {y_coords.max():.6f}]")
    print(f"  Z: [{z_coords.min():.6f}, {z_coords.max():.6f}]")
    
    # List available snapshots in this slice
    print(f"\n{'='*70}")
    print(f"Available snapshots in {SLICE_NAME}_compr:")
    print(f"{'='*70}")
    
    snapshot_files = loader.list_available_slices(SLICE_LOCATION)
    print(f"Total snapshots: {len(snapshot_files)}")
    print("\nFirst 5 snapshots:")
    for snap_file in snapshot_files[:5]:
        print(f"  - {os.path.basename(snap_file)}")
    if len(snapshot_files) > 5:
        print(f"  ... and {len(snapshot_files) - 5} more")
    
    # Load the first snapshot to see available variables
    print(f"\n{'='*70}")
    print("Loading first snapshot to inspect variables...")
    print(f"{'='*70}\n")
    
    first_snapshot = snapshot_files[0]
    print(f"File: {os.path.basename(first_snapshot)}")
    
    # Check what variables are in the snapshot file
    with h5py.File(first_snapshot, "r") as f:
        print(f"Variables available in snapshot:")
        for key in f.keys():
            dataset = f[key]
            print(f"  - {key:20s}: shape={dataset.shape}, dtype={dataset.dtype}")
    
    # Load the snapshot data
    data = loader.load_snapshot(first_snapshot)
    print(f"\nLoaded snapshot data:")
    for key in ['u', 'v', 'w', 'p']:
        values = data[key]
        print(f"  {key}: shape={values.shape}, min={values.min():.6f}, max={values.max():.6f}, mean={values.mean():.6f}")
    
    # Load and display the time-averaged snapshot
    print(f"\n{'='*70}")
    print("Loading time-averaged data from last_snapshot...")
    print(f"{'='*70}\n")
    
    # Find the averaged snapshot file (with avg_ variables)
    avg_snapshot_files = sorted(glob.glob(os.path.join(LAST_SNAPSHOT_DIR, "*-COMP-DATA.h5")))
    
    if avg_snapshot_files:
        avg_snapshot = avg_snapshot_files[0]
        print(f"File: {os.path.basename(avg_snapshot)}")
        
        # Check what variables are in the averaged snapshot
        with h5py.File(avg_snapshot, "r") as f:
            print(f"Variables available in averaged snapshot:")
            for key in f.keys():
                dataset = f[key]
                print(f"  - {key:20s}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # Try to load averaged data
        try:
            avg_data = loader.load_snapshot_avg(avg_snapshot)
            print(f"\nLoaded averaged data:")
            for key in ['avg_u', 'avg_v', 'avg_w', 'avg_p']:
                if key in avg_data:
                    values = avg_data[key]
                    print(f"  {key}: shape={values.shape}, min={values.min():.6f}, max={values.max():.6f}, mean={values.mean():.6f}")
                else:
                    print(f"  {key}: Not found in averaged snapshot")
            print(f"\nAll variables found in averaged snapshot: {[k for k in avg_data.keys() if k.startswith('avg_')]}")
        except Exception as e:
            print(f"\nError loading averaged data: {e}")
    else:
        print("No averaged snapshot files found in last_snapshot directory")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Slice: {SLICE_NAME}")
    print(f"Mesh dimensions: {loader.shape}")
    print(f"Fluid points: {loader.N_points}")
    print(f"Snapshots available: {len(snapshot_files)}")
    print(f"Time-averaged snapshot available: {len(avg_snapshot_files) > 0}")
    print("="*70)