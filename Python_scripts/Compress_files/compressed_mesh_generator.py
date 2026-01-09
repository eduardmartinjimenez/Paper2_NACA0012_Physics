import os, glob, re
import h5py
import numpy as np
import gc

# Define input file path and base name
INPUT_FILE_PATH = "/home/jofre/Members/George/Simulations/3d_ibm_stl_naca0012_Mesh_7/Snapshots/"
INPUT_FILE_NAME = "3d_ibm_stl_naca0012_aoa5_Re10000_1090000.h5"

# Define ouput folder path
OUTPUT_FILE_PATH = "/home/jofre/Members/George/Simulations/3d_ibm_stl_naca0012_Mesh_7/Snapshots/compressed_snapshots/"

# Rename ONLY the output files. If None, it uses INPUT_FILE_BASENAME.
OUTPUT_MESH_NAME = "3d_NACA0012_Re10000_AoA5"

# Define crop region:
X_MIN = -0.5
Y_MIN = -0.5
Z_MIN = -1.0

X_MAX = 8.0
Y_MAX = 1.5
Z_MAX = 1.0

def crop_3d_by_physical_coords(data, coord_array, min_coords, max_coords):
    """
    Crop a 3D array using physical coordinate bounds from a coordinate tensor.

    Parameters:
        data (np.ndarray): The original 3D array with shape (D, H, W).
        coord_array (np.ndarray): Array of shape (D, H, W, 3) with (X, Y, Z) coordinates per voxel.
        min_coords (tuple): Minimum (X, Y, Z) values as a 3-tuple.
        max_coords (tuple): Maximum (X, Y, Z) values as a 3-tuple.

    Returns:
        cropped_data (np.ndarray): Cropped 3D array.
        crop_bounds (tuple): ((z_min, z_max), (y_min, y_max), (x_min, x_max)) index bounds.
    """
    if data.shape != coord_array.shape[:3]:
        raise ValueError(
            "coord_array must have shape (D, H, W, 3) matching data shape (D, H, W)."
        )

    # Create mask for each axis
    mask = np.ones(data.shape, dtype=bool)
    for axis, (min_val, max_val) in enumerate(zip(min_coords, max_coords)):
        # print("axis=",axis," min=",min_val," max=",max_val)
        axis_vals = coord_array[..., axis]
        # print("ax vals ",axis_vals)
        mask &= (axis_vals >= min_val) & (axis_vals <= max_val)
        # print("mask ",axis_vals)

    # Find the bounding box of the region where mask is True
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("No voxels found within the given coordinate range.")

    z_min, y_min, x_min = indices.min(axis=0)
    z_max, y_max, x_max = indices.max(axis=0) + 1  # +1 to include upper bounds

    cropped_data = data[z_min:z_max, y_min:y_max, x_min:x_max]
    # return cropped_data, ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    return cropped_data


def compress_3d_array(data: np.ndarray, mask: np.ndarray):
    """
    Eliminates positions in a 3D NumPy array based on a mask and returns:
    - The values of the non-eliminated positions
    - Their indices in the original array

    Parameters:
    - data (np.ndarray): A 3D array of data.
    - mask (np.ndarray): A 3D boolean array of the same shape as `data`.
                         Positions where `mask` is True will be eliminated.

    Returns:
    - compressed_values (np.ndarray): 1D array of values from `data` where mask is False.
    - compressed_topo (np.ndarray): 2D array of shape (N, 3), with the indices of these values.
    """
    if data.shape != mask.shape:
        raise ValueError("`data` and `mask` must have the same shape")

    # Get the indices where mask is False (i.e., not eliminated)
    compressed_topo = np.argwhere(~mask)

    # Extract the values at those indices
    compressed_values = data[~mask]

    return compressed_values, compressed_topo

if __name__ == "__main__":
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)
    
    # Build full input file path
    INPUT_FULL_PATH = os.path.join(INPUT_FILE_PATH, INPUT_FILE_NAME)
    print(f"Loading mesh from: {INPUT_FULL_PATH}")

    # Import data file
    data_file = h5py.File(INPUT_FULL_PATH, "r")

    ### Import 3D Data
    x_data = data_file["x"][:, :, :]
    y_data = data_file["y"][:, :, :]
    z_data = data_file["z"][:, :, :]
    
    print("Check point 1: 3D coordinate arrays loaded.")

    min_coords = (X_MIN, Y_MIN, Z_MIN)
    max_coords = (X_MAX, Y_MAX, Z_MAX)
    
    coords_array = np.stack(
        (x_data, y_data, z_data), axis=-1
    )  # Shape: (D, H, W, 3)    
    
    # Crop X data to the specified domain
    x_cropped = crop_3d_by_physical_coords(
        x_data, coords_array, min_coords, max_coords
    )

    print("Check point 2: X cropped")

    del x_data
    gc.collect()

    # Crop Y data to the specified domain
    y_cropped = crop_3d_by_physical_coords(
        y_data, coords_array, min_coords, max_coords
    )
    print("Check point 3: Y cropped")

    del y_data
    gc.collect()

    # Crop Z data to the specified domain
    z_cropped = crop_3d_by_physical_coords(
        z_data, coords_array, min_coords, max_coords
    )
    print("Check point 4: Z cropped")
    del z_data
    gc.collect()

    # Import tag_IBM data
    tag_ibm_data = data_file["tag_IBM"][:, :, :]

    ### Crop tag_IBM data to the specified domain
    tag_ibm_cropped = crop_3d_by_physical_coords(
        tag_ibm_data, coords_array, min_coords, max_coords
    )       
    print("Check point 5: tag_ibm cropped")
    del tag_ibm_data
    gc.collect()
    
    ### Compressing data by removing solid points tag_ibm==1
    mask = tag_ibm_cropped > 0.9999

    tag_ibm_compressed, compressed_topo = compress_3d_array(tag_ibm_cropped, mask)

    # Save compressed mesh
    OUTPUT_MESH_PATH = os.path.join(
        OUTPUT_FILE_PATH, f"{OUTPUT_MESH_NAME}-CROP-MESH.h5"
    )

    print("saving file: ", OUTPUT_MESH_PATH)
    with h5py.File(OUTPUT_MESH_PATH, "w") as output_file:
        output_file.create_dataset("x", data=x_cropped, dtype="float32")
        output_file.create_dataset("y", data=y_cropped, dtype="float32")
        output_file.create_dataset("z", data=z_cropped, dtype="float32")
        output_file.create_dataset("tag_IBM", data=tag_ibm_cropped, dtype="float32")
        output_file.create_dataset(
            "compressed_topology", data=compressed_topo, dtype="int"
        )
        output_file.close()
    
    print("Mesh compression completed!")
