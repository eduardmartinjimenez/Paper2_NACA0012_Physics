import os, glob, re
import h5py
import numpy as np
from mpi4py import MPI


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