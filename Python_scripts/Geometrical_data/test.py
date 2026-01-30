import h5py
import numpy as np
import os

# Example 1: Basic h5 file reading
def read_h5_basic(file_path):
    """
    Read and display the structure of an h5 file.
    """
    print(f"\n=== Reading file: {file_path} ===")
    
    with h5py.File(file_path, "r") as f:
        print("\nDatasets and groups in the file:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        
        f.visititems(print_structure)
        
        # List all keys at root level
        print(f"\nRoot level keys: {list(f.keys())}")
        
        return list(f.keys())


# Example 2: Extract specific datasets
def extract_datasets(file_path, dataset_names):
    """
    Extract specific datasets from an h5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5 file
    dataset_names : list
        List of dataset names to extract
    
    Returns:
    --------
    dict : Dictionary with dataset names as keys and numpy arrays as values
    """
    print(f"\n=== Extracting datasets from: {file_path} ===")
    
    data = {}
    with h5py.File(file_path, "r") as f:
        for name in dataset_names:
            if name in f:
                data[name] = f[name][:]
                print(f"Extracted '{name}': shape={data[name].shape}, dtype={data[name].dtype}")
            else:
                print(f"Warning: '{name}' not found in file")
    
    return data


# Example 3: Extract all datasets
def extract_all_datasets(file_path):
    """
    Extract all datasets from an h5 file.
    """
    print(f"\n=== Extracting all datasets from: {file_path} ===")
    
    data = {}
    with h5py.File(file_path, "r") as f:
        def extract_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[:]
                print(f"Extracted '{name}': shape={obj.shape}, dtype={obj.dtype}")
        
        f.visititems(extract_dataset)
    
    return data


# Example 4: Read with slicing (memory efficient for large files)
def read_h5_slice(file_path, dataset_name, slice_indices=None):
    """
    Read a slice of data from a large h5 dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5 file
    dataset_name : str
        Name of the dataset to read
    slice_indices : tuple
        Slice indices (e.g., (slice(0, 100), slice(0, 100)))
    
    Returns:
    --------
    numpy array : The sliced data
    """
    print(f"\n=== Reading slice from {dataset_name} ===")
    
    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in file")
        
        dataset = f[dataset_name]
        print(f"Full dataset shape: {dataset.shape}")
        
        if slice_indices:
            data = dataset[slice_indices]
        else:
            data = dataset[:]
        
        print(f"Extracted slice shape: {data.shape}")
    
    return data


# Example usage
if __name__ == "__main__":
    
    # ===== MODIFY THIS SECTION FOR YOUR FILE =====
    
    # Path to your h5 file
    h5_file = "/path/to/your/file.h5"
    
    # Check if file exists
    if not os.path.exists(h5_file):
        print(f"Error: File not found: {h5_file}")
        print("\nPlease update the 'h5_file' variable with the correct path.")
        print("\nExample paths in your workspace:")
        print("  - Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/...")
        print("  - Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/...")
    else:
        # 1. First, explore the file structure
        keys = read_h5_basic(h5_file)
        
        # 2. Extract specific datasets (modify with actual dataset names)
        # Example: extract velocity components
        # data = extract_datasets(h5_file, ["u", "v", "w", "p"])
        
        # 3. Or extract all datasets
        # all_data = extract_all_datasets(h5_file)
        
        # 4. For large files, use slicing
        # Example: read first 100 points in each dimension
        # u_slice = read_h5_slice(h5_file, "u", (slice(0, 100), slice(0, 100), slice(0, 100)))
        
        # 5. Access the data
        # print(f"\nData keys: {list(data.keys())}")
        # print(f"u velocity range: [{data['u'].min():.4f}, {data['u'].max():.4f}]")
