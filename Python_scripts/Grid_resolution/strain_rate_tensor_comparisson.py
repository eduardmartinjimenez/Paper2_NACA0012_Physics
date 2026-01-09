import h5py
import numpy as np

file1 = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Mean_data/AoA5_Re10000_velocity_RMS_profiles_data_mpi.h5"
file2 = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/Mean_data/AoA5_Re10000_velocity_RMS_profiles_data.h5"

with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
    def compare_groups(g1, g2, path=""):
        keys1 = set(g1.keys())
        keys2 = set(g2.keys())
        
        if keys1 != keys2:
            print(f"Different keys at {path}: {keys1 ^ keys2}")
            return False
        
        all_equal = True
        for key in keys1:
            current_path = f"{path}/{key}" if path else key
            if isinstance(g1[key], h5py.Dataset):
                if not np.allclose(g1[key][()], g2[key][()]):
                    print(f"Data differs at {current_path}")
                    all_equal = False
            elif isinstance(g1[key], h5py.Group):
                if not compare_groups(g1[key], g2[key], current_path):
                    all_equal = False
        
        return all_equal
    
    if compare_groups(f1, f2):
        print("Files are equal")
    else:
        print("Files are NOT equal")

        with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
            def compare_dataset_details(g1, g2, path=""):
                keys1 = set(g1.keys())
                keys2 = set(g2.keys())
                
                for key in keys1 & keys2:
                    current_path = f"{path}/{key}" if path else key
                    if isinstance(g1[key], h5py.Dataset):
                        data1 = g1[key][()]
                        data2 = g2[key][()]
                        
                        if data1.shape != data2.shape:
                            print(f"Shape differs at {current_path}: {data1.shape} vs {data2.shape}")
                        
                        nan_count1 = np.isnan(data1).sum()
                        nan_count2 = np.isnan(data2).sum()
                        if nan_count1 != nan_count2:
                            print(f"NaN count differs at {current_path}: {nan_count1} vs {nan_count2}")
                        
                        mask1 = ~np.isnan(data1)
                        mask2 = ~np.isnan(data2)
                        if mask1.any():
                            print(f"  {current_path} - File1: min={np.nanmin(data1):.6e}, max={np.nanmax(data1):.6e}")
                        if mask2.any():
                            print(f"  {current_path} - File2: min={np.nanmin(data2):.6e}, max={np.nanmax(data2):.6e}")
                    
                    elif isinstance(g1[key], h5py.Group):
                        compare_dataset_details(g1[key], g2[key], current_path)
            
            compare_dataset_details(f1, f2)