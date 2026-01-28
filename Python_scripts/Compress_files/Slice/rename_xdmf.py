import os


# test 
def replace_in_file(file_path, old_text, new_text):
    # Read the content of the file
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Replace the old text with the new text
    new_content = file_content.replace(old_text, new_text)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
def replace_in_files(folder_path, old_text, new_text):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .xdmf file
        if filename.endswith('.xdmf'):
            file_path = os.path.join(folder_path, filename)
            # Perform replacement in the file
            replace_in_file(file_path, old_text, new_text)

# Replace "snapshots_x/slice_" with "slice_" in all .xdmf files within a folder
folder_path = '/gpfs/scratch/upc108/EDU/NACA_0012_AOA12_Re50000_1716x1662x128/slices_data/slice_test/'
old_text = 'slices_data/slice_1/slice_'
new_text = 'slice_'
replace_in_files(folder_path, old_text, new_text)
