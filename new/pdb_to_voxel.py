from Bio.PDB import PDBParser
import numpy as np
import os


def pdb_to_voxel(pdb_file, grid_size=32, padding=1.0):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Get the bounds of the molecule
    atoms = list(structure.get_atoms())
    min_coords = np.min([atom.coord for atom in atoms], axis=0)
    max_coords = np.max([atom.coord for atom in atoms], axis=0)

    # Add padding
    min_coords -= padding
    max_coords += padding

    # Create an empty voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int8)

    # Calculate grid spacing
    grid_spacing = (max_coords - min_coords) / grid_size

    # Fill the voxel grid
    for atom in atoms:
        # Get the voxel indices for the atom
        indices = np.floor((atom.coord - min_coords) / grid_spacing).astype(int)

        # Make sure indices are within bounds
        indices = np.clip(indices, 0, grid_size - 1)

        # Update the voxel value
        voxel_grid[tuple(indices)] = 1  # or some other value based on atom properties

    return voxel_grid


# Directory paths
input_dir = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig"
output_dir = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data//chig"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through each PDB file in the directory
for pdb_filename in os.listdir(input_dir):
    if pdb_filename.endswith(".pdb"):
        pdb_filepath = os.path.join(input_dir, pdb_filename)

        # Convert PDB to voxel
        voxel_data = pdb_to_voxel(pdb_filepath)

        # Save the voxel data (using numpy's save function for simplicity)
        voxel_filename = pdb_filename.replace(".pdb", ".npy")
        voxel_filepath = os.path.join(output_dir, voxel_filename)
        np.save(voxel_filepath, voxel_data)

print("Voxel conversion completed.")

