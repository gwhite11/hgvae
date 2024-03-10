from Bio.PDB import PDBParser
import numpy as np
import os

def pdb_to_voxel(pdb_file, grid_size=32, padding=1.0):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Define the atom types
    atom_types = ["C", "N", "O", "H"]
    num_atom_types = len(atom_types)

    # Get the bounds of the molecule
    atoms = list(structure.get_atoms())
    min_coords = np.min([atom.coord for atom in atoms], axis=0)
    max_coords = np.max([atom.coord for atom in atoms], axis=0)

    # Add the padding
    min_coords -= padding
    max_coords += padding

    # Create an empty voxel grid and label grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int8)
    label_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int8)  # Changed to an integer grid

    # Calculate grid spacing
    grid_spacing = (max_coords - min_coords) / grid_size

    for atom in atoms:
        atom_name = atom.get_name()[0]
        if atom_name in atom_types:
            indices = np.floor((atom.coord - min_coords) / grid_spacing).astype(int)
            indices = np.clip(indices, 0, grid_size - 1)
            voxel_grid[tuple(indices)] = 1
            label_idx = atom_types.index(atom_name)
            label_grid[indices[0], indices[1], indices[2]] = label_idx  # Changed to set the class label as an integer

    return voxel_grid, label_grid


# Directory paths
input_dir = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//pdb_files_5"
output_dir = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data_with_labels"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# In the loop that processes the PDB files:
for pdb_filename in os.listdir(input_dir):
    if pdb_filename.endswith(".pdb"):
        pdb_filepath = os.path.join(input_dir, pdb_filename)

        # Convert PDB to voxel and labels
        voxel_data, label_data = pdb_to_voxel(pdb_filepath)

        # Save the voxel and label data
        voxel_filename = pdb_filename.replace(".pdb", "_voxel.npy")
        label_filename = pdb_filename.replace(".pdb", "_label.npy")
        voxel_filepath = os.path.join(output_dir, voxel_filename)
        label_filepath = os.path.join(output_dir, label_filename)
        np.save(voxel_filepath, voxel_data)
        np.save(label_filepath, label_data)

print("Voxel and label conversion completed.")