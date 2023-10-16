from Bio.PDB import PDBParser
import numpy as np
import os
import re

def pdb_to_voxel(pdb_file, force_file, grid_size=32, padding=1.0):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    with open(force_file, 'r') as f:
        lines = f.readlines()

    pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
    forces = np.array([list(map(float, pattern.findall(line)[:3])) for line in lines])

    # Get the bounds of the molecule
    atoms = list(structure.get_atoms())
    min_coords = np.min([atom.coord for atom in atoms], axis=0)
    max_coords = np.max([atom.coord for atom in atoms], axis=0)

    # Add padding
    min_coords -= padding
    max_coords += padding

    # Create an empty voxel grid for both occupancy and forces
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int8)
    force_grid = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)

    # Calculate grid spacing
    grid_spacing = (max_coords - min_coords) / grid_size

    # Fill the voxel grid and the force grid
    for atom, force in zip(atoms, forces):
        # Get the voxel indices for the atom
        indices = np.floor((atom.coord - min_coords) / grid_spacing).astype(int)

        # Make sure indices are within bounds
        indices = np.clip(indices, 0, grid_size - 1)

        # Update the voxel and force values
        voxel_grid[tuple(indices)] = 1  # or some other value based on atom properties
        force_grid[tuple(indices)] = force

    return voxel_grid, force_grid


# Directory paths
input_dir = "C://Users//gemma//PycharmProjects//pythonProject1//new//input_files"
output_dir = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data_with_energy"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through each PDB file in the directory
for pdb_filename in os.listdir(input_dir):
    if pdb_filename.endswith(".pdb"):
        pdb_filepath = os.path.join(input_dir, pdb_filename)
        force_filepath = os.path.join(input_dir, pdb_filename.replace(".pdb", "_forces.txt"))

        # Convert PDB to voxel and force grids
        voxel_data, force_data = pdb_to_voxel(pdb_filepath, force_filepath)

        # Save the voxel and force data
        voxel_filename = pdb_filename.replace(".pdb", "_voxel.npy")
        force_filename = pdb_filename.replace(".pdb", "_force.npy")

        voxel_filepath = os.path.join(output_dir, voxel_filename)
        force_filepath = os.path.join(output_dir, force_filename)

        np.save(voxel_filepath, voxel_data)
        np.save(force_filepath, force_data)

print("Voxel and force conversion completed.")
