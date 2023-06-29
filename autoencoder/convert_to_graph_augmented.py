import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import os
import glob

# im not sure if I need to do this step for the graph creation - I was having problems adding data
# augmentation to the model and thought this might help with overfitting?

def pdb_to_torch_geometric(pdb_file, output_directory, distance_threshold=5.0, perturb_range=0.1):
    try:
        # Parse the PDB file
        parser = PDBParser()
        structure = parser.get_structure('pdb', pdb_file)

        # Get atom information and build a neighbor search object
        atom_list = [atom for atom in structure.get_atoms()]
        coordinates = [atom.coord for atom in atom_list]
        neighbor_search = NeighborSearch(atom_list)

        # Build edge index
        edge_index = []
        for i, atom in enumerate(atom_list):
            neighbors = neighbor_search.search(atom.coord, distance_threshold)  # Get neighbors within threshold distance
            for neighbor in neighbors:
                j = atom_list.index(neighbor)  # Get index of neighbor atom
                edge_index.append([i, j])  # Add edge

        # Perturb node positions
        perturbed_coordinates = np.array(coordinates) + np.random.uniform(-perturb_range, perturb_range, size=(len(coordinates), 3))
        perturbed_coordinates = torch.tensor(perturbed_coordinates, dtype=torch.float)

        # Create TorchGeometric data
        x = perturbed_coordinates  # Use perturbed atom coordinates as features
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transform to PyTorch format

        data = Data(x=x, edge_index=edge_index)

        # Check if the output file already exists
        output_file = os.path.join(output_directory, os.path.basename(pdb_file) + '.pt')
        if os.path.isfile(output_file):
            print(f"File {output_file} already exists. Skipping...")
            return None

        # Save the graph data
        torch.save(data, output_file)
        return data

    except Exception as e:
        print(f"Failed to process {pdb_file}. Error: {e}")
        return None


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//pdb_files_3'

# Specify the path to your output directory
output_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//pert_graphs'

# Get all pdb files in the directory
pdb_files = glob.glob(os.path.join(pdb_directory, '*.pdb'))

# Iterate over all pdb files and create corresponding graphs
print(f"Found {len(pdb_files)} PDB files.")

# Iterate over all pdb files and create corresponding graphs
for pdb_file in pdb_files:
    print(f"Processing {pdb_file}...")
    pdb_to_torch_geometric(pdb_file, output_directory=output_directory)
    print(f"Finished processing {pdb_file}.")

