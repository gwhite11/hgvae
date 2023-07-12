import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import os
import glob


def pdb_to_torch_geometric(pdb_file, output_directory, distance_threshold=5.0):
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

    # Create TorchGeometric data
    x = torch.tensor(np.array(coordinates), dtype=torch.float)  # Use atom coordinates as features
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transform to PyTorch format

    # Ensure edge_index is two-dimensional
    if edge_index.dim() == 1:
        edge_index = edge_index.unsqueeze(0)

    data = Data(x=x, edge_index=edge_index)

    # Save the graph data
    output_file = os.path.join(output_directory, os.path.basename(pdb_file) + '.pt')
    torch.save(data, output_file)

    return data


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//pdb_files_4'

# Specify the path to your output directory
output_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data_test_2'

# Get all pdb files in the directory
pdb_files = glob.glob(os.path.join(pdb_directory, '*.pdb'))

# Iterate over all pdb files and create corresponding graphs
print(f"Found {len(pdb_files)} PDB files.")

# Iterate over all pdb files and create corresponding graphs
for pdb_file in pdb_files:
    print(f"Processing {pdb_file}...")
    pdb_to_torch_geometric(pdb_file, output_directory=output_directory)
    print(f"Finished processing {pdb_file}.")
