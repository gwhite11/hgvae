import numpy as np
import torch
from torch_geometric.data import Data
import os
import glob
import networkx as nx
import pandas as pd


def distance(a, b):
    return np.linalg.norm(a - b)


def calculate_angle(coords_a, coords_b, coords_c):
    vector_ab = coords_b - coords_a
    vector_bc = coords_b - coords_c
    cosine_angle = np.dot(vector_ab, vector_bc) / (np.linalg.norm(vector_ab) * np.linalg.norm(vector_bc))
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)


# Load the CSV file into a DataFrame
csv_file_path = 'C://Users//gemma//PycharmProjects//pythonProject1//combined_output.csv'
df = pd.read_csv(csv_file_path)


def lookup_features(residue, atom_name):
    # Search for the matching row in the dataframe
    row = df[(df['Residue'] == residue) & (df['Atom Name'] == atom_name)]
    if not row.empty:
        atom_charge = row['Atom Charge'].values[0]
        epsilon = row['Epsilon'].values[0]
        sigma = row['Sigma'].values[0]
        return atom_charge, epsilon, sigma
    else:
        # Return default values if not found
        return 0.0, 0.0, 0.0


def pdb_to_pyg_data(pdb_file, output_directory):
    # Initialize graph
    G = nx.Graph()

    # Atom types and residue types
    atom_types = []
    residue_types = []
    node_idx_map = {}  # Map from PDB atom index to consecutive node index

    # Read molecule from PDB file
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_idx = int(line[6:11].strip())
                atom_symbol = line[12:16].strip()
                atom_coords = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                residue_name = line[17:20].strip()
                residue_id = int(line[22:26].strip())

                node_idx = len(G.nodes)
                node_idx_map[atom_idx] = node_idx  # Store the mapping

                G.add_node(node_idx, symbol=atom_symbol, coords=atom_coords, residue=residue_name)

                atom_types.append(atom_symbol)
                residue_types.append(residue_name)

    # Add bonds to graph using the node index mapping
    with open(pdb_file, 'r') as f:
        prev_atom_idx = None
        for line in f:
            if line.startswith('ATOM'):
                atom_idx = int(line[6:11].strip())
                if prev_atom_idx is not None:
                    start_atom_idx = node_idx_map[prev_atom_idx]
                    end_atom_idx = node_idx_map[atom_idx]
                    bond_type = "unknown"
                    bond_distance = distance(G.nodes[start_atom_idx]['coords'], G.nodes[end_atom_idx]['coords'])
                    G.add_edge(start_atom_idx, end_atom_idx, bond_type=bond_type, distance=bond_distance)
                prev_atom_idx = atom_idx

    # Getting atom charge, epsilon, and sigma for each atom
    features = []
    for residue, atom in zip(residue_types, atom_types):
        atom_charge, epsilon, sigma = lookup_features(residue, atom)
        features.append([atom_charge, epsilon, sigma])

    features = np.array(features)
    coords = np.array([G.nodes[i]['coords'] for i in G.nodes])
    node_features = np.hstack((coords, features))
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge indices
    edge_index = torch.tensor(list(G.edges)).long().t()

    # Calculate angles between connected atoms and include them as additional edge features
    for edge in G.edges:
        start_atom_idx, middle_atom_idx, end_atom_idx = edge[0], edge[1], None
        for neighbor in G.neighbors(middle_atom_idx):
            if neighbor != start_atom_idx:
                end_atom_idx = neighbor
                break

        if end_atom_idx is not None:
            start_coords = G.nodes[start_atom_idx]['coords']
            middle_coords = G.nodes[middle_atom_idx]['coords']
            end_coords = G.nodes[end_atom_idx]['coords']
            angle = calculate_angle(start_coords, middle_coords, end_coords)
            G.edges[edge]['angle'] = angle

    # Create PyTorch Geometric Data object with additional edge features (angles)
    edge_attr = [G.edges[edge].get('angle', 0.0) for edge in G.edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Save the graph data
    output_file = os.path.join(output_directory, os.path.basename(pdb_file) + '.pt')
    torch.save(data, output_file)

    return data


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig'

# Specify the path to your output directory
output_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph'

# Get all pdb files in the directory
pdb_files = glob.glob(os.path.join(pdb_directory, '*.pdb'))

# Iterate over all pdb files and create corresponding graphs
print(f"Found {len(pdb_files)} PDB files.")

for pdb_file in pdb_files:
    try:
        print(f"Processing {pdb_file}...")
        pdb_to_pyg_data(pdb_file, output_directory=output_directory)
        print(f"Finished processing {pdb_file}.")
    except RuntimeError as e:
        print(f"Error processing {pdb_file}: {str(e)}")
        continue
