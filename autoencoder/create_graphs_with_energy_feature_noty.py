import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
import os
import glob
import networkx as nx
import re


def load_energies_for_pdb(pdb_file):
    energies_file = pdb_file.replace('.pdb', '_forces.txt')
    pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

    energies = []
    with open(energies_file, 'r') as f:
        for line in f:
            energies.append(list(map(float, pattern.findall(line)[:3])))

    return np.array(energies)


def distance(a, b):
    return np.linalg.norm(a - b)


def calculate_angle(coords_a, coords_b, coords_c):
    vector_ab = coords_b - coords_a
    vector_bc = coords_b - coords_c
    cosine_angle = np.dot(vector_ab, vector_bc) / (np.linalg.norm(vector_ab) * np.linalg.norm(vector_bc))
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)


def get_unique_types(pdb_files):
    unique_atom_types = set()
    unique_residue_types = set()

    for pdb_file in pdb_files:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_symbol = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    unique_atom_types.add(atom_symbol)
                    unique_residue_types.add(residue_name)

    return list(unique_atom_types), list(unique_residue_types)


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//new//input_files_2'

# Get all pdb files in the directory
pdb_files = glob.glob(os.path.join(pdb_directory, '*.pdb'))

# Get unique atom and residue types from all PDB files
all_atom_types, all_residue_types = get_unique_types(pdb_files)

# Initialize encoders and fit them on all atom and residue types
atom_encoder = OneHotEncoder(categories=[all_atom_types], sparse=False)
residue_encoder = OneHotEncoder(categories=[all_residue_types], sparse=False)
atom_encoder.fit(np.array(all_atom_types).reshape(-1, 1))
residue_encoder.fit(np.array(all_residue_types).reshape(-1, 1))


def pdb_to_pyg_data(pdb_file, output_directory):
    # Initialize graph
    G = nx.Graph()

    # Atom types and residue types
    atom_types = []
    residue_types = []
    node_idx_map = {}  # Map from PDB atom index to consecutive node index
    atom_to_label = {'C': 0, 'O': 1, 'H': 2, 'N': 3}
    atom_labels = []

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

                atom_label = atom_to_label.get(atom_symbol, -1)
                atom_labels.append(atom_label)

    # Add bonds to graph using the node index mapping
    with open(pdb_file, 'r') as f:
        prev_atom_idx = None
        for line in f:
            if line.startswith('ATOM'):
                atom_idx = int(line[6:11].strip())
                if prev_atom_idx is not None:
                    start_atom_idx = node_idx_map[prev_atom_idx]
                    end_atom_idx = node_idx_map[atom_idx]
                    bond_type = "unknown"  # Since the PDB file does not contain bond information, we set it to unknown
                    bond_distance = distance(G.nodes[start_atom_idx]['coords'], G.nodes[end_atom_idx]['coords'])
                    G.add_edge(start_atom_idx, end_atom_idx, bond_type=bond_type, distance=bond_distance)
                prev_atom_idx = atom_idx

    # One-hot encoding
    atom_features = atom_encoder.transform(np.array(atom_types).reshape(-1, 1))
    residue_features = residue_encoder.transform(np.array(residue_types).reshape(-1, 1))

    # Node features
    coords = np.array([G.nodes[i]['coords'] for i in G.nodes])
    # Now, add energies as a node feature
    energies = load_energies_for_pdb(pdb_file)
    # Ensure we have the same number of energy entries as nodes
    assert len(G.nodes) == len(energies), "Mismatch between number of atoms and energies."

    # Node features include the atom type one-hot encodings, residue type one-hot encodings, and energies.
    node_features = np.hstack([atom_features, residue_features, energies])
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge indices and edge features (relative distances)
    edge_index = []
    edge_attr = []
    for start_atom_idx, end_atom_idx, data in G.edges(data=True):
        edge_index.append([start_atom_idx, end_atom_idx])
        edge_attr.append([data['distance']])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create PyTorch Geometric Data object without `y` for energies since they are now part of node features
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Save the graph data
    output_file = os.path.join(output_directory, os.path.basename(pdb_file) + '.pt')
    torch.save(data, output_file)

    return data


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig_with_forces'

# Specify the path to your output directory
output_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_energy'

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
