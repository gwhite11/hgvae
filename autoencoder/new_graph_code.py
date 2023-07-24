from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
import os
import glob
from Bio.PDB import PDBParser, DSSP


def distance(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):  # b is the vertex
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_secondary_structure(pdb_file):
    structure = PDBParser(QUIET=True).get_structure('protein', pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    sec_structure = {}
    for key in list(dssp.keys()):
        sec_structure[key[1]] = dssp[key][2]
    return sec_structure


def pdb_to_pyg_data(pdb_file, output_directory):
    # Read molecule from PDB file
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=True)
    if mol is None:
        raise ValueError(f"Could not read PDB file: {pdb_file}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    # Initialize graph
    G = nx.Graph()

    # Atom types and residue types
    atom_types = []
    residue_types = []
    sec_structures = []

    sec_structure_dict = get_secondary_structure(pdb_file)

    # Get the first chain from the structure
    model = PDBParser(QUIET=True).get_structure('protein', pdb_file)[0]
    first_chain = list(model.get_chains())[0]

    # Add atoms to graph
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        atom_coords = np.array(atom.GetConformer().GetAtomPosition(atom_idx))
        residue_name = atom.GetPDBResidueInfo().GetResidueName()
        residue_id = atom.GetPDBResidueInfo().GetResidueNumber()

        sec_structure = sec_structure_dict.get(residue_id, "C")

        G.add_node(atom_idx, symbol=atom_symbol, coords=atom_coords, residue=residue_name, sec_structure=sec_structure)

        atom_types.append(atom_symbol)
        residue_types.append(residue_name)
        sec_structures.append(sec_structure)

    # Add bonds to graph
    for bond in mol.GetBonds():
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_distance = distance(G.nodes[start_atom_idx]['coords'], G.nodes[end_atom_idx]['coords'])
        G.add_edge(start_atom_idx, end_atom_idx, bond_type=bond_type, distance=bond_distance)

    # Calculate angles
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                atom_angle = angle(G.nodes[neighbors[i]]['coords'], G.nodes[node]['coords'],
                                   G.nodes[neighbors[j]]['coords'])
                G.edges[neighbors[i], node]['angle'] = atom_angle
                G.edges[neighbors[j], node]['angle'] = atom_angle

    # One-hot encoding
    atom_encoder = OneHotEncoder(sparse=False)
    residue_encoder = OneHotEncoder(sparse=False)
    sec_structure_encoder = OneHotEncoder(sparse=False)

    atom_features = atom_encoder.fit_transform(np.array(atom_types).reshape(-1, 1))
    residue_features = residue_encoder.fit_transform(np.array(residue_types).reshape(-1, 1))
    sec_structure_features = sec_structure_encoder.fit_transform(np.array(sec_structures).reshape(-1, 1))

    # Node features
    coords = np.array([G.nodes[i]['coords'] for i in G.nodes])
    node_features = np.hstack((coords, atom_features, residue_features, sec_structure_features))
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge indices
    edge_index = torch.tensor(list(G.edges)).t().contig()

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index)

    # Save the graph data
    output_file = os.path.join(output_directory, os.path.basename(pdb_file) + '.pt')
    torch.save(data, output_file)

    return data


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//pdb_files_4'

# Specify the path to your output directory
output_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data_OHE'

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

