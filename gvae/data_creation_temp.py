
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
import glob
import networkx as nx
import re
from openmm.app import *
from openmm import *
from openmm.unit import *
from simtk import openmm
import os
import traceback


# Need code to collect pdb files - need to get a good representation of every family of protein


# Need to add code to use pdb2pqr on the files selecting parameters amber force field, 300k and pH7


# This code gets the data into the correct format for the simulation using pdb fixer and transform file back from a pqr file type
# Define the input folder and output folder paths
input_folder = 'input_files_11'
output_folder = 'output_files_11'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process PDB files
def process_pdb_file(filepath, output_folder):
    try:
        fixer = PDBFixer(filename=filepath)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens()
    except Exception as e:  # Catching all exceptions to print the error and continue
        print(f"Error processing {filepath}: {e}")
        return

    # Generate the output PDB file path
    output_pdb_name = "fixed_" + os.path.basename(filepath)
    output_pdb_path = os.path.join(output_folder, output_pdb_name)

    # Write the fixed structure to a new PDB file
    with open(output_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

# Iterate through the files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if filename.endswith(".pdb"):
        print(f"Processing PDB file: {filename}...")
        process_pdb_file(file_path, output_folder)
    elif filename.endswith(".pqr"):
        print(f"Converting PQR to PDB: {filename}...")
        # Rename the .pqr file to .pdb
        renamed_file_path = os.path.join(input_folder, filename.replace('.pqr', '.pdb'))
        os.rename(file_path, renamed_file_path)
        # Process the renamed file
        process_pdb_file(renamed_file_path, output_folder)


def simulate_pdb(input_file: str, output_prefix: str, iteration: int, num_steps: int = 1000000, output_intervals: int = 20, gpu_index: str = '0'):
    # Load your protein
    pdb = PDBFile(input_file)
    modeller = Modeller(pdb.topology, pdb.positions)

    # set up
    forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml')
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

    platform = openmm.Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': gpu_index}
    integrator = LangevinIntegrator(300 * kelvin, 1 / picoseconds, 0.002 * picoseconds)
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)

    # Set the initial positions from the Modeller object
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    # Production (folding)
    print(f'Running Production for iteration {iteration}...')

    # Determine the step interval for saving .pdb files
    step_interval = num_steps // output_intervals

    for i in range(output_intervals):
        simulation.step(step_interval)
        PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(),
                          open(f'{output_prefix}_{iteration}_{i + 1}.pdb', 'w'))

        # Saving forces
        forces = simulation.context.getState(getForces=True).getForces()
        with open(f'{output_prefix}_{iteration}_{i + 1}_forces.txt', 'w') as force_file:
            for force in forces:
                force_file.write(f"{force[0]}\t{force[1]}\t{force[2]}\n")

    print(f"Simulation {iteration} completed!")


# Call the function
input_folder = 'C://Users//gemma//PycharmProjects//pythonProject//simulations//input_temp'
output_folder = 'C://Users//gemma//PycharmProjects//pythonProject//simulations//output_temp'

# Create the output folder if it doesn't already exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Number of simulations per protein
num_simulations = 20

# Iterate over all the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".pdb"):
        input_path = os.path.join(input_folder, filename)
        output_prefix = os.path.join(output_folder, os.path.splitext(filename)[0])

        for iteration in range(1, num_simulations + 1):
            try:
                # Call the simulate_pdb function for each iteration
                simulate_pdb(input_path, output_prefix, iteration, 100000, 20, '0')
            except Exception as e:
                print(f"Error occurred in iteration {iteration} while processing file {filename}:")
                print(traceback.format_exc())
                continue

            print(f"Completed iteration {iteration} for file {filename}")



# This code turns the simulation data into a graph object:
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
        print(f"Processing {pdb_file}...")  # Debug print
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_symbol = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    unique_atom_types.add(atom_symbol)
                    unique_residue_types.add(residue_name)

    print(f"Unique atom types: {unique_atom_types}")  # Debug print
    print(f"Unique residue types: {unique_residue_types}")  # Debug print

    return list(unique_atom_types), list(unique_residue_types)


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_energy_6'

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

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Save the graph data
    output_file = os.path.join(output_directory, os.path.basename(pdb_file) + '.pt')
    torch.save(data, output_file)

    return data


# Specify the path to your directory with pdb files
pdb_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_energy_6'

# Specify the path to your output directory
output_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graphs_energy_no_y'

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

