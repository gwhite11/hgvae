import xml.etree.ElementTree as ET


def parse_cluster_data_from_file(file_path):
    clusters = {}
    current_cluster = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Cluster'):
                current_cluster = int(line.split()[1][:-1])  # Extracting cluster number
                clusters[current_cluster] = []
            elif line.startswith('Atom:'):
                parts = line.split('Coordinates:')
                atom_part = parts[0].split()
                atom_name = atom_part[1]
                residue_name = atom_part[3].rstrip(',')

                # Extract and parse coordinates
                coordinates_str = parts[1].strip().split('[')[1].split(']')[0]
                coordinates = [float(coord.strip()) for coord in coordinates_str.split(',')]

                atom_info = {
                    'atom': atom_name,
                    'residue': residue_name,
                    'coordinates': coordinates
                }
                clusters[current_cluster].append(atom_info)

    return clusters


def calculate_cluster_centers(clusters):
    cluster_centers = {}

    for cluster_id, atoms in clusters.items():
        if len(atoms) == 1:
            cluster_centers[cluster_id] = atoms[0]['coordinates']
        else:
            sum_coords = [0.0, 0.0, 0.0]
            for atom in atoms:
                if len(atom['coordinates']) != 3:
                    print(f"Error in cluster {cluster_id}: Atom {atom['atom']} has incorrect coordinates {atom['coordinates']}")
                    continue  # Skip this atom

                for i in range(3):
                    sum_coords[i] += atom['coordinates'][i]

            center_of_mass = [coord / len(atoms) for coord in sum_coords]
            cluster_centers[cluster_id] = center_of_mass

    return cluster_centers


file_path = 'C://Users//gemma//PycharmProjects//pythonProject1//gvae//clusters_info_15.txt'

clusters = parse_cluster_data_from_file(file_path)
cluster_centers = calculate_cluster_centers(clusters)


def parse_amber_forcefield_charges(file_path):
    atomic_properties = {}

    tree = ET.parse(file_path)
    root = tree.getroot()

    for residue in root.findall(".//Residues/Residue"):
        residue_name = residue.get('name')

        for atom in residue.findall(".//Atom"):
            atom_name = atom.get('name')
            atom_charge = float(atom.get('charge'))
            atom_type = atom.get('type')

            key = f"{atom_name}@{residue_name}"

            atomic_properties[key] = {
                'charge': atom_charge,
                'type': atom_type
            }

    return atomic_properties


file_path = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//protein.ff14SB.xml'
atomic_properties = parse_amber_forcefield_charges(file_path)


def aggregate_cluster_properties(clusters, atomic_properties):
    cluster_properties = {}
    basic_atomic_masses = {'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.0}  # Basic atomic mass lookup table

    for cluster_id, atoms in clusters.items():
        total_mass = 0.0
        total_charge = 0.0

        for atom in atoms:
            atom_name = atom['atom']
            atom_residue = atom['residue']
            key = f"{atom_name}@{atom_residue}"

            if key in atomic_properties:
                total_charge += atomic_properties[key]['charge']
                # Use only the first letter of the atom name to determine the mass
                element = atom_name[0]  # First character represents the element
                if element in basic_atomic_masses:
                    total_mass += basic_atomic_masses[element]
                else:
                    print(f"No mass data for element: {element}")
            else:
                print(f"Key not found: {key}")

        cluster_properties[cluster_id] = {
            'mass': total_mass,
            'charge': total_charge
        }

    return cluster_properties


cluster_props = aggregate_cluster_properties(clusters, atomic_properties)
print(cluster_props)