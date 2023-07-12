from Bio.PDB import PDBParser
import torch


def rebuild_molecule(original_pdb_file, coarse_grained_pdb_file, output_pdb_file):
    # Load the original PDB file
    parser = PDBParser()
    structure = parser.get_structure('pdb', original_pdb_file)

    # Load the coarse-grained PDB file
    coarse_grained_structure = parser.get_structure('pdb', coarse_grained_pdb_file)

    # Extract atom information from the original PDB file
    original_atoms = []
    for atom in structure.get_atoms():
        original_atoms.append({
            'atom_name': atom.get_name(),
            'atom_type': atom.get_id(),
            'residue_name': atom.get_parent().get_resname(),
            'residue_id': atom.get_parent().get_id(),
            'coord': atom.coord
        })

    # Extract coordinates from the coarse-grained PDB file
    coarse_grained_coords = []
    for atom in coarse_grained_structure.get_atoms():
        if isinstance(atom.coord, torch.Tensor):  # Skip non-coordinate nodes
            coarse_grained_coords.append(atom.coord.numpy())

    # Check if the number of atoms matches the number of coarse-grained nodes
    if len(original_atoms) != len(coarse_grained_coords):
        raise ValueError("Mismatch in the number of atoms and coarse-grained nodes.")

    # Update the original atom coordinates with the coarse-grained coordinates
    for i, atom in enumerate(original_atoms):
        atom['coord'] = coarse_grained_coords[i]

    # Write the updated atom information to a new PDB file for the coarse-grained version
    with open(output_pdb_file, 'w') as f:
        for atom in original_atoms:
            atom_line = f"ATOM  {atom['atom_type']:4s} {atom['atom_name']:4s} {atom['residue_name']:3s} " \
                        f" {atom['residue_id']:4s}    {atom['coord'][0]:8.3f}{atom['coord'][1]:8.3f}{atom['coord'][2]:8.3f}\n"
            f.write(atom_line)


# for example
original_pdb_file = 'path/to/original.pdb'
coarse_grained_pdb_file = 'path/to/coarse_grained.pdb'
output_pdb_file = 'path/to/reconstructed.pdb'

rebuild_molecule(original_pdb_file, coarse_grained_pdb_file, output_pdb_file)
