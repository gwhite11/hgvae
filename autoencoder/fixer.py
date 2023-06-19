import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# Define the input folder and output folder paths
input_folder = 'input_files'
output_folder = 'output_files'

# Iterate through the PQR files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".pqr"):
        # Load the RNA PQR file
        pqr_path = os.path.join(input_folder, filename)
        fixer = PDBFixer(filename=pqr_path)

        # Check for duplicate atom warning
        try:
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens()
        except UserWarning as warning:
            print(f"Skipping file {filename} due to duplicate atom warning: {warning}")
            continue

        # Generate the output PDB file path
        pdb_name = os.path.splitext(filename)[0] + '.pdb'
        pdb_path = os.path.join(output_folder, pdb_name)

        # Write the RNA to a PDB file
        with open(pdb_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)