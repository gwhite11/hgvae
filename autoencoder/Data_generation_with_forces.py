from openmm.app import *
from openmm import *
from openmm.unit import *
from simtk import openmm
import os
import traceback


def simulate_pdb(input_file: str, output_prefix: str, num_steps: int = 1000000, output_intervals: int = 20,
                 gpu_index: str = '0'):
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
    print('Running Production...')

    # Determine the step interval for saving .pdb files
    step_interval = num_steps // output_intervals

    for i in range(output_intervals):
        simulation.step(step_interval)
        PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(),
                          open(f'{output_prefix}_{i + 1}.pdb', 'w'))

        # Saving forces
        forces = simulation.context.getState(getForces=True).getForces()
        with open(f'{output_prefix}_{i + 1}_forces.txt', 'w') as force_file:
            for force in forces:
                force_file.write(f"{force[0]}\t{force[1]}\t{force[2]}\n")

    print("Simulation completed!")


# Call the function

input_folder = 'C://Users//gemma//PycharmProjects//pythonProject//simulations//input_data_1'
output_folder = 'C://Users//gemma//PycharmProjects//pythonProject//simulations//output_data_with_energy_1'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".pdb"):
        input_path = os.path.join(input_folder, filename)
        # We strip the file extension (.pdb) and use the name as the output prefix
        output_prefix = os.path.join(output_folder, os.path.splitext(filename)[0])

        try:
            # Call the simulate_pdb function
            simulate_pdb(input_path, output_prefix, 1000000, 20, '0')
        except Exception as e:
            print(f"Error occurred while processing file {filename}:")
            print(traceback.format_exc())
            continue

        print(f"Completed processing file {filename}")