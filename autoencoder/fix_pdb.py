def remove_ter_lines(input_pdb, output_pdb):
    with open(input_pdb, 'r') as f_in:
        lines = f_in.readlines()

        # Remove any lines that start with "TER"
        lines = [line for line in lines if not line.startswith('TER')]

        corrected_lines = []
        for line in lines:
            if line.startswith('ATOM'):
                line = line[:54] + line[60:66] + line[54:60] + line[66:]
            corrected_lines.append(line)

    with open(output_pdb, 'w') as f_out:
        f_out.writelines(corrected_lines)


# Use it after generating your PDB file
remove_ter_lines("coarse_grained_graph.pdb_7", "corrected_coarse_grained_graph.pdb")
