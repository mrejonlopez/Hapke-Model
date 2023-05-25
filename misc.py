import os

# Define input file name and output directory
input_file = 'Mastrapa.txt'
output_dir = 'Optical Constants'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define possible values for ice state and temperature
ice_states = ['Crystalline']
temperatures = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

# Loop over ice states and temperatures
for ice_state in ice_states:
    for temperature in temperatures:

        # Define output file name
        output_file = f'{ice_state}_{temperature}.txt'

        # Open output file for writing
        with open(os.path.join(output_dir, output_file), 'w') as f_out:

            # Open input file for reading
            with open(input_file, 'r') as f_in:

                # Loop over lines in input file
                for line in f_in:

                    # Check if current line matches ice state and temperature
                    if line.startswith(ice_state) and int(line[12:15]) == temperature:
                        # Extract columns 3-5 (wave, n, k) from the line
                        cols = line.split()[2:]

                        # Write extracted columns to output file
                        f_out.write('\t'.join(cols) + '\n')
