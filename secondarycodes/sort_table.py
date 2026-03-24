import numpy as np
import os
import sys

# Load the table from the .txt file as strings and skip the header
def load_table_as_strings(file_path):
    with open(file_path, 'r') as f:
        # Skip the header and read the rest of the data as strings
        data = [line.strip().split() for line in f.readlines()[1:]]
    return data

# Load the header separately
def load_header(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()  # Read the first line (header)
    return header

# Convert the data to numerical form for sorting but keep original strings for writing
def sort_table(data):
    # Convert each row to floats for sorting, but keep the original strings
    data_sorted = sorted(data, key=lambda row: [float(x) for x in row])
    return data_sorted

# Remove duplicates based on the first column using a relative tolerance
def remove_duplicates(data, tol=1e-6):
    unique_data = []
    prev_value = None

    for row in data:
        first_value = float(row[0])
        
        if prev_value is None or not np.isclose(first_value, prev_value, rtol=tol):
            unique_data.append(row)
            prev_value = first_value
    
    return unique_data

# Save the sorted and filtered table back to the file, keeping the original precision
def save_table(file_path, sorted_data, header):
    with open(file_path, 'w') as f:
        f.write(header + '\n')  # Write the header back to the file
        for row in sorted_data:
            f.write("\t".join(row) + '\n')  # Write each row in its original string form

# Main function to handle the sorting, removing duplicates, and saving
def sort_and_filter_file(file_path, tol=1e-6):
    # Load the header (first row) and data (remaining rows as strings)
    header = load_header(file_path)
    data = load_table_as_strings(file_path)
    
    # Sort the table by all columns
    sorted_data = sort_table(data)
    
    # Remove duplicates based on the first column with tolerance
    unique_data = remove_duplicates(sorted_data, tol=tol)
    
    # Save the sorted and filtered data back to the file with the header
    save_table(file_path, unique_data, header)

if __name__ == "__main__":
    Pe_ = [10**a for a in np.arange(3., 3.001, 0.125)]
    Gamma_ = [10**a for a in np.arange(-5., -4.99, 0.0625)]
    beta_ = [10**a for a in np.arange(-4.9375, -4.9, 0.125)]
    for Pe in Pe_:
        for Gamma in Gamma_:
            for beta in beta_:
                print('Pe = ', Pe, 'Gamma = ', Gamma, 'beta = ', beta)
                start_folder = f"results/output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}"
                destination_folder = f"results/output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}"
                filename = "gamma_linear_plot.txt"  # Change this to your actual file path
                start_filepath = start_folder + "/" + filename
                destination_filepath = destination_folder + "/" + filename
                
                if os.path.isfile(start_filepath):
                    if os.path.exists(destination_folder)==False:
                        os.mkdir(destination_folder)
                    data = load_table_as_strings(start_filepath)
                    sorted_data = sort_table(data)
                    
                    if len(data) > 1:
                        #Move the files from results/results to results
                        command_line = f"mv results/results/output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}/gamma_linear_plot.txt " + destination_folder
                        os.system(command_line)
                    else:
                        with open(destination_filepath, "a") as file:
                            for row in sorted_data:
                                file.write("\t".join(row) + '\n')
                    
                    # Sort and filter the file
                    sort_and_filter_file(destination_folder + "/" + filename, 1e-6)
                    print(f"Columns sorted (while filtering duplicates) and file '{filename}' overwritten.")
