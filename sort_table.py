import numpy as np
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
    # Check if a file path is provided
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 sort_table.py path/to/your/file.txt [tolerance]")
    else:
        # Get the file path from command-line arguments
        file_path = sys.argv[1]
        
        # Optionally get the tolerance from the command line, or use the default
        tol = float(sys.argv[2]) if len(sys.argv) == 3 else 1e-6
        
        # Sort and filter the file
        sort_and_filter_file(file_path, tol)
