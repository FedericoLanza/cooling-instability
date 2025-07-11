import numpy as np

def filter_multiples(filename, tolerance=1e-8):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Preserve the first empty row
    first_line = lines[0]
    data_lines = lines[1:]
    
    # Parse the file content into a list of lists
    data = [list(map(float, line.split())) for line in data_lines]
    
    # Extract the step size from the second row of the data (third row of the file)
    step = data[1][0]
    
    # Filter rows where the first column is a multiple of step within tolerance
    filtered_data = [row for row in data if abs((row[0] / step) - round(row[0] / step)) < tolerance]
    
    # Write the filtered data back to the file, maintaining the first empty row
    with open(filename, 'w') as f:
        f.write(first_line)
        for row in filtered_data:
            f.write('\t'.join(map(str, row)) + '\n')

# Example usage

Pe_ = [0.0316227766]
Gamma_ = [1]
beta_ = [1e-1]
for Pe in Pe_:
    for Gamma in Gamma_:
        for beta in beta_:
            filename = f"results/output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}/gamma_linear.txt"  # Change this to your actual file path
            filter_multiples(filename)

