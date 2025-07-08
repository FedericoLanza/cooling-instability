import argparse
import os
import math
import numpy as np

def find_value_in_first_column_for_max_in_second(filename):
    with open(filename, 'r') as file:
        # Skip the header
        next(file)
        max_value = float('-inf')
        corresponding_first_column_value = None
        
        # Process each line, splitting by tab
        for line in file:
            columns = line.split()
            first_column_value = float(columns[0])
            second_column_value = float(columns[1])
            
            # Update max_value and corresponding_first_column_value if we find a new max
            if second_column_value > max_value:
                max_value = second_column_value
                corresponding_first_column_value = first_column_value
                
    return corresponding_first_column_value

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--x_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The x variable')
    parser.add_argument('--fixed_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The fixed variable')
    parser.add_argument('--vary_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The vary variable')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--nofit', action='store_true', help='Flag for choosing whether to fit the points around the maximum')

    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    x_variable = args.x_variable
    fixed_variable = args.fixed_variable
    vary_variable = args.vary_variable
    tp = args.tp
    nofit = args.nofit
    
    if args.x_variable == args.vary_variable or args.x_variable == args.fixed_variable or args.vary_variable == args.fixed_variable:
        raise ValueError("The x_variable, vary_variable, and fixed_variable must all be different. Please run the program again accordingly.")
    
    if x_variable == "Pe":
        Pe_ = [10**a for a in np.arange(0., 5.501, 0.25)]
    elif x_variable == "Gamma":
        Gamma_ = [2**a for a in np.arange(1.25, 3.01, 0.25)]
    elif x_variable == "beta":
        beta_ = [10**a for a in np.arange(-4., -0.99, 0.25)]
    
    if fixed_variable == "Pe":
        Pe_ = [100]
    elif fixed_variable == "Gamma":
        Gamma_ = [1]
    elif fixed_variable == "beta":
        beta_ = [1e-3]
    
    if vary_variable == "Pe":
        Pe_ = [0]
    elif vary_variable == "Gamma":
        Gamma_ = [0]
    elif vary_variable == "beta":
        beta_ = [0]
    
    for Pe in Pe_:
        for Gamma in Gamma_:
            for beta in beta_:
                Pe_str = f" --Pe {Pe:.10g}"
                Gamma_str = f" --Gamma {Gamma:.10g}"
                beta_str = f" --beta {beta:.10g}"
                command_linear_model = f"python3 plot_disp_rels.py"
                if vary_variable != "Pe": command_linear_model += Pe_str
                if vary_variable != "Gamma": command_linear_model += Gamma_str
                if vary_variable != "beta": command_linear_model += beta_str
                if tp: command_linear_model += " --tp"
                if nofit: command_linear_model += " --nofit"
                
                # Print the command to be executed
                print(f"Executing: {command_linear_model}")
                    
                # Execute the command
                os.system(command_linear_model)
