import argparse
import os
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
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    tp = args.tp
    
    eps = 1e-3
    tpert = 0.1
    dt = 0.01
    nx = 6000
    Lx = 300
    tmax = 10

    #Pe_ = [10**a for a in np.arange(6, 8.5, 0.5)]
    beta_ = [10**a for a in np.arange(-2., -0.99, 0.25)]
    #Gamma_ = [2**a for a in np.arange(-1., 3., 1) if a != 0]

    Pe_ = [1e5]
    #beta_ = [1e-3]
    Gamma_ = [1]
    
    outpvart = []
    Tvar = []
    if tp == False:
        outpvart = "output_"
        Tvar = "Tu"
    else:
        outpvart = "outppt_"
        Tvar = "Tp"
    
    for Pe in Pe_:
        for Gamma in Gamma_:
            for beta in beta_:
            
                Pe_str = f"Pe_{Pe:.10g}"
                Gamma_str = f"Gamma_{Gamma:.10g}"
                beta_str = f"beta_{beta:.10g}"
                print(Pe_str, " ", Gamma_str, " ", beta_str)
                
                folder_name = "results/" + outpvart + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
                file_path = folder_name + "gamma_linear.txt"
                if os.path.isfile(file_path) == False:
                    if os.path.exists(folder_name) == False:
                        os.mkdir(folder_name) # create folder where to save data (if it does not exist yet)
                    k_ = np.arange(0., 1.025, 0.025)
                    for k in k_:
                    
                        # Construct the command to be executed
                        command_linear_model = "python3 linear_model_" + Tvar + f".py -Pe {Pe} -k {k} -Gamma {Gamma} -beta {beta} -eps {eps} -tpert {tpert} -dt {dt} -nx {nx} -Lx {Lx} -tmax {tmax}"
                    
                        # Print the command to be executed
                        print(f"Executing: {command_linear_model}")
                    
                        # Execute the command
                        os.system(command_linear_model)
                else:
                    k_max = find_value_in_first_column_for_max_in_second(file_path)
                    print("k_max = ", k_max)
                    k_ = np.arange(max(k_max - 0.01, 0.001) , k_max + 0.0011, 0.001)
                        # for k in np.arange(0.120 , 0.140, 0.001):
                    for k in k_:
                        # Construct the command to be executed
                        command_linear_model = f"python3 linear_model_" + Tvar + f".py -Pe {Pe} -k {k} -Gamma {Gamma} -beta {beta} -eps {eps} -tpert {tpert} -dt {dt} -nx {nx} -Lx {Lx} -tmax {tmax}"
                
                        # Print the command to be executed
                        print(f"Executing: {command_linear_model}")
                
                        # Execute the command
                        os.system(command_linear_model)
                
                    # sort the values just generated in gamma_linear.txt
                    command_sort = f"python3 sort_table.py " + folder_name + "gamma_linear.txt"
                    print(f"Executing: {command_sort}")
                    os.system(command_sort)
                        
