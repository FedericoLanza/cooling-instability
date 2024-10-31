import os
import numpy as np

#Pe = 100
#k = 2
#Gamma = 1
#beta = 0.001
eps = 1e-3
tpert = 0.1
dt = 0.01
nx = 1000
Lx = 50
tmax = 10

#Pe_ = [10**a for a in np.arange(0., 4., 1)]
#beta_ = [10**a for a in np.arange(-4.5, 0., 1)]
#Gamma_ = [2**a for a in np.arange(-1., 3., 1)]

Pe_ = [1, 10, 1000]
beta_ = [0.1]
Gamma_ = [1]

#print("Pe_ = ", Pe_)
#print("beta_ = ", beta_)
#exit(0)

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

    
for Pe in Pe_:
    for Gamma in Gamma_:
        for beta in beta_:
        
            Pe_str = f"Pe_{Pe:.10g}"
            Gamma_str = f"Gamma_{Gamma:.10g}"
            beta_str = f"beta_{beta:.10g}"
            print(Pe_str, " ", Gamma_str, " ", beta_str)
            
            folder_name = f"results/output_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
            if os.path.exists(folder_name) == False:
                os.mkdir(folder_name) # create folder where to save data (if it does not exist yet)
                for k in np.arange(0., 10.25, 0.25):
                
                    # Construct the command to be executed
                    command_linear_model = f"python3 linear_model_Tu.py -Pe {Pe} -k {k} -Gamma {Gamma} -beta {beta} -eps {eps} -tpert {tpert} -dt {dt} -nx {nx} -Lx {Lx} -tmax {tmax}"
                
                    # Print the command to be executed
                    print(f"Executing: {command_linear_model}")
                
                    # Execute the command
                    os.system(command_linear_model)
            else:
                file_path = folder_name + "gamma_linear.txt"
                if os.path.isfile(file_path):
                    k_max = find_value_in_first_column_for_max_in_second(file_path)
                    print("k_max = ", k_max)
                    for k in np.arange(max(k_max - 0.1,0.01) , k_max + 0.11, 0.01):
                        # Construct the command to be executed
                        command_linear_model = f"python3 linear_model_Tu.py -Pe {Pe} -k {k} -Gamma {Gamma} -beta {beta} -eps {eps} -tpert {tpert} -dt {dt} -nx {nx} -Lx {Lx} -tmax {tmax}"
                
                        # Print the command to be executed
                        print(f"Executing: {command_linear_model}")
                
                        # Execute the command
                        os.system(command_linear_model)
                
                    # sort the values just generated in gamma_linear.txt
                    command_sort = f"python3 sort_table.py " + folder_name + "gamma_linear.txt"
                    print(f"Executing: {command_sort}")
                    os.system(command_sort)
                    
