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

Pe_ = [10**a for a in np.arange(0.5, 4.5, 1)]
beta_ = [10**a for a in np.arange(-5., -1., 1)]
#Gamma_ = [5*a for a in np.arange(1, 5, 1)]

#Pe_ = [1, 1000]
#beta_ = [0.0001]
Gamma_ = [1]

#print("Pe_ = ", Pe_)
#print("beta_ = ", beta_)
#exit(0)

for Pe in Pe_:
    for Gamma in Gamma_:
        for beta in beta_:
        
            # create folder where to save data (if it does not exists yet)
            Pe_str = f"Pe_{Pe:.10g}"
            Gamma_str = f"Gamma_{Gamma:.10g}"
            beta_str = f"beta_{beta:.10g}"
            folder_name = f"results/output_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
            if os.path.exists(folder_name) == False:
                os.mkdir(folder_name)
            #else:
                #if os.file.exists(folder_name + "gamma_linear.txt")
                #k_max =
            for k in np.arange(0., 10.25, 0.25):
            
                # Construct the command to be executed
                command = f"python3 linear_model_Tu.py {Pe} {k} {Gamma} {beta} {eps} {tpert} {dt} {nx} {Lx} {tmax}"
                
                # Print the command to be executed
                print(f"Executing: {command}")
                
                # Execute the command
                os.system(command)
