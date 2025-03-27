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
    parser.add_argument('--Pe', default=31.6227766, type=float, help="Peclet number")
    parser.add_argument('--Gamma', default=4.0, type=float, help="Heat conductivity")
    # parser.add_argument("--beta", default=1e-1, type=float, help="Viscosity ratio")
    parser.add_argument('--eps', default=1e-3, type=float, help="Perturbation amplide")
    parser.add_argument('--tpert', default=0.1, type=float, help="Perturbation duration")
    parser.add_argument('--dt', default=0.005, type=float, help="Timestep")
    parser.add_argument('--nx', default=1000, type=int, help="Number of mesh points")
    parser.add_argument('--Lx', default=25, type=float, help="System size")
    parser.add_argument('--tmax', default=10.0, type=float, help="Total time")
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--aesthetic', action='store_true', help='Flag for generating data for the plots to present in the article')
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    tp = args.tp
    aesthetic = args.aesthetic
    
    dt = args.dt
    nx_Gamma_1 = []
    Lx_Gamma_1 = []
    tmax = args.tmax
    eps = args.eps
    tpert = args.tpert
    
    Pe = args.Pe
    Gamma = args.Gamma
    # beta = args.beta
    a_ = []
    a_sat = np.arange(-2.125, -1.624, 0.125)
    if (Pe == 1):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 15
        if (Gamma == 0.5):
            a_ = np.arange(-5.25, -4.24, 0.25)
        elif (Gamma == 1.):
            a_ = np.arange(-8.5, -7.49, 0.25)
        else:
            print("not contemplated")
            exit(0)
    elif (Pe == 3.16227766):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 20
        if (Gamma == 0.5):
            a_ = np.arange(-3.25, -2.24, 0.25)
        elif (Gamma == 1.):
            a_ = np.arange(-4., -2.99, 0.25)
        elif (Gamma == 2.):
            a_ = np.arange(-5.25, -4.24, 0.25)
        elif (Gamma == 4.):
            a_ = np.arange(-7.5, -6.49, 0.25)
    elif (Pe == 10):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 20
        if (Gamma == 0.5):
            a_ = np.arange(-2.75, -1.74, 0.25)
        elif (Gamma == 1.):
            a_ = np.arange(-2.75, -1.99, 0.25)
        elif (Gamma == 2.):
            a_ = np.arange(-3., -1.99, 0.25)
        elif (Gamma == 4.):
            a_ = np.arange(-3.25, -2.24, 0.25)
    elif (Pe == 31.6227766):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 25
        if (Gamma == 0.5):
            a_ = a_sat
        elif (Gamma == 1.):
            a_ = np.arange(-2.25, -1.749, 0.125)
        elif (Gamma == 2.):
            a_ = np.arange(-2.25, -1.749, 0.125)
        elif (Gamma == 4.):
            a_ = np.arange(-2.25, -1.749, 0.125)
    elif (Pe == 100):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 50
        a_ = a_sat
    elif (Pe == 316.227766):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 50
        a_ = a_sat
    elif (Pe == 1000):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 75
        a_ = a_sat
    elif (Pe == 3162.27766):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 100
        a_ = a_sat
    elif (Pe == 10000):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 150
        a_ = a_sat
    elif (Pe == 31622.7766):
        nx_Gamma_1 = 1000
        Lx_Gamma_1 = 200
        a_ = a_sat
    elif (Pe == 100000):
        nx_Gamma_1 = 2000
        Lx_Gamma_1 = 400
        a_ = a_sat
    elif (Pe == 316227.766):
        nx_Gamma_1 = 4000
        Lx_Gamma_1 = 800
        a_ = a_sat
    elif (Pe == 1000000):
        nx_Gamma_1 = 6000
        Lx_Gamma_1 = 1200
        a_ = a_sat
    elif (Pe == 3162277.66):
        nx_Gamma_1 = 8000
        Lx_Gamma_1 = 1600
        a_ = a_sat
    else:
        print("not contemplated")
        exit(0)
        
    if tp:
        a_ += 0.375
    
    if aesthetic:
        beta_ = [beta]
    else:
        beta_ = [10**a for a in a_]

    Pe_ = [Pe]
    Gamma_ = [Gamma]
    
    outpvart = []
    Tvar = []
    if tp == False:
        outpvart = "output_"
        Tvar = "Tu"
    else:
        outpvart = "outppt_"
        Tvar = "Tp"
    
    # epsilon = np.finfo(float).eps
    
    for Pe in Pe_:
        k_step1 = []
        if (Pe < 3.16227766):
            k_step1 = 0.005
        elif (Pe >= 3.16227766 and Pe < 10):
            k_step1 = 0.01
        elif (Pe >= 10 and Pe < 31.6227766):
            k_step1 = 0.025
        elif (Pe >= 31.6227766 and Pe < 316.227766):
            k_step1 = 0.05
        elif (Pe >= 316.227766 and Pe < 31622.7766):
            k_step1 = 0.025
        elif (Pe >= 31622.7766 and Pe < 10**6):
            k_step1 = 0.0025
        elif (Pe >= 10**6 and Pe < 10**7):
            k_step1 = 0.0005
        else:
            k_step1 = 0.00025
            
        for Gamma in Gamma_:
            Lx = Lx_Gamma_1/np.sqrt(Gamma)
            nx = round(nx_Gamma_1/np.sqrt(Gamma))
            k_step2 = k_step1*np.sqrt(Gamma)
            
            for beta in beta_:
                k_step = k_step2*(-np.log10(beta))
                
                Pe_str = f"Pe_{Pe:.10g}"
                Gamma_str = f"Gamma_{Gamma:.10g}"
                beta_str = f"beta_{beta:.10g}"
                print(Pe_str, " ", Gamma_str, " ", beta_str)
                
                folder_name = "results/" + outpvart + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
                if aesthetic:
                    file_path = folder_name + "gamma_linear_plot.txt"
                else:
                    file_path = folder_name + "gamma_linear.txt"
                if os.path.isfile(file_path) == False:
                    if os.path.exists(folder_name) == False:
                        os.mkdir(folder_name) # create folder where to save data (if it does not exist yet)
                    if aesthetic:
                        k_ = np.arange(0., 15.01, 0.1)
                    else:
                        k_ = np.arange(0., 41*k_step, k_step)
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
                    k_step_small = k_step/25
                    if aesthetic:
                        k_ = np.arange(0.05, 15., 0.1)
                    else:
                        k_ = np.arange(max(k_max - 10*k_step_small, k_step_small) , k_max + 11*k_step_small, k_step_small)
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
                        
