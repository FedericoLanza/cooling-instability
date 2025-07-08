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
    parser.add_argument('--Pe', default=100, type=float, help="Peclet number")
    parser.add_argument('--Gamma', default=1, type=float, help="Heat conductivity")
    parser.add_argument('--beta', default=1e-3, type=float, help="Viscosity ratio")
    parser.add_argument('--eps', default=1e-3, type=float, help="Perturbation amplide")
    parser.add_argument('--tpert', default=0.1, type=float, help="Perturbation duration")
    parser.add_argument('--dt', default=0.005, type=float, help="Timestep")
    parser.add_argument('--tmax', default=25.0, type=float, help="Total time")
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--find_betac', action='store_true', help='Flag for generating data for finding beta critical')
    parser.add_argument('--aesthetic', action='store_true', help='Flag for generating data for the countor plot to present in the article')
    parser.add_argument('--savexmax', action='store_true', help='Flag for saving xmax')
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    tp = args.tp
    find_betac = args.find_betac
    aesthetic = args.aesthetic
    savexmax = args.savexmax
    
    dt = args.dt
    tmax = args.tmax
    eps = args.eps
    tpert = args.tpert
    
    Pe = args.Pe
    Gamma = args.Gamma
    u0 = 1.
    
    #Pe_all = np.array([10**a for a in np.arange(0., 5., 0.5)])
    Pe_all = [100]
    
    if find_betac:
        a_ = []
        #a_sat = np.arange(-2.125, -1.624, 0.125) # old one
        a_sat = np.arange(-2., -1.749, 0.0625) # new one
        if (Pe == 1):
            if (Gamma == 0.5):
                a_ = np.arange(-5.25, -4.24, 0.25)
            elif (Gamma == 1.):
                a_ = np.arange(-8.5, -7.49, 0.25)
        elif (Pe == 3.16227766):
            if (Gamma == 0.5):
                a_ = np.arange(-3.25, -2.24, 0.25)
            elif (Gamma == 1.):
                a_ = np.arange(-4., -2.99, 0.25)
            elif (Gamma == 2.):
                a_ = np.arange(-5.25, -4.24, 0.25)
            elif (Gamma == 4.):
                a_ = np.arange(-7.5, -6.49, 0.25)
        elif (Pe == 5.623413252):
            if (Gamma == 0.5):
                a_ = np.arange(-2.75, -2.24, 0.125)
            elif (Gamma == 1.):
                a_ = np.arange(-3., -2.374, 0.125)
            elif (Gamma == 2.):
                a_ = np.arange(-3.5, -2.74, 0.125)
            elif (Gamma == 4.):
                a_ = np.arange(-4.5, -3.49, 0.25)
        elif (Pe == 10):
            if (Gamma == 0.5):
                a_ = np.arange(-2.75, -1.74, 0.25)
            elif (Gamma == 1.):
                a_ = np.arange(-2.75, -1.99, 0.25)
            elif (Gamma == 2.):
                a_ = np.arange(-3., -1.99, 0.25)
            elif (Gamma == 4.):
                a_ = np.arange(-3.25, -2.24, 0.25)
        elif (Pe == 17.7827941):
            if (Gamma == 0.5):
                a_ = np.arange(-2.25, -1.749, 0.125)
            elif (Gamma == 1.):
                a_ = np.arange(-2.375, -1.876, 0.125)
            elif (Gamma == 2.):
                #a_ = np.arange(-2.5, -1.999, 0.125)
                a_ = np.arange(-2.375, -2., 0.125)
            elif (Gamma == 4.):
                a_ = np.arange(-2.625, -2.124, 0.125)
        elif (Pe == 31.6227766):
            if (Gamma == 0.5):
                a_ = a_sat
            elif (Gamma == 1.):
                a_ = np.arange(-2.25, -1.749, 0.125)
            elif (Gamma == 2.):
                a_ = np.arange(-2.25, -1.749, 0.125)
            elif (Gamma == 4.):
                a_ = np.arange(-2.25, -1.749, 0.125)
        elif (Pe == 56.23413252):
            if (Gamma == 0.5):
                a_ = a_sat
            elif (Gamma == 1.):
                a_ = np.arange(-2.125, -1.874, 0.0625)
            elif (Gamma == 2.):
                a_ = np.arange(-2.0625, -1.8124, 0.0625)
            elif (Gamma == 4.):
                a_ = np.arange(-2.25, -1.999, 0.0625)
        elif (Pe >= 100):
            a_ = a_sat
        else:
            print("not contemplated")
            exit(0)
        if tp:
            a_ += 0.375
        beta_ = [10**a for a in a_]
    else:
        beta_ = [args.beta]
    if savexmax:
        Pe_ = Pe_all
    else:
        Pe_ = [10**a for a in np.arange(3, 4, 0.125)]
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
        for Gamma in Gamma_:
        
            kappa_eff = 1./Pe + 2*Pe/105
            xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff)
            
            Lx = 10./xi
            nx = max(1000, round(5*Lx))
            k_step2 = xi/400
            
            for beta in beta_:
                k_step = k_step2*(-np.log10(beta))
                #k_step = 0.0125 # Pe=1, beta=1e-3; 2<Gamma<=4 : k_step=0.0125, 4<Gamma<=8 : k_step=0.00625
                if savexmax:
                    k = 1.
                    command_linear_model = "python3 linear_model_" + Tvar + f".py --Pe {Pe} --k {k} --Gamma {Gamma} --beta {beta} --eps {eps} --tpert {tpert} --dt {dt} --nx {nx} --Lx {Lx} --tmax {tmax} --savexmax --plot"
                    
                    # Print the command to be executed
                    print(f"Executing: {command_linear_model}")
                    
                    # Execute the command
                    os.system(command_linear_model)
                    continue
                
                Pe_str = f"Pe_{Pe:.10g}"
                Gamma_str = f"Gamma_{Gamma:.10g}"
                beta_str = f"beta_{beta:.10g}"
                folder_name = "results/" + outpvart + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
                
                if aesthetic:
                    file_path = folder_name + "gamma_linear_plot.txt"
                else:
                    file_path = folder_name + "gamma_linear.txt"
                if os.path.isfile(file_path) == False:
                    if os.path.exists(folder_name) == False:
                        os.mkdir(folder_name) # create folder where to save data (if it does not exist yet)
                    if aesthetic:
                        k_ = np.arange(0., 6.01, 0.1)
                        aesth = " --aesthetic"
                    else:
                        k_ = np.arange(0., 41*k_step, k_step)
                        aesth = ""
                    for k in k_:
                        # Construct the command to be executed
                        command_linear_model = "python3 linear_model_" + Tvar + f".py --Pe {Pe} --k {k} --Gamma {Gamma} --beta {beta} --eps {eps} --tpert {tpert} --dt {dt} --nx {nx} --Lx {Lx} --tmax {tmax} --savegamma{aesth}"
                    
                        # Print the command to be executed
                        print(f"Executing: {command_linear_model}")
                    
                        # Execute the command
                        os.system(command_linear_model)
                else:
                    k_max = find_value_in_first_column_for_max_in_second(file_path)
                    print("k_max = ", k_max)
                    k_step_small = k_step/25
                    if aesthetic:
                        k_ = np.arange(0.05, 5.5, 0.1)
                        aesth = " --aesthetic"
                    else:
                        k_ = np.arange(max(k_max - 10*k_step_small, k_step_small) , k_max + 11*k_step_small, k_step_small)
                        #k_ = np.arange(41*k_step, 51*k_step, k_step)
                        aesth = ""
                    for k in k_:
                        # Construct the command to be executed
                        command_linear_model = f"python3 linear_model_" + Tvar + f".py --Pe {Pe} --k {k} --Gamma {Gamma} --beta {beta} --eps {eps} --tpert {tpert} --dt {dt} --nx {nx} --Lx {Lx} --tmax {tmax} --savegamma{aesth}"
                
                        # Print the command to be executed
                        print(f"Executing: {command_linear_model}")
                
                        # Execute the command
                        os.system(command_linear_model)
                        
                    if aesthetic == False:
                        # sort the values just generated in gamma_linear.txt
                        command_sort = f"python3 sort_table.py " + folder_name + "gamma_linear.txt"
                        print(f"Executing: {command_sort}")
                        os.system(command_sort)
     
     
#        if (Pe < 3.16227766):
#            k_step1 = 0.005
#        elif (Pe >= 3.16227766 and Pe < 10):
#            k_step1 = 0.01
#        elif (Pe >= 10 and Pe < 31.6227766):
#            k_step1 = 0.025
#        elif (Pe >= 31.6227766 and Pe < 316.227766):
#            k_step1 = 0.05
#        elif (Pe >= 316.227766 and Pe < 31622.7766):
#            k_step1 = 0.025
#        elif (Pe >= 31622.7766 and Pe < 10**6):
#            k_step1 = 0.0025
#        elif (Pe >= 10**6 and Pe < 10**7):
#            k_step1 = 0.0005
#        else:
#            k_step1 = 0.00025
