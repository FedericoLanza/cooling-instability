import argparse
import math
import multiprocessing as mp
import numpy as np
import os

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
except Exception:
    rank, size = 0, 1

if rank == 0:
    print(f"[Diag] SLURM_NTASKS={os.getenv('SLURM_NTASKS')}, "
          f"SLURM_CPUS_PER_TASK={os.getenv('SLURM_CPUS_PER_TASK')}, "
          f"cpu_count={mp.cpu_count()}, MPI_size={size}")

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
    parser.add_argument('--Pe', default=1000, type=float, help="Peclet number")
    parser.add_argument('--Gamma', default=1e-5, type=float, help="Heat conductivity")
    parser.add_argument('--beta', default=1e-3, type=float, help="Viscosity ratio")
    parser.add_argument('--eps', default=1e-3, type=float, help="Perturbation amplitude")
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--find_betac', action='store_true', help='Flag for generating data for finding beta critical')
    parser.add_argument('--aesthetic', action='store_true', help='Flag for generating data for the countor plot to present in the article')
    parser.add_argument('--savexmax', action='store_true', help='Flag for saving xmax')
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    eps = args.eps
    tp = args.tp
    find_betac = args.find_betac
    aesthetic = args.aesthetic
    savexmax = args.savexmax

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
        #beta_ = [args.beta]
        beta_ =  [10**a for a in np.arange(-10., -9.99, 0.25)]
    if savexmax:
        Pe_ = Pe_all
    else:
        Pe_ = [args.Pe]
        #Pe_ = [10**a for a in np.arange(0.25, 3., 0.5)]
        #Pe_.remove(1e3)
    Gamma_ = [args.Gamma]
    #Gamma_ = [10**a for a in np.arange(-6.5, -4.99, 0.25)]
    #Gamma_.remove(10**-6.25)
    
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
        #Gamma_ = [0.01/Pe]
        for Gamma in Gamma_:
            #if Pe*Gamma > 1.5e-2:
                #continue
                
            kappa_eff = 1./Pe + 2*Pe/105
            xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff)
            Lx = 40./Gamma
            nx = 1000 # change it in case

            tmax = 20/Gamma
            dt = tmax/1e4
            tpert = dt*5
            
            for beta in beta_:
                psi = -np.log10(beta)
                if psi < 2.5:
                    Lx *= 2**(2.5 - psi)
                    nx *= int(2**(2.5 - psi))
                k_expected = xi * (-1.7135933925582293 + 2.0663122529310142*psi)
                if (abs(psi - 1.) < 1e-4):
                    k_expected *= 5
                
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
                if os.path.isfile(file_path) == False: # if the file does not exist
                    if os.path.exists(folder_name) == False:
                        os.mkdir(folder_name) # create folder where to save data (if it does not exist yet)
                    if aesthetic:
                        k_ = np.arange(0, 2.03e-4, 2e-6)
                        aesth = " --aesthetic"
                    else:
                        k_step = k_expected / 25
                        if (abs(psi - 1.5) < 1e-4):
                            k_center = 1.25*k_expected
                        elif (abs(psi - 1.25) < 1e-4):
                            k_center = 1.5*k_expected
                        elif (abs(psi - 1.) < 1e-4):
                            k_center = 1.7*k_expected
                        else:
                            k_center = k_expected
                        k_left = k_center - 13*k_step if Pe < 10 else k_center - 5*k_step
                        k_right = k_center - k_step if Pe < 10 else k_center + 3*k_step
                        
                        k_ = np.arange(k_left, k_right, k_step)
                        aesth = ""
                    for k in k_:
                        # Construct the command to be executed
                        command_linear_model = "python3 linear_model_" + Tvar + f".py --Pe {Pe} --k {k} --Gamma {Gamma} --beta {beta} --eps {eps} --tpert {tpert} --dt {dt} --nx {nx} --Lx {Lx} --tmax {tmax} --savegamma{aesth}"
                    
                        # Print the command to be executed
                        print(f"Executing: {command_linear_model}")
                    
                        # Execute the command
                        os.system(command_linear_model)
                else: # if the file exists instead
                    #continue
                    k_step = k_expected / 5
                    #k_max = find_value_in_first_column_for_max_in_second(file_path)
                    #print("k_max = ", k_max)
                    if aesthetic:
                        k_ = [a for a in np.arange(5.02e-4, 7.99e-4, 2e-6)]
                        aesth = " --aesthetic"
                    else:
                        k_ = np.arange(k_max - 3*k_step, k_max, k_step)
                        aesth = ""
                    for k in k_:
                        for nok in np.arange(5.2e-4, 8e-4, 2e-5):
                                if abs(k - nok) < 1e-4:
                                    continue
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
