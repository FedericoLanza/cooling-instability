import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read the last value from the second column of a file
def read_table_all_values(file_path):
    x = []
    y = []
    with open(file_path, 'r') as file:
        # Skip the first line (the header)
        next(file)
    
        # Read the remaining lines
        for line in file:
            # Split the line into two parts (x and y values)
            values = line.split()
        
            # Convert to integers (or floats if needed) and append to respective lists
            x.append(float(values[0]))
            y.append(float(values[1]))
            
    return x, y


def read_table_const_spacing(file_path, spacing=0.0001, tolerance=1e-6):
    x = []
    y = []
    
    with open(file_path, 'r') as file:
        # Skip the first line (the header)
        next(file)
    
        # Read the remaining lines
        for line in file:
            # Split the line into parts (k, gamma, err_gamma)
            values = line.split()
            k_val = float(values[0])
            gamma_val = float(values[1])
            
            x.append(k_val)
            y.append(gamma_val)

    # Detect interval where k values are spaced by approx 0.01
    interval_x = []
    interval_y = []
    
    i = 1
    while i < len(x):
        if abs(x[i] - x[i-1] - spacing) < tolerance:
            # Start new interval if it's the first point or a continuation of the interval
            if not interval_x:
                interval_x.append(x[i-1])
                interval_y.append(y[i-1])
            
            # Add the current point to the interval
            interval_x.append(x[i])
            interval_y.append(y[i])
        else:
            # Stop once the interval breaks
            if interval_x:
                break
        i += 1

    return interval_x, interval_y

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('-Pe', type=float, help='Value for Peclet number')
    parser.add_argument('-Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('-beta', type=float, help='Value for viscosity ratio')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    
    return parser.parse_args()

if __name__ == "__main__":

    # Parse the command-line arguments
    args = parse_args() # object containing the values of the parsed argument
    
    Pe = args.Pe
    Gamma = args.Gamma
    beta = args.beta
    
    tp = args.tp
    
    ueps = 0.001
    Lx = 50
    rnd = False
    holdpert = False
    
    u0 = 1.0 # base inlet velocity
    
    # Input paths
    file_gamma_full = f"gamma_full.txt"
    file_gamma_linear = f"gamma_linear.txt"
    
    Pe_ = []
    beta_ = []
    Gamma_ = []
    
    Pe_str = None
    beta_str = None
    Gamma_str = None
    whos_none = None # 0: Pe is None, 1: beta is None, 2: 1: Gamma is None
    
    if Pe == None:
        whos_none = 0
        Pe_ = [10**a for a in np.arange(0, 4, 1)]
        
        beta_str = f"beta_{beta:.10g}"
        Gamma_str = f"Gamma_{Gamma:.10g}"
        output_image = "gamma_linear_" + "_".join([beta_str, Gamma_str]) + ".png"
    else:
        Pe_ = [Pe]
        
    if beta == None:
        whos_none = 1
        #beta_ = [10**a for a in np.arange(-5., -1., 1)]
        beta_ = [10**a for a in np.arange(-1.25, -0.49, 0.125)]
        
        Pe_str = f"Pe_{Pe:.10g}"
        Gamma_str = f"Gamma_{Gamma:.10g}"
        output_image = "gamma_linear_" + "_".join([Pe_str, Gamma_str]) + ".png"
    else:
        beta_ = [beta]
        
    if Gamma == None:
        whos_none = 2
        Gamma_ = [2**a for a in np.arange(-1., 3., 1)]
                
        Pe_str = f"Pe_{Pe:.10g}"
        beta_str = f"beta_{beta:.10g}"
        output_image = "gamma_linear_" + "_".join([Pe_str, beta_str]) + ".png"
    else:
        Gamma_ = [Gamma]
    
    output_folder = "results/"
    if tp == False:
        output_folder += "output_mix/"
    else:
        output_folder += "outppt_mix/"
    #with open(output_folder + "k_max.txt", 'a') as output_file:
    #    output_file.write(f"Pe\t beta\t Gamma\t k_max\n")
    
    fig, ax = plt.subplots(1, 1)
    
    
    for Pe in Pe_:
        if (Pe > 9 and Pe < 101):
            spacing = 0.01
        elif (Pe > 1e4 and Pe < 1e5):
            spacing = 0.0001
        elif (Pe >= 1e5):
            spacing = 0.00001
        else:
            spacing = 0.001
        for beta in beta_:
            for Gamma in Gamma_:
                psi = - math.log(beta)
                kappa = 1./Pe
                kappa_parallel = 2*Pe*u0*u0/105
                kappa_eff = kappa + kappa_parallel # effective constant diffusion for the base state
                
                xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
                print("Pe = ", Pe, ", beta = ", beta, ", Gamma = ", Gamma)
                print("xi = ", xi)
                
                label = None
                if whos_none == 0:
                    Pe_str = f"Pe_{Pe:.10g}"
                    label = Pe_str
                if whos_none == 1:
                    beta_str = f"beta_{beta:.10g}"
                    label = rf"$\beta$ = {beta:.10g}"
                if whos_none == 2:
                    Gamma_str = f"Gamma_{Gamma:.10g}"
                    label = Gamma_str
                
                input_folder = "results/"
                if tp == False:
                    input_folder += "output_"
                else:
                    input_folder += "outppt_"
                input_folder += "_".join([Pe_str, Gamma_str, beta_str]) + "/"
                
                path_gamma_full = os.path.join(input_folder, file_gamma_full)
                path_gamma_linear = os.path.join(input_folder, file_gamma_linear)
                
                #if os.path.isfile(path_gamma_full):
                    #Ly_full_, gamma_full_ = read_table(path_gamma_full)
                    #ax.scatter([2*math.pi/Ly for Ly in Ly_full_], gamma_full_, label="full") # Plot gamma vs k from complete simulations
                max_k = 0.
                
                if os.path.isfile(path_gamma_linear):
                    k_linear_, gamma_linear_ = read_table_all_values(path_gamma_linear)
                    k_fit_, gamma_fit_ = read_table_const_spacing(path_gamma_linear, spacing)
                    
                    #print("k_fit_ = ", k_fit_)
                    #print("gamma_fit_ = ", gamma_fit_)
                    
                    popt, pcov = np.polyfit(k_fit_, gamma_fit_, 2, cov=True)
                    a = popt[0]
                    b = popt[1]
                    c = popt[2]
                    a_var = pcov[0, 0]
                    b_var = pcov[1, 1]
                    c_var = pcov[2, 2]
                    k_max = - b/(2*a)
                    k_max_sigma = np.sqrt( a_var * (b/(2*a**2))**2 + b_var * (1/(2*a))**2 )
                    gamma_max = - b**2/(4*a) + c
                    gamma_max_sigma = np.sqrt( a_var * (b**2/(4*a**2))**2 + b_var * (b/(2*a))**2 + c_var )
                    
                    with open(output_folder + "k_max.txt", 'a') as output_file:
                        output_file.write(f"{Pe}\t{Gamma}\t{beta}\t{k_max}\t{k_max_sigma}\t{gamma_max}\t{gamma_max_sigma}\n")
                    # k_rescaled_ = [k/xi for k in k_linear_]
                    # gamma_rescaled_ = [(gamma + Gamma)/xi for gamma in gamma_linear_]
                    
                    ax.scatter(k_linear_, gamma_linear_, label=label) # Plot gamma vs k from linear stability analysis
                    ax.plot(k_fit_, [a*k**2 + b*k + c for k in k_fit_], color='black', linestyle='solid')
                    max_k = max(k_linear_) if max(k_linear_) > max_k else max_k
                    
                k_ = np.arange(0., max_k + 0.001, 0.001)
                ax.plot(k_, [0 for k in k_], color='black', linestyle='dashed')
                #ax.axvline(xi, color='black', linestyle='dotted', label="xi") # plot value of xi
    
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\gamma$")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize="large")
    
    fig.savefig(output_folder + output_image, dpi=300)
    plt.show()
