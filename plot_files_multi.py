import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read the last value from the second column of a file
def read_table(file_path):
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

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', type=float, help='Value for viscosity ratio')
    return parser.parse_args()

if __name__ == "__main__":

    # Parse the command-line arguments
    args = parse_args() # object containing the values of the parsed argument
    
    Pe = args.Pe
    Gamma = args.Gamma
    beta = args.beta
    
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
    
    if Pe == None:
        Pe_ = [10**a for a in np.arange(0, 4, 1)]
    else:
        Pe_ = [Pe]
        
    if beta == None:
        beta_ = [10**a for a in np.arange(-5., -1., 1)]
    else:
        beta_ = [beta]
        
    if Gamma == None:
        Gamma_ = [1] + [5*a for a in np.arange(1, 5, 1)]
    else:
        Gamma_ = [Gamma]
    
    output_folder = "results/output_mix/"
    #with open(output_folder + "k_max.txt", 'a') as output_file:
    #    output_file.write(f"Pe\t beta\t Gamma\t k_max\n")
    
    fig, ax = plt.subplots(1, 1)
    
    
    for Pe in Pe_:
        for beta in beta_:
            for Gamma in Gamma_:
                psi = - math.log(beta)
                kappa = 1./Pe
                kappa_parallel = 2*Pe*u0*u0/105
                kappa_eff = kappa + kappa_parallel # effective constant diffusion for the base state
                
                xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
                print("xi = ", xi)
                
                Pe_str = f"Pe_{Pe:.10g}"
                Gamma_str = f"Gamma_{Gamma:.10g}"
                beta_str = f"beta_{beta:.10g}"
                
                input_folder= f"results/output_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
                
                path_gamma_full = os.path.join(input_folder, file_gamma_full)
                path_gamma_linear = os.path.join(input_folder, file_gamma_linear)
                
                #if os.path.isfile(path_gamma_full):
                    #Ly_full_, gamma_full_ = read_table(path_gamma_full)
                    #ax.scatter([2*math.pi/Ly for Ly in Ly_full_], gamma_full_, label="full") # Plot gamma vs k from complete simulations
                
                if os.path.isfile(path_gamma_linear):
                    k_linear_, gamma_linear_ = read_table(path_gamma_linear)
                    
                    index_max = np.argmax(gamma_linear_)
                    range = 11
                    k_fit_ = k_linear_[index_max-range//2 : index_max+range//2+1]
                    gamma_fit_ = gamma_linear_[index_max-range//2 : index_max+range//2+1]
                    
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
                    
                    ax.scatter([k for k in k_linear_], gamma_linear_, label=beta_str) # Plot gamma vs k from linear stability analysis
                    ax.plot(k_fit_, [a*k**2 + b*k + c for k in k_fit_], color='black', linestyle='solid')
                    
                k_ = np.arange(0., 16., 0.1)
                ax.plot(k_, [0 for k in k_], color='black', linestyle='dashed')
                #ax.axvline(xi, color='black', linestyle='dotted', label="xi") # plot value of xi
    
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\gamma$")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize="large")
    fig.savefig(output_folder + "gamma_linear_" + "_".join([Pe_str, Gamma_str]) + ".png", dpi=300)
    plt.show()
