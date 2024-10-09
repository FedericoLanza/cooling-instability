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

if __name__ == "__main__":

    Pe = 100
    Gamma = 1
    beta = 1e-3
    ueps = 0.001
    Lx = 50
    rnd = False
    holdpert = False

    u0 = 1.0 # base inlet velocity
    kappa = 1./Pe
    kappa_parallel = 2*Pe*u0*u0/105
    kappa_eff = kappa + kappa_parallel # effective constant diffusion for the base state
    xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
    print("xi = ", xi)
    psi = - math.log(beta)
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    rnd_str = f"rnd_{rnd}"
    holdpert_str = f"holdpert_{holdpert}"
    
    # Input paths
    file_gamma = f"gamma.txt"
    file_gamma_linear = f"gamma_linear.txt"
    
    folder_name = f"results/output_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
    
    path_gamma = os.path.join(folder_name, file_gamma)
    path_gamma_linear = os.path.join(folder_name, file_gamma_linear)
    
    if os.path.isfile(path_gamma):
        Ly_, gamma_ = read_table(path_gamma)
        
    if os.path.isfile(path_gamma_linear):
        Ly_linear_, gamma_linear_ = read_table(path_gamma_linear)
    
    k_ = np.arange(0., 16., 0.1)
    
    # Plot data
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(k_, [0 for k in k_], color='black', linestyle='dashed')
    #ax.axvline(xi, color='black', linestyle='dotted', label="xi")
    ax.scatter([Ly for Ly in Ly_], gamma_, label="original") # Plot gamma vs k from complete simulations
    ax.scatter([Ly for Ly in Ly_linear_[:-2]], gamma_linear_[:-2], label="linear") # Plot gamma vs k from linear stability analysis
    ax.plot(k_, [- Gamma + 0.5 * k * (psi + (kappa_parallel - kappa)*k - math.sqrt((kappa_eff*k)**2 + 2*kappa_eff*psi*k)) for k in k_], label="step base state", color='red') # Plot gamma vs k from linear stability analysis of step function
    #ax.plot(k_, [- Gamma + 0.5 * ((psi - 1)*k + (kappa_parallel - kappa)*k**2 - math.sqrt((kappa_eff*k**2 - 2*(kappa_eff**2)*k**3 + (kappa_eff**3)*k**4 + (2*k - 4*kappa_eff*k**2 + 2*kappa_eff*k**3)*psi ))/math.sqrt(kappa_eff)) for k in k_], label="exp base state", color='green') # Plot gamma vs k from linear stability analysis of step function
    
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\gamma$")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize="large")
    #fig.savefig('results/gamma_compare.png', dpi=300)
    fig.savefig(folder_name + 'gamma_compare.png', dpi=300)
    plt.show()
