import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read the last value from the second column of a file
def read_last_value(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip().split()
        return last_line[1] if len(last_line) >= 2 else None
        
def read_halfway_value(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        halfway_line = lines[4].strip().split()
        return halfway_line[1] if len(halfway_line) >= 2 else None

if __name__ == "__main__":

    Pe = 100
    Gamma = 1
    beta = 0.001
    ueps = 0.001
    Lx = 50
    rnd = False
    holdpert = False

    u0 = 1.0 # base inlet velocity
    Deff = 1./Pe + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    lambda_ = (- u0 + math.sqrt(u0*u0 + 4*Deff*Gamma)) / (2*Deff) # decay constant for the base state
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    rnd_str = f"rnd_{rnd}"
    holdpert_str = f"holdpert_{holdpert}"
    
    Ly_ = [pow(2,a) for a in np.arange(0., 4., 0.25)] # List of wavelengths
    print("Ly_ = ", Ly_)
    T_ = [0.1 * n for n in range(1, 10)] # List of temperature levels
    Ly_umax = [] # List of wavelenghts for umax
    umax_ = [] # List of max velocities at stationary state
    Ly_gamma = [] # List of wavelenghts for gamma
    gamma_ = [] # List of growth rates
    
    # Input files
    file_umax = f"umax.txt"
    file_gamma = f"growth_rates.txt"
    
    # Prepare output files
    with open(f"results/umax_vs_lambda.txt", 'w') as output_file:
            output_file.write("Ly\t umax\n")
    with open(f"results/gamma_vs_lambda.txt", 'w') as output_file:
            output_file.write("Ly\t gamma\n")
    for T in T_:
        with open(f"results/xspan_vs_lambda_T={T:.2f}.txt", 'w') as output_file:
                output_file.write("Ly\t xspan\n")
    
    fig_umax, ax_umax = plt.subplots(1, 1)
    fig_gamma, ax_gamma = plt.subplots(1, 1)
    fig_xspan, ax_xspan = plt.subplots(1, 1)
    
    # Extract umax and gamma from every Ly analyzed and save it
    for Ly in Ly_:
        Ly_str = f"Ly_{Ly:.10g}".format(Ly)
        folder_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "/"
        print(folder_name)
        path_umax = os.path.join(folder_name, file_umax)
        path_gamma = os.path.join(folder_name, file_gamma)
        
        if os.path.exists(folder_name):
            # Extract umax
            if os.path.isfile(path_umax):  # Check if the folder exists and if the data were analyzed
                Ly_umax.append(Ly) # add this value to the list of Ly explored
                umax_last = read_last_value(path_umax)
                if umax_last is not None:
                    umax_.append(float(umax_last))
                    with open(f"results/umax_vs_lambda.txt", 'a') as output_file:
                            output_file.write(f"{Ly}\t{umax_last}\n")
            # Extract gamma
            if os.path.isfile(path_gamma):
                Ly_gamma.append(Ly) # add this value to the list of Ly explored
                gamma_last = read_halfway_value(path_gamma)
                if gamma_last is not None:
                    gamma_.append(float(gamma_last))
                    with open(f"results/gamma_vs_lambda.txt", 'a') as output_file:
                            output_file.write(f"{Ly}\t{gamma_last}\n")
                        
    ax_umax.scatter(Ly_umax, umax_, label="holdpert_False") # Plot umax vs Ly
    ax_gamma.scatter(Ly_gamma, gamma_, label="holdpert_False") # Plot gamma vs Ly
    
    #exit(0)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(T_)))  # Generate colors using colormap
    color_dict = {}  # Dictionary to store colors for each Ly
    
    Ly_xspan = [] # List of wavelenghts for xspan
    for i, T in enumerate(T_):
        color_dict[T] = colors[i]  # Store color for this Ly
    
        xbase = -lambda_ * math.log(T)
        xspan_ = [] # List of values of xspan at stationary state
        
        file_xspan = f"xspan_T={T:.2f}.txt" # input file
        for Ly in Ly_:
            
            Ly_str = f"Ly_{Ly:.10g}".format(Ly)
            folder_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "/"
            path_xspan = os.path.join(folder_name, file_xspan)
            
            if os.path.exists(folder_name): # Check if the folder exists
                # Extract xspan
                if os.path.isfile(path_xspan):  # Check if the data were analyzed
                    if i==0:
                        Ly_xspan.append(Ly) # add this value to the list of Ly explored
                    xspan_last = read_last_value(path_xspan)
                    if xspan_last is not None:
                            xspan_.append(float(xspan_last))
                            with open(f"results/xspan_vs_lambda_T={T:.2f}.txt", 'a') as output_file:
                                output_file.write(f"{Ly}\t{xspan_last}\n")
        ax_xspan.scatter(Ly_xspan, xspan_, label=f"T={T:.2f}", color=color_dict[T]) # Plot xspan vs Ly for this T

    ax_umax.set_xscale('log')
    ax_umax.set_yscale('log')
    ax_umax.set_xlabel("$\lambda$")
    ax_umax.set_ylabel("$(u_{max})_{\text{sat}}$")
    fig_umax.savefig('results/umax_vs_lambda.png', dpi=300)
    
    ax_gamma.set_xscale('log')
    ax_gamma.set_yscale('log')
    ax_gamma.set_xlabel("$\lambda$")
    ax_gamma.set_ylabel("$\gamma$")
    fig_gamma.savefig('results/gamma_vs_lambda.png', dpi=300)
    
    ax_xspan.set_xscale('log')
    ax_xspan.set_yscale('log')
    ax_xspan.set_xlabel("$\lambda$")
    ax_xspan.set_ylabel("$(x_{max} - x_{min})_{\text{sat}}$")
    ax_xspan.legend()
    fig_gamma.savefig('results/xspan_vs_lambda.png', dpi=300)
    
    #plt.scale
    
    #plt.legend()
    plt.show()
