import math
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

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
        
def read_selected_value(file_path, i):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        selected_line = lines[i].strip().split()
        return selected_line[1] if len(selected_line) >= 2 else None

if __name__ == "__main__":

    Pe = 100
    Gamma = 1
    ueps = 0.001
    Ly = 2
    Lx = 50
    rnd = False
    holdpert = False

    u0 = 1.0 # base inlet velocity
    Deff = 1./Pe + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    lambda_ = (- u0 + math.sqrt(u0*u0 + 4*Deff*Gamma)) / (2*Deff) # decay constant for the base state
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Ly_str = f"Ly_{Ly:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    rnd_str = f"rnd_{rnd}"
    holdpert_str = f"holdpert_{holdpert}"
    
    beta_ = [0.00025, 0.0005, 0.001, 0.002, 0.0028284271, 0.004] # List of wavelengths
    print("beta_ = ", beta_)
    T_ = [0.1 * n for n in range(1, 10)] # List of temperature levels
    beta_umax = [] # List of wavelenghts for umax
    umax_ = [] # List of max velocities at stationary state
    beta_gamma = [] # List of wavelenghts for gamma
    gamma_ = [] # List of growth rates
    beta_tstat = [] # List of wavelenghts for gamma
    tstat_ = [] # List of growth rates
    
    # Input files
    file_umax = f"umax.txt"
    file_gamma = f"growth_rates.txt"
    file_tstat = f"tstat.txt"
    
    # Prepare output files
    output_folder = f"results/output_" + "_".join([Pe_str, Gamma_str, Ly_str]) + "/"
    with open(output_folder + 'umax.txt', 'w') as output_file:
            output_file.write("beta\t umax\n")
    with open(output_folder + 'gamma.txt', 'w') as output_file:
            output_file.write("beta\t gamma\n")
    with open(output_folder + 'tstat.txt', 'w') as output_file:
            output_file.write("beta\t tstat\n")
    for T in T_:
        with open(output_folder + f"xspan_T={T:.2f}.txt", 'w') as output_file:
                output_file.write("beta\t xspan\n")
    
    fig_umax, ax_umax = plt.subplots(1, 1)
    fig_gamma, ax_gamma = plt.subplots(1, 1)
    fig_tstat, ax_tstat = plt.subplots(1, 1)
    fig_xspan, ax_xspan = plt.subplots(1, 1)
    fig_gamma2, ax_gamma2 = plt.subplots(1, 1)
    fig_tstat2, ax_tstat2 = plt.subplots(1, 1)
    
    # Extract umax and gamma from every Ly analyzed and save it
    for beta in beta_:
        beta_str = f"beta_{beta:.10g}".format(beta)
        folder_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "/"
        print(folder_name)
        path_umax = os.path.join(folder_name, file_umax)
        path_gamma = os.path.join(folder_name, file_gamma)
        path_tstat = os.path.join(folder_name, file_tstat)
        
        if os.path.exists(folder_name):
            # Extract umax
            if os.path.isfile(path_umax):  # Check if the folder exists and if the data were analyzed
                beta_umax.append(beta) # add this value to the list of Ly explored
                umax_last = read_last_value(path_umax)
                if umax_last is not None:
                    umax_.append(float(umax_last))
                    with open(f"results/umax.txt", 'a') as output_file:
                            output_file.write(f"{beta}\t{umax_last}\n")
            # Extract gamma
            if os.path.isfile(path_gamma):
                beta_gamma.append(beta) # add this value to the list of Ly explored
                gamma_str = read_halfway_value(path_gamma)
                if gamma_str is not None:
                    gamma_.append(float(gamma_str))
                    with open(f"results/gamma.txt", 'a') as output_file:
                            output_file.write(f"{beta}\t{gamma_str}\n")
            # Extract tstat
            if os.path.isfile(path_tstat):
                beta_tstat.append(beta) # add this value to the list of Ly explored
                tstat_str = read_halfway_value(path_tstat)
                if tstat_str is not None:
                    tstat_.append(float(tstat_str))
                    with open(f"results/tstat.txt", 'a') as output_file:
                            output_file.write(f"{beta}\t{tstat_str}\n")
                        
    ax_umax.scatter(beta_umax, umax_, label="holdpert_False") # Plot umax vs Ly
    ax_gamma.scatter(beta_gamma, gamma_, label="holdpert_False") # Plot gamma vs Ly
    ax_tstat.scatter(beta_tstat, tstat_, label="holdpert_False") # Plot tstat vs Ly
    
    #exit(0)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(T_)))  # Generate colors using colormap
    color_dict = {}  # Dictionary to store colors for each Ly
    
    beta_xspan = [] # List of wavelenghts for xspan
    for i, T in enumerate(T_):
        color_dict[T] = colors[i]  # Store color for this Ly
    
        xbase = -lambda_ * math.log(T)
        xspan_ = [] # List of values of xspan at stationary state
        gamma2_ = []
        tstat2_ = []
        
        file_xspan = f"xspan_T={T:.2f}.txt" # input file
        for beta in beta_:
            beta_str = f"beta_{beta:.10g}".format(Ly)
            folder_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str, holdpert_str]) + "/"
            path_xspan = os.path.join(folder_name, file_xspan)
            path_gamma = os.path.join(folder_name, file_gamma)
            path_tstat = os.path.join(folder_name, file_tstat)
            
            if os.path.exists(folder_name): # Check if the folder exists
                # Extract xspan
                if os.path.isfile(path_xspan):  # Check if the data were analyzed
                    if i==0:
                        beta_xspan.append(Ly) # add this value to the list of Ly explored
                    xspan_last = read_last_value(path_xspan)
                    if xspan_last is not None:
                        xspan_.append(float(xspan_last))
                        with open(f"results/xspan_T={T:.2f}.txt", 'a') as output_file:
                            output_file.write(f"{beta}\t{xspan_last}\n")
                # Extract all gamma
                if os.path.isfile(path_gamma):  # Check if the data were analyzed
                    #if i==0:
                    #    Ly_xspan.append(Ly) # add this value to the list of Ly explored
                    gamma_sel = read_selected_value(path_gamma, i)
                    if gamma_sel is not None:
                        gamma2_.append(float(gamma_sel))
                # Extract all gamma
                if os.path.isfile(path_tstat):  # Check if the data were analyzed
                    #if i==0:
                    #    Ly_xspan.append(Ly) # add this value to the list of Ly explored
                    tstat_sel = read_selected_value(path_tstat, i)
                    if tstat_sel is not None:
                        tstat2_.append(float(tstat_sel))
        ax_xspan.scatter(beta_xspan, xspan_, label=f"T={T:.2f}", color=color_dict[T]) # Plot xspan vs Ly for this T
        ax_gamma2.scatter(beta_gamma, gamma2_, label=f"T={T:.2f}", color=color_dict[T]) # Plot gamma vs Ly for this T
        ax_tstat2.scatter(beta_tstat, tstat2_, label=f"T={T:.2f}", color=color_dict[T]) # Plot tstat vs Ly for this T

    #ax_umax.set_xscale('log')
    #ax_umax.set_yscale('log')
    ax_umax.set_xlabel("beta")
    ax_umax.set_ylabel("$u^{\max}_{st}$")
    fig_umax.savefig(output_folder + 'umax.png', dpi=300)
    
    #ax_gamma.set_xscale('log')
    #ax_gamma.set_yscale('log')
    ax_gamma.set_xlabel("beta", fontsize=15)
    ax_gamma.set_ylabel(r"$\gamma$", fontsize=15)
    #ax_gamma.tick_params(labelsize=12)
    fig_gamma.savefig(output_folder + 'gamma.png', dpi=300)
    
    #ax_tstat.set_xscale('log')
    #ax_tstat.set_yscale('log')
    ax_tstat.set_xlabel("beta")
    ax_tstat.set_ylabel("$t_{st}$")
    fig_tstat.savefig(output_folder + 'tstat.png', dpi=300)
    
    #ax_xspan.set_xscale('log')
    #ax_xspan.set_yscale('log')
    ax_xspan.set_xlabel("beta")
    ax_xspan.set_ylabel("$(x_{max} - x_{min})_{st}$")
    ax_xspan.legend()
    fig_xspan.savefig(output_folder + 'xspan.png', dpi=300)
    
    #ax_gamma2.set_xscale('log')
    #ax_gamma2.set_yscale('log')
    ax_gamma2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax_gamma2.set_xlabel(r"$\beta$", fontsize=16)
    ax_gamma2.set_ylabel(r"$\gamma$", fontsize=16)
    ax_gamma2.tick_params(labelsize=11)
    ax_gamma2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_gamma2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax_gamma2.legend(fontsize="large")
    fig_gamma2.savefig(output_folder + 'gamma_all.png', dpi=300)
    
    #ax_tstat2.set_xscale('log')
    #ax_tstat2.set_yscale('log')
    ax_tstat2.set_xlabel("beta")
    ax_tstat2.set_ylabel("$t_{st}$")
    ax_tstat2.legend()
    fig_tstat2.savefig(output_folder + 'tstat_vs_beta_all.png', dpi=300)
    
    #plt.scale
    
    #plt.legend()
    plt.show()
