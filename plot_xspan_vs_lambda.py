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

if __name__ == "__main__":

    Pe = 100
    Gamma = 1
    beta = 0.001
    ueps = 0.1
    Lx = 50
    rnd = False

    u0 = 1.0 # base inlet velocity
    # base state parameters
    Deff = 1./Pe + 2*Pe*u0*u0/105 # effective constant diffusion for the base state
    lambda_ = (- u0 + math.sqrt(u0*u0 + 4*Deff*Gamma)) / 2*Deff # decay constant for the base state
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    rnd_str = f"rnd_{rnd}"
    
    Ly_ = [pow(2,a) for a in np.arange(-1, 5., 0.25)] # List of wavelengths
    print(Ly_)
    T_ = [0.1 * n for n in range(1, 10)] # List of temperature levels
    Ly_explored = [] # List of wavelenght explored
    #Ly_explored2 = [] # List of wavelenghts explored
    
    umax_ = [] # List of values of umax at stationary state
    #umax2_ = [] # List of values of umax at stationary state
    
    file_umax = f"umax.txt" # input file
    
    # Prepare output files
    with open(f"results/umax_vs_lambda.txt", 'w') as output_file:
            output_file.write("Ly\t umax\n")
    for T in T_:
        with open(f"results/xspan_vs_lambda_T={T:.2f}.txt", 'w') as output_file:
                output_file.write("Ly\t xspan\n")
    
    fig_xspan, ax_xspan = plt.subplots(1, 1)
    fig_umax, ax_umax = plt.subplots(1, 1)
    
    for Ly in Ly_:
        Ly_str = f"Ly_{Ly:.10g}".format(Ly)
        
        folder_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str]) + "_holdpert_True/"
        path_umax = os.path.join(folder_name, file_umax)
        #folder2_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str]) + "_holdpert_False/"
        #path_umax2 = os.path.join(folder2_name, file_umax)
        print(folder_name)
        
        if os.path.exists(folder_name) and os.path.isfile(path_umax):  # Check if the folder exists and if the data were analyzed
            Ly_explored.append(Ly)
            
            umax_last = read_last_value(path_umax)
            if umax_last is not None:
                umax_.append(float(umax_last))
                with open(f"results/umax_vs_lambda.txt", 'a') as output_file:
                        output_file.write(f"{Ly}\t{umax_last}\n")
                        
    ax_umax.scatter(Ly_explored, umax_, label="holdpert_False")
    # ax_umax.scatter(Ly_umax2, umax2_float, label="holdpert_True")
    #exit(0)
    print(Ly_explored)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(T_)))  # Generate colors using colormap
    color_dict = {}  # Dictionary to store colors for each Ly
    
    for i, T in enumerate(T_):
        color_dict[T] = colors[i]  # Store color for this Ly
    
        xbase = -lambda_ * math.log(T)
        xspan_ = [] # List of values of xspan at stationary state
        # xspan2_ = [] # List of values of xspan at stationary state
        
        file_xspan = f"xspan_T={T:.2f}.txt" # input file
        for Ly in Ly_:
            
            Ly_str = f"Ly_{Ly:.10g}".format(Ly)
            folder_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str]) + "_holdpert_True/"
            path_xspan = os.path.join(folder_name, file_xspan)
            # path_xspan2 = os.path.join(folder2_name, file_xspan)
            
            if os.path.exists(folder_name) and os.path.isfile(path_xspan):  # Check if the folder exists and if the data were analyzed
                xspan_last = read_last_value(path_xspan)
                if xspan_last is not None:
                        xspan_.append( float(xspan_last) )
                        with open(f"results/xspan_vs_lambda_T={T:.2f}.txt", 'a') as output_file:
                            output_file.write(f"{Ly}\t{xspan_last}\n")
        ax_xspan.scatter(Ly_explored, xspan_, label=f"T={T:.2f}", color=color_dict[T])
        # ax_xspan.scatter(Ly_exploted, xspan2_, label="holdpert_False")

    ax_xspan.set_xscale('log')
    ax_xspan.set_yscale('log')
    ax_xspan.set_xlabel("$\lambda$")
    ax_xspan.set_ylabel("$(x_{max} - x_{min})_{sat}$")
    ax_xspan.legend()
    
    ax_umax.set_xscale('log')
    ax_umax.set_yscale('log')
    ax_umax.set_xlabel("$\lambda$")
    ax_umax.set_ylabel("$(u_{max})_{sat}$")
    #plt.scale
    
    #plt.legend()
    plt.show()
