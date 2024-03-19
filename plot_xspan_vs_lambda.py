import numpy as np
import matplotlib.pyplot as plt
import os

# Function to read the last value from the second column of a file
def read_last_value(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip().split()
        return last_line[1] if len(last_line) >= 2 else None

if __name__ == "__main__":

    T = 0.5
    file_xspan = f"xspan_T={T:.2f}.txt"
    file_umax = f"umax.txt"
    
    Pe = 100
    Gamma = 1
    beta = 0.001
    ueps = 0.1
    Lx = 50
    rnd = False

    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"
    ueps_str = f"ueps_{ueps:.10g}"
    Lx_str = f"Lx_{Lx:.10g}"
    rnd_str = f"rnd_{rnd}"

    Ly_ = [pow(2,a) for a in np.arange(-1, 4.5, 0.5)] # List of wavelengths
    xspan_ = [] # List of values of xspan at stationary state
    xspan2_ = [] # List of values of xspan at stationary state
    umax_ = [] # List of values of umax at stationary state
    umax2_ = [] # List of values of umax at stationary state
    
    # Create or open the output file
    with open(f"xspan_vs_lambda_T={T:.2f}.txt", 'w') as output_file:
        output_file.write("Ly xspan\n")
        
        # Iterate over the folders Ly_{a}
        for Ly in Ly_:
            Ly_str = f"Ly_{Ly:.10g}".format(Ly)
            
            folder_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str]) + "_holdpert_False/"
            path_xspan = os.path.join(folder_name, file_xspan)
            path_umax = os.path.join(folder_name, file_umax)
            
            folder2_name = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str]) + "_holdpert_True/"
            path_xspan2 = os.path.join(folder2_name, file_xspan)
            path_umax2 = os.path.join(folder2_name, file_umax)

            if os.path.exists(folder_name): # Check if the folder exists
                if os.path.isfile(path_xspan): # Check if file for xspan exists
                    last_value = read_last_value(path_xspan)
                    if last_value is not None:
                        output_file.write(f"{Ly}\t{last_value}\n")
                        xspan_.append(last_value)
                if os.path.isfile(path_umax): # Check if file for umax exists
                    last_value = read_last_value(path_umax)
                    if last_value is not None:
                        #output_file.write(f"{Ly}\t{last_value}\n")
                        umax_.append(last_value)
                        
            if os.path.exists(folder2_name): # Check if the folder exists
                if os.path.isfile(path_xspan2): # Check if file exists
                    last_value = read_last_value(path_xspan2)
                    if last_value is not None:
                        # output_file.write(f"{Ly}\t{last_value}\n")
                        xspan2_.append(last_value)
                if os.path.isfile(path_umax2): # Check if file for umax exists
                    last_value = read_last_value(path_umax2)
                    if last_value is not None:
                        # output_file.write(f"{Ly}\t{last_value}\n")
                        umax2_.append(last_value)
                    
    xspan_float = [float(x) for x in xspan_]
    xspan2_float = [float(x) for x in xspan2_]
    umax_float = [float(x) for x in umax_]
    umax2_float = [float(x) for x in umax2_]
    
    print(Ly_)
    print(xspan_float)
    print(xspan2_float)
    print(umax_float)
    print(umax2_float)
    
    fig_xspan, ax_xspan = plt.subplots(1, 1)
    ax_xspan.scatter(Ly_, xspan_float, label="holdpert_False")
    ax_xspan.scatter(Ly_, xspan2_float, label="holdpert_True")
    ax_xspan.set_xscale('log')
    ax_xspan.set_yscale('log')
    ax_xspan.set_xlabel("$\lambda$")
    ax_xspan.set_ylabel("$(x_{max} - x_{min})_{sat}$")
    
    fig_umax, ax_umax = plt.subplots(1, 1)
    ax_umax.scatter(Ly_, umax_float, label="holdpert_False")
    ax_umax.scatter(Ly_, umax2_float, label="holdpert_True")
    ax_umax.set_xscale('log')
    ax_umax.set_yscale('log')
    ax_umax.set_xlabel("$\lambda$")
    ax_umax.set_ylabel("$(u_{max})_{sat}$")
    #plt.scale
    plt.legend()
    plt.show()
