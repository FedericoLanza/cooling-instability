import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def find_min_spacing(filename, tolerance=1e-6):
    # Load data from file, skipping the first row (header)
    data = np.loadtxt(filename, skiprows=1)
    
    # Extract first column
    x_values = data[:, 0]
    #print('x_values = ', x_values)
    # Compute spacing between consecutive values
    spacings = np.diff(x_values)
    
    #print('spacings = ', spacings)
    
    # Find the minimum spacing above the tolerance
    unique_spacings = np.unique(np.round(spacings, int(abs(np.log10(tolerance)))))
    min_spacing = np.min(unique_spacings)
    
    return min_spacing


# Function to read the last value from the second column of a file
def read_table_all_values(file_path):
    x = []
    y = []
    ysigma = []
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
            ysigma.append(float(values[2]))
            
    return x, y, ysigma


def read_table_const_spacing(file_path, spacing, tolerance=1e-6):
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
    parser.add_argument('--Pe', type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', type=float, help='Value for viscosity ratio')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--nofit', action='store_true', help='Flag for choosing whether to fit the points around the maximum')
    parser.add_argument('--both', action='store_true', help='Flag for plotting both data from Tp and Tu')
    parser.add_argument('--find_betac', action='store_true', help='Flag for plotting data for finding betac')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    parser.add_argument('--aesthetic', action='store_true', help='Flag for generating plots to present in the article')
    
    return parser.parse_args()

if __name__ == "__main__":

    # Parse the command-line arguments
    args = parse_args() # object containing the values of the parsed argument
    
    Pe = args.Pe
    Gamma = args.Gamma
    beta = args.beta
    
    tp = args.tp
    nofit = args.nofit
    both = args.both
    find_betac = args.find_betac
    latex = args.latex
    aesthetic = args.aesthetic
    
    ueps = 0.001
    Lx = 50
    rnd = False
    holdpert = False
    
    u0 = 1.0 # base inlet velocity
    
    # Input paths
    file_gamma_full = "gamma_full.txt"
    file_gamma_linear = "gamma_linear_plot.txt" if aesthetic else "gamma_linear.txt"
    
    Pe_ = []
    beta_ = []
    Gamma_ = []
    
    Pe_str = None
    beta_str = None
    Gamma_str = None
    multi_Pe = False
    multi_beta = False
    multi_Gamma = False
    y_variable_str = None
    output_image = "gamma_vs_k_linear" if aesthetic else "gamma_linear"
    
    if Pe == None:
        multi_Pe = True
        Pe_ = [10**a for a in np.arange(1.25, 3.01, 0.5)]
    else:
        Pe_ = [Pe]
        Pe_str_out = f"_Pe_{Pe:.10g}"
        output_image += Pe_str_out
        
    if Gamma == None:
        multi_Gamma = True
        #Gamma_ = [10**a for a in np.arange(-6.5, -4.5, 0.25)]
    else:
        Gamma_ = [Gamma]
        Gamma_str_out = f"_Gamma_{Gamma:.10g}"
        output_image += Gamma_str_out
    
    if beta == None:
        multi_beta = True
        if find_betac:
            a_sat = np.arange(-2., -1.749, 0.0625)
            #a_sat = np.arange(-2.25, -1.749, 0.125)
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
                    a_ = np.arange(-2.5, -1.999, 0.125)
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
            beta_ = [10**a for a in np.arange(-5., -2.99, 0.25)]
            #beta_ = [10**(-2.5)]
    else:
        beta_ = [beta]
        beta_str_out = f"_beta_{beta:.10g}"
        output_image += beta_str_out
        
    output_image += ".pdf"
    
    io_folder = "results/"
    if tp == False:
        io_folder += "output"
    else:
        io_folder += "outppt"
        
    output_folder = io_folder + "_mix/"
    
    
    if both == True:
        io_folder_2 = "results/"
        if tp == False:
            io_folder_2 += "outppt"
        else:
            io_folder_2 += "output"
    
    if latex:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        })
    
    fig, ax = plt.subplots(1, 1)
    
    if False:
        # [-5, -2.25, step=0.25], quadratic fit for gamma
        c_gamma = 0.01169361417
        a_gamma = -0.5552080655
        b_gamma = -2.701975883
        a_k = -0.8856475586
        b_k = -1.60735377
        c_k = 0
        c_gamma_err = 0.0001988853709
        a_gamma_err = 0.003222716832
        b_gamma_err = 0.01243252889
        a_k_err = 0.0009175867527
        b_k_err = 0.007308784492
        c_k_err = 0
        cov_ab_gamma = 6.50484e-06
        cov_ab_k = 1.49779e-05
    else:
        a_gamma = -0.7435226754
        b_gamma = -3.414410586
        c_gamma = 0
        a_k = -0.8856475586
        b_k = -1.60735377
        a_gamma_err = 0.002099624492
        b_gamma_err = 0.0163571008
        a_k_err = 0.0009175867527
        b_k_err = 0.007308784492
        cov_ab_gamma = 6.50484e-06
        cov_ab_k = 1.49779e-05
    
    for Pe in Pe_:
        for beta in beta_:
            psi = - math.log(beta)
            Gamma_ = [0.01/Pe]
            for Gamma in Gamma_:
                print("psi = ", psi)
                kappa = 1./Pe
                kappa_parallel = 2*Pe*u0*u0/105
                kappa_eff = kappa + kappa_parallel # effective constant diffusion for the base state
                
                xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
                
                label = "Data"
                if multi_Pe:
                    x_variable = "Pe"
                    label += f", Pe = {Pe:.10g}"
                if multi_beta:
                    x_variable = "beta"
                    label += rf", $\beta$ = {beta:.10g}"
                if multi_Gamma:
                    x_variable = "Gamma"
                    label += rf", $\Gamma$ = {Gamma:.10g}"
                    
                Pe_str = f"Pe_{Pe:.10g}"
                beta_str = f"beta_{beta:.10g}"
                Gamma_str = f"Gamma_{Gamma:.10g}"
                
                input_folder = io_folder + "_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
                
                path_gamma_full = os.path.join(input_folder, file_gamma_full)
                path_gamma_linear = os.path.join(input_folder, file_gamma_linear)
                
                print(path_gamma_linear)
                
                #if os.path.isfile(path_gamma_full):
                    #Ly_full_, gamma_full_ = read_table(path_gamma_full)
                    #ax.scatter([2*math.pi/Ly for Ly in Ly_full_], gamma_full_, label="full") # Plot gamma vs k from complete simulations
                
                max_k = 0.
                
                if os.path.isfile(path_gamma_linear):
                
                    print("Pe = ", Pe, ", beta = ", beta, ", Gamma = ", Gamma)
                    print("xi = ", xi)
                    
                    # Plot the dispersion relationship for this combination of parameters
                    
                    k_linear_, gamma_linear_, gamma_linear_sigma_ = read_table_all_values(path_gamma_linear)
                    
                    if aesthetic:
                        sum_k = np.float64(-a_k*psi + b_k)
                        sum_gamma = np.float64(-a_gamma*psi + b_gamma + c_gamma*psi**2)
                        k_linear_resc_ = k_linear_/(Gamma * sum_k)
                        gamma_linear_resc_ = gamma_linear_/(Gamma * sum_gamma)
                        ax.scatter(k_linear_resc_, gamma_linear_resc_, label=label)
                    else:
                        ax.errorbar(k_linear_, gamma_linear_, yerr=gamma_linear_sigma_, fmt='o', capsize=3, label=label)
                    
                    # Plot an horizontal line for gamma = 0
                    ax.axhline(y=0., color='black', linestyle='--', linewidth=2)
                    
                    if nofit == False:
                        #spacing = find_min_spacing(path_gamma_linear)
                        #k_fit_, gamma_fit_ = read_table_const_spacing(path_gamma_linear, spacing)
                    
                        max_pos = np.argmax(gamma_linear_)
                        k_fit_, gamma_fit_ = k_linear_[max_pos-3:max_pos+4], gamma_linear_[max_pos-3:max_pos+4]
                    
                        k_plot_ = np.linspace(min(k_fit_), max(k_fit_), 50)
                        
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
                        gamma_max_sigma = np.sqrt( a_var * (b**2/(4*a**2))**2 + b_var * (b/(2*a))**2 + c_var)
                        
                        print('gamma_max = ', gamma_max)
                        print('gamma_max_sigma = ', gamma_max_sigma)
                        
                        with open(io_folder + f"_mix/values_vs_Pe_different_beta_Gamma_123.txt", 'a') as output_file:
                            output_file.write(f"{Pe}\t{Gamma}\t{beta}\t{k_max}\t{k_max_sigma}\t{gamma_max}\t{gamma_max_sigma}\n")
                    
                        ax.plot(k_plot_, [a*k**2 + b*k + c for k in k_plot_], color='black', linestyle='solid')
                
                if both == True:
                    input_folder_2 = io_folder_2 + "_" + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
                    path_gamma_linear_2 = os.path.join(input_folder_2, file_gamma_linear)
                    if os.path.isfile(path_gamma_linear_2):
                        k_linear_2, gamma_linear_2 = read_table_all_values(path_gamma_linear_2)
                        ax.scatter(k_linear_2, gamma_linear_2, marker = '*', label=label) # Plot gamma vs k from linear stability analysis
                        
                #ax.axvline(xi, color='black', linestyle='dotted', label="xi") # plot value of xi
    
    
    if aesthetic:
        ax.set_xlabel(r"$k/(\Gamma(a_{k}\log\beta + b_{k}))$")
        ax.set_ylabel(r"$\gamma/(\Gamma(a_{\gamma}\log\beta + b_{\gamma}))$")
    else:
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\gamma$")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()
    
    if latex:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        })
    
    #fig.savefig(output_folder + output_image, dpi=300)
    plt.show()
