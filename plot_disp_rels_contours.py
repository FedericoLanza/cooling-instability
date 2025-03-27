import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', type=float, help='Value for viscosity ratio')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    Pe = args.Pe
    Gamma = args.Gamma
    beta = args.beta
    tp = args.tp
    latex = args.latex
    
    letter = "p" if tp else "u"
    
    Pe_ = []
    beta_ = []
    Gamma_ = []
    
    multi_Pe = False
    multi_beta = False
    multi_Gamma = False
    
    output_image = "gamma_linear_contour"
    
    if Pe == None:
        multi_Pe = True
        Pe_ = [10**a for a in np.arange(1., 5., 0.5)]
        var_ = Pe_
        y_label = 'Pe'
    else:
        Pe_ = [Pe]
        Pe_str = f"_Pe_{Pe:.10g}"
        output_image += Pe_str
        
    if beta == None:
        multi_beta = True
        beta_ = [10**a for a in np.arange(-4.5, -0.99, 0.5)]
        var_ = beta_
        y_label = r'$\beta$'
    else:
        beta_ = [beta]
        beta_str = f"_beta_{beta:.10g}"
        output_image += beta_str
        
    if Gamma == None:
        multi_Gamma = True
        Gamma_ = [2**a for a in np.arange(-1., 2.01, 1.)]
        var_ = Gamma_
        y_label = r'$\Gamma$'
    else:
        Gamma_ = [Gamma]
        Gamma_str = f"_Gamma_{Gamma:.10g}"
        output_image += Gamma_str
    
    if ( (multi_Pe and multi_Gamma) or (multi_Pe and multi_beta) or (multi_Gamma and multi_beta) or (multi_Pe == False and multi_Gamma == False and multi_beta == False) ):
        print("Please fix the value of two parameters.")
        exit(0)
    print('beta_ = ', beta_)
    print('var_ = ', var_)
    output_image += ".pdf"

    # Initialize a dictionary to store data
    data = {}

    # Iterate over the folders
    for Pe in Pe_:
        for Gamma in Gamma_:
            for beta in beta_:
                
                folder_name = f"results/outp{letter}t_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}"
                file_path = os.path.join(folder_name, "gamma_linear_plot.txt")
                print('file_path = ', file_path)
                if os.path.exists(file_path):
                    print('ok')
                    # Load data, skipping the first row
                    k, gamma, gamma_sigma = np.loadtxt(file_path, skiprows=1, unpack=True)
                    
                    # Filter k values within the range [0, 10]
                    mask = (k >= 0) & (k <= 10)
                    k, gamma = k[mask], gamma[mask]
                    
                    # Store filtered data
                    if multi_Pe:
                        data[Pe] = (k, gamma)
                    if multi_Gamma:
                        data[Gamma] = (k, gamma)
                    if multi_beta:
                        data[beta] = (k, gamma)
    
    # Collect all unique k values within the range [0, 10]
    k_values = sorted(set(k for var in data for k in data[var][0] if 0 <= k <= 10))

    # Create 2D arrays for k, beta, and gamma
    K, B = np.meshgrid(k_values, var_)
    G = np.full_like(K, np.nan, dtype=float)  # Initialize gamma values

    # Fill gamma values where available and interpolate missing values
    for var in var_:
        k_vals, gamma_vals = data[var]
        interp_func = interp1d(k_vals, gamma_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        G[np.where(np.isclose(var_, var))[0][0], :] = interp_func(k_values)

    # Plot the contour map
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(K, B, G, levels=20, cmap='coolwarm')
    plt.colorbar(contour, label=r'$\gamma$')
    plt.contour(K, B, G, levels=10, colors='black', linewidths=0.5)

    # Highlight the gamma = 0 contour with a thick line
    zero_contour = plt.contour(K, B, G, levels=[0], colors='red', linewidths=2.5)

    plt.xscale('linear')
    plt.yscale('log')  # Since values varies exponentially
    plt.xlabel('$k$')
    plt.ylabel(y_label)
    #plt.title(r'Contour plot of $\gamma$ as a function of $k$ and ' + y_label)
    
    plt.savefig( "results/outp{letter}t" + output_image, dpi=300)
    plt.show()

