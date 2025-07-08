import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os


# Function to read the first three columns of a file
def read_table(file_path):
    x = []
    y1 = []
    y2 = []
    with open(file_path, 'r') as file:
    
        # Skip the first line (the header)
        next(file)
        
        # Read the remaining lines
        for line in file:
            # Split the line into two parts (x and y values)
            values = line.split()
        
            # Convert to integers (or floats if needed) and append to respective lists
            x.append(float(values[0]))
            y1.append(float(values[1]))
            y2.append(float(values[2]))
            
    return x, y1, y2

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', default=100.0, type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', default=1.0, type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', default=0.001, type=float, help='Value for viscosity ratio')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    
    return parser.parse_args()

if __name__ == "__main__":

    # Parse the command-line arguments
    args = parse_args() # object containing the values of the parsed argument
    
    Pe = args.Pe
    Gamma = args.Gamma
    beta = args.beta
    
    tp = args.tp
    latex = args.latex
         
    ueps = 0.001
    Lx = 50
    rnd = False
    holdpert = False

    u0 = 1.0 # base inlet velocity
    
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"

    file_xmax_vs_Pe = f"xmax_vs_Pe_{Gamma_str}_{beta_str}.txt"
    
    if latex:
        plt.rcParams.update({
            "text.usetex": True,  # Use LaTeX for proper formatting
            "font.family": "serif",  # Match JFM style
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 18,  # Default text size (JFM uses ~8pt for labels)
            "axes.titlesize": 20,  # Title size (slightly larger)
            "axes.labelsize": 18,  # Axis labels (JFM ~8pt)
            "xtick.labelsize": 18,  # Tick labels
            "ytick.labelsize": 18,
            "legend.fontsize": 12,  # Legend size
            "figure.figsize": (6, 4.5),  # Keep plots compact (JFM prefers small plots)
            "lines.linewidth": 1.5,  # Thin lines
            "lines.markersize": 8,  # Small but visible markers
            
        })
    
    fig, ax = plt.subplots(1, 1)
    
    path_folder = "results/"
    if tp == False:
        path_folder += "output_mix"
    else:
        path_folder += "outppt_mix"
    
    output_image = "xmax_vs_Pe.pdf"
    path_file = os.path.join(path_folder, file_xmax_vs_Pe)
    path_image = os.path.join(path_folder, output_image)
    
    if os.path.isfile(path_file):
        Pe_, xmax_, Tmax_ = read_table(path_file)
        kappa_eff_ = [1./Pe + 2*Pe*u0*u0/105 for Pe in Pe_]
        inv_xi_ = [(2*kappa_eff) / (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) for kappa_eff in kappa_eff_] # decay constant for the base state
        ax.scatter(Pe_, xmax_, label=r"$x_{\max}$", alpha=0.6) # Plot xmax vs Pe
        ax.plot(Pe_, inv_xi_, label=r"$1/\xi$", alpha=0.6) # Plot 1/xi vs Pe
        ax.plot(Pe_, [2*inv_xi for inv_xi in inv_xi_], label=r"$2/\xi$", alpha=0.6) # Plot 2/xi vs Pe
    
    ax.set_xlabel(r"Pe")
    ax.set_ylabel(r"$x_{\max}$")
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(path_image, dpi=300)
    plt.show()
