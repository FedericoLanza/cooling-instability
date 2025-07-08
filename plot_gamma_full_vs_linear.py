import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os

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
    parser.add_argument('--Pe', default=100.0, type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', default=1.0, type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', default=0.001, type=float, help='Value for viscosity ratio')
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
    kappa = 1./Pe
    kappa_parallel = 2*Pe*u0*u0/105
    kappa_eff = kappa + kappa_parallel # effective constant diffusion for the base state
    xi = (- u0 + math.sqrt(u0*u0 + 4*kappa_eff*Gamma)) / (2*kappa_eff) # decay constant for the base state
    print("xi = ", xi)
    psi = - math.log(beta)
    
    # Input paths
    file_gamma_full = f"gamma_full.txt"
    file_gamma_linear = f"gamma_linear_plot.txt"
    
    Pe_str = f"Pe_{Pe:.10g}"
    Gamma_str = f"Gamma_{Gamma:.10g}"
    beta_str = f"beta_{beta:.10g}"

    fig, ax = plt.subplots(1, 1)
    
    path_folder = "results/"
    if tp == False:
        path_folder += "output_"
    else:
        path_folder += "outppt_"
    #input_folder = path_folder + "_".join([Pe_str, Gamma_str, beta_str]) + "/"
    input_folder = path_folder + "/"
    output_folder = path_folder + "mix/"
    
    path_gamma_full = os.path.join(input_folder, file_gamma_full)
    path_gamma_linear = os.path.join(input_folder, file_gamma_linear)
    
    k_max_lim = 10.
    
    if os.path.isfile(path_gamma_full):
        Ly_full_, gamma_full_ = read_table(path_gamma_full)
        #ax.scatter([2*math.pi/Ly for Ly in Ly_full_], gamma_full_, label="full") # Plot gamma vs k from complete simulations
        
    if os.path.isfile(path_gamma_linear):
        k_linear_, gamma_linear_ = read_table(path_gamma_linear)
        max_k = max(k_linear_)
        
        ax.plot([k for k in k_linear_], gamma_linear_, marker="o", label="LSA", alpha=0.6, markersize=5) # Plot gamma vs k from linear stability analysis
        
        # Get current axis limits
        k_min, k_max_lim = ax.get_xlim()
        gamma_min, gamma_max_lim = ax.get_ylim()
        
        gamma_max = np.max(gamma_linear_)
        k_max = k_linear_[np.argmax(gamma_linear_)]
        
        # Add dashed lines
        ax.plot([k_max, k_max], [ax.get_ylim()[0], gamma_max], linestyle="dotted", linewidth=1., color="black", alpha=0.7)  # Vertical line
        ax.plot([ax.get_xlim()[0], k_max], [gamma_max, gamma_max], linestyle="dotted", linewidth=1., color="black", alpha=0.7)  # Horizontal line
        
        # Add labels
        ax.text(k_max, ax.get_ylim()[0]-0.02, r"$k_{\max}$", ha="center", va="center")  # x-axis label
        ax.text(ax.get_xlim()[0], gamma_max, r"$\gamma_{\max}$", ha="center", va="center")   # y-axis label
        
        # Highlight the max point
        ax.scatter([k_max], [gamma_max], color="black", s=10, zorder=3, label="Maximum")
        
    k_step = 0.025
    k_ = np.arange(k_min, max_k + k_step, k_step)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0., 0.], linestyle="--", linewidth=1.3, color="black", alpha=1.)  # Horizontal line
    
    #ax.axvline(xi, color='black', linestyle='dotted', label="xi")
    #ax.plot(k_, [- Gamma + 0.5 * k * (psi + (kappa_parallel - kappa)*k - math.sqrt((kappa_eff*k)**2 + 2*kappa_eff*psi*k)) for k in k_], label="step base state", color='red') # Plot gamma vs k from linear stability analysis of step function
    #ax.plot(k_, [- Gamma + 0.5 * ((psi - 1)*k + (kappa_parallel - kappa)*k**2 - math.sqrt((kappa_eff*k**2 - 2*(kappa_eff**2)*k**3 + (kappa_eff**3)*k**4 + (2*k - 4*kappa_eff*k**2 + 2*kappa_eff*k**3)*psi ))/math.sqrt(kappa_eff)) for k in k_], label="exp base state", color='green') # Plot gamma vs k from linear stability analysis of step functio
    ax.set_xlim(k_min, k_max_lim)
    ax.set_ylim(gamma_min, gamma_max_lim)
    
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\gamma$")
    #ax.legend(fontsize="large")
    fig.tight_layout()
    fig.savefig("results/imgs/gamma_vs_k.pdf", dpi=300)
    plt.show()
