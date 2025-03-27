import argparse
import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX-style rendering
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.size": 12,
#})


def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    tp = args.tp
    
    # File paths
    io_folder = "results/"
    if tp:
        io_folder += "outppt_mix/"
    else:
        io_folder += "output_mix/"
    output_path = io_folder + "betac_vs_Pe.png"

    # Create the output figure
    plt.figure(figsize=(8, 6))

    # Array for all values of Gamma considered
    Gamma_ = [2**a for a in np.arange(-1., 3., 1)]

    for Gamma in Gamma_:

        file_path = io_folder + f"beta_roots_quadratic_Gamma_{Gamma:.10g}.txt"

        # Load data, skipping the first row (header)
        data = np.loadtxt(file_path, skiprows=1)

        # Extract columns
        Pe_, beta_c_, beta_c_sigma_ = data[:, 0], data[:, 1], data[:, 2]

        # Plot data
        plt.errorbar(Pe_, beta_c_, yerr = beta_c_sigma_, marker = 'o', label=fr'$\Gamma = {Gamma:.10g}$')

    # Add horizontal dashed lines
    if tp:
        plt.axhline(y=np.exp(-3.03), color='k', linestyle='--', label=r'$\beta_c(Pe\to \infty)$')
    else:
        plt.axhline(y=np.exp(-4.27), color='gray', linestyle='--', label=r'$\beta_c(Pe\to \infty)$')

    # Labels and title
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Pe")
    plt.ylabel(r"$\beta_c$")
    if tp:
        plt.title(r"$\beta_c$ vs Pe, constant $\Delta P$")
    else:
        plt.title(r"$\beta_c$ vs Pe, constant $u_0$")
    plt.legend()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
