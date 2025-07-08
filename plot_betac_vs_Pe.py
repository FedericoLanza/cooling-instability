import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

 #Enable LaTeX-style rendering
    

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
    "xtick.labelsize": 24,  # Tick labels
    "ytick.labelsize": 24,
    "legend.fontsize": 12,  # Legend size
    "lines.linewidth": 1.5,
    "lines.markersize": 8,
    "figure.subplot.wspace": 0.35,  # Horizontal spacing
    "figure.subplot.bottom": 0.15,  # Space for x-labels
    "figure.subplot.left": 0.15,  # Space for y-labels
    "axes.labelpad": 8,
})


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
    output_path = io_folder + "betac_vs_Pe.pdf"

    # Create the output figure
    plt.figure(figsize=(8, 6))

    # Array for all values of Gamma considered
    Gamma_ = [2**a for a in np.arange(-1., 3., 1)]


    # **Create a color gradient**
    norm = mcolors.LogNorm(vmin=min(Gamma_), vmax=max(Gamma_))  # Log scale normalization
    colormap = cm.viridis  # Choose colormap (viridis, plasma, inferno, etc.)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)  # Create a color scale
    sm.set_array([])  # Required for colorbar
    
    for Gamma in Gamma_:
    
        color = colormap(norm(Gamma))
        
        file_path = io_folder + f"beta_roots_quadratic_Gamma_{Gamma:.10g}.txt"

        # Load data, skipping the first row (header)
        data = np.loadtxt(file_path, skiprows=1)

        # Extract columns
        Pe_, beta_c_, beta_c_sigma_ = data[:, 0], data[:, 1], data[:, 2]

        # Plot data
        #plt.errorbar(Pe_, beta_c_, yerr = beta_c_sigma_, marker = 'o', label=fr'$\Gamma = {Gamma:.10g}$', color=color)
        plt.plot(Pe_, beta_c_, marker = 'o', label=fr'$\Gamma = {Gamma:.10g}$', color=color)

    # Add horizontal dashed lines
    if tp:
        plt.axhline(y=np.exp(-3.03), color='k', linestyle='--')
    else:
        plt.axhline(y=np.exp(-4.27), color='gray', linestyle='--')
        #plt.text(x, y, r'$\beta_c(Pe\to \infty)$')
    
    # Labels and title
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Pe")
    plt.ylabel(r"$\beta_c$")

    plt.legend(frameon=False,fontsize="medium")

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
