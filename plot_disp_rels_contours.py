import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', type=float, help='Value for viscosity ratio')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    parser.add_argument('--norm', action='store_true', help='Flag for having the same range values for all plots')
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    Pe = args.Pe
    Gamma = args.Gamma
    beta = args.beta
    tp = args.tp
    latex = args.latex
    norm = args.norm
    
    letter = "p" if tp else "u"
    
    Pe_ = []
    beta_ = []
    Gamma_ = []
    
    multi_Pe = False
    multi_beta = False
    multi_Gamma = False
    
    output_image = "gamma_vs_k_linear_cmap"
    
    # Plot the contour map
    if latex:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 28,
        "axes.titlesize": 28,
        "axes.labelsize": 28,  # Axis labels (JFM ~8pt)
        "xtick.labelsize": 28,  # Tick labels
        "ytick.labelsize": 28
        })
    
    if Pe == None:
        multi_Pe = True
        Pe_ = [10**a for a in np.arange(0.5, 5.26, 0.125)]
        var_ = Pe_
        y_label = 'Pe'
        fig_label = "($c$)"
    else:
        Pe_ = [Pe]
        Pe_str = f"_Pe_{Pe:.10g}"
        output_image += Pe_str
        
    if Gamma == None:
        multi_Gamma = True
        Gamma_ = [2**a for a in np.arange(-2.5, 2.001, 0.125)]
        var_ = Gamma_
        y_label = r'$\Gamma$'
        fig_label = "($b$)"
    else:
        Gamma_ = [Gamma]
        Gamma_str = f"_Gamma_{Gamma:.10g}"
        output_image += Gamma_str
        
    if beta == None:
        multi_beta = True
        beta_ = [10**a for a in np.arange(-4.5, -0.499, 0.125)]
        var_ = beta_
        y_label = r'$\beta$'
        fig_label = "($a$)"
    else:
        beta_ = [beta]
        beta_str = f"_beta_{beta:.10g}"
        output_image += beta_str
    
    if ( (multi_Pe and multi_Gamma) or (multi_Pe and multi_beta) or (multi_Gamma and multi_beta) or (multi_Pe == False and multi_Gamma == False and multi_beta == False) ):
        print("Please fix the value of two parameters.")
        exit(0)

    output_image += "_norm.pdf" if norm else ".pdf"

    # Initialize a dictionary to store data
    data = {}

    # Iterate over the folders
    for Pe in Pe_:
        for Gamma in Gamma_:
            for beta in beta_:
                
                folder_name = f"results/outp{letter}t_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}"
                file_path = os.path.join(folder_name, "gamma_linear_plot.txt")
                #print('file_path = ', file_path)
                if os.path.exists(file_path):
                    #print('ok')
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
    tolerance_decimals = 5  # adjust as needed
    k_values = sorted(set(round(k, tolerance_decimals) for var in data for k in data[var][0] if 0 <= k <= 6))


    # Create 2D arrays for k, var, and gamma
    K, B = np.meshgrid(k_values, var_)
    G = np.full_like(K, np.nan, dtype=float)  # Initialize gamma values
    
    
    # Fill gamma values where available and interpolate missing values
    for var in var_:
        k_vals, gamma_vals = data[var]
        interp_func = interp1d(k_vals, gamma_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        G[np.where(np.isclose(var_, var))[0][0], :] = interp_func(k_values)

    if (len(Pe_) == 1):
        G = gaussian_filter(G, sigma=1)
    
    # Define colormap normalization with white centered at gamma = 0
    if norm:
        vmin, vmax = -3.5, 3.5
    else:
        vmin, vmax = np.nanmin(G), np.nanmax(G)
    #norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap('coolwarm')
    
    fig = plt.figure(figsize=(6, 8))
    ax = plt.gca()  # Get the current axis

    # Define the contours
    levels = 50
    contour = plt.contourf(B, K, G, levels=levels, cmap=cmap, norm=norm)
    
    # Define and adjust the colorbar
    if (norm==False):
        cbar = plt.colorbar(contour, ax=ax, location="top", fraction=0.08)
        cbar.set_label(r'$\gamma$', rotation=0, labelpad=2)
        cbar.ax.xaxis.set_label_coords(0.5, 3.1)
        cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Reduce to nbins=nbins ticks
        
    # Plot the contours
    plt.contour(B, K, G, levels=10, colors='black', linewidths=0.5)
    # linestyles='solid',
    
    # Highlight the gamma = 0 contour with a thick line
    zero_contour = plt.contour(B, K, G, levels=[0], colors='black', linewidths=1.5)

    plt.xscale('log')
    plt.yscale('linear')  # Since values varies exponentially
    
    ax.set_xlabel(y_label)
    vshift = -0.075 if multi_beta else -0.075 # horizontal shift of the y-label (in order to not overlap with the axis numbers)
    ax.xaxis.set_label_coords(0.5, vshift)
    plt.ylim(0, 5.5)
    plt.ylabel('$k$')
    if multi_Pe:
        plt.xlim(np.sqrt(10), 1e5)
        ax.set_xticks([10, 10**2, 10**3, 10**4])
        title_str = rf"$\Gamma = {Gamma:.10g}$, $\beta = 10^{{-3}}$"
        plt.ylabel('$k$', color='white')
    elif multi_Gamma:
        ax.set_xticks([2**-1, 2**0, 2**1])
        title_str = rf"Pe $ = 10^{{2}}$, $\beta = 10^{{-3}}$"
        plt.ylabel('$k$', color='white')
    elif multi_beta:
        ax.set_xticks([10**-4, 10**-3, 10**-2, 10**-1])
        title_str = fr"Pe $ = 10^{{2}}$, $\Gamma = {Gamma:.10g}$"

    hshift = -0.1
    ax.yaxis.set_label_coords(hshift, 0.5)

    fig.text(0.5, 0.015, title_str, ha='center')
    fig.subplots_adjust(bottom=-0.2)
    #fig.subplots_adjust(top=0.1)
    #ax.set_title(title_str, pad=75)
    
    ylab_xpos_l_b = ax.yaxis.get_label().get_position()[0]  # horizontal position of y-label
    yshift = 0.29 if multi_beta else 0.29
    fig.text(ylab_xpos_l_b + yshift, 0.999, fig_label, verticalalignment='top', horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(f"results/outp{letter}t_mix", output_image), dpi=300, bbox_inches='tight')
    plt.show()

