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

def load_data(file_path):
    data = np.loadtxt(file_path, skiprows=1, unpack=True)
    k = data[0, :]
    gamma = data[1, :]
    gamma_sigma = data[2, :]
    return k, gamma, gamma_sigma

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Pe', type=float, help='Value for Peclet number')
    parser.add_argument('--Gamma', type=float, help='Value for heat transfer ratio')
    parser.add_argument('--beta', type=float, help='Value for viscosity ratio')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--latex', action='store_true', help='Flag for plotting in LaTeX style')
    parser.add_argument('--normalize', action='store_true', help='Flag for having the same range values for all plots')
    parser.add_argument('--filter', action='store_true', help='Flag for filtering with Gaussian filter')
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args() # object containing the values of the parsed argument
    
    Pe = args.Pe
    Gamma = args.Gamma
    beta = args.beta
    tp = args.tp
    latex = args.latex
    normalize = args.normalize
    filter = args.filter
    
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
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
        "xtick.labelsize": 24,  # Tick labels
        "ytick.labelsize": 24,
        "figure.subplot.wspace": 0.35,
        "figure.subplot.bottom": 0.15,  # Space for x-labels
        
        })
    
    if Pe == None:
        multi_Pe = True
        Pe_ = [10**a for a in np.arange(-3., 3.01, 0.125)]
        var_ = Pe_
        y_label = 'Pe'
        fig_label = "($c$)"
    else:
        Pe_ = [Pe]
        Pe_str = f"_Pe_{Pe:.10g}"
        output_image += Pe_str
        
    if Gamma == None:
        multi_Gamma = True
        Gamma_ = [10**a for a in np.arange(-6.5, -4.49, 0.03125)]
        for a in np.arange(-6.46875, -5.5, 0.0625):
            Gamma_.remove(10**a)
        Gamma_.append(10**-4.921875)
        Gamma_.sort()
        print(Gamma_)
        var_ = Gamma_
        y_label = r'$\Gamma$'
        fig_label = "($b$)"
    else:
        Gamma_ = [Gamma]
        Gamma_str = f"_Gamma_{Gamma:.10g}"
        output_image += Gamma_str
        
    if beta == None:
        multi_beta = True
        beta_ = [10**a for a in np.arange(-5., -0.99, 0.0625)]
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

    output_image += "_normalized.pdf" if normalize else ".pdf"

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
                    # Load data, skipping the first row
                    k, gamma, gamma_sigma = load_data(file_path)

                    # Filter k values within the range [kmin, kmax]
                    mask = (k >= 0) & (k <= 2.02e-4)
                    k, gamma = k[mask], gamma[mask]
                    
                    # Store filtered data
                    if multi_Pe:
                        data[Pe] = (k, gamma)
                    if multi_Gamma:
                        data[Gamma] = (k, gamma)
                    if multi_beta:
                        data[beta] = (k, gamma)
    
    # Collect all unique k values within the range [0, 6]
    tolerance_decimals = 6  # adjust as needed
    k_values = sorted(set(round(k, tolerance_decimals) for var in data for k in data[var][0] if (0 <= k and k <= 2.02e-4) ))
    
    #print("k_values = ", k_values)

    # Create 2D arrays for k, var, and gamma
    K, B = np.meshgrid(k_values, var_)
    G = np.full_like(K, np.nan, dtype=float)  # Initialize gamma values
    
    # Fill gamma values where available and interpolate missing values
    for var in var_:
        k_vals, gamma_vals = data[var]
        interp_func = interp1d(k_vals, gamma_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        G[np.where(np.isclose(var_, var))[0][0], :] = interp_func(k_values)

    if filter:
        G = gaussian_filter(G, sigma=0.5)
    
    # Define colormap normalization with white centered at gamma = 0
    if normalize:
        vmin, vmax = -5e-5, 5.5e-5
        levels = np.arange(vmin, vmax + 0.1e-5, (vmax - vmin)/21)
    else:
        vmin, vmax = np.nanmin(G), np.nanmax(G)
        levels = 20

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap('coolwarm')
    
    fig = plt.figure(figsize=(6, 8))
    ax = plt.gca()  # Get the current axis

    deltav = 19 if multi_Gamma else 19
    # Define the contours
    contourf = ax.contourf(
    B, K, G,
    levels=256,                     # dense levels → smooth colors
    cmap=cmap,
    norm=norm
    )
    black_levels = np.arange(vmin, vmax + 0.1e-5, (vmax - vmin)/deltav)

    contour_lines = ax.contour(
    B, K, G,
    levels=black_levels,
    colors='black',
    linewidths=0.7
    )

    # Make the γ=0 line thicker
    ax.contour(
    B, K, G,
    levels=[0],
    colors='black',
    linewidths=2
    )
    #contour = plt.contourf(B, K, G, levels=levels, cmap=cmap)
    
    # Define and adjust the colorbar
    if (not normalize):
        cbar = plt.colorbar(contour, ax=ax, location="top", fraction=0.2)
        cbar.set_label(r'$\gamma$', rotation=0, labelpad=2)
        cbar.ax.xaxis.set_label_coords(0.5, 3.1)
        cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Reduce to nbins=nbins ticks
        
    # Plot the contours
    #plt.contour(B, K, G, levels=levels, colors='black', linewidths=0.5)
    # linestyles='solid',
    
    # Highlight the gamma = 0 contour with a thick line
    zero_contour = plt.contour(B, K, G, levels=[0], colors='black', linewidths=3)
    if multi_Gamma:
        plt.xscale('linear')
    else:
        plt.xscale('log')
    plt.yscale('linear')
    
    ax.set_xlabel(y_label)
    vshift = -0.075 # horizontal shift of the y-label (in order to not overlap with the axis numbers)
    ax.xaxis.set_label_coords(0.5, vshift)
    plt.ylim(0, 2e-4)
    plt.ylabel('$k$')
    if multi_Pe:
        #plt.xlim([10**-1, 10**3])
        ax.set_xticks([10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3])
        title_str = rf"$\Gamma = 10^{{-5}}$, $\beta = 10^{{-3}}$"
        plt.ylabel('$k$', color='white')
    elif multi_Gamma:
        #ax.set_xticks([10**-6, 10**-5])
        title_str = rf"Pe $ = 10^{{3}}$, $\beta = 10^{{-3}}$"
        plt.ylabel('$k$', color='white')
    elif multi_beta:
        ax.set_xticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1])
        title_str = fr"Pe $ = 10^{{3}}$, $\Gamma = 10^{{-5}}$"
        plt.ylabel('$k$', labelpad=8)
        
    formatter = ticker.ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2)) # Example: scientific notation for exponents outside -3 to 4
    if (multi_Gamma):
        ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    #ax.xaxis.set_major_formatter(formatter)
    
    #hshift = -0.1
    #ax.yaxis.set_label_coords(hshift, 0.5)

    fig.text(0.5, 0.015, title_str, ha='center')
   # fig.subplots_adjust(bottom=-0.2)
    #fig.subplots_adjust(top=0.1)
    #ax.set_title(title_str, pad=75)
    
    #ylab_xpos_l_b = ax.yaxis.get_label().get_position()[0]  # horizontal position of y-label
    #yshift = 0.29 if multi_beta else 0.29
    #fig.text(ylab_xpos_l_b + yshift, 0.999, fig_label, verticalalignment='top', horizontalalignment='right')
    
    #plt.tight_layout()
    plt.savefig(os.path.join(f"results/outp{letter}t_mix", output_image), dpi=300, bbox_inches='tight')
    plt.show()

