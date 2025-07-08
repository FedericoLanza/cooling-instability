# Python program for plotting k_max or gamma_max as a function of Pe, Gamma, beta

import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
#import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

 #Enable LaTeX-style rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
    "xtick.labelsize": 20,  # Tick labels
    "ytick.labelsize": 20,
    "figure.subplot.wspace": 0.3,  # Horizontal spacing
    "figure.subplot.hspace": 0.3,  # Vertical spacing
    "figure.subplot.left": 0.09,
    "figure.subplot.right": 0.975,
    "figure.subplot.top": 0.95,
    "figure.subplot.bottom": 0.075,  # Space for x-labels
})

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot k_max or gamma_max vs chosen variable for different values of another variable, while keeping the third fixed.')
    parser.add_argument('--x_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The variable to plot on the x-axis.')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--loglog', action='store_true', help='Flag for plotting data in loglog scale.')
    parser.add_argument('--rescale', action='store_true', help='Flag for plotting some data rescaled.')
    return parser.parse_args()

# Create name of the input/output folder
def create_folder_name(tp):
    folder_name = []
    if (tp == False): # tu
        folder_name = 'results/output_mix/'
    else: # tp
        folder_name = 'results/outppt_mix/'
    return folder_name
    
# Load and filter the data
def load_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    Pe = data[:, 0]
    Gamma = data[:, 1]
    beta = data[:, 2]
    k_max = data[:, 3]
    k_max_sigma = data[:, 4]
    gamma_max = data[:, 5]
    gamma_max_sigma = data[:, 6]
    return Pe, Gamma, beta, k_max, k_max_sigma, gamma_max, gamma_max_sigma

# Plotting function
def plot_variable_vs_x(x_variable, vary_variable, fixed_variable, fixed_value, y_variable, folder_name, ax, rescale):
    
    # Load data
    file_path = folder_name + f'values_vs_{x_variable}_different_{vary_variable}_{fixed_variable}_{fixed_value:.10g}.txt'
    #file_path = folder_name + f'values_vs_{x_variable}_different_{vary_variable}.txt'
    Pe, Gamma, beta, k_max, k_max_sigma, gamma_max, gamma_max_sigma = load_data(file_path)
    
    # Create mappings between variable names and data columns
    variables = {
        'Pe': Pe,
        'Gamma': Gamma,
        'beta': beta
    }

    # Select y-data and its associated error based on user choice
    if y_variable == 'k_max':
        y_data = k_max
        y_err = k_max_sigma
        y_label = r'$k_{\max}$'
        
    else:  # y_variable == 'gamma_max'
        y_data = gamma_max
        y_err = gamma_max_sigma
        y_label = r'$\gamma_{\max}$'

    # Filter data for the fixed variable
    mask_fixed = variables[fixed_variable] == fixed_value

    # Apply the mask to all data arrays
    Pe_filtered = Pe[mask_fixed]
    Gamma_filtered = Gamma[mask_fixed]
    beta_filtered = beta[mask_fixed]
    y_filtered = y_data[mask_fixed]
    y_err_filtered = y_err[mask_fixed]

    # Find unique values of the variable to vary
    vary_values = np.unique(variables[vary_variable][mask_fixed])

    # **Create a color gradient**
    norm = mcolors.LogNorm(vmin=min(vary_values), vmax=max(vary_values))  # Log scale normalization
    #colormap = cm.viridis if left_or_right == 0 else cm.viridis.reversed() # Choose colormap (viridis, plasma, inferno, etc.)
    colormap = cm.viridis
    #colormap = sns.color_palette("rocket", as_cmap=True)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)  # Create a color scale
    sm.set_array([])  # Required for colorbar

    # Plotting
    for val in vary_values:
        # Mask for the current vary_variable value within the filtered data
        mask_vary = (variables[vary_variable][mask_fixed] == val)

        # Get the data for the current vary_variable value
        x_data = variables[x_variable][mask_fixed][mask_vary]
        y_plot_data = y_filtered[mask_vary]
        y_err_plot_data = y_err_filtered[mask_vary]

        # **Assign color based on log of vary_variable**
        color = colormap(norm(val))
        
        # Define labels and markers
        if vary_variable == 'Pe':
            if val == 1:
                vary_variable_label = rf'{vary_variable} = 1'
            elif val == 10:
                vary_variable_label = rf'{vary_variable} = 10'
            else:
                exponent = int(np.log10(val))
                vary_variable_label = rf'{vary_variable} = $10^{{{exponent}}}$'
            marker = 'o'
        elif vary_variable == 'Gamma':
            vary_variable_label = rf'$\{vary_variable}$ = {val}'
            marker = 's'
        elif vary_variable == 'beta':
            exponent = int(np.log10(val))
            vary_variable_label = rf'$\{vary_variable}$ = $10^{{{exponent}}}$'
            marker = '^'
            
        if (rescale == True and x_variable == 'beta' and y_variable == 'k_max'):
            if vary_variable == 'Gamma':
                Pe_value = fixed_value
                Gamma_value = val
            elif vary_variable == 'Pe':
                Gamma_value = fixed_value
                Pe_value = val
            kappa_eff = 1./Pe_value + 2.*Pe_value/105
            xi = (-1 + np.sqrt(1 + 4 * Gamma_value * kappa_eff)) / (2 * kappa_eff)
            ax.scatter(x_data, y_plot_data/(xi), label=vary_variable_label, marker=marker, color=color)
        else:
            # Scatter plot without error bars
            ax.scatter(x_data, y_plot_data, label=vary_variable_label, marker=marker, color=color, alpha=1.)
        
        #elif y_variable == 'gamma_max':
            # Scatter plot with error bars
            #ax.errorbar(x_data, y_plot_data, yerr=y_err_plot_data, label=vary_variable_label, capsize=3, fmt=marker, color=color)
        
    ax.set_xscale('log')
    
    
    
    if (x_variable == 'beta'and fixed_variable == 'Gamma'):
        if (y_variable == 'gamma_max'):
            for idx, Pe_value in enumerate(vary_values):
                if (Pe_value > 10**2):
                    mask_beta = y_plot_data > 0
                    model, cov = np.polyfit(-np.log10(x_data[mask_beta]), y_plot_data[mask_beta], 2, cov=True)
                    a = model[0]
                    b = model[1]
                    c = model[2]
                    sigma_a = np.sqrt(cov[0, 0])
                    sigma_b = np.sqrt(cov[1, 1])
                    sigma_c = np.sqrt(cov[2, 2])
                    print("Pe = ", Pe_value)
                    print("gamma_max = (", a, " pm ", sigma_a, ")*log(beta)**2 + (", b, " pm ", sigma_b, ")*log(beta) + ", c, " pm ", sigma_c)
                    #ax.plot(x_data[mask_beta], [(a*np.log10(x)**2 - b*np.log10(x) + c) for x in x_data[mask_beta]], color='black', linestyle='--', linewidth=2) # Plot xi curves
        elif (y_variable == 'k_max'):
            for idx, Pe_value in enumerate(vary_values):
                if (Pe_value > 10**2):
                    mask_beta = x_data < 1e-2
                    model, cov = np.polyfit(-np.log10(x_data[mask_beta]), y_plot_data[mask_beta], 1, cov=True)
                    a = model[0]*np.sqrt(Pe_value)
                    b = model[1]*np.sqrt(Pe_value)
                    sigma_a = np.sqrt(cov[0, 0])
                    sigma_b = np.sqrt(cov[1, 1])
                    print("Pe = ", Pe_value)
                    print("k_max = sqrt(Gamma/Pe)*(", a, " pm ", sigma_a, ")*log(beta) +(", b, " pm ", sigma_b, ")")
                    #ax.plot(x_data[mask_beta], [np.sqrt(1./Pe_value)*(-a*np.log10(x) + b) for x in x_data[mask_beta]], color='black', linestyle='--', linewidth=2) # Plot xi curves
                
        
    a1 = -2.8977944814107963
    a0 = 9.392975223880983
    
    b2 = 0.1258613583258935
    b1 = -0.7360202734455011
    b0 = -1.8893240970789122
    
    def k_max_large_Pe(Pe_, Gamma_, beta_):
        return np.sqrt(Gamma_ / Pe_)*( a1 * np.log10(beta_) + a0)
    
    def gamma_max_large_Pe(Pe_, Gamma_, beta_):
        return Gamma_ * ( b2 * np.log10(beta_)**2 + b1 * np.log10(beta_) +  b0)
        
    if x_variable == 'Pe':
        x_label = 'Pe'
    
        Pe_min = 10**3
        Pe_max = np.max(Pe_filtered)
        Pe_curve = np.logspace(np.log10(Pe_min), np.log10(Pe_max), 100) # Generate Pe values for the curve
        
        #  Plot the xi function if y_variable is k_max
        if y_variable == 'k_max':
            ax.set_yscale('log')
            
            #k_eff_curve = 1./Pe_curve + 2.*Pe_curve/105  # k_eff = 1/Pe + 2*Pe/105
            #xi_curve = np.pi*(-1 + np.sqrt(1 + 4 * Gamma_value * k_eff_curve)) / (2 * k_eff_curve)

            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                #ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower center')
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-2):
                        continue
                    color = colormap(norm(beta_value))
                    #mask_inset = x_data > 10**3
                    #ax_inset.scatter(x_data[mask_inset], y_plot_data[mask_inset], marker=marker, color=color, alpha=1.)
                    k_max_large_Pe_ = [k_max_large_Pe(Pe_, Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    #ax_inset.set_xscale('log')
                    #ax_inset.set_yscale('log')
                
            elif vary_variable == 'Gamma':
                # Plot xi for each Gamma value in vary_values
                beta_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Pe_, Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
        
        elif y_variable == 'gamma_max':
            
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-2):
                        continue
                    color = colormap(norm(beta_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Pe_, Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    
            elif vary_variable == 'Gamma':
                beta_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Pe_, Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    
    if x_variable == 'Gamma':
        x_label = r'$\Gamma$'
        
        Gamma_min = np.min(Gamma_filtered)
        Gamma_max = np.max(Gamma_filtered)
        Gamma_curve = np.logspace(np.log10(Gamma_min), np.log10(Gamma_max), 100)  # Generate Gamma values for the curve
        
        if y_variable == 'k_max':
            if loglog:
                ax.set_yscale('log')
            #Gamma_curve = np.logspace(Gamma_min, Gamma_max, 100)
            
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-2):
                        continue
                    color = colormap(norm(beta_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Pe_value, Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    ax.plot(Gamma_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                
            elif vary_variable == 'Pe':
                beta_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    k_max_large_Pe_ = [k_max_large_Pe(Pe_value, Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    color = colormap(norm(Pe_value))
                    ax.plot(Gamma_curve, k_max_large_Pe_, color='black', linestyle='--', linewidth=2 ) # Plot xi curves
                    
        elif y_variable == 'gamma_max':
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-2):
                        continue
                    color = colormap(norm(beta_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Pe_value, Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    ax.plot(Gamma_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                
            elif vary_variable == 'Pe':
                beta_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    color = colormap(norm(Pe_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Pe_value, Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    color = colormap(norm(Pe_value))
                    ax.plot(Gamma_curve, gamma_max_large_Pe_, color='black', linestyle='--', linewidth=2 ) # Plot xi curves
    
    if x_variable == 'beta':
        x_label = r'$\beta$'
        
        beta_min = np.min(beta_filtered)
        beta_max = 10**-2
        beta_curve = np.logspace(np.log10(beta_min), np.log10(beta_max), 100)  # Generate beta values for the curve
        
        if y_variable == 'k_max':
            
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Pe_value, Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    color = colormap(norm(Pe_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Pe_value, Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, k_max_large_Pe_, color='black', linestyle='--', linewidth=2) # Plot xi curves
                
        elif y_variable == 'gamma_max':
            
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Pe_value, Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    color = colormap(norm(Pe_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Pe_value, Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, gamma_max_large_Pe_, color='black', linestyle='--', linewidth=2) # Plot xi curves
            
    fixed_variable_label = []
    if fixed_variable == 'Pe':
        fixed_variable_label = rf'{fixed_variable} = {fixed_value}'
    else:
        fixed_variable_label = rf'$\{fixed_variable}$ = {fixed_value}'
    
    # Add labels and title
    ax.set_xlabel(x_label)
    if (rescale and y_variable == 'k_max'):
        ax.set_ylabel(y_label + rf"/$\xi$")
    else:
        ax.set_ylabel(y_label)
    #ax.yaxis.set_label_coords(-0.16, 0.5)

    # Plot a horizontal dashed line at gamma_max = 0 when y_variable is gamma_max
    if y_variable == 'gamma_max':
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
    if ((y_variable == 'gamma_max' and x_variable != 'Pe') or (y_variable == 'gamma_max' and x_variable == 'Pe')):
        if fixed_variable == 'Pe':
            if fixed_value == 1:
                legend_title = rf'{fixed_variable} = 1'
            elif fixed_value == 10:
                legend_title = rf'{fixed_variable} = 10'
            else:
                exponent = int(np.log10(fixed_value))
                legend_title = rf'{fixed_variable} = $10^{{{exponent}}}$'
        elif fixed_variable == 'Gamma':
            legend_title = rf'$\{fixed_variable}$ = {fixed_value}'
        elif fixed_variable == 'beta':
            exponent = int(np.log10(fixed_value))
            legend_title = rf'$\{fixed_variable}$ = $10^{{{exponent}}}$'
        
        # Add legend
        if x_variable == 'Gamma' or x_variable == 'beta':
            leg = ax.legend(frameon=False, title=legend_title, fontsize=20, handletextpad=0.25, handlelength=1.2, labelspacing=0.4, loc="best", ncol=2, columnspacing=0.5)
            leg.set_title(legend_title, prop={'size': 20})
        else:
            leg = ax.legend(frameon=False, title=legend_title, fontsize=16, handletextpad=0.25, handlelength=1.2, labelspacing=0.4, loc="best")
            leg.set_title(legend_title, prop={'size': 16})
        title = leg.get_title()
        title.set_ha('left')   # Set the horizontal alignment: 'center', 'left', or 'right'

fixed_values = {
    "Pe": 100,
    "Gamma": 1,
    "beta": 1e-3,
}

# Main function to execute the script
if __name__ == '__main__':

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define the parsed arguments
    args = parse_arguments()
    x_variable = args.x_variable
    tp = args.tp
    loglog = args.loglog
    rescale = args.rescale
    
    folder_name = create_folder_name(args.tp)
    loglog_str = "_loglog" if loglog else ""
    rescale_str = "_rescaled" if rescale else ""
    
    parameters = ['Pe', 'Gamma', 'beta']
    y_variables = ['gamma_max', 'k_max']
    parameters.remove(x_variable)
    
    for i in [0,1]:
        vary_variable = parameters[i]
        fixed_variable = parameters[abs(i-1)]
        fixed_value = fixed_values[fixed_variable]
        for j in [0,1]:
            y_variable = y_variables[j]
            ax = axes[i,j]
            plot_variable_vs_x(x_variable, vary_variable, fixed_variable, fixed_value, y_variable, folder_name, ax, rescale)
            
    labels = ['($a$)', '($b$)', '($c$)', '($d$)']
    for idx, axi in enumerate(axes.flat):
        pos = axi.get_position()
        xx = pos.x0  # left edge of the subplot
        yy = pos.y1  # top edge of the subplot
        x_offset = 0.03 if idx % 2 == 0 else 0.03
        y_offset = 0.04
        # Place the label slightly to the left of the y-label and near the top
        fig.text(xx - x_offset, yy + y_offset, labels[idx], fontsize=24, verticalalignment='top', horizontalalignment='right')
        print(f"Label {labels[idx]} at: x = {xx - x_offset:.3f}, y = {yy + y_offset:.3f}")
        
    #fig.tight_layout()
    
    # Save the plot in the results/output_mix directory
    plt.savefig(folder_name + f'max_values_vs_{x_variable}{loglog_str}{rescale_str}.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
