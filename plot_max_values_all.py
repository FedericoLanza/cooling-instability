# Python program for plotting k_max or gamma_max as a function of Pe, Gamma, beta

import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
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
    parser.add_argument('--plot_Pe', action='store_true', help='Choose whether to plot the data vs Pe (True), or vs the other variables (False).')
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

def load_data_full(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    n_peaks = data[:, 7]
    Ly = data[:, 8]
    gamma_max_full = data[:, 9]
    return n_peaks, Ly, gamma_max_full

# Plotting function
def plot_variable_vs_x(x_variable, vary_variable, fixed_variable, fixed_value, y_variable, folder_name, ax, rescale):
    
    # Load data
    file_path = folder_name + f'values_vs_{x_variable}_different_{vary_variable}_{fixed_variable}_{fixed_value:.10g}.txt'
    #file_path = folder_name + f'values_vs_{x_variable}_different_{vary_variable}.txt'
    Pe, Gamma, beta, k_max, k_max_sigma, gamma_max, gamma_max_sigma = load_data(file_path)
    
    file_path_full = folder_name + 'compare.txt'
    n_peaks, Ly, gamma_max_full = load_data_full(file_path_full)
    
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
    if plot_Pe:
        mask_fixed = True
    else:
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

    # Values found from fitting several points (k_max, gamma_max) in the region \Gamma \in [1e-7, 1e-5], \beta \in [1e-5, 1e-2], \Peclet = 10^3.
    if False:
        # [-5, -3, step=0.25]
        a_k = -0.8956065306
        b_k = -1.701252574
        a_k_err = 0.0006128950766
        b_k_err = 0.005476120772
        cov_ab_k = 3.30823e-06
        a_gamma = -0.7670123872
        b_gamma = -3.63371958
        a_gamma_err = 0.001437601667
        b_gamma_err = 0.01272552975
        cov_ab_gamma = 1.80316e-05
    
    else:
        # [-5, -2.5, step=0.25]
        a_k = -0.8895926296
        b_k = -1.643656674
        a_k_err = 0.0007957415133
        b_k_err = 0.006619171203
        cov_ab_k = 5.14507e-06
        a_gamma = -0.7530394513
        b_gamma = -3.500970335
        a_gamma_err = 0.001803608438
        b_gamma_err = 0.01475799352
        cov_ab_gamma = 2.59932e-05
    
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
            exponent = np.log10(val)
            vary_variable_label = rf'$\{vary_variable}$ = $10^{{{exponent:.3g}}}$'
            marker = 's'
        elif vary_variable == 'beta':
            exponent = np.log10(val)
            vary_variable_label = rf'$\{vary_variable}$ = $10^{{{exponent:.2g}}}$'
            marker = 'o' if plot_Pe else '^'
            
        if (rescale == True and x_variable == 'Pe'):
            if vary_variable == 'Gamma':
                beta_value = fixed_value
                Gamma_value = val
            elif vary_variable == 'beta':
                Gamma_value = 0.01
                beta_value = val
            psi_value = -np.log(beta_value)
            a_c, b_c, a_c_err, b_c_err, cov_ab = ((a_gamma, b_gamma, a_gamma_err, b_gamma_err, cov_ab_gamma) if (y_variable == 'gamma_max') else (a_k, b_k, a_k_err, b_k_err, cov_ab_k))
            sumab = -a_c*psi_value + b_c
            
            y_plot_data_resc = [y * x / (0.01 * sumab) for y, x in zip(y_plot_data, x_data)]
            y_plot_data_resc_err = [np.sqrt( (sigmay * x/(sumab * 0.01))**2 + ((a_c_err*psi_value)**2 + b_c_err**2 - 2*psi_value*cov_ab)*(x*y)**2/(sumab**2 * 0.01)**2 ) for sigmay, y, x in zip(y_err_plot_data, y_plot_data, x_data)]
            
            
            # Plot the values with error bars
            ax.errorbar(x_data, y_plot_data_resc, yerr=y_plot_data_resc_err, label=vary_variable_label, capsize=3, fmt=marker, color=color)
        else:
            # Scatter plot the values without error bars
            ax.scatter(x_data, y_plot_data, label=vary_variable_label, marker=marker, color=color, alpha=1.)

        #elif y_variable == 'gamma_max':
            # Scatter plot with error bars
            #ax.errorbar(x_data, y_plot_data, yerr=y_err_plot_data, label=vary_variable_label, capsize=3, fmt=marker, color=color)
        
        if ( x_variable == 'beta' and fixed_variable == 'Gamma' and abs(val - 1e3) < 1e-2 ):
            Gamma = fixed_value
            if (y_variable == 'gamma_max'):
                mask_beta = x_data > 0
                model, cov = np.polyfit(-np.log(x_data[mask_beta]), y_plot_data[mask_beta], 1, cov=True)
                a = model[0]
                b = model[1]
                sigma_a = np.sqrt(cov[0, 0])
                sigma_b = np.sqrt(cov[1, 1])
                print("gamma_max = Gamma * ( (", a/Gamma, " pm ", sigma_a/Gamma, ") * psi + (", b/Gamma, " pm ", sigma_b/Gamma, ") ) ")
                ax.plot(x_data[mask_beta], [(a * np.log(x) + b) for x in x_data[mask_beta]], color='black', linestyle='-', linewidth=2) # Plot xi curves
                
            elif (y_variable == 'k_max'):
                mask_beta = x_data < 1e-2
                model, cov = np.polyfit(-np.log(x_data[mask_beta]), y_plot_data[mask_beta], 1, cov=True)
                a = model[0]
                b = model[1]
                sigma_a = np.sqrt(cov[0, 0])
                sigma_b = np.sqrt(cov[1, 1])
                print("k_max = Gamma * ( (", a/Gamma, " pm ", sigma_a/Gamma, ") * psi + (", b/Gamma, " pm ", sigma_b/Gamma, ") )")
                ax.plot(x_data[mask_beta], [(a * np.log(x) + b) for x in x_data[mask_beta]], color='black', linestyle='-', linewidth=2) # Plot xi curves
                        
    ax.set_xscale('log')
    
    def k_max_large_Pe(Gamma_, beta_):
        return Gamma_ * ( a_k * np.log(beta_) + b_k)
    
    def gamma_max_large_Pe(Gamma_, beta_):
        return Gamma_ * ( a_gamma * np.log(beta_) + b_gamma)
        
    if x_variable == 'Pe':
    
        if rescale:
            ax.axhline(1, color='black', linestyle='--', linewidth=1)
        else:
            ax.set_yscale('log')
        x_label = 'Pe'
    
        Pe_min = 10
        Pe_max = np.max(Pe_filtered)
        Pe_curve = np.logspace(np.log10(Pe_min), np.log10(Pe_max), 100) # Generate Pe values for the curve
        
        #  Plot the xi function if y_variable is k_max
        if y_variable == 'k_max' and rescale == False:
            
            #k_eff_curve = 1./Pe_curve + 2.*Pe_curve/105  # k_eff = 1/Pe + 2*Pe/105
            #xi_curve = np.pi*(-1 + np.sqrt(1 + 4 * Gamma_value * k_eff_curve)) / (2 * k_eff_curve)

            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                #ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower center')
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-3):
                        continue
                    color = colormap(norm(beta_value))
                    #mask_inset = x_data > 10**3
                    #ax_inset.scatter(x_data[mask_inset], y_plot_data[mask_inset], marker=marker, color=color, alpha=1.)
                    k_max_large_Pe_ = [k_max_large_Pe(Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    #ax_inset.set_xscale('log')
                    #ax_inset.set_yscale('log')
                
            elif vary_variable == 'Gamma':
                # Plot xi for each Gamma value in vary_values
                beta_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
            print('okk')
        elif (y_variable == 'gamma_max' and rescale == False):
            
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-3):
                        continue
                    color = colormap(norm(beta_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    
            elif vary_variable == 'Gamma':
                beta_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Gamma_value, beta_value) for Pe_ in Pe_curve]
                    ax.plot(Pe_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
            print('okg')
                    
    if x_variable == 'Gamma':
    
        ax.set_yscale('log')
        x_label = r'$\Gamma$'
        
        Gamma_min = np.min(Gamma_filtered)
        Gamma_max = np.max(Gamma_filtered)
        Gamma_curve = np.logspace(np.log10(Gamma_min), np.log10(Gamma_max), 100)  # Generate Gamma values for the curve
        Gamma_full = np.logspace(np.log10(Gamma_min),np.log10(Gamma_max), 9)
        
        if y_variable == 'k_max':
            #Gamma_curve = np.logspace(Gamma_min, Gamma_max, 100)
            
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-2.49):
                        continue
                    color = colormap(norm(beta_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    ax.plot(Gamma_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                
                # Plot k^* from full problem simulation
                k_max_full = [2*np.pi*float(n_peaks[i])/Ly[i] for i in range(0,len(Gamma_full))]
                k_max_full_sigma = [2*np.pi/Ly[i] for i in range(0,len(Gamma_full))]
                ax.errorbar(Gamma_full, k_max_full, yerr=k_max_full_sigma, label=r'$k_*$', fmt='o', color='red', capsize=5, alpha=0.6)
                
            elif vary_variable == 'Pe':
                beta_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    k_max_large_Pe_ = [k_max_large_Pe(Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    color = colormap(norm(Pe_value))
                    ax.plot(Gamma_curve, k_max_large_Pe_, color='black', linestyle='--', linewidth=2 ) # Plot xi curves
                    
        elif y_variable == 'gamma_max':
        
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, beta_value in enumerate(vary_values):
                    if (beta_value > 10**-2.49):
                        continue
                    color = colormap(norm(beta_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    ax.plot(Gamma_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                
                # Plot gamma^* from full problem simulation
                ax.scatter(Gamma_full, gamma_max_full, label=r'$\beta$ = $10^{-3}$, full problem', marker='o', color='red')
                #label=r'$\beta = 10^{-3}$, full problem',
                
            elif vary_variable == 'Pe':
                beta_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    color = colormap(norm(Pe_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Gamma_, beta_value) for Gamma_ in Gamma_curve]
                    color = colormap(norm(Pe_value))
                    ax.plot(Gamma_curve, gamma_max_large_Pe_, color='black', linestyle='--', linewidth=2 ) # Plot xi curves
    
    if x_variable == 'beta':
        x_label = r'$\beta$'
        
        beta_min = np.min(beta_filtered)
        beta_max = 10**-2.5
        beta_curve = np.logspace(np.log10(beta_min), np.log10(beta_max), 100)  # Generate beta values for the curve
        
        if y_variable == 'k_max':
            #ax.set_yscale('log')
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, k_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    color = colormap(norm(Pe_value))
                    k_max_large_Pe_ = [k_max_large_Pe(Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, k_max_large_Pe_, color='black', linestyle='--', linewidth=2) # Plot xi curves
                
        elif y_variable == 'gamma_max':
            
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                for idx, Gamma_value in enumerate(vary_values):
                    color = colormap(norm(Gamma_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, gamma_max_large_Pe_, color=color, linestyle='--', linewidth=2) # Plot xi curves
                    
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                for idx, Pe_value in enumerate(vary_values):
                    if (Pe_value < 1000):
                        continue
                    color = colormap(norm(Pe_value))
                    gamma_max_large_Pe_ = [gamma_max_large_Pe(Gamma_value, beta_) for beta_ in beta_curve]
                    ax.plot(beta_curve, gamma_max_large_Pe_, color='black', linestyle='--', linewidth=2) # Plot xi curves
            
    fixed_variable_label = []
    if fixed_variable == 'Pe':
        fixed_variable_label = rf'{fixed_variable} = {fixed_value}'
    else:
        fixed_variable_label = rf'$\{fixed_variable}$ = {fixed_value}'
    
    # Add labels and title
    ax.set_xlabel(x_label)
    if (rescale and y_variable == 'gamma_max'):
        ax.set_ylabel(y_label + rf"/$(\Gamma(a_\gamma + \psi b_\gamma))$")
    elif (rescale and y_variable == 'k_max'):
        ax.set_ylabel(y_label + rf"/$(\Gamma(a_k + \psi b_k))$")
    else:
        ax.set_ylabel(y_label)
    #ax.yaxis.set_label_coords(-0.16, 0.5)

    # Plot a horizontal dashed line at gamma_max = 0 when y_variable is gamma_max
    if (y_variable == 'gamma_max' and rescale==False):
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
    if (y_variable == 'gamma_max'):
        if fixed_variable == 'Pe':
            if fixed_value == 1:
                legend_title = rf'{fixed_variable} = 1'
            elif fixed_value == 10:
                legend_title = rf'{fixed_variable} = 10'
            else:
                exponent = int(np.log10(fixed_value))
                legend_title = rf'{fixed_variable} = $10^{{{exponent}}}$'
        elif fixed_variable == 'Gamma':
            legend_title = rf'$\{fixed_variable}$ = 0.01/Pe'
        elif fixed_variable == 'beta':
            exponent = int(np.log10(fixed_value))
            legend_title = rf'$\{fixed_variable}$ = $10^{{{exponent}}}$'
            
        # Add legend
        if x_variable == 'beta':
            loc = 'best'
            leg = ax.legend(frameon=False, fontsize=13, handletextpad=0.25, handlelength=1.2, labelspacing=0.4, loc=loc, ncol=2, columnspacing=0.5)
            leg.set_title(legend_title, prop={'size': 13})
        elif x_variable == 'Gamma':
            handles, labels = ax.get_legend_handles_labels()
            
            # Swap the two blocks
            handles_main = handles[:6]
            labels_main  = labels[:6]
            handles_last = handles[6:]
            labels_last  = labels[6:]
            
            loc = 'best'
            leg = ax.legend(handles_main, labels_main, frameon=False, fontsize=12, handletextpad=0.25, handlelength=1.2, labelspacing=0.4, loc=loc, ncol=2, columnspacing=0.5)
            leg.set_title(legend_title, prop={'size': 12})
            
            # Add it manually so second legend doesn't overwrite it
            ax.add_artist(leg)

            # Second legend (single entry)
            leg2 = ax.legend(handles_last, labels_last, ncol=2, frameon=False, fontsize=12, handletextpad=-0.2, loc='upper left', bbox_to_anchor=(-0.015, 0.72))  # <-- adjust vertically
        else:
            leg = ax.legend(frameon=False, title=legend_title, fontsize=18, handletextpad=0.25, handlelength=1.2, labelspacing=0.4, loc="best")
            leg.set_title(legend_title, prop={'size': 18})
        title = leg.get_title()
        title.set_ha('left')   # Set the horizontal alignment: 'center', 'left', or 'right'

fixed_values = {
    "Pe": 1000,
    "Gamma": 123,
    "beta": 1e-3,
}

# Main function to execute the script
if __name__ == '__main__':

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define the parsed arguments
    args = parse_arguments()
    
    plot_Pe = args.plot_Pe
    tp = args.tp
    loglog = args.loglog
    rescale = args.rescale
    
    folder_name = create_folder_name(args.tp)
    loglog_str = "_loglog" if loglog else ""
    rescale_str = "_rescaled" if rescale else ""
    
    parameters = ['Gamma', 'beta']
    y_variables = ['gamma_max', 'k_max']
    
    if plot_Pe:
    
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        x_variable = 'Pe'
        vary_variable = 'beta'
        fixed_variable = 'Gamma'
        fixed_value = fixed_values[fixed_variable]
        x_variable_str = x_variable
        
        for j in [0,1]:
                y_variable = y_variables[j]
                ax = axes[j]
                plot_variable_vs_x(x_variable, vary_variable, fixed_variable, fixed_value, y_variable, folder_name, ax, rescale)
    else:
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        fixed_variable = 'Pe'
        fixed_value = fixed_values[fixed_variable]
        x_variable_str = 'Gamma_and_beta'
        
        for i in [0,1]:
            x_variable = parameters[i]
            vary_variable = parameters[abs(i-1)]
            
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
    plt.savefig(folder_name + f'max_values_vs_{x_variable_str}{loglog_str}{rescale_str}.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
