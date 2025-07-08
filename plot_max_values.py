# Python program for plotting k_max or gamma_max as a function of Pe, Gamma, beta

import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

 #Enable LaTeX-style rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 21,
    "axes.titlesize": 24,
    "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
    "xtick.labelsize": 20,  # Tick labels
    "ytick.labelsize": 20
})

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot k_max or gamma_max vs chosen variable for different values of another variable, while keeping the third fixed.')
    parser.add_argument('--x_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The variable to plot on the x-axis.')
    parser.add_argument('--y_variable', type=str, required=True, choices=['k_max', 'gamma_max'], help='The variable to plot on the y-axis.')
    parser.add_argument('--vary_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The variable for which to plot different values.')
    parser.add_argument('--fixed_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The variable to keep fixed.')
    parser.add_argument('--fixed_value', type=float, required=True, help='The value of the fixed variable.')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    parser.add_argument('--loglog', action='store_true', help='Flag for plotting data in loglog scale.')
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
def plot_variable_vs_x(x_variable, vary_variable, fixed_variable, fixed_value, y_variable, loglog, folder_name):
    
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
    colormap = cm.viridis.reversed() # Choose colormap (viridis, plasma, inferno, etc.)
    #colormap = sns.color_palette("rocket", as_cmap=True)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)  # Create a color scale
    sm.set_array([])  # Required for colorbar

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for val in vary_values:
        # Mask for the current vary_variable value within the filtered data
        mask_vary = (variables[vary_variable][mask_fixed] == val)

        # Get the data for the current vary_variable value
        x_data = variables[x_variable][mask_fixed][mask_vary]
        y_plot_data = y_filtered[mask_vary]
        y_err_plot_data = y_err_filtered[mask_vary]
        
        # Rescale data
        #if y_variable == 'k_max' and vary_variable == 'beta':
        #    y_plot_data = [-yy/np.log(val) for yy in y_plot_data]
        #    y_err_plot_data = y_err_filtered[mask_vary]


        # **Assign color based on log of vary_variable**
        color = colormap(norm(val))
        
        # Define labels
        vary_variable_label = []
        if vary_variable == 'Pe':
            vary_variable_label = rf'{vary_variable} = {val}'
        else:
            vary_variable_label = rf'$\{vary_variable}$ = {val}'
        
        if y_variable == 'k_max':
            # Scatter plot without error bars
            plt.scatter(x_data, y_plot_data, label=vary_variable_label, color=color)
        elif y_variable == 'gamma_max':
            # Scatter plot with error bars
            plt.errorbar(x_data, y_plot_data, yerr=y_err_plot_data, label=vary_variable_label, capsize=3, fmt='o', color=color)
        
    plt.xscale('log')
    if loglog:
        plt.yscale('log')
    
    x_label = []
    if x_variable == 'Pe':
        x_label = 'Pe'
    
        #  Plot the xi function if y_variable is k_max
        if y_variable == 'k_max':
            Pe_min = np.min(Pe_filtered)
            Pe_max = np.max(Pe_filtered)
            Pe_curve = np.logspace(np.log10(Pe_min), np.log10(Pe_max), 100)  # Generate Pe values for the curve
            k_eff_curve = 1./Pe_curve + 2.*Pe_curve/105  # k_eff = 1/Pe + 2*Pe/105

            # Check if we should use a fixed Gamma or vary Gamma
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_value * k_eff_curve)) / (2 * k_eff_curve)
                #xi_curve_resc = [xi_curve*np.log()]
                plt.plot(Pe_curve, xi_curve, color='black', linestyle='--', linewidth=2, label=r'$\xi = (-1 \pm \sqrt{1 + 4 \Gamma \kappa_{eff}})/(2\kappa_{eff})$') # Plot xi curve

            elif vary_variable == 'Gamma':
                # Plot xi for each Gamma value in vary_values
                for idx, Gamma_val in enumerate(vary_values):
                    xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_val * k_eff_curve)) / (2 * k_eff_curve)
                    #plt.plot(Pe_curve, xi_curve, linestyle='--', linewidth=2, label=r'$\xi$ = $\xi$(Pe, $\Gamma$ = ' + f'{Gamma_val}' + ')' ) # Plot xi curves
        
    if x_variable == 'Gamma':
        x_label = r'$\Gamma$'
        
        if y_variable == 'k_max':
                
            Gamma_min = np.min(Gamma_filtered)
            Gamma_max = np.max(Gamma_filtered)
            Gamma_curve = np.logspace(np.log10(Gamma_min), np.log10(Gamma_max), 100)  # Generate Gamma values for the curve
            
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                kappa_eff = 1./Pe_value + 2*Pe_value/105
                xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_curve * kappa_eff)) / (2 * kappa_eff)
                plt.plot(Gamma_curve, 2*xi_curve, color='black', label=r'$\xi$ = $\xi$(Pe = 100, $\Gamma$)', linestyle='--', linewidth=2) # Plot xi curve
                
            elif vary_variable == 'Pe':
                # Plot xi for each Gamma value in vary_values
                for idx, Pe_val in enumerate(vary_values):
                    kappa_eff = 1. / Pe_val + 2 * Pe_val / 105
                    xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_curve * kappa_eff)) / (2 * kappa_eff)
                    #plt.plot(Gamma_curve, xi_curve, linestyle='--', linewidth=2, label=r'$\xi$ = $\xi$(Pe = ' + f'{Pe_val}' ', $\Gamma$)' ) # Plot xi curves
                    
    if x_variable == 'beta':
        x_label = r'$\beta$'
        
        if y_variable == 'k_max':
            beta_min = np.min(Gamma_filtered)
            beta_max = np.max(Gamma_filtered)
            beta_curve = np.logspace(np.log10(beta_min), np.log10(beta_max), 100)  # Generate beta values for the curve
            if fixed_variable == 'Pe':
                beta_fit_curve = - Gamma_curve
                plt.plot(beta_curve, beta_fit_curve, color='black', label=r'$\xi$ = $\xi$(Pe = 100, $\Gamma$)', linestyle='--', linewidth=2) # Plot beta curve
                
            
        
    fixed_variable_label = []
    if fixed_variable == 'Pe':
        fixed_variable_label = rf'{fixed_variable} = {fixed_value}'
    else:
        fixed_variable_label = rf'$\{fixed_variable}$ = {fixed_value}'
    
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label, rotation=0)
    ax.yaxis.set_label_coords(-0.14, 0.5)
    #plt.ylabel(y_label + r'/($\log$ $\beta$)')
    #plt.title(rf'{y_label} vs ' + x_label + ' for ' + fixed_variable_label + rf' = {fixed_value}')
    #plt.title(y_label + ' vs ' + x_label + ' for ' + fixed_variable_label)
    # Remove grid
    plt.grid(False)

    # Plot a horizontal dashed line at gamma_max = 0 when y_variable is gamma_max
    if y_variable == 'gamma_max':
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        
    if y_variable == 'k_max':
        # Add legend
        plt.legend(frameon=False, handletextpad=0.25)

    #plt.tight_layout()
    fig.subplots_adjust(left=0.15, bottom=0.125)
    
    # Save the plot in the results/output_mix directory
    
    loglog_str = "_loglog" if loglog else ""
    plt.savefig(folder_name + f'{y_variable}_vs_{x_variable}_different_{vary_variable}_fixed_{fixed_variable}_{fixed_value}{loglog_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# Main function to execute the script
if __name__ == '__main__':
    args = parse_arguments()
    
    folder_name = create_folder_name(args.tp)
    
    # Validate the combination of inputs
    if args.x_variable == args.vary_variable or args.x_variable == args.fixed_variable or args.vary_variable == args.fixed_variable:
        raise ValueError("The x_variable, vary_variable, and fixed_variable must all be different.")

    # Call the plotting function with parsed arguments
    plot_variable_vs_x(args.x_variable, args.vary_variable, args.fixed_variable, args.fixed_value, args.y_variable, args.loglog, folder_name)
