# Python program for plotting k_max or gamma_max as a function of Pe, Gamma, beta

import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot k_max or gamma_max vs chosen variable for different values of another variable, while keeping the third fixed.')
    parser.add_argument('--x_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The variable to plot on the x-axis.')
    parser.add_argument('--y_variable', type=str, required=True, choices=['k_max', 'gamma_max'], help='The variable to plot on the y-axis.')
    parser.add_argument('--vary_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The variable for which to plot different values.')
    parser.add_argument('--fixed_variable', type=str, required=True, choices=['Pe', 'beta', 'Gamma'], help='The variable to keep fixed.')
    parser.add_argument('--fixed_value', type=float, required=True, help='The value of the fixed variable.')
    return parser.parse_args()

# Load and filter the data
def load_data(file_path='results/outppt_mix/k_max.txt'):
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
def plot_variable_vs_x(x_variable, vary_variable, fixed_variable, fixed_value, y_variable):
    # Load data
    Pe, Gamma, beta, k_max, k_max_sigma, gamma_max, gamma_max_sigma = load_data()

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
        y_label = 'k_max'
    else:  # y_variable == 'gamma_max'
        y_data = gamma_max
        y_err = gamma_max_sigma
        y_label = 'gamma_max'

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

    # Plotting
    plt.figure(figsize=(8, 6))

    for val in vary_values:
        # Mask for the current vary_variable value within the filtered data
        mask_vary = (variables[vary_variable][mask_fixed] == val)

        # Get the data for the current vary_variable value
        x_data = variables[x_variable][mask_fixed][mask_vary]
        y_plot_data = y_filtered[mask_vary]
        y_err_plot_data = y_err_filtered[mask_vary]
        
        if y_variable == 'k_max' and vary_variable == 'beta': # rescale data
            y_plot_data = [-yy/np.log(val) for yy in y_plot_data]
            y_err_plot_data = y_err_filtered[mask_vary]

        # Scatter plot with error bars
        plt.errorbar(x_data, y_plot_data, yerr=y_err_plot_data, label=f'{vary_variable} = {val}', capsize=3, fmt='o')

    if x_variable == 'Pe':
        plt.xscale('log')
        
        #  Plot the xi function if y_variable is k_max
        if y_variable == 'k_max':
            Pe_min = np.min(Pe_filtered)
            Pe_max = np.max(Pe_filtered)
            Pe_curve = np.logspace(np.log10(Pe_min), np.log10(Pe_max), 100)  # Generate Pe values for the curve
            k_eff_curve = 1 / Pe_curve + 2 * Pe_curve / 105  # k_eff = 1/Pe + 2*Pe/105

            # Check if we should use a fixed Gamma or vary Gamma
            if fixed_variable == 'Gamma':
                Gamma_value = fixed_value
                xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_value * k_eff_curve)) / (2 * k_eff_curve)
                xi_curve_resc = [xi*np.log()]
                plt.plot(Pe_curve, xi_curve, color='black', linestyle='--', linewidth=2, label=r'$\xi = (-1 \pm \sqrt{1 + 4 \Gamma \kappa_{eff}})/(2\kappa_{eff})$') # Plot xi curve

            elif vary_variable == 'Gamma':
                # Plot xi for each Gamma value in vary_values
                for idx, Gamma_val in enumerate(vary_values):
                    xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_val * k_eff_curve)) / (2 * k_eff_curve)
                    plt.plot(Pe_curve, xi_curve, linestyle='--', linewidth=2, label=r'$\xi$ = $\xi$(Pe, $\Gamma$ = ' + f'{Gamma_val}' + ')' ) # Plot xi curves
        
    if x_variable == 'Gamma':
        plt.xscale('log')
        plt.yscale('log')
        
        if y_variable == 'k_max':
            Gamma_min = np.min(Gamma_filtered)
            Gamma_max = np.max(Gamma_filtered)
            Gamma_curve = np.logspace(np.log10(Gamma_min), np.log10(Gamma_max), 100)  # Generate Gamma values for the curve
            
            if fixed_variable == 'Pe':
                Pe_value = fixed_value
                kappa_eff = 1. / Pe_value + 2 * Pe_value / 105
                xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_curve * kappa_eff)) / (2 * kappa_eff)
                plt.plot(Gamma_curve, xi_curve, color='black', label=r'$\xi$ = $\xi$(Pe = 100, $\Gamma$)', linestyle='--', linewidth=2) # Plot xi curve
                
            elif vary_variable == 'Pe':
                # Plot xi for each Gamma value in vary_values
                for idx, Pe_val in enumerate(vary_values):
                    kappa_eff = 1. / Pe_val + 2 * Pe_val / 105
                    xi_curve = (-1 + np.sqrt(1 + 4 * Gamma_curve * kappa_eff)) / (2 * kappa_eff)
                    plt.plot(Gamma_curve, xi_curve, linestyle='--', linewidth=2, label=r'$\xi$ = $\xi$(Pe = ' + f'{Pe_val}' ', $\Gamma$)' ) # Plot xi curves
        
    if x_variable == 'beta':
        plt.xscale('log')
    
    # Add labels and title
    plt.xlabel(x_variable)
    plt.ylabel(y_label)
    #plt.ylabel(y_label + r'/($\log$ $\beta$)')
    plt.title(f'{y_label} vs {x_variable} for {fixed_variable} = {fixed_value}')

    # Remove grid
    plt.grid(False)

    # Plot a horizontal dashed line at gamma_max = 0 when y_variable is gamma_max
    if y_variable == 'gamma_max':
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Add legend
    plt.legend()

    # Save the plot in the results/output_mix directory
    plt.savefig(f'results/outppt_mix/{y_label}_vs_{x_variable}_different_{vary_variable}_fixed_{fixed_variable}_{fixed_value}_pt.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main function to execute the script
if __name__ == '__main__':
    args = parse_arguments()

    # Validate the combination of inputs
    if args.x_variable == args.vary_variable or args.x_variable == args.fixed_variable or args.vary_variable == args.fixed_variable:
        raise ValueError("The x_variable, vary_variable, and fixed_variable must all be different.")

    # Call the plotting function with parsed arguments
    plot_variable_vs_x(args.x_variable, args.vary_variable, args.fixed_variable, args.fixed_value, args.y_variable)
