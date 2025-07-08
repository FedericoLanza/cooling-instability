import argparse
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--Gamma', type=float, required=True, help='Value of Gamma to use in the filename.')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    return parser.parse_args()

def linear_model(log_beta, a, b):
    return a * log_beta + b

def quadratic_model(log_beta, a2, b2, c2):
    return a2 * log_beta**2 + b2 * log_beta + c2

def process_file(folder_path, Gamma):
    # Load data
    filename = f"values_vs_beta_different_Pe_Gamma_{Gamma:.10g}_zoom.txt"
    file_path = folder_path + filename
    df = pd.read_csv(file_path, sep='\t')  # Assuming tab-separated values
    
    results_linear = []
    results_quadratic = []
    
    plt.figure(figsize=(8, 6))
    
    # Get unique Pe values and define a colormap
    unique_pe = sorted(df['Pe'].unique())
    norm = mcolors.LogNorm(vmin=min(unique_pe), vmax=max(unique_pe))  # Logarithmic scale for Pe
    colormap = cm.viridis
    
    for pe in unique_pe:
        group = df[df['Pe'] == pe]
        beta = group['beta'].values
        log_beta = np.log(beta)
        gamma_max = group['gamma_max'].values
        gamma_max_sigma = group['gamma_max_sigma'].values
        
        # Fit the data with a linear model in log(beta)
        popt_linear, pcov_linear = opt.curve_fit(linear_model, log_beta, gamma_max, sigma=gamma_max_sigma)
        a, b = popt_linear
        sigma_a, sigma_b = np.sqrt(np.diag(pcov_linear))
        
        # Compute root (log(beta) where gamma_max = 0) for linear model
        if a != 0:
            log_beta_root = -b / a
            beta_root = np.exp(log_beta_root)
            
            # Error propagation for beta_root
            sigma_log_beta_root = np.sqrt((sigma_b / a) ** 2 + (b * sigma_a / a**2) ** 2)
            sigma_beta_root = beta_root * sigma_log_beta_root
            
            results_linear.append((pe, beta_root, sigma_beta_root))
        
        # Fit the data with a quadratic model in log(beta)
        popt_quadratic, pcov_quadratic = opt.curve_fit(quadratic_model, log_beta, gamma_max, sigma=gamma_max_sigma)
        a2, b2, c2 = popt_quadratic
        
        # Compute roots for quadratic model
        discriminant = b2**2 - 4*a2*c2
        if discriminant >= 0:
            log_beta_root1 = (-b2 + np.sqrt(discriminant)) / (2*a2)
            log_beta_root2 = (-b2 - np.sqrt(discriminant)) / (2*a2)
            beta_root1, beta_root2 = np.exp(log_beta_root1), np.exp(log_beta_root2)
            
            # Select the root closest to the beta value whose gamma_max is closest to zero
            closest_idx = np.argmin(np.abs(gamma_max))
            closest_beta = beta[closest_idx]
            beta_root_quadratic = min([beta_root1, beta_root2], key=lambda x: abs(x - closest_beta))
            
            sigma_log_beta_root1 = np.sqrt((2 * a2 * log_beta_root1 + b2)**2 * np.diag(pcov_quadratic)[0] + np.diag(pcov_quadratic)[1] + np.diag(pcov_quadratic)[2])
            sigma_beta_root_quadratic = beta_root_quadratic * sigma_log_beta_root1
            
            results_quadratic.append((pe, beta_root_quadratic, sigma_beta_root_quadratic))
        
        # Assign color based on Pe value
        color = colormap(norm(pe))
        
        # Plot data and fitted lines
        plt.errorbar(beta, gamma_max, yerr=gamma_max_sigma, fmt='o', color=color, label=f'Pe={pe}')
        beta_fit = np.logspace(np.log10(min(beta)), np.log10(max(beta)), 100)
        gamma_fit_linear = linear_model(np.log(beta_fit), *popt_linear)
        gamma_fit_quadratic = quadratic_model(np.log(beta_fit), *popt_quadratic)
        #plt.plot(beta_fit, gamma_fit_linear, '--', color=color, label=f'Linear Fit Pe={pe}')
        plt.plot(beta_fit, gamma_fit_quadratic, ':', color=color)
    if tp:
        plt.axvline(x=np.exp(-3.03), color='gray', linestyle='--', label=r'$\beta_c(Pe\to \infty)$')
    else:
        plt.axvline(x=np.exp(-4.27), color='gray', linestyle='--', label=r'$\beta_c(Pe\to \infty)$')
    plt.axhline(0, color='black', linewidth=1)
    plt.xscale('log')
    plt.xlabel('Beta')
    plt.ylabel('Gamma Max')
    plt.legend()
    plt.title('Best Fit Lines for Different Pe Values (Log Scale)')
    
    # Save plot
    plot_filename = folder_path + f"best_fit_plot_Gamma_{Gamma:.10g}.pdf"
    plt.savefig(plot_filename)
    plt.show()
    
    # Save results
    results_linear_df = pd.DataFrame(results_linear, columns=['Pe', 'beta_root', 'beta_root_sigma'])
    results_linear_df.to_csv(folder_path + f'beta_roots_linear_Gamma_{Gamma:.10g}.txt', sep='\t', index=False)
    
    results_quadratic_df = pd.DataFrame(results_quadratic, columns=['Pe', 'beta_root', 'beta_root_sigma'])
    results_quadratic_df.to_csv(folder_path + f'beta_roots_quadratic_Gamma_{Gamma:.10g}.txt', sep='\t', index=False)
    
    return results_linear_df, results_quadratic_df

if __name__ == "__main__":
    args = parse_args()
    Gamma = args.Gamma
    tp = args.tp
    
    outpvart = "output" if not tp else "outppt"
    folder_path = "results/" + outpvart + "_mix/"
    
    df_results_linear, df_results_quadratic = process_file(folder_path, Gamma)
