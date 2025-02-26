import argparse
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    return parser.parse_args()

def linear_model(log_beta, a, b):
    return a * log_beta + b

def process_file(folder_path):
    # Load data
    filename = "values_vs_beta_different_Pe_zoom.txt"
    file_path = folder_path + filename
    df = pd.read_csv(file_path, sep='\t')  # Assuming tab-separated values
    
    results = []
    plt.figure(figsize=(8, 6))
    
    for pe, group in df.groupby('Pe'):
        beta = group['beta'].values
        log_beta = np.log(beta)
        gamma_max = group['gamma_max'].values
        gamma_max_sigma = group['gamma_max_sigma'].values
        
        # Fit the data with a linear model in log(beta)
        popt, pcov = opt.curve_fit(linear_model, log_beta, gamma_max, sigma=gamma_max_sigma)
        a, b = popt
        sigma_a, sigma_b = np.sqrt(np.diag(pcov))
        
        # Compute root (log(beta) where gamma_max = 0)
        if a != 0:
            log_beta_root = -b / a
            beta_root = np.exp(log_beta_root)
            
            # Error propagation for beta_root
            sigma_log_beta_root = np.sqrt((sigma_b / a) ** 2 + (b * sigma_a / a**2) ** 2)
            sigma_beta_root = beta_root * sigma_log_beta_root
            
            results.append((pe, beta_root, sigma_beta_root))
        
        # Plot data and fitted line
        plt.errorbar(beta, gamma_max, yerr=gamma_max_sigma, fmt='o', label=f'Pe={pe}')
        beta_fit = np.logspace(np.log10(min(beta)), np.log10(max(beta)), 100)
        gamma_fit = linear_model(np.log(beta_fit), *popt)
        plt.plot(beta_fit, gamma_fit, '--', label=f'Fit Pe={pe}')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.xscale('log')
    plt.xlabel('Beta')
    plt.ylabel('Gamma Max')
    plt.legend()
    plt.title('Best Fit Lines for Different Pe Values (Log Scale)')
    
    # Save plot
    plot_filename = folder_path + "best_fit_plot.png"
    plt.savefig(plot_filename)
    plt.show()
    
    # Save results
    results_df = pd.DataFrame(results, columns=['Pe', 'beta_root', 'beta_root_sigma'])
    results_df.to_csv(folder_path + 'beta_roots.txt', sep='\t', index=False)
    
    return results_df

if __name__ == "__main__":

    args = parse_args()
    tp = args.tp
    
    outpvart = []
    if tp == False:
        outpvart = "output"
    else:
        outpvart = "outppt"
    folder_path = "results/" + outpvart + "_mix/"
    
    df_results = process_file(folder_path)
    # print(df_results)
