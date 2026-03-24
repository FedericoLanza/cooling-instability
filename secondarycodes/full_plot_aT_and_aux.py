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
    parser.add_argument('--x_variable', type=str, required=True, choices=['k', 'Pe', 'beta', 'Gamma', 'none'], help='The variable to plot on the x-axis.')
    parser.add_argument('--tp', action='store_true', help='Flag for analyzing the data coming from linear_model_tp.py instead of linear_model_tu.py')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    x_variable = args.x_variable
    tp = args.tp
    
    # File paths
    io_folder = "results/"
    if tp:
        io_folder += "outppt_tfull_mix/"
    else:
        io_folder += "output_tfull_mix/"
    
    file_path = io_folder + f"values.txt"
    
    # Load data, skipping the first row (header)
    data = np.loadtxt(file_path, skiprows=1)
    
    # Extract columns
    Pe_, Gamma_, beta_, k_ = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    mask_k = [True for i in data[:, 0]]
    mask_Pe = [True for i in data[:, 0]]
    mask_Gamma = [True for i in data[:, 0]]
    mask_beta = [True for i in data[:, 0]]
    
    if (x_variable == 'k'):
        k_fixed = None
        Pe_fixed = 100
        Gamma_fixed = 1.
        beta_fixed = 1e-3
        xlabel_str = r"$k$"
        fixed_str = f"Pe_{Pe_fixed}_Gamma_{Gamma_fixed}_beta_{beta_fixed}"
        n_x = 3
        
    if (x_variable == 'Pe'):
        k_fixed = 2*np.pi/3
        Pe_fixed = None
        Gamma_fixed = 1.
        beta_fixed = 1e-3
        xlabel_str = r"Pe"
        fixed_str = f"k_{k_fixed}_Gamma_{Gamma_fixed}_beta_{beta_fixed}"
        n_x = 0
        
    if (x_variable == 'Gamma'):
        k_fixed = 0.19634954084936207
        Pe_fixed = 10000
        Gamma_fixed = None
        beta_fixed = 1e-3
        xlabel_str = r"$\Gamma$"
        fixed_str = f"k_{k_fixed}_Pe_{Pe_fixed}_beta_{beta_fixed}"
        n_x = 1
    
    if (x_variable == 'beta'):
        k_fixed = np.pi
        Pe_fixed = 100
        Gamma_fixed = 1
        beta_fixed = None
        xlabel_str = r"$\psi$"
        fixed_str = f"k_{k_fixed}_Pe_{Pe_fixed}_Gamma_{Gamma_fixed}"
        n_x = 2
    
    output_path_1 = io_folder + f"a_vs_{x_variable}_{fixed_str}.pdf"
    output_path_2 = io_folder + f"max_vs_{x_variable}_{fixed_str}.pdf"
    output_path_3 = io_folder + f"w_vs_{x_variable}_{fixed_str}.pdf"
    
    mask_k = abs(k_ - k_fixed)/k_fixed < 1e-5 if (k_fixed != None) else [True for i in k_]
    mask_Pe = abs(Pe_ - Pe_fixed)/Pe_fixed < 1e-5 if (Pe_fixed != None) else [True for i in Pe_]
    mask_Gamma = abs(Gamma_ - Gamma_fixed)/Gamma_fixed < 1e-5 if (Gamma_fixed != None) else [True for i in Gamma_]
    mask_beta = abs(beta_ - beta_fixed)/beta_fixed < 1e-5 if (beta_fixed != None) else [True for i in beta_]
    
    mask = np.logical_and(np.logical_and(np.logical_and(mask_k, mask_Pe), mask_Gamma), mask_beta)
    
    data = data[mask,:]
    x = data[:,n_x]
    aTm, aTm_sigma, auxm, auxm_sigma = data[:, 4], data[:, 5], data[:, 8], data[:, 9] # coefficients
    length_Tm, length_uxm = data[:, 12], data[:, 13] # lenghts
    x_ux_maxmax, ux_maxmax, x_Tprime_maxmax, Tprime_maxmax = data[:, 14], data[:, 15], data[:, 16], data[:, 17] # finger maximum and relative positions
    wT_direct, wux_direct, wT_fit, wT_fit_sigma, wux_fit, wux_fit_sigma = data[:, 18], data[:, 19], data[:, 20], data[:, 21], data[:, 22], data[:, 23] # peak width
    
    if (x_variable == 'beta'):
        x = [-np.log10(beta) for beta in x]
    
    # Plot finger lengths from indirect and direct measuments
    fig1, ax1 = plt.subplots(1, 2, figsize=(15, 5))
    
    indirect_length_Tm_sigma = [(1/a**2)*s for a, s in zip(aTm, aTm_sigma)]
    indirect_length_uxm_sigma = [(1/a**2)*s for a, s in zip(auxm, auxm_sigma)]
    direct_length_Tm_sigma = [(1./16) for l in length_Tm]
    direct_length_uxm_sigma = [(1./16) for l in length_uxm]
    print("indirect_length_Tm_sigma = ", indirect_length_Tm_sigma)
    print("direct_length_Tm_sigma = ", direct_length_Tm_sigma)
    
    ax1[0].errorbar(x, Tprime_maxmax/aTm, yerr=indirect_length_Tm_sigma, capsize=3, fmt='o', label=r"$T'_{max}/(a_{T})$")
    ax1[1].errorbar(x, (ux_maxmax - 1.)/auxm, yerr=indirect_length_uxm_sigma, capsize=3, fmt='o', label=r"$u_{x,max}/(a_{u_x})$")
    ax1[0].errorbar(x, length_Tm - x_Tprime_maxmax, yerr=direct_length_Tm_sigma, capsize=3, fmt='o', label=r"$l_T - x_{T'_{max}}$", color='red')
    ax1[1].errorbar(x, length_uxm - x_ux_maxmax, yerr=direct_length_uxm_sigma, capsize=3, fmt='o', label=r"$l_{u_x} - x_{u_{x,max}}$", color='red')
    
    domain = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 25)
    print(domain)
    
    if (x_variable == 'Pe'):
        ax1[0].plot(domain, [3*Pe**0.5 for Pe in domain], linestyle="dotted", color="black")
        ax1[1].plot(domain, [1*Pe**0.5 for Pe in domain], linestyle="dotted", color="black")
        ax1[0].plot(domain, [Pe**0.6 for Pe in domain], linestyle="dotted", color="green")
        ax1[1].plot(domain, [Pe**0.6 for Pe in domain], linestyle="dotted", color="green")

    if (x_variable == 'Gamma'):
        ax1[0].plot(domain, [300/Gamma**0.5 for Gamma in domain], linestyle="dotted", color="black")
        ax1[1].plot(domain, [120/Gamma**0.5 for Gamma in domain], linestyle="dotted", color="black")
        ax1[0].plot(domain, [900/Gamma**0.6 for Gamma in domain], linestyle="dotted", color="green")
        ax1[1].plot(domain, [180/Gamma**0.6 for Gamma in domain], linestyle="dotted", color="green")
    
    if (x_variable == 'beta'):
        ax1[0].plot(domain, [5*(psi)**0.5 for psi in domain], linestyle="dotted", color="black")
        ax1[1].plot(domain, [10*(psi)**0.5 for psi in domain], linestyle="dotted", color="black")
        ax1[0].plot(domain, [3.5*(psi) for psi in domain], linestyle="dotted", color="green")
        ax1[1].plot(domain, [3*(psi) for psi in domain], linestyle="dotted", color="green")
        
    if (x_variable == 'k'):
        ax1[0].plot(domain, [20*k**(-1) for k in domain], linestyle="dotted", color="black")
        ax1[1].plot(domain, [20*k**(-1) for k in domain], linestyle="dotted", color="black")
        ax1[0].plot(domain, [25*k**(-0.66) for k in domain], linestyle="dotted", color="green")
        ax1[1].plot(domain, [22*k**(-0.66) for k in domain], linestyle="dotted", color="green")
        
    #[axi.semilogx() for axi in ax1]
    #[axi.semilogy() for axi in ax1]
    [axi.set_xlabel(xlabel_str) for axi in ax1]
    ax1[0].set_ylabel(r"$l_{T_m}$")
    ax1[1].set_ylabel(r"$l_{u_{x,m}}$")
    
    [axi.legend() for axi in ax1]
    
    # Plot peaks maximal value and relative position
    fig2, ax2 = plt.subplots(2, 2, figsize=(15, 12))
    
    ax2[0,0].scatter(x, x_Tprime_maxmax)
    ax2[0,1].scatter(x, Tprime_maxmax)
    ax2[1,0].scatter(x, x_ux_maxmax)
    ax2[1,0].scatter(x, x_ux_maxmax/x_Tprime_maxmax, color='red', label=r"$x_{u_{x,max}}/x_{T'_{max}}$")
    ax2[1,1].scatter(x, ux_maxmax)
    ax2[1,1].scatter(x, ux_maxmax/Tprime_maxmax, color='red', label=r"$u_{x,max}/T'_{max}$")
    
    if (x_variable != 'k'):
        [[ax2[i,j].semilogx() for i in [0,1]] for j in [0,1]]
        [[ax2[i,j].semilogy() for i in [0,1]] for j in [0,1]]
    
    [[ax2[i,j].set_xlabel(xlabel_str) for i in [0,1]] for j in [0,1]]
    ax2[0,0].set_ylabel(r"$x_{T'_{max}}$")
    ax2[0,1].set_ylabel(r"$T'_{max}$")
    ax2[1,0].set_ylabel(r"$x_{u_{x,max}}$")
    ax2[1,1].set_ylabel(r"$u_{x,max}$")

    ax2[1,1].legend()
    ax2[1,0].legend()
    
    # Plot peaks width
    fig3, ax3 = plt.subplots(1, 2, figsize=(15, 5))
    
    w_direct_sigma = [0.005 for i in wT_direct]
    ax3[0].errorbar(x, wT_fit, yerr=wT_fit_sigma, capsize=3, fmt='o', label='fit')
    ax3[1].errorbar(x, wux_fit, yerr=wux_fit_sigma, capsize=3, fmt='o', label='fit')
    ax3[0].errorbar(x, wT_direct, yerr=w_direct_sigma, capsize=3, fmt='o', label='direct')
    ax3[1].errorbar(x, wux_direct, yerr=w_direct_sigma, capsize=3, fmt='o', label='direct')
    
    if (x_variable != 'k'):
        [axi.semilogx() for axi in ax3]
        [axi.semilogy() for axi in ax3]
    
    [axi.set_xlabel(xlabel_str) for axi in ax3]
    ax3[0].set_ylabel(r"$w_{T}$")
    ax3[1].set_ylabel(r"$w_{u_{x}}$")

    ax3[0].legend()

    # Save the plots
    fig1.savefig(output_path_1, dpi=300, bbox_inches='tight')
    fig2.savefig(output_path_2, dpi=300, bbox_inches='tight')
    fig3.savefig(output_path_3, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
