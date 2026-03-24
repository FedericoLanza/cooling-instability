#!/usr/bin/env python3
"""
compute_plot.py

Usage:
    python compute_plot.py --Pe <Pe_value> --Gamma <Gamma_value>

This script expects two files at paths:
 output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta_l:.10g}/gamma_linear_plot.txt
 output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta_r:.10g}/gamma_linear_plot.txt

Each file is expected to have a header line, then at least two columns:
 k  gamma

It computes transformed variables as specified by the user and plots f0 vs k_tilde.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

 #Enable LaTeX-style rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,  # Default text size (JFM uses ~8pt for labels)
    "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
    "xtick.labelsize": 24,  # Tick labels
    "ytick.labelsize": 24,
    "legend.fontsize": 12,  # Legend size
    "lines.linewidth": 1.5,  # Thin lines
    "lines.markersize": 8,  # Small but visible markers
    "figure.subplot.wspace": 0.35,  # Horizontal spacing
    "figure.subplot.bottom": 0.19,  # Space for x-labels
    "axes.labelpad": 8, #default is 5
})
    
def read_kgamma(path):
    """
    Read file skipping the first header row and return k, gamma as 1D numpy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        if data.size < 2:
            raise ValueError(f"File {path} does not contain two columns after header.")
        k = np.atleast_1d(data[0])
        gamma = np.atleast_1d(data[1])
    else:
        if data.shape[1] < 2:
            raise ValueError(f"File {path} must have at least two columns.")
        k = data[:, 0]
        gamma = data[:, 1]
    return k, gamma

def main():
    parser = argparse.ArgumentParser(description="Compute f0 from two gamma_linear_plot.txt datasets.")
    parser.add_argument("--Pe", type=float, required=True, help="Pe value")
    parser.add_argument("--Gamma", type=float, required=True, help="Gamma value")
    parser.add_argument("--tol", type=float, default=1e-12, help="Tolerance used for interpolation decisions (unused normally).")
    parser.add_argument("--npoints", type=int, default=None,
                        help="Number of points in common k_tilde grid. Default: min(len(file1), len(file2))")
    args = parser.parse_args()

    Pe = args.Pe
    Gamma = args.Gamma

    # set betas
    beta_l = 10.0**(-10.25)           # 10^{-5}
    beta_r = 10.0**(-10)         # 10^{-4.875}

    # compute psi values
    psi_l = -np.log(beta_l)          # psi_l = log(beta_l)
    psi_r = -np.log(beta_r)         # psi_r = -log(beta_r)

    # build file paths using .10g formatting
    path_template = "results/output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}/gamma_linear_plot.txt"
    path_l = path_template.format(Pe=Pe, Gamma=Gamma, beta=beta_l)
    path_r = path_template.format(Pe=Pe, Gamma=Gamma, beta=beta_r)

    print("Reading files:")
    print(" left file:", path_l)
    print(" right file:", path_r)

    try:
        k_l, gamma_l = read_kgamma(path_l)
        k_r, gamma_r = read_kgamma(path_r)
    except Exception as e:
        print("Error while reading input files:", e)
        sys.exit(1)

    # ensure arrays are 1D numpy floats and sorted by k (important for interpolation)
    k_l = np.asarray(k_l, dtype=float)
    gamma_l = np.asarray(gamma_l, dtype=float)
    sort_idx = np.argsort(k_l)
    k_l = k_l[sort_idx]
    gamma_l = gamma_l[sort_idx]

    k_r = np.asarray(k_r, dtype=float)
    gamma_r = np.asarray(gamma_r, dtype=float)
    sort_idx = np.argsort(k_r)
    k_r = k_r[sort_idx]
    gamma_r = gamma_r[sort_idx]

    # compute k_tilde and gamma_tilde for each dataset
    # careful with parentheses: k_tilde = k / (Gamma * psi)
    k_tilde_l = k_l / (Gamma * psi_l)
    gamma_tilde_l = (gamma_l / Gamma + 1.0) / psi_l

    k_tilde_r = k_r / (Gamma * psi_r)
    gamma_tilde_r = (gamma_r / Gamma + 1.0) / psi_r

    # determine overlapping k_tilde range
    kmin = max(k_tilde_l.min(), k_tilde_r.min())
    kmax = min(k_tilde_l.max(), k_tilde_r.max())
    if kmax <= kmin:
        print("Warning: no overlap in k_tilde ranges between the two files.")
        print(f" left range: [{k_tilde_l.min()}, {k_tilde_l.max()}], right range: [{k_tilde_r.min()}, {k_tilde_r.max()}]")
        # still attempt to proceed by using union and interpolating (may extrapolate)
        kmin = min(k_tilde_l.min(), k_tilde_r.min())
        kmax = max(k_tilde_l.max(), k_tilde_r.max())

    # choose number of common points
    if args.npoints is None:
        n_common = min(len(k_tilde_l), len(k_tilde_r))
        n_common = max(50, n_common)  # ensure at least some points if small files
    else:
        n_common = args.npoints

    common_k = np.linspace(kmin, kmax, n_common)

    # interpolate gamma_tilde onto common k_tilde grid
    # np.interp requires x to be increasing; we've sorted k_tilde arrays by k, but psi may be negative causing k_tilde to flip sign ordering.
    # ensure strictly increasing x arrays for interpolation: we'll sort k_tilde arrays and corresponding gamma_tilde.
    def prepare_for_interp(x, y):
        order = np.argsort(x)
        x_s = x[order]
        y_s = y[order]
        # If there are duplicate x values, np.interp will still work but it's safer to collapse duplicates by averaging
        # Remove duplicates:
        unique_x, inv = np.unique(x_s, return_index=True)
        if unique_x.size < x_s.size:
            # average y for identical x's
            new_x = []
            new_y = []
            i = 0
            while i < x_s.size:
                xi = x_s[i]
                j = i
                ys = []
                while j < x_s.size and np.isclose(x_s[j], xi):
                    ys.append(y_s[j])
                    j += 1
                new_x.append(xi)
                new_y.append(np.mean(ys))
                i = j
            x_s = np.array(new_x)
            y_s = np.array(new_y)
        return x_s, y_s

    x_l, y_l = prepare_for_interp(k_tilde_l, gamma_tilde_l)
    x_r, y_r = prepare_for_interp(k_tilde_r, gamma_tilde_r)

    # Now interpolate (will linearly extrapolate outside by using left/right values of np.interp)
    gamma_l_on_common = np.interp(common_k, x_l, y_l, left=y_l[0], right=y_l[-1])
    gamma_r_on_common = np.interp(common_k, x_r, y_r, left=y_r[0], right=y_r[-1])

    # compute f1 and f0 pointwise:
    denom = (1.0 / psi_l - 1.0 / psi_r)
    if np.isclose(denom, 0.0):
        raise ValueError("Denominator (1/psi_l - 1/psi_r) is numerically zero or too small.")

    f1 = (gamma_l_on_common - gamma_r_on_common) / denom
    f0 = gamma_l_on_common - f1 / psi_l
    f0b = gamma_r_on_common - f1 / psi_r

    mask_f1 = f1 < 1
    mask_f0 = f0 > -1
    mask_f1[1] = False
    #mask_f1[51] = False
    #mask_f1[52] = False
    f1 = f1[mask_f1]
    f0 = f0[mask_f1]
    common_k = common_k[mask_f1]
    
    if True:
        # --- Fit ----------------------------------------------------------
        f0_max_pos = np.argmax(f0)
        
        k0_range, f0_range = common_k[f0_max_pos-3:f0_max_pos+4], f0[f0_max_pos-3:f0_max_pos+4]
        coeffs_0, cov_0 = np.polyfit(k0_range, f0_range, deg=2, cov=True)
        
        poly_0 = np.poly1d(coeffs_0)
        k0_fit = np.linspace(k0_range.min(), k0_range.max(), 50)
        f0_fit = poly_0(k0_fit)
        
        km0 = - coeffs_0[1]/(2*coeffs_0[0])
        f0_km0 = - coeffs_0[1]**2/(4*coeffs_0[0]) + coeffs_0[2]
        f0pp_km0 = 2*coeffs_0[0]
        
        k1_range, f1_range = common_k[f0_max_pos-3:f0_max_pos+4], f1[f0_max_pos-3:f0_max_pos+4]
        coeffs_1, cov_1 = np.polyfit(k1_range, f1_range, deg=2, cov=True)
        
        poly_1 = np.poly1d(coeffs_1)
        k1_fit = np.linspace(k1_range.min(), k1_range.max(), 50)
        f1_fit = poly_1(k1_fit)
        
        f1_km0 = coeffs_1[0]*km0**2 + coeffs_1[1]*km0 + coeffs_1[2]
        f1p_km0 = 2*coeffs_1[0]*km0 + coeffs_1[1]
        
        km1 = - f1p_km0 / f0pp_km0
        
        gammam0 = f0_km0
        gammam1 = f1_km0
        
        km0_sigma = np.sqrt( cov_0[0,0] * (coeffs_0[1]/(2*coeffs_0[0]**2))**2 + cov_0[1,1] * (1/(2*coeffs_0[0]))**2)
        km1_sigma = np.sqrt( ((2*coeffs_1[0]*km0_sigma)**2 + cov_1[0,0]*(2*km0)**2 + cov_1[1,1])/f0pp_km0**2 + cov_0[0,0]*(f1p_km0/2*coeffs_0[0])**2 )
        gammam0_sigma = np.sqrt( cov_0[0,0] * (coeffs_0[1]**2/(4*coeffs_0[0]**2))**2 + cov_0[1,1] * (coeffs_0[1]/(2*coeffs_0[0]))**2 + cov_0[2,2] )
        gammam1_sigma = np.sqrt( cov_1[0,0]*km0**4 + ((2*coeffs_1[0]*km0 + coeffs_1[1]) * km0_sigma)**2 + cov_1[1,1]*km0 + cov_1[2,2] )
        
        print("a_k = ", km0, "pm ", km0_sigma, ", b_k = ", km1, "pm ", km1_sigma)
        print("a_gamma = ", gammam0, "pm ", gammam0_sigma, ", b_gamma = ", gammam1 - 1, "pm ", gammam1_sigma,)

        maxgammatilde = np.max(f0 + f1/psi_l)
        gamma_max_l = (maxgammatilde*psi_l - 1)*Gamma
        print(f"gamma_max(psi = {psi_l}) = {gamma_max_l:.10g}")
        print("b_gamma = ", gamma_max_l/Gamma - psi_l*gammam0)
        
    # Plot scatter f0 vs k_tilde
    fig, ax = plt.subplots(1, 2, figsize=(15., 5.))
    
    ax[0].plot(common_k, f0, linewidth=2)
    #ax[0].scatter(common_k, f0b, s=15, label='gamma_r_on_common - f1 / psi_r')
    ax[0].plot(k0_fit, f0_fit, linewidth=2)
    ax[1].plot(common_k, f1, linewidth=2)
    ax[1].plot(k1_fit, f1_fit, linewidth=2, color='red')

    [axi.set_xlim(-0.005, max(common_k)) for axi in ax]
    [axi.set_xlabel(r'$\tilde{\tilde{k}} = k/(\Gamma\psi)$ ') for axi in ax]
    ax[0].set_ylabel(r'$f_0(\tilde{\tilde{k}})$')
    ax[1].set_ylabel(r'$f_1(\tilde{\tilde{k}})$')
    [axi.set_xticks([0, 0.5, 1, 1.5, 2, 2.5]) for axi in ax]

    # Save figure
    outname = f"results/output_mix/f0_and_f1.pdf"
    plt.savefig(outname, dpi=200)
    print(f"Saved scatter plot to {outname}")
    
    #fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

