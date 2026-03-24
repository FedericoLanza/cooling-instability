# Python program for fitting k_max or gamma_max as a function of Gamma and beta

#!/usr/bin/env python3
import argparse
import numpy as np

def load_data(path, fit, sigma_is_std=False):
    """
    Reads: col2=x, col3=beta, col4=y, col5=variance(or std) on y.
    Ignores other columns.
    """
    if (fit == "k"):
        usecols = (1,2,3,4)
    elif (fit == "gamma"):
        usecols = (1,2,5,6)
    data = np.genfromtxt(path, comments="#", usecols=usecols, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    x, beta, y, errcol = data.T
    if sigma_is_std:
        var = errcol**2
    else:
        var = errcol
    return x, beta, y, var

def wls_fit(x, beta, y, var):
    """
    Weighted LS for y = x*(-a*log(beta) + b) = a*(-x*log(beta)) + b*(x)
    Returns fit results and uncertainties.
    """
    # Design matrix: columns = [-x*lpg(beta), x]
    X = np.column_stack([x * np.log(beta), x])
    y = np.asarray(y)
    var = np.asarray(var)

    # Mask valid rows
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(var) & (var > 0)
    X = X[mask]
    y = y[mask]
    var = var[mask]

    if X.shape[0] < 2:
        raise ValueError("Not enough valid rows to fit (need at least 2).")

    w = 1.0 / var

    # Weighted normal equations
    XtW = X.T * w
    XtWX = XtW @ X
    XtWy = XtW @ y

    # Solve for [a, b]
    params = np.linalg.solve(XtWX, XtWy)
    a_hat, b_hat = params

    # Residuals and chi^2
    y_hat = X @ params
    resid = y - y_hat
    chi2 = np.sum((resid**2) * w)
    dof = X.shape[0] - X.shape[1]

    # Covariance matrix
    cov_known = np.linalg.inv(XtWX)
    cov_scaled = cov_known * (chi2 / dof) if dof > 0 else np.full_like(cov_known, np.nan)

    se_known = np.sqrt(np.diag(cov_known))
    se_scaled = np.sqrt(np.diag(cov_scaled))

    return {
        "a": a_hat,
        "b": b_hat,
        "se_a_known": se_known[0],
        "se_b_known": se_known[1],
        "se_a_scaled": se_scaled[0],
        "se_b_scaled": se_scaled[1],
        "cov_ab_known":  cov_known[0,1],
        "cov_ab_scaled": cov_scaled[0,1],
        "rho_ab_known":  cov_known[0,1] / (se_known[0]*se_known[1]),
        "rho_ab_scaled": cov_scaled[0,1] / (se_scaled[0]*se_scaled[1]),
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof if dof > 0 else np.nan,
        "n_used": X.shape[0]
    }

def main():
    ap = argparse.ArgumentParser(description="Weighted fit for y = x*(-a*log(beta) + b)")
    ap.add_argument("--file", default="results/output_mix/values_vs_Gamma_different_beta_Pe_1000_fit.txt", help="Path to file.txt")
    ap.add_argument("--sigma-is-std", action="store_true",
                    help="Set if the 5th column is standard deviation (σ), not variance (σ²).")
    ap.add_argument("--fit", type=str, required=True, choices=['k', 'gamma'], help='The variable to plot on the x-axis.')

    args = ap.parse_args()

    x, beta, y, var = load_data(args.file, args.fit, sigma_is_std=args.sigma_is_std)
    res = wls_fit(x, beta, y, var)


    print("# Fit for finding ", args.fit, " from the model y = x*(-a*log(beta) + b)")
    print(f"Used rows: {res['n_used']}, dof: {res['dof']}")
    print(f"a = {res['a']:.10g}")
    print(f"b = {res['b']:.10g}")
    print()
    print("Uncertainties assuming provided variances are exact:")
    print(f"  SE(a) = {res['se_a_known']:.10g}")
    print(f"  SE(b) = {res['se_b_known']:.10g}")
    print()
    print("Uncertainties scaled by reduced chi² (if variances are only up to a scale):")
    print(f"  SE(a) = {res['se_a_scaled']:.10g}")
    print(f"  SE(b) = {res['se_b_scaled']:.10g}")
    print()
    print(f"chi² = {res['chi2']:.6g},  reduced chi² = {res['reduced_chi2']:.6g}")
    print(f"cov(a,b) [scaled] = {res['cov_ab_scaled']:.6g}")
    print(f"corr(a,b) [scaled] = {res['rho_ab_scaled']:.6g}")

if __name__ == "__main__":
    main()
