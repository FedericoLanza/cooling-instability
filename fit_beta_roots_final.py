import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

data = np.loadtxt("results/output_mix/values_vs_beta_different_Gamma_Pe_1000.txt")

Gamma_all = data[:, 1]
beta_all  = data[:, 2]
f_all     = data[:, 5]   # y-values
sigma_all = data[:, 6]   # uncertainties


# ---------------------------------------------------------
# 2. FILTER DATA around the root region
# ---------------------------------------------------------

beta_min = 10**(-2.5)
beta_max = 10**(-1.5)

mask = (beta_all >= beta_min) & (beta_all <= beta_max)

Gamma_all = Gamma_all[mask]
beta_all  = beta_all[mask]
f_all     = f_all[mask]
sigma_all = sigma_all[mask]

x_all = np.log(beta_all)     # <-- log(beta)


# ---------------------------------------------------------
# 3. Organize by Gamma
# ---------------------------------------------------------

Gammas = np.unique(Gamma_all)
NG = len(Gammas)

idx_by_Gamma = [np.where(Gamma_all == g)[0] for g in Gammas]


# ---------------------------------------------------------
# 4. MODEL: quadratic in log(beta) with shared root x0
#
# f = a*(x-x0)^2 + b*(x-x0) + c
#
# ---------------------------------------------------------

def residuals(params):
    x0 = params[0]   # shared root in log(beta)
    resid = []
    offset = 1
    
    for ig, idx in enumerate(idx_by_Gamma):
        a = params[offset + 3*ig + 0]
        b = params[offset + 3*ig + 1]
        c = params[offset + 3*ig + 2]

        x = x_all[idx]
        f_obs = f_all[idx]
        sig   = sigma_all[idx]

        f_model = a*(x - x0)**2 + b*(x - x0) + c

        resid.append((f_model - f_obs)/sig)

    return np.concatenate(resid)


# ---------------------------------------------------------
# 5. INITIAL GUESS
# ---------------------------------------------------------

x0_guess = np.median(x_all)
params0 = [x0_guess] + [1.0, 0.0, 0.0] * NG


# ---------------------------------------------------------
# 6. FIT
# ---------------------------------------------------------

result = least_squares(residuals, params0)

x0 = result.x[0]
beta_crit = np.exp(x0)

print("\nEstimated shared root (critical beta):", beta_crit)


# Extract per-Gamma parameters
coeffs = result.x[1:].reshape(NG, 3)


# ---------------------------------------------------------
# 7. PLOT (quadratic fits in log space)
# ---------------------------------------------------------

plt.figure(figsize=(8,6))
colors = plt.cm.viridis(np.linspace(0,1,NG))

x_plot = np.linspace(np.log(beta_min), np.log(beta_max), 300)
beta_plot = np.exp(x_plot)

for ig, g in enumerate(Gammas):
    idx = idx_by_Gamma[ig]
    a, b, c = coeffs[ig]

    plt.errorbar(beta_all[idx], f_all[idx], yerr=sigma_all[idx],
                 fmt='o', color=colors[ig], markersize=4, alpha=0.7)

    f_fit = a*(x_plot - x0)**2 + b*(x_plot - x0) + c
    plt.plot(beta_plot, f_fit, color=colors[ig], lw=2)

plt.axvline(beta_crit, color='k', ls='--', label=r'$\beta_{crit}$')

plt.xscale('log')
plt.xlabel(r'$\beta$')
plt.ylabel('f')
plt.title("Quadratic model in log(beta)")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.show()
