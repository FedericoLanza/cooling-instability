import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX-style rendering
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.size": 12,
#})

# File paths
file_path_1 = "results/output_mix/beta_roots.txt"
file_path_2 = "results/outppt_mix/beta_roots.txt"
output_path = "results/output_mix/betac_vs_Pe.png"

# Load data, skipping the first row (header)
data_1 = np.loadtxt(file_path_1, skiprows=1)
data_2 = np.loadtxt(file_path_2, skiprows=1)

# Extract columns
Pe_1, beta_c_1, beta_c_sigma_1 = data_1[:, 0], data_1[:, 1], data_1[:, 2]
Pe_2, beta_c_2, beta_c_sigma_2 = data_2[:, 0], data_2[:, 1], data_2[:, 2]

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(Pe_1, beta_c_1, yerr = beta_c_sigma_1, marker = 'o', linestyle='None', color='b', label=r'Data, constant $u_0$')
plt.errorbar(Pe_2, beta_c_2, yerr = beta_c_sigma_2, marker = 's', linestyle='None', color='r', label=r'Data, constant $\Delta P$')

# Add horizontal dashed lines
plt.axhline(y=0.01398178315, color='k', linestyle='--', label=r'$\beta_c(Pe\to \infty)$ constant $u_0$')
plt.axhline(y=0.048315638, color='gray', linestyle='--', label=r'$\beta_c(Pe\to \infty)$, constant $\Delta P$')

# Labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Pe")
plt.ylabel(r"$\beta_c$")
plt.title(r"$\beta_c$ vs Pe")
plt.legend()

# Save the plot
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
