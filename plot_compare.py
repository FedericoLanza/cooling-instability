import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
"font.size": 24,
"axes.labelsize": 24,  # Axis labels (JFM ~8pt)
"xtick.labelsize": 24,  # Tick labels
"ytick.labelsize": 24,
"legend.fontsize": 20,  # Legend size
"lines.linewidth": 1.5,
"lines.markersize": 8,
"figure.subplot.wspace": 0.35,  # Horizontal spacing
"figure.subplot.bottom": 0.15,  # Space for x-labels
"axes.labelpad": 8,
"lines.markersize": 6,
})


# Initialize lists to store data
Pe = []
k_max = []
k_max_full = []
k_max_full_sigma = []
n_peaks = []
Ly = []
gamma_max = []
gamma_max_full = []

# Read data from the file
folder_name = "results/output_mix/"
with open(folder_name + "compare.txt", "r") as file:
    # Skip the header
    next(file)
    # Read each line after the header
    for line in file:
        values = line.split()
        # Check if the line has the expected number of columns (9 in this case)
        if len(values) == 10:
            Pe.append(float(values[0]))
            k_max.append(float(values[3]))
            n_peaks.append((values[7]))
            Ly.append(float(values[8]))
            gamma_max.append(float(values[5]))
            gamma_max_full.append(float(values[9]))

Pe = np.array(Pe)
k_max = np.array(k_max)
n_peaks = np.array(n_peaks)
Ly = np.array(Ly)
gamma_max = np.array(gamma_max)
gamma_max_full = np.array(gamma_max_full)

mask = np.array(gamma_max_full) > 0

# Apply mask
Pe_masked = Pe[mask]
n_peaks_masked = n_peaks[mask]
Ly_masked = Ly[mask]
n_points_inst = len(Pe_masked)
print(n_points_inst)
k_max_full = [2*np.pi*float(n_peaks_masked[i])/Ly_masked[i] for i in range(0,n_points_inst)]
k_max_full_sigma = [2*np.pi/Ly_masked[i] for i in range(0,n_points_inst)]

# Create figure
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Plot gamma_max and gamma_max_full vs Pe
ax[0].scatter(Pe, gamma_max, label=r'$\gamma_{\max}$', marker='o')
ax[0].scatter(Pe, gamma_max_full, label=r'$\gamma_*$', marker='s', color='orange')
ax[0].axhline(0, color='gray', linestyle='--', linewidth=1.5)
ax[0].set_xlabel('Pe')             # Double the default font size
ax[0].set_ylabel('Growth rate of fastest mode', fontsize=18)    # Double the default font size
ax[0].set_xscale('log')
leg0 = ax[0].legend(frameon=False, loc="center right")

# Plot k_max and k_max_full vs Pe
ax[1].errorbar(Pe_masked, k_max_full, yerr=k_max_full_sigma, label=r'$k_*$', fmt='s', color='orange', capsize=5, alpha=0.6)
ax[1].scatter(Pe, k_max, label=r'$k_{\max}$', marker='o')
ax[1].set_xlabel('Pe')             # Double the default font size
ax[1].set_ylabel('Wavenumber of fastest mode', fontsize=18)         # Double the default font size
ax[1].set_xscale('log')
ax[1].set_yscale('log')
leg1 = ax[1].legend(frameon=False)

ylab_xpos_l_b = ax[0].yaxis.get_label().get_position()[0]  # horizontal position of y-label
ylab_xpos_r_b = ax[0].yaxis.get_label().get_position()[1]  # horizontal position of y-label
fig.text(ylab_xpos_l_b + 0.075, 0.98, "($a$)", verticalalignment='top', horizontalalignment='right')
fig.text(ylab_xpos_r_b + 0.025, 0.98, "($b$)", verticalalignment='top', horizontalalignment='right')

# Show the second plot
#fig.tight_layout()
fig.savefig(folder_name + f'max_values_vs_Pe_compare.pdf', dpi=600, bbox_inches='tight')

plt.show()
