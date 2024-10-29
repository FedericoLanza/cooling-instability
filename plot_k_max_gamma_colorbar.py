import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the data from the .txt file
data = np.loadtxt('results/output_mix/k_max.txt', skiprows=1)  # Skip the header row

# Assign columns to corresponding variables
Pe = data[:, 0]
Gamma = data[:, 1]
beta = data[:, 2]
k_max = data[:, 3]
k_max_sigma = data[:, 4]
gamma_max = data[:, 5]

# Filter data for Gamma == 1
gamma_value = 1  # You can change this value to filter other Gamma values
mask_gamma = Gamma == gamma_value

# Apply mask to all relevant variables
Pe = Pe[mask_gamma]
Gamma = Gamma[mask_gamma]
beta = beta[mask_gamma]
k_max = k_max[mask_gamma]
k_max_sigma = k_max_sigma[mask_gamma]
gamma_max = gamma_max[mask_gamma]

# Create a colormap for gamma_max values (negative -> blue, positive -> red)
norm = mcolors.TwoSlopeNorm(vmin=np.min(gamma_max), vcenter=0, vmax=np.max(gamma_max))
cmap = plt.cm.bwr  # Blue-White-Red colormap, where negative -> blue, positive -> red

# Find unique beta values for plotting
beta_values = np.unique(beta)

# Plotting
plt.figure(figsize=(8, 6))

for b in beta_values:
    # Select values corresponding to the current beta
    mask = beta == b
    Pe_subset = Pe[mask]
    k_max_subset = k_max[mask]
    k_max_sigma_subset = k_max_sigma[mask]
    gamma_max_subset = gamma_max[mask]
    
    # Scatter plot with colors based on gamma_max values
    sc = plt.scatter(Pe_subset, k_max_subset, c=gamma_max_subset, cmap=cmap, norm=norm, label=f'beta = {b}', s=100, edgecolor='k', marker='o')

    # Add error bars
    plt.errorbar(Pe_subset, k_max_subset, yerr=k_max_sigma_subset, fmt='none', capsize=3, color='gray')

# Set the x-axis to a logarithmic scale
plt.xscale('log')

# Adding labels and title
plt.xlabel('Pe (log scale)')
plt.ylabel('k_max')
plt.title(f'k_max vs Pe with color-coded gamma_max for Gamma = {gamma_value}')

# Add color bar
cbar = plt.colorbar(sc)
cbar.set_label('gamma_max', rotation=270, labelpad=15)

plt.legend()

# Save the plot as a .png file
plt.savefig('k_max_gamma_max_colored_plot_filtered.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
