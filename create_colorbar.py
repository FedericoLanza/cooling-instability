import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,  # Axis labels (JFM ~8pt)
    "xtick.labelsize": 24,  # Tick labels
    "ytick.labelsize": 24,
    "figure.subplot.wspace": 0.35,  # Horizontal spacing
    "figure.subplot.bottom": 0.15,  # Space for x-labels
    "figure.subplot.left": 0.05,  # Space for x-labels
    "figure.subplot.right": 0.4,  # Space for x-labels
})

# Define your custom colormap (or use a built-in one like 'viridis')
cmap = plt.get_cmap('coolwarm')  # or your own custom colormap

# Define normalization from -1 to 1
norm = mcolors.Normalize(vmin=-3.5, vmax=3.5)

# Create a dummy scalar mappable to use for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # necessary dummy data

# Create the colorbar
fig, ax = plt.subplots(figsize=(0.8, 8.))  # adjust size as needed
cbar = fig.colorbar(sm, cax=ax, orientation='vertical')
cbar.set_label(r'$\gamma$', labelpad=2)

plt.tight_layout()
plt.savefig("results/output_mix/colorbar.pdf", dpi=300, bbox_inches='tight')
plt.show()
