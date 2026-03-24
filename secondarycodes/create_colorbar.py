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
    "figure.subplot.right": 0.35,  # Space for x-labels
})

# Define your custom colormap (or use a built-in one like 'viridis')
cmap = plt.get_cmap('coolwarm')  # or your own custom colormap

# Define normalization from -1 to 1
norm = mcolors.Normalize(vmin=-5e-5, vmax=5.5e-5)

# Create a dummy scalar mappable to use for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # necessary dummy data

# Create the colorbar
fig, ax = plt.subplots(figsize=(1.3, 8.))  # adjust size as needed
cbar = fig.colorbar(sm, cax=ax, orientation='vertical')
cbar.set_label(r'$\gamma$', labelpad=2)

ticks = np.arange(-4e-5, 5e-5 + 0.1e-5, 1e-5)
tick_labels = [f"{t/1e-5:.0f}" for t in ticks]   # gives -4, -3, ..., 4, 5

cbar.set_ticks(ticks)
cbar.set_ticklabels(tick_labels)

# Colorbar label with exponent
cbar.set_label(r'$\gamma\;(\times 10^{-5})$', labelpad=6)

#plt.tight_layout()
plt.savefig("results/output_mix/colorbar.pdf", dpi=300, bbox_inches='tight')
plt.show()
