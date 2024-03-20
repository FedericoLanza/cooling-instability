import numpy as np
import matplotlib.pyplot as plt
import os

file_name = "xspan_T=0.50.txt"

Pe = 100
Gamma = 1
beta = 0.001
ueps = 0.1
Lx = 50
rnd = False

Pe_str = f"Pe_{Pe:.10g}"
Gamma_str = f"Gamma_{Gamma:.10g}"
beta_str = f"beta_{beta:.10g}"
ueps_str = f"ueps_{ueps:.10g}"
Lx_str = f"Lx_{Lx:.10g}"
rnd_str = f"rnd_{rnd}"

Ly_ = [pow(2,a) for a in np.arange(-1, 5.0, 0.5)]

colors = plt.cm.viridis(np.linspace(0, 1, len(Ly_)))  # Generate colors using colormap
color_dict = {}  # Dictionary to store colors for each Ly

for i, Ly in enumerate(Ly_):
    Ly_str = f"Ly_{Ly:.10g}".format(Ly)
    color_dict[Ly] = colors[i]  # Store color for this Ly
    
    file_path = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str]) + "_holdpert_True/" + file_name # path for file
    #file2_path = "results/" + "_".join([Pe_str, Gamma_str, beta_str, ueps_str, Ly_str, Lx_str, rnd_str]) + "_holdpert_False/" + file_name # path for file
    # Read data from the file
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data = [line.split() for line in file.readlines()]
            x_values = [float(row[0]) for row in data]
            y_values = [float(row[1]) for row in data]
            plt.plot(x_values, y_values, label=f"$L_y={Ly:.7g}$", color=color_dict[Ly])
    #if os.path.isfile(file2_path):
    #    with open(file2_path, 'r') as file:
    #        data2 = [line.split() for line in file.readlines()]
    #        x2_values = [float(row[0]) for row in data2]
    #        y2_values = [float(row[1]) for row in data2]
    #        plt.plot(x2_values, y2_values, label=f"$L_y={Ly:.7g}$", linestyle='dotted', color=color_dict[Ly])

plt.semilogy()
plt.xlabel("t")
plt.ylabel("$x_{max}-x_{min}$")
#plt.scale
plt.legend()
plt.show()
