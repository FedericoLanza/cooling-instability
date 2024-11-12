import matplotlib.pyplot as plt

# Initialize lists to store data
Pe = []
k_max = []
k_max_full = []
gamma_max = []
gamma_max_full = []

# Read data from the file
with open("results/output_mix/compare.txt", "r") as file:
    # Skip the header
    next(file)
    # Read each line after the header
    for line in file:
        values = line.split()
        # Check if the line has the expected number of columns (9 in this case)
        if len(values) == 9:
            Pe.append(float(values[0]))
            k_max.append(float(values[3]))
            k_max_full.append(float(values[7]))
            gamma_max.append(float(values[5]))
            gamma_max_full.append(float(values[8]))

# Plot k_max and k_max_full vs Pe
plt.figure(figsize=(10, 6))
plt.scatter(Pe, k_max, label='linear', marker='o')
plt.scatter(Pe, k_max_full, label='full', marker='s')
plt.xlabel('$Pe$', fontsize=20)             # Double the default font size
plt.ylabel('$k_{max}$', fontsize=20)         # Double the default font size
plt.xscale('log')
plt.title(r"$k_{max}$ vs $Pe$ for $\beta=0.001$ and $\Gamma = 1$", fontsize=18 * 1.5)  # 1.5 times bigger
plt.legend()

# Show the first plot
plt.show()

# Plot gamma_max and gamma_max_full vs Pe
plt.figure(figsize=(10, 6))
plt.scatter(Pe, gamma_max, label='linear', marker='o')
plt.scatter(Pe, gamma_max_full, label='full', marker='s')
plt.xlabel('$Pe$', fontsize=20)             # Double the default font size
plt.ylabel('$\gamma_{max}$', fontsize=20)    # Double the default font size
plt.xscale('log')
plt.title(r"$\gamma_{max}$ vs $Pe$ for $\beta=0.001$ and $\Gamma = 1$", fontsize=18 * 1.5)  # 1.5 times bigger
plt.legend()

# Show the second plot
plt.show()
