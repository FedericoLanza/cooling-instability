import matplotlib.pyplot as plt

# Specify the file path
file_name = "xratio_T=0.50.txt"

file_path1 = "results/Pe_100.0_Gamma_1.0_beta_0.001_ueps_0.1_Ly_1.0_Lx_10.0_rnd_False.0/" + file_name
file_path2 = "results/Pe_100.0_Gamma_1.0_beta_0.001_ueps_0.1_Ly_2.0_Lx_20.0_rnd_False/" + file_name
file_path4 = "results/Pe_100.0_Gamma_1.0_beta_0.001_ueps_0.1_Ly_4.0_Lx_50.0_rnd_False/" + file_name
file_path8 = "results/Pe_100.0_Gamma_1.0_beta_0.001_ueps_0.1_Ly_8.0_Lx_50.0_rnd_False.0/" + file_name
file_path16 = "results/Pe_100.0_Gamma_1.0_beta_0.001_ueps_0.1_Ly_16.0_Lx_50.0_rnd_False/" + file_name

# Read data from the file
with open(file_path1, 'r') as file:
    # Assuming data is space-separated, adjust delimiter if needed
    data1 = [line.split() for line in file.readlines()]

with open(file_path2, 'r') as file:
    # Assuming data is space-separated, adjust delimiter if needed
    data2 = [line.split() for line in file.readlines()]
    
with open(file_path4, 'r') as file:
    # Assuming data is space-separated, adjust delimiter if needed
    data4 = [line.split() for line in file.readlines()]
    
with open(file_path8, 'r') as file:
    # Assuming data is space-separated, adjust delimiter if needed
    data8 = [line.split() for line in file.readlines()]
    
with open(file_path16, 'r') as file:
    # Assuming data is space-separated, adjust delimiter if needed
    data16 = [line.split() for line in file.readlines()]

# Extract columns
x1_values = [float(row[0]) for row in data1]
y1_values = [float(row[1]) for row in data1]
x2_values = [float(row[0]) for row in data2]
y2_values = [float(row[1]) for row in data2]
x4_values = [float(row[0]) for row in data4]
y4_values = [float(row[1]) for row in data4]
x8_values = [float(row[0]) for row in data8]
y8_values = [float(row[1]) for row in data8]
x16_values = [float(row[0]) for row in data16]
y16_values = [float(row[1]) for row in data16]

# Plot the data
plt.plot(x1_values, y1_values, label="Ly=1")
plt.plot(x2_values, y2_values, label="Ly=2")
plt.plot(x4_values, y4_values, label="Ly=4")
plt.plot(x8_values, y8_values, label="Ly=8")
plt.plot(x16_values, y16_values, label="Ly=16")
plt.semilogy()
plt.xlabel("t")
plt.ylabel("$x_{max}-x_{min}$")
#plt.scale
plt.legend()
plt.show()
