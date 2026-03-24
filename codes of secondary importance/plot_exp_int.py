import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi

def DeltaP(Pe, Gamma, L, psi, u0):
    xi = np.sqrt(105/2)*np.sqrt(Gamma/Pe)/u0
    term1 = expi(-psi)
    term2 = expi(-psi * np.exp(-xi*L))
    return u0/xi * (term1 - term2)
    
def DeltaPHelfrich(Gamma, L, psi, u0):
    xi = Gamma/u0
    term1 = expi(-psi)
    term2 = expi(-psi * np.exp(-xi*L))
    return u0/xi * (term1 - term2)
    
def derDeltaP(Pe, Gamma, L, psi, u0):
    xi = np.sqrt(105/2)*np.sqrt(Gamma/Pe)/u0
    term1 = expi(-psi)
    term2 = expi(-psi * np.exp(-xi*L))
    term3 = - L * np.exp(-psi * np.exp(-xi*L))
    return 2./xi * (term1 - term2) + term3
    
def derDeltaPHelfrich(Gamma, L, psi, u0):
    xi = Gamma/u0
    term1 = expi(-psi)
    term2 = expi(-psi * np.exp(-xi*L))
    term3 = - L * np.exp(-psi * np.exp(-xi*L))
    return 2./xi * (term1 - term2) + term3

# Define parameters
Pe = 1000   # Example value
Gamma = 1 # Example value
L = 1     # Example value

dilation = 7.

# Define u0 range
u0_values_H = np.linspace(0., 2.5, 100)
u0_values = np.linspace(0., 1, 400)  # Avoid u0 = 0 to prevent singularity

# Define psi values
#psi_values = [2.9, 3.0, 3.1, 3.44]
psi_values = [2.8, 2.9, 3.0, 3.1, 3.2, 3.44]

# Plot the function for different psi values
fig, ax = plt.subplots(1, 2, figsize=(15,5))

for psi in psi_values:
    DeltaP_values = DeltaP(Pe, Gamma, L, psi, u0_values)
    DeltaPHelfrich_values = DeltaPHelfrich(Gamma, L, psi, u0_values_H)
    derDeltaP_values = derDeltaP(Pe, Gamma, L, psi, u0_values)
    derDeltaPHelfrich_values = derDeltaPHelfrich(Gamma, L, psi, u0_values_H)
    ax[0].plot(u0_values, DeltaP_values, label=fr'$\psi = {psi}$')
    #ax[0].plot(u0_values_H, DeltaPHelfrich_values, label=fr'$\psi = {psi}$')
    ax[1].plot(u0_values, derDeltaP_values, label=fr'$\psi = {psi}$')
    #ax[1].plot(u0_values_H, derDeltaPHelfrich_values, linestyle='-.', label=fr'$\psi = {psi}$')
    ax[1].axhline(y=0., color='black', linestyle='--')
    
[axi.set_xlabel(r'$u_0$') for axi in ax]
ax[0].set_ylabel(r'$\Delta P(u_0)$')
ax[1].set_ylabel(r'$d\Delta P(u_0)/du_0$')
ax[0].legend()
plt.show()

exit(0)

for psi in psi_values:
    DeltaP_values_Gamma_1 = DeltaP(Pe, dilation*Gamma, L, psi, u0_values)
    DeltaP_values_Gamma_2 = np.sqrt(dilation) * DeltaP(Pe, Gamma, L, psi, u0_values/np.sqrt(dilation))
    DeltaP_values_Pe_1 = DeltaP(dilation*Pe, Gamma, L, psi, u0_values)
    DeltaP_values_Pe_2 = 1./np.sqrt(dilation) * DeltaP(Pe, Gamma, L, psi, u0_values*np.sqrt(dilation))
    #plt.plot(u0_values, DeltaP_values, label=fr'$\psi = {psi}$')
    plt.plot(u0_values, DeltaP_values_Gamma_1, linestyle = '-', label=fr'$\Gamma$ 1')
    plt.plot(u0_values, DeltaP_values_Gamma_2, linestyle = '--', label=fr'$\Gamma$ 2')
    plt.plot(u0_values, DeltaP_values_Pe_1, linestyle = '-', label=fr'Pe 1')
    plt.plot(u0_values, DeltaP_values_Pe_2, linestyle = '--', label=fr'Pe 2')
