import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define parameters
N = 40  # number of Fock states (for each mode)
chi = 0.1  # nonlinear interaction strength
alpha = 2.0  # coherent state amplitude for mode b (visible)
tlist = np.linspace(0, 10, 100)  # time array

# Create annihilation operators for two modes (a: IR, b: visible)
a = tensor(destroy(N), qeye(N))  # Mode a (IR, 1550 nm), acting only on the first Hilbert space
b = tensor(qeye(N), destroy(N))  # Mode b (visible, 775 nm), acting only on the second Hilbert space

# Define the interaction Hamiltonian: H = i(chi * a * a * b^dagger - chi * a^dagger * a^dagger * b)
H = 1j * chi * (a * a * b.dag() - a.dag() * a.dag() * b)

# Define the initial state: vacuum state for mode a and coherent state for mode b
psi0 = tensor(basis(N, 0), coherent(N, alpha))  # |0_a> ⊗ |α_b>

# Quadrature operators for mode b (to check squeezing)
X_b = (b + b.dag()) / 2  # X quadrature
P_b = (b - b.dag()) / (2j)  # P quadrature

# Time evolution using mesolve
result = mesolve(H, psi0, tlist, [], [a.dag() * a, b.dag() * b, X_b**2 , P_b**2, X_b , P_b])

# Extract results: photon numbers and quadrature variances
n_a = result.expect[0]  # Photon number in mode a (IR)
n_b = result.expect[1]  # Photon number in mode b (visible)
var_X_b = result.expect[2]-result.expect[4]**2  # Variance of X quadrature for mode b
var_P_b = result.expect[3] - result.expect[5]**2 # Variance of P quadrature for mode b

# Plot the photon numbers in both modes over time
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(tlist, n_a, label='Photon number in mode a (IR)')
plt.plot(tlist, n_b, label='Photon number in mode b (Visible)')
plt.xlabel('Time')
plt.ylabel('Photon number')
plt.title('Photon Number Evolution')
plt.legend()

# Plot the variances of X and P quadratures for mode b
plt.subplot(212)
plt.plot(tlist, var_X_b, label='Variance of X_b')
plt.plot(tlist, var_P_b, label='Variance of P_b')
plt.axhline(0.25, color='k', linestyle='--', label='Vacuum Limit')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.yscale('log')
plt.title('Quadrature Variances of Mode b')
plt.legend()
plt.tight_layout()
plt.show()

# Time evolution using mesolve
result = mesolve(H, psi0, tlist, [], [])

# Extract states at times t = 0, 5, 10
state_t0 = result.states[0]   # state at t = 0
state_t5 = result.states[50]  # state at t = 5 (approximately halfway through tlist)
state_t10 = result.states[-1] # state at t = 10

# Partial trace to get the state of mode b at t = 0, 5, 10
rho_b_t0 = state_t0.ptrace(1)  # trace out mode a
rho_b_t5 = state_t5.ptrace(1)
rho_b_t10 = state_t10.ptrace(1)

# Define the phase space grid for the Wigner function
xvec = np.linspace(-5, 5, 500)

# Calculate the Wigner functions for mode b at the selected times
W_b_t0 = wigner(rho_b_t0, xvec, xvec)
W_b_t5 = wigner(rho_b_t5, xvec, xvec)
W_b_t10 = wigner(rho_b_t10, xvec, xvec)

# Plot the Wigner distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot Wigner at t = 0
plot_wigner(rho_b_t0, fig=fig, ax=axes[0], xvec=xvec)
axes[0].set_title("Wigner distribution of mode b at t = 0")

# Plot Wigner at t = 5
plot_wigner(rho_b_t5, fig=fig, ax=axes[1], xvec=xvec)
axes[1].set_title("Wigner distribution of mode b at t = 5")

# Plot Wigner at t = 10
plot_wigner(rho_b_t10, fig=fig, ax=axes[2], xvec=xvec)
axes[2].set_title("Wigner distribution of mode b at t = 10")

plt.show()