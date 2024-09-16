import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.constants import c as c_const, pi

# Load the data from the text file
data_dir = r'E:\Sim_Code\Sim_results'
filename = os.path.join(data_dir, '1550_dispersion')

# Define column names, and skip the header row
columns = ['f', 'neff_real', 'neff_imag', 'loss', 'beta_real', 'beta_imag', 'overlap', 'vg', 'D']

# Read the CSV file, skipping the first row (headers), and using ',' as the delimiter
data = pd.read_csv(filename, skiprows=1, names=columns, delimiter=',')

# Convert all columns to numeric, coercing errors to NaN if any invalid values are found
data = data.apply(pd.to_numeric, errors='coerce')

# Drop any rows that have NaN values
data = data.dropna()

# Extract frequency (f) in Hz, real part of neff, and group velocity dispersion (D)
frequency = data['f']  # In Hz
wavelength = c_const / frequency  # Calculate wavelength in meters
D = data['D']  # This is D = d(tau_g)/d(lambda)

### 1. Calculate beta_2 from beta_real ###
# Calculate angular frequency omega (rad/s)
omega = 2 * pi * frequency

# Extract neff_real and calculate beta_real
beta_real = data['beta_real']

# Calculate first and second derivatives of beta_real w.r.t. omega
d_beta_domega = np.gradient(beta_real, omega)  # First derivative
beta_2_from_beta_real = np.gradient(d_beta_domega, omega)  # Second derivative

### 2. Calculate beta_2 from GVD (D) ###
# Calculate beta_2 using the formula: beta_2 = -lambda^2 / (2 * pi * c) * D
beta_2_from_D = -(wavelength**2) / (2 * pi * c_const) * D

### Plotting beta_2 from beta_real and from D ###

plt.figure(figsize=(10, 6))

# Plot beta_2 from beta_real
plt.plot(wavelength * 1e9, beta_2_from_beta_real*1e27, label='Beta_2 (from beta_real)', color='blue', linestyle='-', linewidth=2)

# Plot beta_2 from D
plt.plot(wavelength * 1e9, beta_2_from_D*1e27, label='Beta_2 (from D)', color='red', linestyle='--', linewidth=2)

# Add a reference line at Beta_2 = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Reference Line (Beta_2 = 0)')

# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Beta_2 (ps^2/km)', fontsize=14)
plt.title('Comparison of Beta_2 from beta_real and GVD (D)', fontsize=16, fontweight='bold')

# Add a legend to distinguish the two curves and the reference line
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot as an image (optional)
plt.savefig('beta_2_comparison_with_reference_line_1550.png', dpi=300)
