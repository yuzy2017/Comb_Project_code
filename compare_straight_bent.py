import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.constants import c as c_const, pi

# Load the data from the text file
data_dir = r'E:\Sim_Code\Sim_results'
filename = os.path.join(data_dir, '1550_dispersion')
filename_bent = os.path.join(data_dir, '1550_bent_dispersion')

# Define column names, and skip the header row
columns = ['f', 'neff_real', 'neff_imag', 'loss', 'beta_real', 'beta_imag', 'overlap', 'vg', 'D']

# Read the CSV file, skipping the first row (headers), and using ',' as the delimiter
data = pd.read_csv(filename, skiprows=1, names=columns, delimiter=',')
data_bent = pd.read_csv(filename, skiprows=1, names=columns, delimiter=',')
# Convert all columns to numeric, coercing errors to NaN if any invalid values are found
data = data.apply(pd.to_numeric, errors='coerce')
data_bent = data_bent.apply(pd.to_numeric, errors='coerce')
# Drop any rows that have NaN values
data = data.dropna()
data_bent = data_bent.dropna()
# Extract frequency (f) in Hz, real part of neff, and group velocity dispersion (D)
frequency = data['f']  # In Hz
freq_bent = data_bent['f']

wavelength = c_const / frequency  # Calculate wavelength in meters
wl_bent = c_const/freq_bent
D = data['D']  # This is D = d(tau_g)/d(lambda)
D_bent = data_bent['D']
### 1. Calculate beta_2 from beta_real ###
# Calculate angular frequency omega (rad/s)
omega = 2 * pi * frequency
omega_bent = 2 * pi * freq_bent

# Extract neff_real and calculate beta_real
beta_real = data['beta_real']
beta_real_bent = data_bent['beta_real']

neff = data['neff_real']
neff_bent = data_bent['neff_real']



### 2. Calculate beta_2 from GVD (D) ###
# Calculate beta_2 using the formula: beta_2 = -lambda^2 / (2 * pi * c) * D
beta_2 = -(wavelength**2) / (2 * pi * c_const) * D
beta_2_bent = -(wl_bent**2) / (2 * pi * c_const) * D_bent

### Plotting beta_2 from beta_real and from D ###

plt.figure(figsize=(10, 6))

# Plot beta_2 from beta_real
plt.plot(wavelength * 1e9, beta_2*1e27, label='Beta_2 (straight)', color='blue', linestyle='-', linewidth=2)

# Plot beta_2 from D
plt.plot(wavelength * 1e9, beta_2_bent*1e27, label='Beta_2 (bent)', color='red', linestyle='--', linewidth=2)

# Add a reference line at Beta_2 = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Reference Line (Beta_2 = 0)')

# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Beta_2 (ps^2/km)', fontsize=14)
plt.title('Comparison of Beta_2 from D straight and bent', fontsize=16, fontweight='bold')

# Add a legend to distinguish the two curves and the reference line
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot as an image (optional)
plt.savefig('beta_2_comparison_straight_and_bent_1550.png', dpi=300)

### Plotting beta_real ###

plt.figure(figsize=(10, 6))

# Plot beta_2 from beta_real
plt.plot(wavelength * 1e9, beta_real*1e6, label='Beta_real (straight)', color='blue', linestyle='-', linewidth=2)
#plt.plot(wavelength * 1e9, neff, label='Beta_real (straight)', color='blue', linestyle='-', linewidth=2)

# Plot beta_2 from D
plt.plot(wavelength * 1e9, beta_real_bent*1e6, label='Beta_real (bent)', color='red', linestyle='--', linewidth=2)
#plt.plot(wavelength * 1e9, neff_bent, label='Beta_real (bent)', color='red', linestyle='--', linewidth=2)


# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Beta_real (1/um)', fontsize=14)
plt.title('Comparison of Beta_real straight and bent', fontsize=16, fontweight='bold')

# Add a legend to distinguish the two curves and the reference line
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot as an image (optional)
plt.savefig('beta_real_comparison_straight_and_bent_1550.png', dpi=300)
