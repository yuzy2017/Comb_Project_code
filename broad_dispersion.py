import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c as c_const, pi
import os


# Load the data from the text file
data_dir = r'E:\Sim_Code\Sim_results'
filename = os.path.join(data_dir, 'broad_dispersion')


# Define column names, and skip the header row
columns = ['f','neff_1(real)','neff_1(imag)','loss_1','beta_1(real)','beta_1(imag)','vg_1','D_1','neff_2(real)','neff_2(imag)','loss_2','beta_2(real)','beta_2(imag)','vg_2','D_2','neff_3(real)','neff_3(imag)','loss_3','beta_3(real)','beta_3(imag)','vg_3','D_3','neff_4(real)','neff_4(imag)','loss_4','beta_4(real)','beta_4(imag)','vg_4','D_4']

# Read the CSV file, skipping the first row (headers), and using ',' as the delimiter
data = pd.read_csv(filename, skiprows=1, names=columns, delimiter=',')

# Convert all columns to numeric, coercing errors to NaN if any invalid values are found
data = data.apply(pd.to_numeric, errors='coerce')

# Drop any rows that have NaN values
data = data.dropna()

# Convert frequency from Hz to angular frequency (rad/s)
frequency = data['f']  # In Hz
wavelength = c_const / frequency  # Wavelength in meters
D_3 = data['D_4']  # GVD for the third mode

omega = 2*pi*frequency

### 1. Calculate beta_2 from D_3 ###
# Calculate beta_2 using the formula: beta_2 = -lambda^2 / (2 * pi * c) * D
beta_2_from_D = -(wavelength**2) / (2 * pi * c_const) * D_3

### Plot beta_2 from D ###
plt.figure(figsize=(10, 6))

# Plot beta_2 from D
plt.plot(wavelength * 1e9, beta_2_from_D, label='Beta_2 (from D_3)', color='red', linestyle='--', linewidth=2)

# Add a reference line at Beta_2 = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Reference Line (Beta_2 = 0)')

# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Beta_2 (s^2/m)', fontsize=14)
plt.title('Beta_2 from D_3 across Wavelengths', fontsize=16, fontweight='bold')

# Add a legend to distinguish the curve and the reference line
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

### 2. Identify mode crossings by comparing neff between modes ###
# Extract neff for different modes
neff_1 = data['neff_1(real)']
neff_2 = data['neff_2(real)']
neff_3 = data['neff_3(real)']
neff_4 = data['neff_4(real)']

# Plot neff for the different modes to visually identify crossings
plt.figure(figsize=(10, 6))

# Plot neff for each mode
plt.plot(wavelength * 1e9, neff_1, label='Mode 1 (neff)', color='blue')
plt.plot(wavelength * 1e9, neff_2, label='Mode 2 (neff)', color='green')
plt.plot(wavelength * 1e9, neff_3, label='Mode 3 (neff)', color='red')
plt.plot(wavelength * 1e9, neff_4, label='Mode 4 (neff)', color='purple')

# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Effective Index (neff)', fontsize=14)
plt.title('Effective Index (neff) for Different Modes', fontsize=16, fontweight='bold')

# Add a legend to distinguish the modes
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

### 3. Identify mode crossing points programmatically ###
# Calculate the difference between neff of different modes
neff_diff_3_4 = np.abs(neff_3 - neff_4)  # Difference between mode 1 and mode 3
neff_diff_2_3 = np.abs(neff_2 - neff_3)  # Difference between mode 2 and mode 3

# Define a threshold for mode crossing (e.g., modes are considered crossing if their neff difference is below a small threshold)
threshold = 0.01

# Find points where the difference is smaller than the threshold
crossing_points_3_4 = wavelength[neff_diff_3_4 < threshold]
crossing_points_2_3 = wavelength[neff_diff_2_3 < threshold]

# Print the wavelengths where mode crossings are detected
print(f"Mode crossings between Mode 3 and Mode 4 occur at wavelengths (in nm): {crossing_points_3_4 * 1e9}")
print(f"Mode crossings between Mode 2 and Mode 3 occur at wavelengths (in nm): {crossing_points_2_3 * 1e9}")

# Concatenate the selected slices of neff_3 and neff_4
neff_TE = np.concatenate([neff_3[0:11], neff_4[11:78], neff_3[78:]])

# Calculate beta from the selected neff_TE
beta_TE = neff_TE * omega / c_const

# Calculate the first derivative of beta_TE w.r.t. omega
#d_beta_domega = np.gradient(beta_TE, omega)
d_n_dlambda = np.gradient(neff_TE, wavelength)
# Calculate the second derivative (beta_2) of beta_TE w.r.t. omega
#beta_2_from_neff_TE = np.gradient(d_beta_domega, omega)
d_n_2_dlambda = np.gradient(d_n_dlambda, wavelength)
### Plotting beta_2 from the selected neff_TE ###

plt.figure(figsize=(10, 6))

# Plot beta_2 from neff_TE
plt.plot(wavelength * 1e9,-d_n_2_dlambda*wavelength/c_const * 1e6, label='D(GVD) (from selected neff_TE)', color='green', linestyle='-', linewidth=2)

# Add a reference line at Beta_2 = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Reference Line (Beta_2 = 0)')

# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=14)
#plt.ylabel('Beta_2 (ps^2/km)', fontsize=14)
plt.ylabel('GVD (ps/nm/km)',fontsize= 14)
plt.title('Beta_2 from selected neff_TE across Frequencies', fontsize=16, fontweight='bold')

# Add a legend to distinguish the curve and the reference line
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot as an image (optional)
plt.savefig('beta_2_from_neff_TE.png', dpi=300)


# Load the data from the text file
file_path = 'E:/Sim_Code/LNbO3_sampled_ne_no_no.txt'
data = pd.read_csv(file_path, sep='\t', header=None)

# Convert the range from microns to nanometers for consistency with the dataset
wavelength_min = 700  # 0.7 µm in nm
wavelength_max = 1700  # 1.7 µm in nm

# Filter the dataset based on the wavelength range
data_filtered = data[(data[0] >= wavelength_min) & (data[0] <= wavelength_max)]

# Extract relevant columns: wavelength (column 0) and refractive indices (ne - column 1)
wavelength1 = data_filtered[0].values  # Wavelength (column 0)
n_e1 = data_filtered[1].values  # Extraordinary refractive index (ne - column 1)

# Fit a 24th order polynomial to the ne data
poly_coeffs = np.polyfit(wavelength1, n_e1, 24)

# Generate the fitted refractive index values using the polynomial
poly_fit_ne = np.polyval(poly_coeffs, wavelength*1e9)


# Calculate first and second derivatives of the refractive index with respect to wavelength
dn_dlambda = np.gradient(poly_fit_ne, wavelength)  # First derivative
d2n_dlambda2 = np.gradient(dn_dlambda, wavelength)  # Second derivative

# Calculate GVD using the formula
GVD = -(wavelength / (c_const)) * d2n_dlambda2*1e6

plt.figure(figsize=(10, 6))

# Plot beta_2 from neff_TE
plt.plot(wavelength * 1e9,-d_n_2_dlambda*wavelength/c_const * 1e6, label='D(GVD) (from selected neff_TE)', color='green', linestyle='-', linewidth=2)
plt.plot(wavelength * 1e9,GVD, label='D(GVD) (from material)', color='blue', linestyle='-', linewidth=2)
plt.plot(wavelength * 1e9,-GVD-d_n_2_dlambda*wavelength/c_const * 1e6, label='Waveguide Dispersion', color='red', linestyle='-', linewidth=2)
# Add a reference line at Beta_2 = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Reference Line (Beta_2 = 0)')
#plt.xlim([1000,1700])
plt.ylim([-1000,1000])
# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=14)
#plt.ylabel('Beta_2 (ps^2/km)', fontsize=14)
plt.ylabel('GVD (ps/nm/km)',fontsize= 14)
plt.title('Beta_2 from selected neff_TE across Frequencies', fontsize=16, fontweight='bold')

# Add a legend to distinguish the curve and the reference line
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()