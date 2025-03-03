import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq
from numpy import sqrt,max,min,cos,sin

# --- Load Data ---
data_file = Path("..") / "20250225" / "Confocal" / "20250225-2048-12_confocal_xy_data.dat"
print(f"解析后的绝对路径: {data_file.resolve()}")

if not data_file.exists():
    raise FileNotFoundError("error 文件不存在，请检查路径是否正确！")

# Read the tab-separated file
data = pd.read_csv(data_file, delimiter="\t", skiprows=np.arange(0, 18, 1))

# Remove unwanted columns
data = data.drop(["count rate /Dev2/AI1 (Hz)", "count rate /Dev2/AI0 (Hz)", 'z position (m)', "Unnamed: 6"], axis=1)
# Shifting the origin to (0,0) and rescale the coordinate to real distance
data['#x position (m)'] = (data['#x position (m)']-data['#x position (m)'].min())*1e6*0.606
data['y position (m)'] = ((data['y position (m)']-data['y position (m)'].min())*1e6)*0.368
data.drop(data[data['#x position (m)'] < 20].index, inplace = True)
# Rename columns for easier access
data.columns = ["x", "y", "z"]

# Sort the data by x position (for correct spatial ordering)
data = data.sort_values(by=["x"], ignore_index=True)
# x =data['x'].to_numpy()
# y = data['y'].to_numpy()
z = data['z'].to_numpy()
# --- Normalize SHG Intensity ---
z_min = min(z)
z_max = max(z)
data["z_norm"] = sqrt(z)/sqrt(z_max) # Normalize to [0,1]

# --- Find Poling Period Using Peaks ---
peaks, _ = find_peaks(data["z_norm"], height=0.5, distance=2)  # Adjust height & distance as needed
poling_periods = np.diff(data["x"].iloc[peaks])  # Extract distances between peaks

# --- Compute FFT to Analyze Periodicity ---
fft_values = np.abs(fft(data["z_norm"].to_numpy(dtype=np.float64)))

freqs = fftfreq(len(data["z_norm"]), d=np.mean(np.diff(data["x"])))  # Frequency in spatial domain

# --- Plot Results ---
plt.figure(figsize=(12, 5))

# 1. Raw & Normalized SHG Intensity
plt.subplot(2, 1, 1)
plt.plot(data["x"], data["z_norm"], label="Normalized Intensity", linewidth=1.5)
plt.scatter(data["x"].iloc[peaks], data["z_norm"].iloc[peaks], color='red', label="Detected Peaks")
plt.xlabel("X Position (m)")
plt.ylabel("Intensity")
plt.title("SHG Intensity & Poling Peaks")
plt.legend()

# 2. FFT Spectrum
plt.subplot(2, 1, 2)
plt.plot(freqs[:len(freqs)//2], fft_values[:len(freqs)//2])
plt.xlabel("Spatial Frequency (1/m)")
plt.ylabel("Amplitude")
plt.title("FFT of Poling Pattern")

plt.tight_layout()
plt.show()

# --- Print Analysis Results ---
print(f"Average Poling Period: {np.mean(poling_periods):.4f} m")
print(f"Detected {len(peaks)} peaks in SHG intensity profile")
