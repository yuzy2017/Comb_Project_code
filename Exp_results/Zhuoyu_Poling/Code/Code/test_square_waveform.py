import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Define parameters
L = 1000  # Total length of signal
T = 1.0   # Time period of one domain
N = 2**12  # Number of sample points
x = np.linspace(0, L, N)

# Generate square wave without inversion
square_wave_no_inv = np.sign(np.sin(2 * np.pi * x / T))

# Generate square wave with domain inversion (flip every second period)
square_wave_inv = square_wave_no_inv.copy()
for i in range(len(x)):
    if int(x[i] / T) % 2 == 1:  # Flip every second domain
        square_wave_inv[i] *= -1

# Compute FFT
fft_no_inv = fft(square_wave_no_inv)
fft_inv = fft(square_wave_inv)
freqs = fftfreq(N, d=x[1] - x[0])  # Frequency axis

# Plot the waveforms
plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(x[:500], square_wave_no_inv[:500], label='No Inversion')
plt.plot(x[:500], square_wave_inv[:500], label='With Inversion', linestyle='dashed')
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.title("Square Waves with and without Domain Inversion")
plt.legend()

# Plot the Fourier spectrum
plt.subplot(2,1,2)
plt.semilogy(freqs[:N//2], np.abs(fft_no_inv[:N//2]), label="No Inversion")
plt.semilogy(freqs[:N//2], np.abs(fft_inv[:N//2]), label="With Inversion", linestyle='dashed')
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Fourier Spectrum Comparison")
plt.legend()

plt.tight_layout()
plt.show()
