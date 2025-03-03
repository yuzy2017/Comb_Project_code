import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# **Step 1: Structuring Data**
data = {
    "Pulse Number": [1, 2, 3, 4, 5],
    "400V":  ["0.96\n0.1905", "0.40\n0.0584", "0.64\n0.1156", "1.04\n0.0248", "0.80\n0.0661"],
    "350V (50ms)":  ["0.84\n0.0766", "0.56\n0.4629", "0.88\n0.2176", "0.56\n0.0091", "0.96\n0.0862"],
    "350V (25ms)":  ["0.88\n0.1586", "0.80\n0.0939", "0.42\n0.3263", "0.35\n0.0471", "0.45\n0.2508"]
}

df = pd.DataFrame(data)

# **Step 2: Extract Duty Cycle & QPM FFT Coefficients**
pulse_numbers = df["Pulse Number"].values
voltages = ["400V", "350V (50ms)", "350V (25ms)"]

# **Function to Extract Duty Cycle & QPM FFT**
def extract_dc_qpm(df, voltages):
    duty_cycles, qpm_coeffs = [], []
    for voltage in voltages:
        dc_values, qpm_values = [], []
        for val in df[voltage]:
            parts = val.split("\n")  # Split duty cycle and QPM coefficient
            dc_values.append(float(parts[0]))  # First value is Duty Cycle
            qpm_values.append(float(parts[1]))  # Second value is QPM Coefficient
        duty_cycles.append(dc_values)
        qpm_coeffs.append(qpm_values)
    return np.array(duty_cycles), np.array(qpm_coeffs)

# **Extract Data**
duty_cycles, qpm_coeffs = extract_dc_qpm(df, voltages)

# **Step 3: Plot 400V Only**
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Pulse Number", fontsize=14)
ax1.set_ylabel("Duty Cycle", color="b", fontsize=14)
ax1.plot(pulse_numbers, duty_cycles[0], marker="o", linestyle="-", color="b", label="Duty Cycle")
ax1.tick_params(axis='y', labelcolor="b")

ax2 = ax1.twinx()  # Create a second y-axis
ax2.set_ylabel("QPM FFT Coefficient", color="r", fontsize=14)
ax2.plot(pulse_numbers, qpm_coeffs[0], marker="s", linestyle="--", color="r", label="QPM FFT Coeff")
ax2.tick_params(axis='y', labelcolor="r")

plt.title("400V: Duty Cycle & QPM FFT vs. Pulse Number", fontsize=16)
fig.tight_layout()
plt.savefig("400V_DutyCycle_QPM_vs_PulseNumber.png", dpi=600, transparent=True, bbox_inches='tight')
plt.show()

# **Step 4: Plot 350V (50ms & 25ms)**
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Pulse Number", fontsize=14)
ax1.set_ylabel("Duty Cycle", color="b", fontsize=14)
for i in range(1, 3):  # Skip 400V, plot only 350V (50ms & 25ms)
    ax1.plot(pulse_numbers, duty_cycles[i], marker="o", linestyle="-", label=f"Duty Cycle @ {voltages[i]}")

ax1.tick_params(axis='y', labelcolor="b")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()  # Create a second y-axis
ax2.set_ylabel("QPM FFT Coefficient", color="r", fontsize=14)
for i in range(1, 3):
    ax2.plot(pulse_numbers, qpm_coeffs[i], marker="s", linestyle="--", label=f"QPM FFT @ {voltages[i]}")

ax2.tick_params(axis='y', labelcolor="r")
ax2.legend(loc="upper right")

plt.title("350V (50ms & 25ms): Duty Cycle & QPM FFT vs. Pulse Number", fontsize=16)
fig.tight_layout()
plt.savefig("350V_DutyCycle_QPM_vs_PulseNumber.png", dpi=600, transparent=True, bbox_inches='tight')
plt.show()
