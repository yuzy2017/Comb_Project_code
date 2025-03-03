import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# **Step 1: Structuring Data**
data = {
    "Voltage (V)": [600, 550, 500, 450],  # Voltages
    "10s": ["0.93\n0.1939", "0.80\n0.1740", "NaN", "NaN"],
    "25s": ["0.72\n0.1161", "0.88\n0.2711", "0.80\n0.1181", "0.80\n0.1224"],
    "50s": ["0.40\n0.1088", "0.48\n0.5907", "0.88\n0.1837", "0.93\n0.1885"],
    "75s": ["0.56\n0.1961", "0.48\n0.1314", "0.40\n0.6361", "0.93\n0.1224"]
}

df = pd.DataFrame(data)

# **Step 2: Extract Duty Cycle & QPM FFT Coefficients**
voltages = df["Voltage (V)"].values
hold_times = ["10s", "25s", "50s", "75s"]  # Hold times as labels

duty_cycles = []
qpm_coeffs = []

for hold_time in hold_times:
    dc_values = []
    qpm_values = []

    for val in df[hold_time]:
        if val == "NaN":  # Skip invalid entries
            dc_values.append(np.nan)
            qpm_values.append(np.nan)
        else:
            parts = val.split("\n")  # Split duty cycle and QPM coefficient
            dc_values.append(float(parts[0]))  # First value is Duty Cycle
            qpm_values.append(float(parts[1]))  # Second value is QPM Coefficient

    duty_cycles.append(dc_values)
    qpm_coeffs.append(qpm_values)

duty_cycles = np.array(duty_cycles)
qpm_coeffs = np.array(qpm_coeffs)

# **Step 3: Plot Duty Cycle vs. Voltage**
plt.figure(figsize=(10, 5))
for i, hold_time in enumerate(hold_times):
    plt.plot(voltages, duty_cycles[i], marker="o", linestyle="-", label=f"Duty Cycle @ {hold_time}")

plt.xlabel("Voltage (V)", fontsize=14)
plt.ylabel("Duty Cycle", fontsize=14)
plt.title("Duty Cycle vs. Voltage for Different Hold Times", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("DutyCycle_vs_Voltage.png", dpi=600, transparent=True, bbox_inches='tight')
plt.show()

# **Step 4: Plot QPM FFT Coefficients vs. Voltage**
plt.figure(figsize=(10, 5))
for i, hold_time in enumerate(hold_times):
    plt.plot(voltages, qpm_coeffs[i], marker="s", linestyle="--", label=f"QPM Coeff @ {hold_time}")

plt.xlabel("Voltage (V)", fontsize=14)
plt.ylabel("QPM FFT Coefficient", fontsize=14)
plt.title("QPM FFT Coefficient vs. Voltage for Different Hold Times", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("QPM_Coeff_vs_Voltage.png", dpi=600, transparent=True, bbox_inches='tight')
plt.show()
