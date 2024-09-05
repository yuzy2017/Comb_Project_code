import numpy as np
from scipy.constants import c as c_const, hbar, pi, epsilon_0

def calculate_optical_parameters(lambda_s, lambda_i, lambda_pump_vis, lambda_pump_ir, chi2, neff, FSR, overlap, n2, Aeff, Qp, Qp1, Qs, Qi):
    # Convert parameters where necessary
    Aeff_m2 = Aeff * 1e-12  # Convert µm² to m²
    lambda_s = lambda_s * 1e-9  # Convert nm to m
    lambda_i = lambda_i * 1e-9
    lambda_pump_ir = lambda_pump_ir * 1e-9
    lambda_pump_vis = lambda_pump_vis * 1e-9
    chi2 = chi2 * 1e-12  # Convert pm/V to m/V and then convert to chi2_eff
    FSR = FSR * 1e9  # Convert GHz to Hz
    overlap = overlap * 1e6  # convert 1/um to 1/m

    # Calculate angular frequencies
    omega_s = 2 * pi * c_const / lambda_s
    omega_i = 2 * pi * c_const / lambda_i
    omega_p = 2 * pi * c_const / lambda_pump_vis

    # Calculate the resonator length
    L_res = c_const / (neff * FSR)
    # Fi = Qi / 2 * lambda_i / (2 * pi * neff * L_res)
    # Fs = Qs / 2 * lambda_s / (2 * pi * neff * L_res)
    # Fp = Qp * lambda_pump_ir / (2 * pi * neff * L_res)
    # Calculate 'g'
    g = 9 * chi2 ** 2 / 32 * overlap ** 2 * omega_s * omega_i / neff ** 4 * hbar * omega_p / (epsilon_0 * neff ** 2 * L_res)
    # g = (2*pi*1e6)**2
    # Calculate threshold powers
    P_th_spdc = (hbar * omega_s * omega_i * omega_p ** 2) / (64 * g) * (Qp1 / (Qs * Qi * Qp))
    Efficiency_SHG = (4 * g * Qi * Qs * Qp1) / (hbar * omega_s ** 2 * omega_i ** 2)# this equation from On-chi chi2 microring optical parametric oscillation
    P_th_cas =np.sqrt( P_th_spdc / Efficiency_SHG)
    P_th_kerr = (1.54 * pi / 2 * neff**2 * Aeff_m2 * L_res) / (n2 * lambda_pump_ir * Qp**2) * (Qp1 / (2 * Qp))

    return g, Efficiency_SHG, P_th_cas, P_th_kerr

def main():
    # Define input parameters
    lambda_s = 1550.0
    lambda_i = 1570.0
    lambda_pump_vis = 780.0
    lambda_pump_ir = 1560
    chi2 = 20
    neff = 2.0
    FSR = 50
    overlap = 0.74
    n2 = 2.54e-17
    Aeff = 1
    Qp = 2e5
    Qp1 = 4e5
    Qs = 1e6
    Qi = 1e6

    # Calculate results
    g, Efficiency_SHG, P_th_cas, P_th_kerr = calculate_optical_parameters(
        lambda_s, lambda_i, lambda_pump_vis, lambda_pump_ir, chi2, neff, FSR, overlap, n2, Aeff, Qp, Qp1, Qs, Qi
    )

    # Print results
    print(f"g = {g:.4e}")
    print(f"Efficiency_SHG = {Efficiency_SHG:.4e}")
    print(f"P_th_cas = {P_th_cas:.4e} W")
    print(f"P_th_kerr = {P_th_kerr:.4e} W")

if __name__ == "__main__":
    main()
