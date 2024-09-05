import streamlit as st
import numpy as np
from scipy.constants import c as c_const, hbar, pi, epsilon_0

# Set up the Streamlit app
st.title("Interactive Calculation for Optical Parameters")

# Define input parameters using Streamlit input widgets
st.markdown("<b>\lambda_s (nm):</b>", unsafe_allow_html=True)
lambda_s = st.number_input(r"$\lambda_s$ (nm):", value=1550.0)

st.markdown("<b>\lambda_i (nm):</b>", unsafe_allow_html=True)
lambda_i = st.number_input(r"$\lambda_i$ (nm):", value=1570.0)

st.markdown("<b>\lambda_{\text{pump, vis}} (nm):</b>", unsafe_allow_html=True)
lambda_pump_vis = st.number_input(r"$\lambda_{\text{pump, vis}}$ (nm):", value=780.0)

st.markdown("<b>\lambda_{\text{pump, ir}} (nm):</b>", unsafe_allow_html=True)
lambda_pump_ir = st.number_input(r"$\lambda_{\text{pump, ir}}$ (nm):", value=1560)

st.markdown("<b>\chi^{(2)} (pm/V):</b>", unsafe_allow_html=True)
chi2 = st.number_input(r"$\chi^{(2)}$ (pm/V):", value=40)

st.markdown("<b>n_{\text{eff}}:</b>", unsafe_allow_html=True)
neff = st.number_input(r"$n_{\text{eff}}$:", value=2.0)

st.markdown("<b>FSR (GHz):</b>", unsafe_allow_html=True)
FSR = st.number_input(r"FSR (GHz):", value=50)

st.markdown("<b>\text{Overlap}:</b>", unsafe_allow_html=True)
overlap = st.number_input(r"$\text{Overlap} (1/um)$:", value=0.74)

st.markdown("<b>n_2 (m^2/W):</b>", unsafe_allow_html=True)
n2 = st.number_input(r"$n_2$ (m$^2$/W):", value=2.54e-17)

st.markdown("<b>A_{\text{eff}} (um^2):</b>", unsafe_allow_html=True)
Aeff = st.number_input(r"$A_{\text{eff}}$ (µm$^2$):", value=1)

st.markdown("<b>Q_p:</b>", unsafe_allow_html=True)
Qp = st.number_input(r"$Q_p$:", value=2e5)

st.markdown("<b>Q_{p1}:</b>", unsafe_allow_html=True)
Qp1 = st.number_input(r"$Q_{p1}$:", value=4e5)

st.markdown("<b>Q_s:</b>", unsafe_allow_html=True)
Qs = st.number_input(r"$Q_s$:", value=1e6)

st.markdown("<b>Q_i:</b>", unsafe_allow_html=True)
Qi = st.number_input(r"$Q_i$:", value=1e6)

# Calculate results when the button is pressed
if st.button('Calculate'):
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
    g = 9 * chi2 ** 2 / 32 * overlap ** 2 * omega_s * omega_i / neff ** 4 * hbar * omega_p / (
                epsilon_0 * neff ** 2 * L_res)
    # g = (2*pi*1e6)**2
    # Calculate threshold powers
    P_th_spdc = (hbar * omega_s * omega_i * omega_p ** 2) / (64 * g) * (Qp1 / (Qs * Qi * Qp))
    Efficiency_SHG = (4 * g * Qi * Qs * Qp1) / (
                hbar * omega_s ** 2 * omega_i ** 2)  # this equation from On-chi chi2 microring optical parametric oscillation
    P_th_cas = np.sqrt(P_th_spdc / Efficiency_SHG)
    #P_th_kerr = (1.54 * pi / 2 * neff ** 2 * Aeff_m2 * L_res) / (n2 * lambda_pump_ir * Qp ** 2) * (Qp1 / (2 * Qp))
    P_th_kerr = (1.54 * pi / 2 * neff ** 2 * 1/overlap**2 * L_res) / (n2 * lambda_pump_ir * Qp ** 2) * (Qp1 / (2 * Qp))
    # Display the results using st.latex for LaTeX formatting
    st.latex(rf"\chi^{{(2)}} \text{{ interaction coefficient }} g = {g:.4e}")
    st.latex(rf"\text{{SHG efficiency }} \eta = {Efficiency_SHG:.4e}")
    st.latex(rf"P_{{\text{{th, cas}}}} = {P_th_cas*1e3:.2f} \, \mathrm{{mW}}")
    st.latex(rf"P_{{\text{{th, kerr}}}} = {P_th_kerr*1e3:.2f} \, \mathrm{{mW}}")

# Display the equations and meanings of variables using LaTeX
st.latex(r"""
g = \frac{9 \chi^{(2)^2} \cdot \text{overlap}^2 \cdot \hbar \omega_s \omega_i \omega_{p,vis}}{16 \varepsilon_0 L_{\text{res}} \cdot n_{\text{eff}}^6}
""")
st.latex(r"""
Efficiency SHG= \frac{P_{\text{SHG}}}{P_{\text{pump}}^2}
""")
st.latex(r"""
P_{\text{th, cas}} = \frac{\hbar \omega_s \omega_i \omega_{p,vis}^2}{64 g} \frac{Q_{p1}}{Q_s Q_i Q_p}
""")
st.latex(r"""
P_{\text{th, kerr}} = 1.54 \frac{\pi}{2} \frac{n_{\text{eff}}^2  L_{\text{res}}}{overlap^2 n_2 \lambda_{p,ir} Q_p^2} \frac{Q_{p}}{2 Q_{p1}}
""")

st.markdown("""
### Variable Meanings:
- $\lambda_s, \lambda_i$: Signal and idler wavelengths  
- $\lambda_{pump, vis}, \lambda_{pump, ir}$: Pump wavelengths in visible and infrared  
- $\chi^2$: Second-order nonlinear susceptibility  
- $n_{eff}$: Effective refractive index  
- FSR: Distance between neighboring lines in the frequency domain  
- overlap: Overlap integral of the modes ,notice $V_{eff}={L_res}{overlap^2} $
- $n_2$: Nonlinear refractive index  
- $A_{eff}$: Effective mode area, dummy here replaced by $1/overlap^2 $
- $Q_p, Q_s, Q_i$: Loaded quality factors of the pump, signal, and idler modes
- $Q_{p1}$: Intrinsic quality factor of the pump mode  
""")
