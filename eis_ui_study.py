import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import schemdraw
import schemdraw.elements as elm

# EIS model: Rs + [Rct || Cdl] + Warburg
def cell_impedance(freq, Rs, Rct, Cdl, sigma):
    omega = 2 * np.pi * freq
    Zc = 1 / (1j * omega * Cdl)               # Capacitive impedance
    Zw = sigma / np.sqrt(1j * omega)          # Warburg element
    Z_parallel = 1 / (1/Rct + 1/Zc)
    return Rs + Z_parallel + Zw

def plot_nyquist(Z_stack, Z_cells=None):
    fig, ax = plt.subplots()
    ax.plot(Z_stack.real, -Z_stack.imag, 'o-', label="Total Stack", color='blue')
    if Z_cells:
        for i, Z in enumerate(Z_cells):
            ax.plot(Z.real, -Z.imag, '--', alpha=0.6, label=f"Cell {i+1}")
    ax.set_xlabel("Z' (Real part, Ω)")
    ax.set_ylabel("-Z'' (Imaginary part, Ω)")
    ax.set_title("Nyquist Plot: EIS Response")
    ax.legend()
    ax.grid(True)
    return fig

def plot_bode(freq, Z_stack, Z_cells=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Magnitude plot
    ax1.semilogx(freq, 20 * np.log10(np.abs(Z_stack)), 'o-', label="Total Stack", color='blue')
    if Z_cells:
        for i, Z in enumerate(Z_cells):
            ax1.semilogx(freq, 20 * np.log10(np.abs(Z)), '--', alpha=0.6, label=f"Cell {i+1}")
    ax1.set_ylabel("Magnitude |Z| (dB)")
    ax1.grid(True, which='both', ls='--')
    ax1.legend()
    
    # Phase plot
    ax2.semilogx(freq, np.angle(Z_stack, deg=True), 'o-', color='blue')
    if Z_cells:
        for i, Z in enumerate(Z_cells):
            ax2.semilogx(freq, np.angle(Z, deg=True), '--', alpha=0.6)
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which='both', ls='--')
    
    fig.suptitle("Bode Plot: Magnitude and Phase of EIS Response")
    return fig

def draw_equivalent_circuit():
    d = schemdraw.Drawing()
    d.config(unit=2.5)
    d += elm.SourceSin().label("AC Source", loc='bottom')
    d += elm.Resistor().label("Rs", loc='bottom')
    d.push()
    d += elm.Line().right()
    d += elm.Resistor().down().label("Rct", loc='left')
    d += elm.Line().right()
    d += elm.Capacitor().up().label("Cdl", loc='right')
    d += elm.Line().left()
    d.pop()
    d += elm.Line().right()
    d += elm.Inductor().label("Warburg σ", loc='bottom')
    d += elm.Line().right().label("To Analyzer", loc='bottom')
    return d.get_imagedata('png')

def main():
    st.set_page_config(layout="wide")
    st.title("🔋 EIS Stack Simulator with Circuit & Explanation")

    # Sidebar: What is EIS?
    with st.sidebar:
        st.header("ℹ️ What is EIS?")
        st.markdown("""
        **Electrochemical Impedance Spectroscopy (EIS)**:
        - A method to study battery behavior using a small AC voltage.
        - Measures impedance (resistance + reactance) at various frequencies.
        - Reveals:
            - **Rs**: Ohmic/electrolyte resistance
            - **Rct**: Charge transfer resistance
            - **Cdl**: Double-layer capacitance
            - **σ**: Diffusion (Warburg impedance)

        Used to detect:
        - Battery aging
        - Cell imbalance
        - Degradation
        """)

    # Show circuit
    st.subheader("📘 Equivalent Circuit Model (Randles + Warburg)")
    st.image(draw_equivalent_circuit(), caption="Equivalent Circuit: Rs + [Rct || Cdl] + Warburg")

    # Cell stack input
    st.subheader("🔧 Cell Stack Configuration")
    num_cells = st.slider("Number of Cells in Stack (Series)", 1, 5, 3)

    freq = np.logspace(-1, 5, 300)
    Z_cells = []

    # Per-cell parameters
    for i in range(num_cells):
        st.markdown(f"### Cell {i+1}")
        cols = st.columns(4)
        Rs = cols[0].slider(f"Rs (Ω) - Cell {i+1}", 0.01, 2.0, 0.1, key=f"Rs_{i}")
        Rct = cols[1].slider(f"Rct (Ω) - Cell {i+1}", 0.1, 10.0, 1.0, key=f"Rct_{i}")
        log_cdl = cols[2].slider(f"log10(Cdl [F]) - Cell {i+1}", -6.0, -2.0, -4.0, key=f"Cdl_{i}")
        Cdl = 10 ** log_cdl
        sigma = cols[3].slider(f"σ (Ω·s⁻½) - Cell {i+1}", 0.01, 1.0, 0.1, key=f"sigma_{i}")
        Z = cell_impedance(freq, Rs, Rct, Cdl, sigma)
        Z_cells.append(Z)

    # Stack impedance (series)
    Z_stack = np.sum(Z_cells, axis=0)

    # Show plots
    st.subheader("📉 Nyquist Plot")
    st.pyplot(plot_nyquist(Z_stack, Z_cells))

    st.subheader("📈 Bode Plot")
    st.pyplot(plot_bode(freq, Z_stack, Z_cells))

    # Diagnostic
    st.subheader("🧪 EIS-Based Diagnostics")
    for i, Z in enumerate(Z_cells):
        idx_1Hz = np.argmin(np.abs(freq - 1))
        st.markdown(f"**Cell {i+1}**:")
        st.write(f"- Impedance at 1 Hz: **{np.abs(Z[idx_1Hz]):.2f} Ω**")
        tau = (1 / (2 * np.pi)) * (np.argmax(-Z.imag))
        st.write(f"- Approx. τ (semicircle freq): **{tau:.2f} s**")

    st.markdown("""
    ---
    ### 🧠 Interpretation Tips:

    **Nyquist Plot:**
    - Plots imaginary part (-Z'') vs real part (Z') of impedance.
    - The **leftmost intercept** on the real axis corresponds to **Rs** (electrolyte resistance).
    - The **semicircle diameter** roughly equals **Rct** (charge transfer resistance).
    - The **width** of the semicircle relates to the time constant τ = Rct × Cdl.
    - The **low-frequency tail** (straight line) shows the Warburg diffusion impedance.

    **Bode Plot:**
    - Plots magnitude (|Z| in dB) and phase (degrees) vs frequency on log scale.
    - The **magnitude plot** shows how impedance varies across frequencies.
    - The **phase plot** indicates capacitive (phase near -90°) and resistive (phase near 0°) behaviors.
    - At high frequency, the phase approaches 0°, dominated by Rs.
    - At intermediate frequencies, phase dips indicating charge transfer and capacitive effects.
    - At low frequencies, phase moves due to diffusion (Warburg).

    Both plots combined give comprehensive insight into battery internal processes.

    You can simulate battery degradation by:
    - Increasing Rct
    - Decreasing Cdl
    - Increasing σ
    """)

if __name__ == "__main__":
    main()
