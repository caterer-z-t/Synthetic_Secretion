import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
All contacting $\beta$-cells were considered to form a functional GJ with 
a strength (in pS) picked from a normal distribution of mean 40 pS and 
standard deviation 1 pS. Each cell may form 1–5 GJ connections with other 
cells, yielding a total mean GJ conductance of 40–200 pS. This is supported 
both by experimental and by simulation data from small clusters of dispersed 
human $\beta$-cells (100–200 pS; Loppini et al. (2015)) and $\beta$-cells 
recorded in intact mouse islets (<170 pS, Zhang et al. (2008); 50–120 unitary 
strength, Moreno et al. (2005); 20 pS, Perez-Armendariz et al. (1991)).
"""

def coupledBetaCells(t, x, gs1=5, gj_conductance=40):
    """
    Two coupled beta cells with a gap junction.
    """
    # Parameters
    gs2 = 32
    gl = 25
    gk = 1300
    gca = 280
    vca = 100
    vk = -80
    vl = -40
    lambda_ = 1.1
    cm = 4524
    vm = -22
    sm = 7.5
    vn = -9
    sn = 10
    tnbar = 9.09
    vs1, vs2 = -40, -42
    ss1, ss2 = 0.5, 0.4
    taus1, taus2 = 1000, 120000

    # Unpack variables for two beta cells
    v1, n1, s1_1, s2_1, v2, n2, s1_2, s2_2 = x

    # Gating variables
    minf1 = 1 / (1 + np.exp((vm - v1) / sm))
    ninf1 = 1 / (1 + np.exp((vn - v1) / sn))
    taun1 = tnbar / (1 + np.exp((v1 - vn) / sn))
    s1inf1 = 1 / (1 + np.exp((vs1 - v1) / ss1))
    s2inf1 = 1 / (1 + np.exp((vs2 - v1) / ss2))

    minf2 = 1 / (1 + np.exp((vm - v2) / sm))
    ninf2 = 1 / (1 + np.exp((vn - v2) / sn))
    taun2 = tnbar / (1 + np.exp((v2 - vn) / sn))
    s1inf2 = 1 / (1 + np.exp((vs1 - v2) / ss1))
    s2inf2 = 1 / (1 + np.exp((vs2 - v2) / ss2))

    # Ionic currents for cell 1
    ica1 = gca * minf1 * (v1 - vca)
    ik1 = gk * n1 * (v1 - vk)
    il1 = gl * (v1 - vl)
    is1_1 = gs1 * s1_1 * (v1 - vk)
    is2_1 = gs2 * s2_1 * (v1 - vk)

    # Ionic currents for cell 2
    ica2 = gca * minf2 * (v2 - vca)
    ik2 = gk * n2 * (v2 - vk)
    il2 = gl * (v2 - vl)
    is1_2 = gs1 * s1_2 * (v2 - vk)
    is2_2 = gs2 * s2_2 * (v2 - vk)

    # Gap junction current
    igj = gj_conductance * (v2 - v1)

    # ODEs for cell 1
    v1dot = -(ica1 + ik1 + il1 + is1_1 + is2_1 - igj) / cm
    n1dot = lambda_ * (ninf1 - n1) / taun1
    s1_1dot = (s1inf1 - s1_1) / taus1
    s2_1dot = (s2inf1 - s2_1) / taus2

    # ODEs for cell 2
    v2dot = -(ica2 + ik2 + il2 + is1_2 + is2_2 + igj) / cm
    n2dot = lambda_ * (ninf2 - n2) / taun2
    s1_2dot = (s1inf2 - s1_2) / taus1
    s2_2dot = (s2inf2 - s2_2) / taus2

    return [v1dot, n1dot, s1_1dot, s2_1dot, v2dot, n2dot, s1_2dot, s2_2dot]


def simulateCoupledCells(gs1=5, gj_conductance=40, v0=0.0, tmax=90000):
    """
    Simulates two coupled beta cells.
    """
    tspan = (0, tmax)
    x0 = [-50, 0, 0.0, 0.0, -50, 0, 0.0, v0]  # Initial conditions for both cells

    sol = solve_ivp(
        lambda t, x: coupledBetaCells(t, x, gs1, gj_conductance),
        tspan,
        x0,
        method="RK45",
        max_step=5.0,
    )

    tsec = sol.t / 1000  # Convert ms to seconds
    v1, n1, s1_1, s2_1, v2, n2, s1_2, s2_2 = sol.y

    # Plotting
    plt.figure(figsize=(10, 10))

    # Voltage traces
    plt.subplot(3, 1, 1)
    plt.plot(tsec, v1, color="black", linewidth=2, label="V1")
    plt.plot(tsec, v2, color="red", linewidth=2, label="V2")
    plt.legend()
    plt.title("Coupled Beta Cells with Gap Junction")
    plt.ylabel("V (mV)")
    plt.box(False)

    # s1 and s2 dynamics
    plt.subplot(3, 1, 2)
    plt.plot(tsec, s1_1, color="black", linewidth=2, label="s1_1")
    plt.plot(tsec, s1_2, color="red", linewidth=2, label="s1_2")
    plt.xlabel("t (s)")
    plt.ylabel("s1 Dynamics")
    plt.box(False)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(s1_1, v1, color="black", linewidth=2, label="V1 vs s1_1")
    plt.plot(s1_2, v2, color="red", linewidth=2, label="V2 vs s1_2")
    plt.xlabel("s1")
    plt.ylabel("V (mV)")
    plt.box(False)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    simulateCoupledCells(gs1=5, gj_conductance=40, v0=0.6, tmax=90000)


if __name__ == "__main__":
    main()
