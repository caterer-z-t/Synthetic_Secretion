import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def singleBetaCell(t, x, gs1=5):
    # Parameters most likely to vary
    gs2 = 32
    # autos1 = 1
    # autos2 = 1
    # Other parameters
    taus1 = 1000
    taus2 = 120000
    tnbar = 9.09
    vs1 = -40
    vs2 = -42
    ss1 = 0.5
    ss2 = 0.4
    s1knot = 1
    s2knot = 1
    gl = 25
    vl = -40
    gk = 1300
    vk = -80
    gca = 280
    vca = 100
    lambda_ = 1.1
    cm = 4524
    vm = -22
    sm = 7.5
    vn = -9
    sn = 10
    # Unpack variables
    v, n, s1, s2 = x
    # Gating variables
    minf = 1.0 / (1.0 + np.exp((vm - v) / sm))
    ninf = 1.0 / (1.0 + np.exp((vn - v) / sn))
    taun = tnbar / (1.0 + np.exp((v - vn) / sn))
    s1inf = 1.0 / (1.0 + np.exp((vs1 - v) / ss1))
    s2inf = 1.0 / (1.0 + np.exp((vs2 - v) / ss2))
    # Ionic currents
    ica = gca * minf * (v - vca)
    ik = gk * n * (v - vk)
    il = gl * (v - vl)
    is1 = gs1 * s1 * (v - vk)
    is2 = gs2 * s2 * (v - vk)
    vdot = -(ica + ik + il + is1 + is2) / cm
    ndot = lambda_ * (ninf - n) / taun
    s1dot = (s1inf - s1) / taus1
    s2dot = (s2inf - s2) / taus2
    return [vdot, ndot, s1dot, s2dot]


# Call function
def call_SBC(gs1=5, v0=0.0, tmax=90000):
    tspan = (0, tmax)
    x0 = [-50, 0, 0.0, v0]
    # Solve ODE
    sol = solve_ivp(
        lambda t, x: singleBetaCell(t, x, gs1), tspan, x0, method="RK45", max_step=5.0
    )
    tsec = sol.t / 1000  # convert ms to seconds
    v, n, s1, s2 = sol.y
    # Plotting
    plt.figure(figsize=(10, 10))
    # Top panel
    plt.subplot(3, 1, 1)
    plt.plot(tsec, v, color="black", linewidth=2)
    plt.axis([0, tsec[-1], -70, -10])
    plt.title("Phantom model")
    plt.ylabel("V (mV)")
    plt.box(False)
    # Middle panel
    plt.subplot(3, 1, 2)
    plt.plot(tsec, s1, color="black", linewidth=2, label="s1")
    plt.plot(tsec, s2, color="red", linewidth=2, label="s2")
    plt.axis([0, tsec[-1], 0, 1])
    plt.xlabel("t (s)")
    plt.ylabel("s₁ and s₂")
    plt.box(False)
    # Bottom panel
    plt.subplot(3, 1, 3)
    plt.plot(s1, v, color="black", linewidth=2)
    plt.axis([0, 1, -70, -10])
    plt.xlabel("s₁")
    plt.ylabel("V (mV)")
    plt.box(False)
    plt.tight_layout()
    plt.show()
    # # Write to file
    # output = np.vstack((tsec, v, n, s1, s2)).T
    # np.savetxt("data.dat", output, fmt='%9.5f', header='t(s) v n s1 s2')


def main():
    # Example usage (you can modify gs1, v0, tmax as in Figs 2–4)
    call_SBC(gs1=5, v0=0.6, tmax=90000)


if __name__ == "__main__":
    main()