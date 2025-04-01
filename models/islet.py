import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# set working directory to the location of this script
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def phantom(t, x, gs1=5, gs2=32):
    # Parameters
    lambda_ = 1.1
    gca, gk, gl = 280, 1300, 25
    vca, vk, vl = 100, -80, -40
    cm = 4524
    tnbar, vm, vn, sm, sn = 9.09, -22, -9, 7.5, 10
    vs1, vs2, ss1, ss2 = -40, -42, 0.5, 0.4
    taus1, taus2 = 1000, 120000

    # Variables
    v, n, s1, s2 = x

    # Activation and time-constant functions
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

    # Differential equations
    dvdt = -(ica + ik + il + is1 + is2) / cm
    dndt = lambda_ * (ninf - n) / taun
    ds1dt = (s1inf - s1) / taus1
    ds2dt = (s2inf - s2) / taus2

    return [dvdt, dndt, ds1dt, ds2dt]


# Initial conditions
x0 = [-50, 0, 0.0, 0.6]
t_span = (0, 90000)
t_eval = np.linspace(*t_span, 5000)  # Points for evaluation

# Solve the system
sol = solve_ivp(phantom, t_span, x0, t_eval=t_eval, method="LSODA")

# Extract solutions
tsec = sol.t / 1000  # Convert time to seconds
v, n, s1, s2 = sol.y

# Plot results
plt.figure(figsize=(8, 10))

# Top panel
plt.subplot(3, 1, 1)
plt.plot(tsec, v, "k", linewidth=2)
plt.axis([0, 90, -70, -10])
plt.title("Phantom Model")
plt.ylabel("V (mV)")
plt.box(False)

# Middle panel
plt.subplot(3, 1, 2)
plt.plot(tsec, s1, "k", linewidth=2, label="s1")
plt.plot(tsec, s2, "r", linewidth=2, label="s2")
plt.axis([0, 90, 0, 1])
plt.xlabel("t (s)")
plt.ylabel("s1 and s2")
plt.legend()
plt.box(False)

# Bottom panel
plt.subplot(3, 1, 3)
plt.plot(s1, v, "k", linewidth=2)
plt.axis([0, 1, -70, -10])
plt.xlabel("s1")
plt.ylabel("V (mV)")
plt.box(False)

plt.tight_layout()
plt.show()

# Save data
output_data = np.column_stack((tsec, v, n, s1, s2))
np.savetxt("../data/data.dat", output_data, fmt="%9.5f")
