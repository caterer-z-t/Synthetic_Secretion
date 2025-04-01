import numpy as np
from scipy.integrate import solve_ivp

# Define parameters
K0 = 0.00241
K1 = 0.000434
K2 = 1.62e-05
K3 = 1.51e-05
Km0 = 0.02
Km1 = 0.064
Km2 = 0.018
Km3 = 0.014
I0 = 5.0
E0 = 0.2


# Define the system of ODEs
def model(t, y):
    S, ES0, ES1, ES2, ES3, P = y

    v0 = K0 * E0 * S / (Km0 + S)
    v1 = K1 * ES0 / (Km1 + ES0)
    v2 = K2 * ES1 / (Km2 + ES1)
    v3 = K3 * ES2 / (Km3 + ES2)

    dS = -v0
    dES0 = v0 - v1
    dES1 = v1 - v2
    dES2 = v2 - v3
    dES3 = v3
    dP = v3  # Product formation

    return [dS, dES0, dES1, dES2, dES3, dP]


# Initial conditions
S0 = 20.0
ES0_0 = 0.0
ES1_0 = 0.0
ES2_0 = 0.0
ES3_0 = 0.0
P0 = 0.0
y0 = [S0, ES0_0, ES1_0, ES2_0, ES3_0, P0]

# Time span for integration
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)

# Solve the ODE system
solution = solve_ivp(model, t_span, y0, t_eval=t_eval, method="RK45")

# Extract results
time = solution.t
S, ES0, ES1, ES2, ES3, P = solution.y

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time, S, label="S (Substrate)")
plt.plot(time, P, label="P (Product)")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.show()
