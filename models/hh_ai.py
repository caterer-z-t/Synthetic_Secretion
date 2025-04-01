# In[1]: Imports
import numpy as np
import matplotlib.pyplot as plt

# In[2]: HH Constants and Helper Functions
# Constants from Table 2.1
Cm = 1.0  # Membrane capacitance (uF/cm^2)
ENa = 55  # Sodium reversal potential (mV)
EK = -77  # Potassium reversal potential (mV)
EL = -65  # Leak reversal potential (mV)
gNa = 40  # Maximum sodium conductance (mS/cm^2)
gK = 35  # Maximum potassium conductance (mS/cm^2)
gL = 0.3  # Leak conductance (mS/cm^2)

# Time parameters
dt = 0.01  # Time step (ms)
T = 100  # Total time (ms)
t = np.arange(0, T, dt)


# Function for alpha and beta values - directly from Table 2.1
def alpha_n(V):
    return 0.02 * (V - 25) / (1 - np.exp(-(V - 25) / 9))


def beta_n(V):
    return -0.002 * (V - 25) / (1 - np.exp((V - 25) / 9))


def alpha_m(V):
    return 0.182 * (V + 35) / (1 - np.exp(-(V + 35) / 9))


def beta_m(V):
    return -0.124 * (V + 35) / (1 - np.exp((V + 35) / 9))


def alpha_h(V):
    return 0.25 * np.exp(-(V + 90) / 12)


def beta_h(V):
    return 0.25 * np.exp((V + 62) / 6) / np.exp((V + 90) / 12)


# Calculate steady-state values and time constants
def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))


def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))


def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))


def tau_n(V):
    return 1.0 / (alpha_n(V) + beta_n(V))


def tau_m(V):
    return 1.0 / (alpha_m(V) + beta_m(V))


def tau_h(V):
    return 1.0 / (alpha_h(V) + beta_h(V))

# In[3]: Simulation of HH Model
# Initialize variables at resting potential
V = -65  # Initial membrane potential (mV)
n = n_inf(V)  # Initial potassium activation variable
m = m_inf(V)  # Initial sodium activation variable
h = h_inf(V)  # Initial sodium inactivation variable

# Arrays to store results
V_trace = np.zeros(len(t))
n_trace = np.zeros(len(t))
m_trace = np.zeros(len(t))
h_trace = np.zeros(len(t))
INa_trace = np.zeros(len(t))
IK_trace = np.zeros(len(t))
IL_trace = np.zeros(len(t))

# Initial conditions
V_trace[0] = V
n_trace[0] = n
m_trace[0] = m
h_trace[0] = h

# External current (similar to Figure 2.6)
I_ext = np.zeros(len(t))
# Apply a 1ms current pulse at t=1ms
pulse_start = int(1 / dt)
pulse_duration = int(1 / dt)  # 1ms duration
I_ext[pulse_start : pulse_start + pulse_duration] = 10  # Current pulse (uA/cm^2)

# Simulation loop - using the differential equations from the document
for i in range(1, len(t)):
    # Compute ionic currents
    INa = gNa * (m**3) * h * (V - ENa)
    IK = gK * (n**4) * (V - EK)
    IL = gL * (V - EL)

    # Store currents
    INa_trace[i - 1] = INa
    IK_trace[i - 1] = IK
    IL_trace[i - 1] = IL

    # Compute membrane potential update (equation 2.4)
    dV = (-INa - IK - IL + I_ext[i - 1]) / Cm
    V = V + dV * dt

    # Update gating variables (equation 2.6)
    n = n + (n_inf(V) - n) * dt / tau_n(V)
    m = m + (m_inf(V) - m) * dt / tau_m(V)
    h = h + (h_inf(V) - h) * dt / tau_h(V)

    # Store values
    V_trace[i] = V
    n_trace[i] = n
    m_trace[i] = m
    h_trace[i] = h

# Store final currents
INa = gNa * (m**3) * h * (V - ENa)
IK = gK * (n**4) * (V - EK)
IL = gL * (V - EL)
INa_trace[-1] = INa
IK_trace[-1] = IK
IL_trace[-1] = IL

# In[4]: Plotting the Results of HH Model
# Plot results (similar to Figure 2.6)
plt.figure(figsize=(15, 10))

# Plot membrane potential (Figure 2.6A)
plt.subplot(3, 1, 1)
plt.plot(t, V_trace)
plt.axhline(y=-65, color="r", linestyle="--", alpha=0.5, label="Resting potential")
plt.title("Hodgkin-Huxley Neuron Model - Action Potential")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

# Plot gating variables (Figure 2.6B)
plt.subplot(3, 1, 2)
plt.plot(t, m_trace, label="m (Na+ activation)")
plt.plot(t, h_trace, label="h (Na+ inactivation)")
plt.plot(t, n_trace, label="n (K+ activation)")
plt.title("Gating Variables")
plt.ylabel("Gating Value")
plt.legend()

# Plot ionic currents (Figure 2.6C)
plt.subplot(3, 1, 3)
plt.plot(t, INa_trace, label="INa (Na+ current)")
plt.plot(t, IK_trace, label="IK (K+ current)")
plt.plot(t, IL_trace, label="IL (Leak current)")
plt.title("Ionic Currents")
plt.xlabel("Time (ms)")
plt.ylabel("Current (μA/cm²)")
plt.legend()

plt.tight_layout()
plt.show()

# In[5]: Simulating HH Model with External Current
# Function to simulate with different input currents
def simulate_HH(I_amplitude, duration=100, dt=0.01, pulse_start=1, pulse_duration=1):
    """
    Simulate Hodgkin-Huxley model with a given input current amplitude.
    Returns the membrane potential trace.
    """
    # Time parameters
    t = np.arange(0, duration, dt)

    # Initialize variables at resting potential
    V = -65  # Initial membrane potential (mV)
    n = n_inf(V)  # Initial potassium activation variable
    m = m_inf(V)  # Initial sodium activation variable
    h = h_inf(V)  # Initial sodium inactivation variable

    # Arrays to store results
    V_trace = np.zeros(len(t))
    V_trace[0] = V

    # External current
    I_ext = np.zeros(len(t))
    start_idx = int(pulse_start / dt)
    end_idx = start_idx + int(pulse_duration / dt)
    I_ext[start_idx:end_idx] = I_amplitude

    # Simulation loop
    for i in range(1, len(t)):
        # Compute ionic currents
        INa = gNa * (m**3) * h * (V - ENa)
        IK = gK * (n**4) * (V - EK)
        IL = gL * (V - EL)

        # Compute membrane potential update
        dV = (-INa - IK - IL + I_ext[i - 1]) / Cm
        V = V + dV * dt

        # Update gating variables
        n = n + (n_inf(V) - n) * dt / tau_n(V)
        m = m + (m_inf(V) - m) * dt / tau_m(V)
        h = h + (h_inf(V) - h) * dt / tau_h(V)

        # Store values
        V_trace[i] = V

    return t, V_trace


# Demonstrate threshold behavior (Figure 2.8B)
plt.figure(figsize=(10, 6))
t, V_below_threshold = simulate_HH(7)  # Subthreshold current
t, V_above_threshold = simulate_HH(8)  # Suprathreshold current

plt.plot(t, V_below_threshold, "--", label="Below threshold")
plt.plot(t, V_above_threshold, label="Above threshold")
plt.axhline(y=-65, color="r", linestyle="--", alpha=0.5, label="Resting potential")
plt.title("Threshold Effect in Hodgkin-Huxley Model")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.xlim(0, 10)  # Focus on first 10ms
plt.tight_layout()
plt.show()


# Demonstrate refractoriness (Figure 2.9)
def simulate_refractoriness(second_pulse_time, I_amplitude=10):
    """Simulate refractoriness with two pulses"""
    duration = 100
    t = np.arange(0, duration, dt)

    # Initialize variables at resting potential
    V = -65  # Initial membrane potential (mV)
    n = n_inf(V)  # Initial potassium activation variable
    m = m_inf(V)  # Initial sodium activation variable
    h = h_inf(V)  # Initial sodium inactivation variable

    # Arrays to store results
    V_trace = np.zeros(len(t))
    V_trace[0] = V

    # External current with two pulses
    I_ext = np.zeros(len(t))
    I_ext[int(20 / dt) : int(21 / dt)] = I_amplitude  # First pulse at t=20ms
    I_ext[int(second_pulse_time / dt) : int((second_pulse_time + 1) / dt)] = (
        I_amplitude  # Second pulse
    )

    # Simulation loop
    for i in range(1, len(t)):
        # Compute ionic currents
        INa = gNa * (m**3) * h * (V - ENa)
        IK = gK * (n**4) * (V - EK)
        IL = gL * (V - EL)

        # Compute membrane potential update
        dV = (-INa - IK - IL + I_ext[i - 1]) / Cm
        V = V + dV * dt

        # Update gating variables
        n = n + (n_inf(V) - n) * dt / tau_n(V)
        m = m + (m_inf(V) - m) * dt / tau_m(V)
        h = h + (h_inf(V) - h) * dt / tau_h(V)

        # Store values
        V_trace[i] = V

    return t, V_trace, I_ext


# Plot refractoriness for different second pulse times
plt.figure(figsize=(12, 8))
t, V_35ms, I_35ms = simulate_refractoriness(35)
t, V_45ms, I_45ms = simulate_refractoriness(45)
t, V_55ms, I_55ms = simulate_refractoriness(55)

plt.plot(t, V_35ms, label="Second pulse at 35ms")
plt.plot(t, V_45ms, label="Second pulse at 45ms")
plt.plot(t, V_55ms, label="Second pulse at 55ms")

# Mark the pulses
plt.plot(t, I_35ms * 2 - 70, "k-", alpha=0.3)  # Scale and shift for visibility

plt.axhline(y=-65, color="r", linestyle="--", alpha=0.5, label="Resting potential")
plt.title("Refractoriness in Hodgkin-Huxley Model")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.xlim(0, 80)  # Focus on relevant time window
plt.tight_layout()
plt.show()
