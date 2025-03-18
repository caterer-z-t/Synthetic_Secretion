# In[0]: Hodgkin-Huxley Model Equations

# equation 2.3
# I(t) = I_{c}(t) + \sum_{k} I_{k}(t)

# equation 2.4
# C \frac{du}{dt} = - \sum_{k} I_{k}(t) + I(t)

# equation 2.5
# \sum_{k} I_{k}(t) = g_{Na} m^{3} h (V - E_{Na}) + g_{K} n^{4} (V - E_{K}) + g_{L} (V - E_{L})

# equation 2.6
# \dot{x} = \frac{dx}{dt} = - \frac{1}{\tau_{x}(V)} (x - x_{0}(V)); where x = n, m, h

# equation 2.7
# m(t) = m_{0}(V_1) + [m_0(V_0) - m_0(V_1)] e^{-\frac{t - t_0}{\tau_m(V_0)}}
# h(t) = h_{0}(V_1) + [h_0(V_0) - h_0(V_1)] e^{-\frac{t - t_0}{\tau_h(V_0)}}

# equation 2.8
# n(t) = n_{0}(V_1) + [n_0(V_0) - n_0(V_1)] e^{-\frac{t - t_0}{\tau_n(V_0)}}

# equation 2.9
# \dot{x} = \alpha_{x}(V) (1 - x) - \beta_{x}(V) x ; where x = n, m, h

# In[1]: Imports
import numpy as np
import matplotlib.pyplot as plt

# In[2]: Hodgkin-Huxley Class
class HodgkinHuxley:
    def __init__(self):
        # Constants from Table 2.1
        self.Cm = 1.0  # Membrane capacitance (uF/cm^2)
        self.ENa = 55  # Sodium reversal potential (mV)
        self.EK = -77  # Potassium reversal potential (mV)
        self.EL = -65  # Leak reversal potential (mV)
        self.gNa = 40  # Maximum sodium conductance (mS/cm^2)
        self.gK = 35  # Maximum potassium conductance (mS/cm^2)
        self.gL = 0.3  # Leak conductance (mS/cm^2)

        # Time parameters
        self.dt = 0.01  # Time step (ms)
        self.T = 100  # Total time (ms)
        self.t = np.arange(0, self.T, self.dt)

    # In[3]: Helper functions
    # Function for alpha and beta values - directly from Table 2.1
    def _alpha_n(self, V):
        return 0.02 * (V - 25) / (1 - np.exp(-(V - 25) / 9))

    def _beta_n(self, V):
        return -0.002 * (V - 25) / (1 - np.exp((V - 25) / 9))

    def _alpha_m(self, V):
        return 0.182 * (V + 35) / (1 - np.exp(-(V + 35) / 9))

    def _beta_m(self, V):
        return -0.124 * (V + 35) / (1 - np.exp((V + 35) / 9))

    def _alpha_h(self, V):
        return 0.25 * np.exp(-(V + 90) / 12)

    def _beta_h(self, V):
        return 0.25 * np.exp((V + 62) / 6) / np.exp((V + 90) / 12)

    # Calculate steady-state values and time constants
    def n_inf(self, V):
        return self._alpha_n(V) / (self._alpha_n(V) + self._beta_n(V))

    def m_inf(self, V):
        return self._alpha_m(V) / (self._alpha_m(V) + self._beta_m(V))

    def h_inf(self, V):
        return self._alpha_h(V) / (self._alpha_h(V) + self._beta_h(V))

    def tau_n(self, V):
        return 1.0 / (self._alpha_n(V) + self._beta_n(V))

    def tau_m(self, V):
        return 1.0 / (self._alpha_m(V) + self._beta_m(V))

    def tau_h(self, V):
        return 1.0 / (self._alpha_h(V) + self._beta_h(V))

    # Compute ionic currents
    def _compute_Ina(self, V, m, h):
        return self.gNa * (m**3) * h * (V - self.ENa)

    def _compute_Ik(self, V, n):
        return self.gK * (n**4) * (V - self.EK)

    def _compute_Il(self, V):
        return self.gL * (V - self.EL)

    # Update membrane potential
    def _update_V(self, V, I_ext, m, h, n):
        INa = self._compute_Ina(V, m, h)
        IK = self._compute_Ik(V, n)
        IL = self._compute_Il(V)
        dV = (-INa - IK - IL + I_ext) / self.Cm
        return V + dV * self.dt

    def _prepare_HH_model(self):
        # Voltage trace
        self.V = -65  # Initial membrane potential (mV)
        self.V_trace = np.zeros(len(self.t))
        self.V_trace[0] = self.V

        self.n_trace = np.zeros(len(self.t))
        self.n_trace[0] = self.n_inf(self.V)  # Initial potassium activation variable

        self.m_trace = np.zeros(len(self.t))
        self.m_trace[0] = self.m_inf(self.V)  # Initial sodium activation variable

        self.h_trace = np.zeros(len(self.t))
        self.h = self.h_inf(self.V)
        self.h_trace[0] = self.h_inf(self.V)

        self.INa_trace = np.zeros(len(self.t))
        self.IK_trace = np.zeros(len(self.t))
        self.IL_trace = np.zeros(len(self.t))

        self.n = self.n_inf(self.V)
        self.m = self.m_inf(self.V)
        self.h = self.h_inf(self.V)

        # External current (similar to Figure 2.6)
        self.I_ext = np.zeros(len(self.t))
        # Apply a 1ms current pulse at t=1ms
        self.pulse_start = int(1 / self.dt)
        self.pulse_duration = int(1 / self.dt)  # 1ms duration
        self.I_ext[self.pulse_start : self.pulse_start + self.pulse_duration] = 10  # Current pulse (\mu A/cm^2)

    # In[4]: Main Hodgkin-Huxley model functions

    # Simulate Hodgkin-Huxley model
    def simulate_HH(self):
        
        self._prepare_HH_model()

        for i in range(1, len(self.t)):
            # Compute ionic currents
            self.INa_trace[i - 1] = self._compute_Ina(self.V, self.m, self.h)
            self.IK_trace[i - 1] = self._compute_Ik(self.V, self.n)
            self.IL_trace[i - 1] = self._compute_Il(self.V)

            # Update membrane potential
            self.V = self._update_V(self.V, self.I_ext[i - 1], self.m, self.h, self.n)
            self.V_trace[i] = self.V

            self.m_trace[i] = self.m
            self.n_trace[i] = self.n
            self.h_trace[i] = self.h

            # Update gating variables
            self.n = self.n + (self.n_inf(self.V) - self.n) * self.dt / self.tau_n(self.V)
            self.m = self.m + (self.m_inf(self.V) - self.m) * self.dt / self.tau_m(self.V)
            self.h = self.h + (self.h_inf(self.V) - self.h) * self.dt / self.tau_h(self.V)

        return self.t, self
    
    def simulate_HH_with_current(
            self, 
            I_amp = 10, 
            duration = 100,
            dt = 0.01,
            pulse_start = 1,
            pulse_duration = 1
            ):
        
        self.t = np.arange(0, duration, dt)
        self.dt = dt
        self.I_ext = np.zeros(len(self.t))
        self.pulse_start = int(pulse_start / dt)
        self.pulse_duration = int(pulse_duration / dt)
        self.I_ext[self.pulse_start : self.pulse_start + self.pulse_duration] = I_amp

        return self.simulate_HH()

    # In[5]: Plotting functions
    def plot_results(self):
        plt.figure(figsize=(15, 10))

        # Plot membrane potential
        plt.subplot(3, 1, 1)
        plt.plot(self.t, self.V_trace)
        plt.axhline(y=-65, color="r", linestyle="--", alpha=0.5, label="Resting potential")
        plt.title("Hodgkin-Huxley Neuron Model - Action Potential")
        plt.ylabel("Membrane Potential (mV)")
        plt.legend()

        # Plot gating variables
        plt.subplot(3, 1, 2)
        plt.plot(self.t, self.m_trace, label=r"m (Na$^+$ activation)")
        plt.plot(self.t, self.h_trace, label=r"h (Na$^+$ inactivation)")
        plt.plot(self.t, self.n_trace, label=r"n (K$^+$ activation)")
        plt.title("Gating Variables")
        plt.ylabel("Gating Value")
        plt.legend()

        # Plot ionic currents
        plt.subplot(3, 1, 3)
        plt.plot(self.t, self.INa_trace, label=r"INa (Na$^+$ current)")
        plt.plot(self.t, self.IK_trace, label=r"IK (K$^+$ current)")
        plt.plot(self.t, self.IL_trace, label="IL (Leak current)")
        plt.title("Ionic Currents")
        plt.xlabel("Time (ms)")
        plt.ylabel(r"Current $(\mu \text{A}/\text{cm}^2)$")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_current_results(self, above_threshold, below_threshold):
        plt.figure(figsize=(10,6))

        plt.plot(self.t, above_threshold, label="Above threshold")
        plt.plot(self.t, below_threshold, label="Below threshold")
        plt.title("Hodgkin-Huxley Neuron Model - Action Potential")
        plt.ylabel("Membrane Potential (mV)")
        plt.xlabel("Time (ms)")
        plt.legend()
        plt.show()

# In[3]: Simulate Hodgkin-Huxley model
def main():
    # Create Hodgkin-Huxley model
    hh_model = HodgkinHuxley()

    # Simulate model
    hh_model.simulate_HH()

    # Plot results
    hh_model.plot_results()

    # Simulate HH Model w/ different input currents
    _, hh_model_below_threshold = hh_model.simulate_HH_with_current(
        I_amp=7, duration=100, dt=0.01, pulse_start=1, pulse_duration=1
    )

    _, hh_model_above_threshold = hh_model.simulate_HH_with_current(
        I_amp=8, duration=100, dt=0.01, pulse_start=1, pulse_duration=1
    )
    hh_model.plot_current_results(hh_model_below_threshold.V_trace, hh_model_above_threshold.V_trace)

if __name__ == "__main__":
    main()
