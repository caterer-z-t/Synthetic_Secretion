import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class SingleBetaCell:
    def __init__(
        self,
        gs1=5,
        gs2=32,
        taus1=1000,
        taus2=120000,
        tnbar=9.09,
        vs1=-40,
        vs2=-42,
        ss1=0.5,
        ss2=0.4,
        s1knot=1,
        s2knot=1,
        gl=25,
        vl=-40,
        gk=1300,
        vk=-80,
        gca=280,
        vca=100,
        lambda_=1.1,
        cm=4524,
        vm=-22,
        sm=7.5,
        vn=-9,
        sn=10,
    ):
        # Store all parameters as instance variables
        self.gs1 = gs1
        self.gs2 = gs2
        self.taus1 = taus1
        self.taus2 = taus2
        self.tnbar = tnbar
        self.vs1 = vs1
        self.vs2 = vs2
        self.ss1 = ss1
        self.ss2 = ss2
        self.s1knot = s1knot
        self.s2knot = s2knot
        self.gl = gl
        self.vl = vl
        self.gk = gk
        self.vk = vk
        self.gca = gca
        self.vca = vca
        self.lambda_ = lambda_
        self.cm = cm
        self.vm = vm
        self.sm = sm
        self.vn = vn
        self.sn = sn

    def dynamics(self, t, x):
        """Calculate the dynamics of the beta cell model"""
        # Unpack variables
        v, n, s1, s2 = x

        # Gating variables
        minf = 1.0 / (1.0 + np.exp((self.vm - v) / self.sm))
        ninf = 1.0 / (1.0 + np.exp((self.vn - v) / self.sn))
        taun = self.tnbar / (1.0 + np.exp((v - self.vn) / self.sn))
        s1inf = 1.0 / (1.0 + np.exp((self.vs1 - v) / self.ss1))
        s2inf = 1.0 / (1.0 + np.exp((self.vs2 - v) / self.ss2))

        # Ionic currents
        ica = self.gca * minf * (v - self.vca)
        ik = self.gk * n * (v - self.vk)
        il = self.gl * (v - self.vl)
        is1 = self.gs1 * s1 * (v - self.vk)
        is2 = self.gs2 * s2 * (v - self.vk)

        # Differential equations
        vdot = -(ica + ik + il + is1 + is2) / self.cm
        ndot = self.lambda_ * (ninf - n) / taun
        s1dot = (s1inf - s1) / self.taus1
        s2dot = (s2inf - s2) / self.taus2

        return np.array([vdot, ndot, s1dot, s2dot])

    def simulate(self, v0=0.0, n0=0.0, s10=0.0, s20=0.6, tmax=90000, max_step=5.0):
        """Simulate the beta cell model"""
        tspan = (0, tmax)
        x0 = [-50, n0, s10, s20]

        # Solve ODE
        sol = solve_ivp(self.dynamics, tspan, x0, method="RK45", max_step=max_step)

        return sol

    def plot_results(self, sol):
        """Plot the simulation results"""
        tsec = sol.t / 1000  # convert ms to seconds
        v, n, s1, s2 = sol.y

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
        plt.legend()
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

    def run_simulation(self, gs1=5, v0=0.6, tmax=90000):
        """Run a simulation with the given parameters and plot the results"""
        # Update gs1 if needed
        self.gs1 = gs1

        # Run simulation
        sol = self.simulate(s20=v0, tmax=tmax)

        # Plot results
        self.plot_results(sol)

        return sol

    def save_data(self, sol, filename="data.dat"):
        """Save simulation results to a file"""
        tsec = sol.t / 1000  # convert ms to seconds
        v, n, s1, s2 = sol.y
        output = np.vstack((tsec, v, n, s1, s2)).T
        np.savetxt(filename, output, fmt="%9.5f", header="t(s) v n s1 s2")


def main():
    # Create instance of SingleBetaCell with default parameters
    model = SingleBetaCell()

    # Example usage (you can modify gs1, v0, tmax as in Figs 2–4)
    model.run_simulation(gs1=5, v0=0.6, tmax=90000)

if __name__ == "__main__":
    # Run the main function
    main()
