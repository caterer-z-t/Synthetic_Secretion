import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class SingleDeltaCell:
    def __init__(self, **kwargs):
        """
        Initialize a single delta cell model based on Briant et al.

        Parameters:
        -----------
        **kwargs: Parameter overrides for the delta cell model
        """
        # Default parameters
        self.params = {
            "gkatp": 0.01,
            "gna": 11,
            "gka": 1.2,
            "gkdr": 4.5,
            "vna": 60,
            "vk": -75,
            "vca": 65,
            "vl": -40,
            "cm": 4e-3,
            "gcal": 0.65,
            "gcat": 0.05,
            "gcan": 0.3,
            "gl": 0.1,
            "vcalm": -30,
            "scalm": 10,
            "vcalh": -33,
            "scalh": -5,
            "tcalh1": 60,
            "tcalh2": 51,
            "vcatm": -49,
            "scatm": 4,
            "vcath": -52,
            "scath": -5,
            "tcatm1": 15,
            "tcatm2": 0,
            "tcath1": 20,
            "tcath2": 5,
            "vcanm": -5,
            "scanm": 10,
            "vcanh": -33,
            "scanh": -5,
            "tcanh1": 2,
            "tcanh2": 2,
            "vnam": -30,
            "vnah": -52,
            "snam": 4,
            "snah": -8,
            "tnah1": 120,
            "tnah2": 0.5,
            "vkam": -45,
            "skam": 10,
            "vkah": -68,
            "skah": -10,
            "taukam": 0.1,
            "tkah1": 60,
            "tkah2": 5,
            "taukm": 1,
            "vkdrm": -25,
            "skdrm": 23,
            "V2m": -28.8749674,
            "sm": -5.45289598,
            "V2h": -45.388663,
            "sh": 4.99005762,
            "a1taum": 0.0001072,
            "b1taum": 12,
            "c1taum": 40,
            "a2taum": 0.000152,
            "b2taum": -20,
            "c2taum": 22.69,
            "a1tauh": 0.0001636,
            "b1tauh": -15,
            "c1tauh": 8.6856,
            "a2tauh": 0.0001857,
            "b2tauh": 5,
            "c2tauh": 35.35,
        }

        # Update parameters with any provided values
        self.params.update(kwargs)

        # Default initial conditions
        self.init = {
            "v": -52.91,
            "mcal": 0.092,
            "hcal": 0.857,
            "mcat": 0.27,
            "hcat": 0.534,
            "mcan": 0.008,
            "hcan": 0.857,
            "mna": 0.003,
            "hna": 0.339,
            "mka": 0.312,
            "hka": 0.182,
            "mkdr": 0.237,
        }

    def dynamics(self, t, y):
        """
        Calculate the dynamics of the delta cell model

        Parameters:
        -----------
        t : float
            Current time
        y : array-like
            State variables: [v, mcal, hcal, mcat, hcat, mcan, hcan, mna, hna, mka, hka, mkdr]

        Returns:
        --------
        dydt : array
            Derivatives of state variables
        """
        v, mcal, hcal, mcat, hcat, mcan, hcan, mna, hna, mka, hka, mkdr = y
        p = self.params

        # Steady states and time constants
        mcalinf = 1 / (1 + np.exp(-(v - p["vcalm"]) / p["scalm"]))
        hcalinf = 1 / (1 + np.exp(-(v - p["vcalh"]) / p["scalh"]))
        taucalm = (1 / (np.exp(-(v + 23) / 20) + np.exp((v + 23) / 20))) + 0.05
        taucalh = (p["tcalh1"] / (np.exp(-v / 20) + np.exp(v / 20))) + p["tcalh2"]

        mcatinf = 1 / (1 + np.exp(-(v - p["vcatm"]) / p["scatm"]))
        hcatinf = 1 / (1 + np.exp(-(v - p["vcath"]) / p["scath"]))
        taucatm = (p["tcatm1"] / (np.exp(-(v + 50) / 12) + np.exp((v + 50) / 12))) + p[
            "tcatm2"
        ]
        taucath = (p["tcath1"] / (np.exp(-(v + 50) / 15) + np.exp((v + 50) / 15))) + p[
            "tcath2"
        ]

        mcaninf = 1 / (1 + np.exp(-(v - p["vcanm"]) / p["scanm"]))
        hcaninf = 1 / (1 + np.exp(-(v - p["vcanh"]) / p["scanh"]))
        taucanm = (1 / (np.exp(-(v + 23) / 20) + np.exp((v + 23) / 20))) + 0.05
        taucanh = (p["tcanh1"] / (np.exp(-v / 20) + np.exp(v / 20))) + p["tcanh2"]

        mnainf = 1 / (1 + np.exp((v - p["V2m"]) / p["sm"]))
        hnainf = 1 / (1 + np.exp((v - p["V2h"]) / p["sh"]))
        taunam = 1e3 * (
            p["a1taum"] * np.exp(-(((v - p["b1taum"]) / p["c1taum"]) ** 2))
            + p["a2taum"] * np.exp(-(((v - p["b2taum"]) / p["c2taum"]) ** 2))
        )
        taunah = 1e3 * (
            p["a1tauh"] * np.exp(-(((v - p["b1tauh"]) / p["c1tauh"]) ** 2))
            + p["a2tauh"] * np.exp(-(((v - p["b2tauh"]) / p["c2tauh"]) ** 2))
        )

        mkainf = 1 / (1 + np.exp(-(v - p["vkam"]) / p["skam"]))
        hkainf = 1 / (1 + np.exp(-(v - p["vkah"]) / p["skah"]))
        taukah = (p["tkah1"] / (np.exp(-(v - 5) / 20) + np.exp((v - 5) / 20))) + p[
            "tkah2"
        ]

        mkdrinf = 1 / (1 + np.exp(-(v - p["vkdrm"]) / p["skdrm"]))
        taukdrm = p["taukm"] * (
            (1.5 / (np.exp(-(v + 10) / 25) + np.exp((v + 10) / 25))) + 15
        )

        # Ionic currents
        Ical = p["gcal"] * mcal**2 * hcal * (v - p["vca"])
        Icat = p["gcat"] * mcat**3 * hcat * (v - p["vca"])
        Ican = p["gcan"] * mcan * hcan * (v - p["vca"])
        Ina = p["gna"] * mna**5 * hna * (v - p["vna"])
        Ika = p["gka"] * mka * hka * (v - p["vk"])
        Ikdr = p["gkdr"] * mkdr**4 * (v - p["vk"])
        Ikatp = p["gkatp"] * (v - p["vk"])
        Il = p["gl"] * (v - p["vl"])

        # Differential equations
        dydt = np.zeros(12)
        dydt[0] = -(Ical + Icat + Ican + Ina + Ikdr + Ikatp + Ika + Il) / p["cm"]  # dv
        dydt[1] = (mcalinf - mcal) / (1 * taucalm)  # dmcal
        dydt[2] = (hcalinf - hcal) / (1 * taucalh)  # dhcal
        dydt[3] = (mcatinf - mcat) / (1 * taucatm)  # dmcat
        dydt[4] = (hcatinf - hcat) / (1 * taucath)  # dhcat
        dydt[5] = (mcaninf - mcan) / (1 * taucanm)  # dmcan
        dydt[6] = (hcaninf - hcan) / (1 * taucanh)  # dhcan
        dydt[7] = (mnainf - mna) / taunam  # dmna
        dydt[8] = (hnainf - hna) / taunah  # dhna
        dydt[9] = (mkainf - mka) / p["taukam"]  # dmka
        dydt[10] = (hkainf - hka) / taukah  # dhka
        dydt[11] = (mkdrinf - mkdr) / taukdrm  # dmkdr

        return dydt

    def simulate(self, tmax=2000, max_step=5.0):
        """
        Simulate the delta cell model

        Parameters:
        -----------
        tmax : float
            Maximum simulation time in ms
        max_step : float
            Maximum step size for the solver

        Returns:
        --------
        sol : OdeSolution
            Solution object from solve_ivp
        """
        # Initial conditions vector
        y0 = [
            self.init[k]
            for k in [
                "v",
                "mcal",
                "hcal",
                "mcat",
                "hcat",
                "mcan",
                "hcan",
                "mna",
                "hna",
                "mka",
                "hka",
                "mkdr",
            ]
        ]

        tspan = (0, tmax)

        # Solve ODE
        sol = solve_ivp(
            self.dynamics,
            tspan,
            y0,
            method="LSODA",
            max_step=max_step,
            rtol=1e-8,
            atol=1e-8,
        )

        return sol

    def plot_results(self, sol, figure_size=(12, 8)):
        """
        Plot the simulation results

        Parameters:
        -----------
        sol : OdeSolution
            Solution from simulate method
        """
        plt.figure(figsize=figure_size)

        # Plot voltage
        plt.plot(sol.t, sol.y[0], color="black", linewidth=2)
        plt.ylabel("V (mV)")
        plt.title("Delta Cell Model")

        plt.tight_layout()
        plt.show()


def main():
    cell = SingleDeltaCell()
    sol = cell.simulate(tmax=1000)
    _ = cell.plot_results(sol, figure_size=(10, 4))


if __name__ == "__main__":
    # Run the main function
    main()
