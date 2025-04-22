import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

from single_beta_cell import SingleBetaCell


class BetaCellNetwork:
    def __init__(
        self, num_cells=10, mean_gj=40, std_gj=1, min_connections=1, max_connections=5
    ):
        """
        Initialize a network of coupled beta cells

        Parameters:
        -----------
        num_cells : int
            Number of beta cells in the network
        mean_gj : float
            Mean gap junction conductance in pS
        std_gj : float
            Standard deviation of gap junction conductance in pS
        min_connections : int
            Minimum number of connections per cell
        max_connections : int
            Maximum number of connections per cell
        """
        self.num_cells = num_cells
        self.mean_gj = mean_gj  # pS
        self.std_gj = std_gj  # pS
        self.min_connections = min_connections
        self.max_connections = max_connections

        # Create individual beta cells with slightly varied parameters
        self.cells = []
        for i in range(num_cells):
            # Add slight variation to parameters
            # gs1 = np.random.normal(5, 0.2)
            # gs2 = np.random.normal(32, 1.0)
            gs1 = np.random.normal(5, 0.05)
            gs2 = np.random.normal(32, 0.2)
            self.cells.append(SingleBetaCell(gs1=gs1, gs2=gs2))

        # Create connectivity matrix (adjacency matrix with weights)
        self.create_network()

    def create_network(self):
        """Create network connectivity with gap junctions"""
        # Initialize adjacency matrix with zeros
        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells))
        self.gj_matrix = np.zeros((self.num_cells, self.num_cells))

        # Create a graph for visualization
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_cells))

        # Assign 3D positions for visualization
        self.positions = {}
        for i in range(self.num_cells):
            # Randomly position cells in 3D space
            self.positions[i] = (np.random.rand(), np.random.rand(), np.random.rand())

        # For each cell, determine connections
        for i in range(self.num_cells):
            # Determine number of connections for this cell
            num_connections = np.random.randint(
                self.min_connections, self.max_connections + 1
            )

            # Find cells that are not yet connected to cell i
            possible_connections = [
                j
                for j in range(self.num_cells)
                if j != i and self.adjacency_matrix[i, j] == 0
            ]

            # If we have fewer possible connections than needed, use all available
            num_connections = min(num_connections, len(possible_connections))

            # Choose random cells to connect with
            if possible_connections:
                # Calculate distances between cell i and all other cells
                distances = {}
                for j in possible_connections:
                    pos_i = self.positions[i]
                    pos_j = self.positions[j]
                    dist = np.sqrt(
                        (pos_i[0] - pos_j[0]) ** 2
                        + (pos_i[1] - pos_j[1]) ** 2
                        + (pos_i[2] - pos_j[2]) ** 2
                    )
                    distances[j] = dist

                # Sort by distance and prefer closer cells (with some randomness)
                sorted_cells = sorted(distances.items(), key=lambda x: x[1])
                preferred_cells = [
                    cell for cell, _ in sorted_cells[: num_connections * 2]
                ]

                if len(preferred_cells) > num_connections:
                    connected_cells = np.random.choice(
                        preferred_cells, num_connections, replace=False
                    )
                else:
                    connected_cells = preferred_cells

                # Create connections with random gap junction conductances
                for j in connected_cells:
                    # Generate a random conductance from normal distribution
                    conductance = np.random.normal(self.mean_gj, self.std_gj)
                    conductance = max(1, conductance)  # Ensure positive conductance

                    # Set the adjacency matrix (symmetric)
                    self.adjacency_matrix[i, j] = conductance
                    self.adjacency_matrix[j, i] = conductance

                    gj_value = max(0, np.random.normal(self.mean_gj, self.std_gj))
                    self.adjacency_matrix[i, j] = self.adjacency_matrix[j, i] = 1
                    self.gj_matrix[i, j] = self.gj_matrix[j, i] = gj_value

                    # make the size of the edge proportional to gj_value
                    # self.graph.edges[i, j]['weight'] = gj_value

                    # Add edge to graph
                    self.graph.add_edge(i, j, weight=conductance)

    def network_dynamics(self, t, X):

        """
        Coupled dynamics of the beta cell network
        Parameters:
        -----------
        t : float
            Current time
        X : array
            State variables for all cells: [v1, n1, s11, s21, v2, n2, s12, s22, ...]
        Returns:
        --------
        dXdt : array
            Derivatives of state variables
        """
        dXdt = np.zeros_like(X)
        for i, cell in enumerate(self.cells):
            idx = 4 * i
            vi = X[idx]
            vi_neighbors = 0
            g_total = 0
            for j in range(self.num_cells):
                if i != j and self.adjacency_matrix[i, j]:
                    vj = X[4 * j]
                    gj = self.gj_matrix[i, j]
                    vi_neighbors += gj * vj
                    g_total += gj
            v_neighbors = vi_neighbors / g_total if g_total > 0 else vi
            dXdt[idx : idx + 4] = cell.dynamics(
                t, X[idx : idx + 4], v_neighbors, g_total
            )
        return dXdt

    def dynamics(self, t, x):
        """
        Coupled dynamics of all cells in the network

        Parameters:
        -----------
        t : float
            Current time
        x : array
            State variables for all cells: [v1, n1, s11, s21, v2, n2, s12, s22, ...]

        Returns:
        --------
        dxdt : array
            Derivatives of state variables
        """
        # Reshape to get state for each cell
        x_reshaped = x.reshape(self.num_cells, 4)

        # Initialize the derivative array
        dxdt = np.zeros_like(x_reshaped)

        # Calculate intrinsic dynamics for each cell
        for i in range(self.num_cells):
            dxdt[i] = self.cells[i].dynamics(t, x_reshaped[i])

        # Add gap junction currents to voltage derivatives
        for i in range(self.num_cells):
            v_i = x_reshaped[i, 0]  # Voltage of cell i

            # Sum up gap junction currents
            gj_current = 0
            for j in range(self.num_cells):
                if self.adjacency_matrix[i, j] > 0:
                    v_j = x_reshaped[j, 0]  # Voltage of cell j
                    gj = self.adjacency_matrix[
                        i, j
                    ]  # * 1e-12  # Convert pS to S # TODO
                    gj_current += gj * (v_j - v_i)

            # Add gap junction current to voltage derivative (I = g*dV)
            dxdt[i, 0] += gj_current / self.cells[i].cm

        # Return flattened array
        return dxdt.flatten()

    def simulate(self, tmax=90000, max_step=10.0):
        """
        Simulate the network of coupled beta cells

        Parameters:
        -----------
        tmax : float
            Maximum simulation time in ms
        max_step : float
            Maximum step size for solver

        Returns:
        --------
        sol : OdeSolution
            Solution object from solve_ivp
        """

        x0 = np.array([-50, 0, 0, 0.6] * self.num_cells)
        tspan = (0, tmax)

        # Solve the system
        sol = solve_ivp(
            self.network_dynamics, tspan, x0, method="RK45", max_step=max_step
        )

        return sol

    def plot_network(self):
        """Visualize the network connectivity"""
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection="3d")

        # Extract 3D positions
        pos = self.positions
        xs = [pos[i][0] for i in range(self.num_cells)]
        ys = [pos[i][1] for i in range(self.num_cells)]
        zs = [pos[i][2] for i in range(self.num_cells)]

        # Plot nodes
        ax.scatter(xs, ys, zs, s=100, c="blue", alpha=0.7)

        # Plot edges
        for i, j in self.graph.edges():
            weight = self.adjacency_matrix[i, j]
            x = [pos[i][0], pos[j][0]]
            y = [pos[i][1], pos[j][1]]
            z = [pos[i][2], pos[j][2]]
            # Line width proportional to weight
            lw = 0.5 + 2.0 * weight / self.mean_gj
            ax.plot(x, y, z, "gray", alpha=0.5, linewidth=lw)

        # Add node labels
        for i in range(self.num_cells):
            ax.text(pos[i][0], pos[i][1], pos[i][2], str(i), fontsize=12)

        ax.set_title(f"Beta Cell Network (n={self.num_cells})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

    def plot_results(self, sol):
        """
        Plot simulation results

        Parameters:
        -----------
        sol : OdeSolution
            Solution from simulate method
        """
        # Convert time from ms to seconds
        tsec = sol.t / 1000

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))

        # Plot voltages
        ax1 = fig.add_subplot(3, 1, 1)
        for i in range(self.num_cells):
            v = sol.y[i * 4]
            ax1.plot(
                tsec, v, linewidth=1, alpha=0.7, label=f"Cell {i}" if i < 5 else ""
            )

        ax1.set_xlim(0, tsec[-1])
        ax1.set_ylim(-70, -10)
        ax1.set_ylabel("V (mV)")
        ax1.set_title(f"Membrane Potentials of {self.num_cells} Coupled Beta Cells")
        if self.num_cells <= 5:
            ax1.legend(loc="upper right")
        else:
            ax1.legend(loc="upper right", ncol=2)

        # Plot s1 variables
        ax2 = fig.add_subplot(3, 1, 2)
        for i in range(self.num_cells):
            s1 = sol.y[i * 4 + 2]
            ax2.plot(tsec, s1, linewidth=1, alpha=0.7)

        ax2.set_xlim(0, tsec[-1])
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("s₁")
        ax2.set_title("Slow Variable s₁")

        # Plot s2 variables
        ax3 = fig.add_subplot(3, 1, 3)
        for i in range(self.num_cells):
            s2 = sol.y[i * 4 + 3]
            ax3.plot(tsec, s2, linewidth=1, alpha=0.7)

        ax3.set_xlim(0, tsec[-1])
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("t (s)")
        ax3.set_ylabel("s₂")
        ax3.set_title("Slow Variable s₂")

        plt.tight_layout()
        plt.show()

        # Plot a phase space of s1 vs V for the first few cells
        plt.figure(figsize=(12, 8))
        for i in range(min(5, self.num_cells)):
            v = sol.y[i * 4]
            s1 = sol.y[i * 4 + 2]
            plt.plot(s1, v, linewidth=1, alpha=0.7, label=f"Cell {i}")

        plt.xlim(0, 1)
        plt.ylim(-70, -10)
        plt.xlabel("s₁")
        plt.ylabel("V (mV)")
        plt.title("Phase Space (s₁ vs V)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    # Create a network of 10 beta cells w/ gap junctions
    network = BetaCellNetwork(
        num_cells=10, mean_gj=40, std_gj=8, min_connections=1, max_connections=5
    )

    # No gap junctions, de-synchronized
    # network = BetaCellNetwork(
    #    num_cells=10, mean_gj=0, std_gj=0, min_connections=1, max_connections=5
    # )

    # Visualize the network
    network.plot_network()

    # Simulate network dynamics
    sol = network.simulate(tmax=90000)

    # Plot results
    network.plot_results(sol)


if __name__ == "__main__":
    main()
