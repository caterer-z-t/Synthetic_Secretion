import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx

from single_delta_cell import SingleDeltaCell  # Assuming this is the correct import path

class DeltaCellNetwork:
    def __init__(
        self,
        num_cells=10,
        mean_gj=40,
        std_gj=1,
        min_connections=1,
        max_connections=5,
        param_variation=0.02,
    ):
        """
        Initialize a network of coupled delta cells

        Parameters:
        -----------
        num_cells : int
            Number of delta cells in the network
        mean_gj : float
            Mean gap junction conductance in pS
        std_gj : float
            Standard deviation of gap junction conductance in pS
        min_connections : int
            Minimum number of connections per cell
        max_connections : int
            Maximum number of connections per cell
        param_variation : float
            Standard deviation for parameter variation as a fraction of the mean
        """
        self.num_cells = num_cells
        self.mean_gj = mean_gj  # pS
        self.std_gj = std_gj  # pS
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.param_variation = param_variation

        # Create individual delta cells with slightly varied parameters
        self.cells = []
        for i in range(num_cells):
            # Create a new cell with slightly varied parameters
            varied_params = {}

            # Choose a few parameters to vary
            base_cell = SingleDeltaCell()
            varying_params = ["gcal", "gcat", "gcan", "gna", "gka", "gkdr"]

            for param in varying_params:
                base_value = base_cell.params[param]
                varied_params[param] = np.random.normal(
                    base_value, base_value * param_variation
                )

            self.cells.append(SingleDeltaCell(**varied_params))

        # Create connectivity matrix (adjacency matrix with weights)
        self.create_network()

    def create_network(self):
        """Create network connectivity with gap junctions"""
        # Initialize adjacency matrix with zeros
        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells))

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
                self.min_connections, min(self.max_connections + 1, self.num_cells)
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

                    # Add edge to graph
                    self.graph.add_edge(i, j, weight=conductance)

    def dynamics(self, t, x):
        """
        Coupled dynamics of all cells in the network

        Parameters:
        -----------
        t : float
            Current time
        x : array
            State variables for all cells: [v1, mcal1, hcal1, ..., v2, mcal2, hcal2, ...]

        Returns:
        --------
        dxdt : array
            Derivatives of state variables
        """
        # Reshape to get state for each cell (12 variables per cell)
        x_reshaped = x.reshape(self.num_cells, 12)

        # Initialize the derivative array
        dxdt = np.zeros_like(x_reshaped)

        # Calculate intrinsic dynamics for each cell
        for i in range(self.num_cells):
            # Get the dynamics for this cell
            cell_state = x_reshaped[i, :]
            dxdt[i, :] = self.cells[i].dynamics(t, cell_state)

        # Add gap junction currents to voltage derivatives
        for i in range(self.num_cells):
            v_i = x_reshaped[i, 0]  # Voltage of cell i

            # Sum up gap junction currents
            gj_current = 0
            for j in range(self.num_cells):
                if self.adjacency_matrix[i, j] > 0:
                    v_j = x_reshaped[j, 0]  # Voltage of cell j
                    gj = self.adjacency_matrix[i, j] * 1e-12  # Convert pS to S
                    gj_current += gj * (v_j - v_i)

            # Add gap junction current to voltage derivative (I = g*dV)
            dxdt[i, 0] += gj_current / self.cells[i].params["cm"]

        # Return flattened array
        return dxdt.flatten()

    def simulate(self, tmax=2000, max_step=5.0):
        """
        Simulate the network of coupled delta cells

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
        tspan = (0, tmax)

        # Initialize states with slight variation
        x0 = np.zeros(self.num_cells * 12)

        for i in range(self.num_cells):
            # Get initial conditions for this cell
            cell_init = self.cells[i].init
            init_vars = [
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

            # Add slight variation to initial conditions
            for j, var in enumerate(init_vars):
                base_value = cell_init[var]
                # Add small random variation (1%)
                variation = base_value * 0.01 * np.random.randn()
                x0[i * 12 + j] = base_value + variation

        # Solve the system
        sol = solve_ivp(
            self.dynamics,
            tspan,
            x0,
            method="LSODA",
            max_step=max_step,
            rtol=1e-6,
            atol=1e-6,
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

        ax.set_title(f"Delta Cell Network (n={self.num_cells})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

    def plot_results(self, sol, figure_size=(10, 8)):
        """
        Plot simulation results

        Parameters:
        -----------
        sol : OdeSolution
            Solution from simulate method
        """
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=figure_size)

        # Plot voltages
        for i in range(self.num_cells):
            v = sol.y[i * 12]
            plt.plot(
                sol.t, v, linewidth=1, alpha=0.7, label=f"Cell {i}" if i < 5 else ""
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.title("Delta Cell Network Membrane Potentials")
        if self.num_cells <= 5:
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="upper right", ncol=2)


# Example usage
def main():
    # Create a delta cell network
    network = DeltaCellNetwork(num_cells=5, mean_gj=50)

    # Visualize the network
    network.plot_network()

    # Simulate
    sol = network.simulate(tmax=2000)

    # Plot results
    network.plot_results(sol, figure_size=(12, 6))


if __name__ == "__main__":
    main()
