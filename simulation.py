import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power, norm
from scipy.linalg import expm

from utils import kron

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


class Simulation:
    def __init__(self, N: int, T: float, r: int, order: int, periodic_boundary: bool):
        self.N = N
        self.T = T
        self.h_coeff = np.random.uniform(0, 1, N - 1)
        self.r = r
        self.order = order
        self.periodic = periodic_boundary
        self.layer_unitaries = []
        self.layers = []
        self.Href = self.get_exact_solution()
        self.params = []
        self.trotterization = self.get_segmented_trotterization()

    def get_single_hamiltonian(self, k: int, t: float) -> np.ndarray:
        exponent = lambda h: (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + h * np.kron(Z, np.eye(2)))
        exponential = expm(exponent(self.h_coeff[k]) * -1j * t)
        return exponential

    def get_hamiltonian_layer(self, parity: int, t: float):
        """
        Args:
        :parity: 0 if the layer should represent even Hamiltonians, 1 otherwise.

        :return: A list of ndarrays corresponding to even/odd Hamiltonians belonging to the same circuit layer.
        """
        V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
        V = expm(V * -1j * t)
        layer = [V] * (math.ceil(self.N / 2) - 1) if parity == 0 else [V] * (math.floor(self.N / 2))

        # fill with identity if necessary
        if parity == 1 and self.N % 2 == 1:
            layer.append(np.eye(2))

        elif parity == 0:
            if self.N % 2 == 1:
                layer = [np.eye(2)] + layer
            elif self.periodic and self.N % 2 == 0:
                layer = [V] + layer
            elif not self.periodic and self.N % 2 == 0:
                layer = [np.eye(2)] + layer + [np.eye(2)]

        return layer

    def get_block_hamiltonian(self, k: int, t: float) -> np.ndarray:
        exponent = lambda h: (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + h * np.kron(Z, np.eye(2)))
        exponential = expm(exponent(1) * -1j * t)
        identity_1 = np.eye(2 ** k) if k > 0 else 1
        identity_2 = np.eye(2 ** (self.N - k - 2)) if k < self.N - 2 else 1

        hamiltonian = kron((identity_1, exponential, identity_2))
        assert hamiltonian.shape == (2 ** self.N, 2 ** self.N), \
            "The single Hamiltonian has wrong shape: {} with k : {}".format(hamiltonian.shape, k)

        return hamiltonian

    def get_parity_hamiltonian(self, t: float, parity: int) -> np.ndarray:
        '''
        Returns the matrix corresponding to one layer of the given parity.

        :param t: time in seconds
        :param parity: 0 if the layer is even, 1 otherwise
        :return: (2**self.N, 2**self.N) - sized layer unitary corresponding to the given parity
        '''

        hamiltonian = np.eye(2 ** self.N)
        if parity == 0:
            # calculate the Hamiltonian for even terms
            if self.periodic and self.N % 2 == 0:
                for k in range(1, math.floor(self.N / 2) + 1):
                    hamiltonian = hamiltonian @ self.get_block_hamiltonian(2 * k - 2, t)
                hamiltonian = np.reshape(hamiltonian, (2, 2 ** (self.N - 1), 2, 2 ** (self.N - 1)))
                hamiltonian = np.transpose(hamiltonian, (1, 0, 3, 2))
                hamiltonian = np.reshape(hamiltonian, (2 ** self.N, 2 ** self.N))

            else:
                for k in range(1, math.ceil(self.N / 2)):
                    hamiltonian = hamiltonian @ self.get_block_hamiltonian(2 * k - 1, t)

        else:
            # calculate the Hamiltonian for odd terms
            for k in range(1, math.floor(self.N / 2) + 1):
                hamiltonian = hamiltonian @ self.get_block_hamiltonian(2 * k - 2, t)

        return hamiltonian

    def _get_second_order_trotterization(self, t: float, power) -> np.ndarray:
        odd_H = self.get_parity_hamiltonian(t / 2, 1)
        even_H = self.get_parity_hamiltonian(t, 0)
        exponent = (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + 1 * np.kron(Z, np.eye(2)))
        exponential = lambda param: expm(exponent * -1j * param)
        # TODO: This way, in order to use the optimization, we need to start the simulation first. Change it maybe?
        self.layers.extend([(odd_H, even_H, odd_H)] * power)
        self.layer_unitaries.extend([(exponential(t / 2), exponential(t), exponential(t / 2))] * power)
        self.params.extend([t / 2, t, t / 2] * power)
        return odd_H @ even_H @ odd_H

    def get_kth_order_trotterization(self, t: float, order: int, power: int):
        if order == 2:
            return self._get_second_order_trotterization(t, power)
        else:
            k = order / 2
            p_k = self.get_pk(k)
            t_1 = matrix_power(self.get_kth_order_trotterization(p_k * t, 2 * k - 2, 2 * power), 2)
            t_2 = self.get_kth_order_trotterization((1 - 4 * p_k) * t, 2 * k - 2, power)
            t_3 = matrix_power(self.get_kth_order_trotterization(p_k * t, 2 * k - 2, 2 * power), 2)
            return t_1 @ t_2 @ t_3

    def get_segmented_trotterization(self):
        return matrix_power(self.get_kth_order_trotterization(self.T / self.r, self.order, 1), self.r)

    def get_exact_solution(self) -> np.ndarray:
        ref_sol: np.ndarray
        kron_sum = lambda h: np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + h * np.kron(Z, np.eye(2))
        hamiltonian = 0
        for k in range(self.N - 1):
            identity_1 = np.eye(2 ** k) if k > 0 else 1
            identity_2 = np.eye(2 ** (self.N - k - 2)) if k < self.N - 2 else 1
            term = np.kron(np.kron(identity_1, kron_sum(1)), identity_2)
            hamiltonian += term

        if self.periodic:
            coupling = lambda pauli: np.kron(np.kron(pauli, np.eye(2 ** (self.N - 2))), pauli)
            hamiltonian += coupling(X) + coupling(Y) + coupling(Z) + 1 * np.kron(Z, np.eye(2 ** (self.N - 1)))

        ref_sol = expm(-1j * hamiltonian * self.T)
        return ref_sol

    def get_pk(self, k: int):
        return 1 / (4 - 4 ** (1 / (2 * k - 1)))

    def plot_eigenvalues(self):
        eigv = np.linalg.eigvals(self.get_exact_solution())
        plt.plot(range(len(eigv)), eigv)
        plt.show()


def get_k(N: int, t: float, epsilon: float, k: int):
    r_2k = t * ((N * t / epsilon) ** (1 / (2 * k)))
    delta = t / r_2k

    if not 1 / (2 * k) <= delta:
        return get_k(N, t, epsilon, k + 1)
    else:
        return r_2k, k


def plot_r_graph(epsilon: float):
    N = list(range(10, 110, 10))
    for n in N:
        t = n
        r_2k, _ = get_k(n, t, epsilon, 2)
        plt.scatter(n, r_2k)

    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def plot_error(N: int, T: float):
    K = [1, 2, 3, 4]
    R = [2 ** x for x in range(9, 2, -1)]
    delta_t = [T / r for r in R]
    print(delta_t)
    colors = ["r", "b", "g", "m"]
    for k_idx, k in enumerate(K):
        empirical_errors = []
        theoretical_errors = []
        for t_idx, t in enumerate(delta_t):
            simulation = Simulation(N, T, R[t_idx], k * 2, False)
            trotterization = simulation.simulate()
            exact_sol = simulation.get_exact_solution()
            error = norm(trotterization - exact_sol)
            empirical_errors.append(error)
            theoretical_errors.append(N * t ** (2 * k) * T)
        plt.plot(delta_t, empirical_errors, label="k=" + str(k), color=colors[k_idx])
        plt.plot(delta_t, theoretical_errors, label="analytic k=" + str(k), linestyle="dashed", color=colors[k_idx])

    plt.xlabel(r'$\delta t$')
    plt.ylabel("Error")

    plt.title(r'$N={}, T={}$'.format(N, T))

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
