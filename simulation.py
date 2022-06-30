import math
import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power, norm
from scipy.linalg import expm

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


class Simulation:
    def __init__(self, N: int, T: float, r: int, order: int):
        self.N = N
        self.T = T
        self.h_coeff = np.random.uniform(0, 1, N - 1)
        self.r = r
        self.order = order

    def simulate(self):
        return self.get_segmented_trotterization()

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
        # TODO: Coefficients differ and we have different Vs
        V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
        V = expm(V * -1j * t)
        layer = [V] * (math.ceil(self.N / 2) - 1) if parity == 0 else [V] * (math.floor(self.N / 2))
        # fill with identity if necessary
        # TODO: Generalize to even parity also
        if parity == 1 and self.N % 2 == 1:
            layer.append(np.eye(2))

        return layer

    def get_block_hamiltonian(self, k: int, t: float) -> np.ndarray:
        exponent = lambda h: (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + h * np.kron(Z, np.eye(2)))
        # TODO: Temporarily fixed the h coefficient
        exponential = expm(exponent(1) * -1j * t)
        identity_1 = np.eye(2 ** k) if k > 0 else 1
        identity_2 = np.eye(2 ** (self.N - k - 2)) if k < self.N - 2 else 1

        hamiltonian = np.kron(np.kron(identity_1, exponential), identity_2)
        assert hamiltonian.shape == (2 ** self.N, 2 ** self.N), \
            "The single Hamiltonian has wrong shape: {} with k : {}".format(str(hamiltonian.shape), str(k))

        return hamiltonian

    def get_parity_hamiltonian(self, t: float, parity: int) -> np.ndarray:
        hamiltonian_product = np.eye(2 ** self.N)
        if parity == 0:
            # calculate the Hamiltonian for even terms
            for k in range(1, math.ceil(self.N / 2)):
                hamiltonian_product = hamiltonian_product @ self.get_block_hamiltonian(2 * k - 1, t)
        else:
            # calculate the Hamiltonian for odd terms
            for k in range(1, math.floor(self.N / 2) + 1):
                hamiltonian_product = hamiltonian_product @ self.get_block_hamiltonian(2 * k - 2, t)

        return hamiltonian_product

    def get_second_order_trotterization(self, t: float) -> np.ndarray:
        odd_H = self.get_parity_hamiltonian(t / 2, 1)
        even_H = self.get_parity_hamiltonian(t, 0)
        return odd_H @ even_H @ odd_H

    def get_kth_order_trotterization(self, t: float, order: int):
        if order == 2:
            return self.get_second_order_trotterization(t)
        else:
            k = order / 2
            p_k = self.get_pk(k)
            t_1 = matrix_power(self.get_kth_order_trotterization(p_k * t, 2 * k - 2), 2)
            t_2 = self.get_kth_order_trotterization((1 - 4 * p_k) * t, 2 * k - 2)
            t_3 = matrix_power(self.get_kth_order_trotterization(p_k * t, 2 * k - 2), 2)
            return t_1 @ t_2 @ t_3

    def get_segmented_trotterization(self):
        result = np.eye(2 ** self.N)
        for segment in range(self.r):
            print("\rSegment {}|{} is being processed.".format(str(segment + 1), str(self.r)), end="")
            result = result @ self.get_kth_order_trotterization(self.T / self.r, self.order)

        return result

    def get_exact_solution(self) -> np.ndarray:
        ref_sol: np.ndarray
        kron_sum = lambda h: np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + h * np.kron(Z, np.eye(2))
        hamiltonian = 0
        for k in range(self.N - 1):
            identity_1 = np.eye(2 ** k) if k > 0 else 1
            identity_2 = np.eye(2 ** (self.N - k - 2)) if k < self.N - 2 else 1
            # TODO: Temporarily disable h
            term = np.kron(np.kron(identity_1, kron_sum(1)), identity_2)
            hamiltonian += term

        ref_sol = expm(-1j * hamiltonian * self.T)
        return ref_sol

    def get_pk(self, k: int):
        return 1 / (4 - 4 ** (1 / (2 * k - 1)))


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
            simulation = Simulation(N, T, R[t_idx], k * 2)
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


if __name__ == "__main__":
    random.seed(22)
    N = 7
    T = 10
    epsilon = 0.001
    r_2k, k = get_k(N, T, epsilon, 2)
    print("r: ", r_2k)
    print("k: ", k)

    #plot_r_graph(epsilon)
    plot_error(3, 5)

    simulation = Simulation(N, T, math.ceil(r_2k), 2 * k)
    trotterization = simulation.simulate()
    exact_sol = simulation.get_exact_solution()

    norm = norm(exact_sol - trotterization)
    print(norm)

    print(np.trace(np.conj(trotterization).T @ exact_sol) - 2 ** N)
    print(np.allclose(trotterization, exact_sol, atol=epsilon))
