import pickle

import matplotlib.pyplot as plt

from calculations import Derivation
from simulation import Simulation
from utils import *


class Optimization:
    def __init__(self, simulation: Simulation, derivation: Derivation):
        self.N = simulation.N
        self.simulation = simulation
        self.non_optimized_unitary_blocks = simulation.layer_unitaries[:int(len(simulation.layer_unitaries) / simulation.r)]  # to be optimized unitaries fom every layer as a block of odd even odd unitaries
        self.unoptimized_layers = self.get_unoptimized_layers()
        self.deriv = derivation

    def get_unoptimized_layers(self):
        # TODO: Change the names
        unoptimized_layers = [np.eye(2**self.N)]

        non_optimized_layers = self.simulation.layers[:int(len(self.simulation.layer_unitaries) / self.simulation.r)]
        for i in range(len(non_optimized_layers) - 1, -1, -1):
            odd_layer_1, even_layer, odd_layer_2 = non_optimized_layers[i]
            if i != len(non_optimized_layers) - 1:
                previous = unoptimized_layers[len(unoptimized_layers) - 1]
            else:
                previous = np.eye(2 ** self.N)

            d1 = odd_layer_2 @ previous
            d2 = even_layer @ d1
            d3 = odd_layer_1 @ d2

            unoptimized_layers.append(d1)
            unoptimized_layers.append(d2)
            if i != 0:
                unoptimized_layers.append(d3)
            else:
                pass

        return list(reversed(unoptimized_layers))

    def plot_optimization(self, diff_norm_arr, grad_norm_arr, title):
        plt.plot(diff_norm_arr, label="Diff. norm")
        plt.legend()
        plt.title(title)
        plt.show()

        plt.semilogy(grad_norm_arr, label="Grad norm")
        plt.legend()
        plt.title(title)
        plt.show()

    def optimize_even_layer(self, time_step, U, left_circuit, right_circuit):
        f = lambda v: np.real(np.trace((-2 * self.simulation.Href.conj().T +
                                        (left_circuit @ self.reshape_even_layer(v) @ right_circuit).conj().T) @
                                       left_circuit @ self.reshape_even_layer(v) @ right_circuit))

        diff_norm = lambda v: np.linalg.norm(
            left_circuit @ self.reshape_even_layer(v) @ right_circuit - self.simulation.Href)

        start_layer_func = lambda Uopt: [np.eye(2)] + [Uopt] * 2 + [np.eye(2)] if not self.simulation.periodic else [
                                                                                                                        Uopt] * 3
        left_matrix = right_circuit @ (-2 * self.simulation.Href.conj().T) @ left_circuit
        derivative_func = lambda Uopt: 4 * 2 * np.trace(
            Uopt.conj().T @ Uopt) * 2 * Uopt if not self.simulation.periodic else 3 * (
                np.trace(Uopt.conj().T @ Uopt) ** 2) * 2 * Uopt
        return self.optimize(U, diff_norm, start_layer_func, left_matrix, derivative_func, "Even optimization", False,
                             time_step, f)

    def optimize_odd_layer(self, time_step, U, left_circuit, right_circuit):
        f = lambda v: np.real(np.trace((-2 * self.simulation.Href.conj().T +
                                        (left_circuit @ kron((v, v, v)) @ right_circuit).conj().T) @
                                       left_circuit @ kron((v, v, v)) @ right_circuit))
        diff_norm = lambda v: np.linalg.norm(left_circuit @
                                             kron((v, v,
                                                   v)) @ right_circuit - self.simulation.Href)
        left_matrix = right_circuit @ (-2 * self.simulation.Href.conj().T) @ left_circuit

        title = "Odd layer optimization"
        start_layer_func = lambda Uopt: [Uopt] * 3
        derivative_func = lambda Uopt: 3 * (np.trace(Uopt.conj().T @ Uopt) ** 2) * 2 * Uopt
        return self.optimize(U, diff_norm, start_layer_func, left_matrix, derivative_func, title, True, time_step)

    def get_eta(self, time_step):
        if time_step >= 0.5:
            return 1e-5
        if time_step >= 0.4:
            return 1e-5
        if time_step >= 0.3:
            return 1e-4
        if time_step >= 0.2:
            return 5e-5
        return 1e-2

    def optimize(self, U, diff_norm, start_layer_func, left_matrix, derivative_func, title, is_odd, time_step,
                 f=None):
        eta = self.get_eta(time_step)
        Uopt = U.copy()
        diff_norm_arr = []
        grad_norm_arr = []
        for k in range(5000):
            print("Diff. Norm: ", diff_norm(Uopt))
            if is_odd:
                G = self.deriv.differentiate_odd_layer(start_layer=start_layer_func(Uopt), left_matrix=left_matrix).T
            else:
                G = self.deriv.differentiate_even_layer(start_layer=start_layer_func(Uopt), left_matrix=left_matrix).T
            G += derivative_func(Uopt)
            G = G.real
            if not is_odd:
                # print(np.linalg.norm(G-eval_numerical_gradient(f, Uopt)))
                pass

            # approx_G = eval_numerical_gradient(f, Uopt, h=1e-6)
            # print(np.linalg.norm(approx_G - G))
            # assert np.allclose(approx_G, G)

            G = project_unitary_tangent(Uopt, G)
            Uopt = polar_decomp(Uopt - eta * G)[0]
            diff_norm_arr.append(diff_norm(Uopt))
            grad_norm_arr.append(np.linalg.norm(G))
            print("gradient norm:", np.linalg.norm(G))

        self.plot_optimization(diff_norm_arr, grad_norm_arr, title)
        return Uopt

    def reshape_even_layer(self, even_V):
        if self.simulation.periodic:
            matrix = kron((even_V, even_V, even_V))
            matrix = np.reshape(matrix, (2, 2 ** (self.N - 1), 2, 2 ** (self.N - 1)))
            matrix = np.transpose(matrix, (1, 0, 3, 2))
        else:
            matrix = kron((np.eye(2), even_V, even_V, np.eye(2)))
        return np.reshape(matrix, (2 ** self.N, 2 ** self.N))

    def optimize_trotterization(self, time_step: float, r: int):
        file_name = "btime_step_" + str(round(time_step, 2))
        try:
            optimized_unitary = pickle.load(open(file_name, "rb"))
        except (OSError, IOError) as e:
            optimized_unitary = np.eye(2 ** self.N)
            for idx, unitary_block in enumerate(self.non_optimized_unitary_blocks):
                U_odd_1, U_even, U_odd_2 = unitary_block
                opt_odd_1 = self.optimize_odd_layer(time_step, U_odd_1, left_circuit=optimized_unitary,
                                                    right_circuit=self.unoptimized_layers[idx * 3])
                optimized_unitary = optimized_unitary @ kron((opt_odd_1, opt_odd_1, opt_odd_1))
                opt_even = self.optimize_even_layer(time_step, U_even, left_circuit=optimized_unitary,
                                                   right_circuit=self.unoptimized_layers[idx * 3 + 1])
                optimized_unitary = optimized_unitary @ self.reshape_even_layer(opt_even)
                opt_odd_2 = self.optimize_odd_layer(time_step, U_odd_2, left_circuit=optimized_unitary,
                                                    right_circuit=self.unoptimized_layers[idx * 3 + 2])
                optimized_unitary = optimized_unitary @ kron((opt_odd_2, opt_odd_2, opt_odd_2))

            pickle.dump(optimized_unitary, open(file_name, "wb"))

        circuit = optimized_unitary ** r

        return circuit


def plot_comparison(N=6, T=0.5, order=2, periodic=False):
    trott_err_arr = []
    trott_fourth_err_arr = []
    opt_trorr_err_arr = []
    r_range = range(2, 9)
    for r in r_range:
        print("Processing r=" + str(r))
        time_step = T / r
        step_simulation = Simulation(N, time_step, 1, order, periodic)
        deriv = Derivation(step_simulation, time_step)
        optimized_trotterization = optimize_trotterization(deriv, time_step, r, N)

        trott_simulation = Simulation(N, T, r, order, periodic)
        trotterization = trott_simulation.simulate()
        exact_solution = trott_simulation.get_exact_solution()

        trott_fourth_order = Simulation(N, T, r, 4, periodic).simulate()

        trott_err_arr.append(np.linalg.norm(trotterization - exact_solution))
        opt_trorr_err_arr.append(np.linalg.norm(optimized_trotterization - exact_solution))
        trott_fourth_err_arr.append(np.linalg.norm(trott_fourth_order - exact_solution))

    plt.plot(r_range, trott_err_arr, label="Trott. error")
    plt.plot(r_range, opt_trorr_err_arr, label="Opt. trott. error")
    # plt.plot(r_range, trott_fourth_err_arr, label="4. Trott. error")

    plt.xlabel("r")
    plt.legend()
    plt.show()


def compare():
    # define parameters
    N = 6
    T = 0.1
    r = 10
    time_step = T / r
    order = 2
    periodic = False

    total_simulation = Simulation(N, 1, r, order, periodic)
    step_simulation = Simulation(N, 0.1, 1, order, periodic)

    deriv = Derivation(step_simulation, 0.1)

    trotterization = total_simulation.simulate()
    optimized_trotterization = optimize_trotterization(deriv, 0.1, r, N)
    exact_solution = total_simulation.get_exact_solution()

    print("Difference between trotterization and exact solution:", np.linalg.norm(trotterization - exact_solution))
    print("Difference between optimized trotterization and exact solution:",
          np.linalg.norm(optimized_trotterization - exact_solution))


if __name__ == '__main__':
    # plot_comparison()
    step_simulation = Simulation(6, T=0.6, r=1, order=4, periodic_boundary=False)
    derivv = Derivation(step_simulation, t=0.6)
    # step simulation must be done first in order to initalize the non optimized unitaries and layers!
    trotterization = step_simulation.simulate()

    optimization = Optimization(step_simulation, derivv)
    optimized_trotterization = optimization.optimize_trotterization(time_step=0.6, r=1)
    exact_solution = step_simulation.get_exact_solution()

    print("Difference between trotterization and exact solution:", np.linalg.norm(trotterization - exact_solution))
    print("Difference between optimized trotterization and exact solution:",
          np.linalg.norm(optimized_trotterization - exact_solution))

    '''

    exact = step_simulation.get_parity_hamiltonian(0.1, 0)
    testV = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
    testV = expm(testV * -1j * 0.1)
    test = reshape_even_layer(testV)
    print(np.linalg.norm(exact - test))
    print(exact - test)
    print(test)
    '''
