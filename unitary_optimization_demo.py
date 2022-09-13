import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from calculations import Derivation
from simulation import Simulation
from utils import *
from gradient_check import eval_numerical_gradient
import pickle


class Optimization:
    def __init__(self, simulation: Simulation, derivation: Derivation):
        self.N = simulation.N
        self.simulation = simulation
        self.non_optimized_unitary_blocks = simulation.layer_unitaries[:len(simulation.layer_unitaries)/simulation.r]  # to be optimized unitaries fom every layer as a block of odd even odd unitaries
        self.non_optimized_layers = simulation.layers[:len(simulation.layer_unitaries)/simulation.r]
        self.optimized_unitary = np.eye(2 ** self.N,
                                        2 ** self.N)  # the matrix which corresponds to the already optimized layers
        self.unoptimized_unitary = self.get_unoptimized_unitary()
        self.derivation = derivation

    def get_unoptimized_unitary(self):
        unoptimized_unitary = np.eye(self.N**2, self.N**2)
        for idx, unitary_block in enumerate(self.non_optimized_layers):
            odd_layer_1, even_layer, odd_layer_2 = unitary_block
            if idx == 0:
                # if it is the first unitary block, do not include the first odd layer as it will be the first unitary that will be optimized
                unoptimized_unitary = unoptimized_unitary @ even_layer @ odd_layer_2
            else:
                unoptimized_unitary = odd_layer_1 @ even_layer @ odd_layer_2


    def plot_optimization(self, diff_norm_arr, grad_norm_arr, title):
        plt.plot(diff_norm_arr, label="Diff. norm")
        plt.legend()
        plt.title(title)
        plt.show()

        plt.semilogy(grad_norm_arr, label="Grad norm")
        plt.legend()
        plt.title(title)
        plt.show()

    def optimize_even_layer(self, time_step, U, deriv: Derivation, first_odd):
        first_odd = kron((first_odd, first_odd, first_odd))
        f = lambda v: np.real(np.trace((-2 * self.simulation.Href.conj().T +
                                        (first_odd @ self.reshape_even_layer(v) @ deriv.odd).conj().T) @
                                       first_odd @ self.reshape_even_layer(v) @ deriv.odd))

        diff_norm = lambda v: np.linalg.norm(
            first_odd @ self.reshape_even_layer(v) @ deriv.odd - self.simulation.Href)

        start_layer_func = lambda Uopt: [np.eye(2)] + [Uopt] * 2 + [np.eye(2)] if not self.simulation.periodic else [Uopt] * 3
        left_matrix = deriv.odd @ (-2 * self.simulation.Href.conj().T) @ first_odd
        derivative_func = lambda Uopt: 4 * 2 * np.trace(
            Uopt.conj().T @ Uopt) * 2 * Uopt if not self.simulation.periodic else 3 * (
                np.trace(Uopt.conj().T @ Uopt) ** 2) * 2 * Uopt
        return self.optimize(U, deriv, diff_norm, start_layer_func, left_matrix, derivative_func, "Even optimization", False,
                        time_step, f)

    def optimize_odd_layer(self, time_step, U, deriv: Derivation, left_circuit, right_circuit):
        if is_first_layer:
            f = lambda v: np.real(np.trace((-2 * self.simulation.Href.conj().T +
                                            (kron((v, v, v)) @ deriv.even @ deriv.odd).conj().T) @
                                           kron((v, v, v)) @ deriv.even @ deriv.odd))
            diff_norm = lambda v: np.linalg.norm(
                kron((v, v,
                      v)) @ deriv.even @ deriv.odd - self.simulation.Href)
            left_matrix = deriv.even @ deriv.odd @ (-2 * deriv.Href.conj().T)
        else:
            first_odd, even = previous_layers
            first_odd = kron((first_odd, first_odd, first_odd))
            even = self.reshape_even_layer(even)
            f = lambda v: np.real(np.trace((-2 * self.simulation.Href.conj().T +
                                            (first_odd @ even @ kron((v, v, v))).conj().T) @
                                           first_odd @ even @ kron((v, v, v))))
            diff_norm = lambda v: np.linalg.norm(
                first_odd @ even @ kron(
                    (v, v, v)) - self.simulation.Href)
            left_matrix = (-2 * self.simulation.Href.conj().T) @ first_odd @ even

        title = "First odd layer optimization" if is_first_layer else "Second odd layer optimization"
        start_layer_func = lambda Uopt: [Uopt] * 3
        derivative_func = lambda Uopt: 3 * (np.trace(Uopt.conj().T @ Uopt) ** 2) * 2 * Uopt
        return self.optimize(U, deriv, diff_norm, start_layer_func, left_matrix, derivative_func, title, True, time_step)

    def get_eta(self, time_step):
        if time_step >= 0.5:
            return 1e-6
        if time_step >= 0.4:
            return 1e-5
        if time_step >= 0.3:
            return 1e-4
        if time_step >= 0.2:
            return 5e-5
        return 1e-2

    def optimize(self, U, deriv, diff_norm, start_layer_func, left_matrix, derivative_func, title, is_odd, time_step,
                 f=None):
        eta = self.get_eta(time_step)
        Uopt = U.copy()
        diff_norm_arr = []
        grad_norm_arr = []
        for k in range(5000):
            print("Diff. Norm: ", diff_norm(Uopt))
            if is_odd:
                G = deriv.differentiate_odd_layer(start_layer=start_layer_func(Uopt), left_matrix=left_matrix).T
            else:
                G = deriv.differentiate_even_layer(start_layer=start_layer_func(Uopt), left_matrix=left_matrix).T
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

    def optimize_trotterization(self, deriv: Derivation, time_step: float, r: int):
        '''
        time_step = pickle.load(open("time_step", "rb"))

        circuit = np.eye(2 ** N)
        for segment in range(r):
            circuit = circuit @ time_step

        return circuit
        '''
        file_name = "btime_step_" + str(round(time_step, 2))
        try:
            step_brick = pickle.load(open(file_name, "rb"))
        except (OSError, IOError) as e:
            # TODO: When to divide time step by 2?
            # TODOS:
            # TODO: in simulation, find a way to store one single matrix from every layer
            # TODO: in simulation, find a way to mark which unitary belongs to which layer (odd or even)
            # 1. TODO: singleV is to be accessed from non optimized layers
            # 2. TODO: in a loop, we go through layers and as we optimize each layer, we update the optimized unitary
            # 3. TODO: rework trace derivative: what is now left_matrix?
            for unitary_block, layer in zip(self.non_optimized_unitary_blocks, self.non_optimized_layers):
                U_odd_1, U_even, U_odd_2 = unitary_block
                opt_odd_1 = self.optimize_odd_layer(time_step, U_odd_1, deriv, True)
                opt_even =
                opt_odd_2 =

            first_odd_V = self.optimize_odd_layer(time_step, single_V, deriv, True)
            even_V = self.optimize_even_layer(time_step, single_V, deriv, first_odd_V)
            second_odd_V = self.optimize_odd_layer(time_step, single_V, deriv, False, (first_odd_V, even_V))

            even_layer = self.reshape_even_layer(even_V)
            step_brick = kron((first_odd_V, first_odd_V, first_odd_V)) @ even_layer @ kron(
                (second_odd_V, second_odd_V, second_odd_V))
            pickle.dump(step_brick, open(file_name, "wb"))

        circuit = np.eye(2 ** self.N)
        for segment in range(r):
            circuit = circuit @ step_brick

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
    step_simulation = Simulation(6, 0.1, 1, 2, True)
    deriv = Derivation(step_simulation, 0.1)
    '''
        trotterization = step_simulation.simulate()
    optimized_trotterization = optimize_trotterization(deriv, 0.1, 1, 6)
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
