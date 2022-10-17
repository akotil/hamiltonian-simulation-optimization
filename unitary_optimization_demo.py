import matplotlib.pyplot as plt

from calculations import Derivative
from simulation import Simulation
from utils import *
from numpy.linalg import matrix_power
from scipy.linalg import expm


class Optimization:
    def __init__(self, simulation: Simulation, derivation: Derivative):
        self.N = simulation.N
        self.sim = simulation
        self.deriv = derivation
        self.non_optimized_unitaries, self.merged_layers = self.merge_layers()
        self.unoptimized_layers = self.get_unoptimized_layers(self.merged_layers)

    def merge_layers(self):
        merged_unitaries = []
        merged_layers = []
        exponent = (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + 1 * np.kron(Z, np.eye(2)))
        exponential = lambda param: expm(exponent * -1j * param)
        no_layers = len(self.sim.params)
        for layer_idx, parameter in enumerate(self.sim.params):
            if (layer_idx - 1) % 3 == 0:
                # the layer is even
                merged_unitaries.append(exponential(parameter))
                merged_layers.append(self.sim.get_parity_hamiltonian(parameter, 0))
                if layer_idx + 2 < no_layers:
                    # this is not the last even layer, merge the following two odd layers
                    odd_1_param = self.sim.params[layer_idx + 1]
                    odd_2_param = self.sim.params[layer_idx + 2]
                    merged_unitaries.append(exponential(odd_1_param + odd_2_param))
                    merged_layers.append(self.sim.get_parity_hamiltonian(odd_1_param + odd_2_param, 1))
            else:
                # the layer is odd
                # only the first and the last layer won't get merged, so add them to the list
                if layer_idx == 0 or layer_idx == no_layers - 1:
                    merged_unitaries.append(exponential(parameter))
                    merged_layers.append(self.sim.get_parity_hamiltonian(parameter, 1))

        return merged_unitaries, merged_layers

    def get_unoptimized_layers(self, layers):
        unoptimized_layers = [np.eye(2 ** self.N)]

        for i in range(len(layers) - 1, -1, -1):
            current_layer = layers[i]
            if i != len(layers) - 1:
                previous = unoptimized_layers[len(unoptimized_layers) - 1]
            else:
                previous = np.eye(2 ** self.N)

            if i != 0:
                unoptimized_layers.append(current_layer @ previous)

        return list(reversed(unoptimized_layers))

    def plot_optimization(self, diff_norm_arr, grad_norm_arr, title):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(title)
        ax1.plot(diff_norm_arr, label="Diff. norm")
        ax1.set_title("Diff. norm")
        ax2.semilogy(grad_norm_arr, label="Grad norm")
        ax2.set_title("Grad norm")
        plt.show()

    def optimize_even_layer(self, time_step, U, left_circuit, right_circuit):
        f = lambda v: np.real(np.trace((-2 * self.sim.Href.conj().T +
                                        (left_circuit @ self.reshape_even_layer(v) @ right_circuit).conj().T) @
                                       left_circuit @ self.reshape_even_layer(v) @ right_circuit))

        diff_norm = lambda v: np.linalg.norm(
            left_circuit @ self.reshape_even_layer(v) @ right_circuit - self.sim.Href)

        start_layer_func = lambda Uopt: [np.eye(2)] + [Uopt] * 2 + [np.eye(2)] if not self.sim.periodic else [
                                                                                                                        Uopt] * 3
        left_matrix = right_circuit @ (-2 * self.sim.Href.conj().T) @ left_circuit
        deriv_func = lambda Uopt: 4 * 2 * np.trace(
            Uopt.conj().T @ Uopt) * 2 * Uopt if not self.sim.periodic else 3 * (
                np.trace(Uopt.conj().T @ Uopt) ** 2) * 2 * Uopt
        return self.optimize(U, diff_norm, start_layer_func, left_matrix, deriv_func, "Even optimization", False,
                             time_step, f)

    def optimize_odd_layer(self, time_step, U, left_circuit, right_circuit):
        f = lambda v: np.real(np.trace((-2 * self.sim.Href.conj().T +
                                        (left_circuit @ kron((v, v, v)) @ right_circuit).conj().T) @
                                       left_circuit @ kron((v, v, v)) @ right_circuit))
        diff_norm = lambda v: np.linalg.norm(left_circuit @ kron((v, v, v)) @ right_circuit - self.sim.Href)

        left_matrix = right_circuit @ (-2 * self.sim.Href.conj().T) @ left_circuit

        start_layer_func = lambda Uopt: [Uopt] * 3
        deriv_func = lambda Uopt: 3 * (np.trace(Uopt.conj().T @ Uopt) ** 2) * 2 * Uopt
        return self.optimize(U, diff_norm, start_layer_func, left_matrix, deriv_func, "Odd layer optimization", True, time_step)

    def get_optimization_params(self, time_step):
        if time_step >= 0.3:
            return 1e-6, 5000
        if time_step >= 0.2:
            return 1e-5, 2500
        if time_step >= 0.1:
            return 1e-4, 2500
        return 1e-3, 2500

    def optimize(self, U, diff_norm, start_layer_func, left_matrix, derivative_func, title, is_odd, time_step,
                 f=None):
        eta, no_iter = self.get_optimization_params(time_step)
        Uopt = U.copy()
        diff_norm_arr = []
        grad_norm_arr = []
        for k in range(no_iter):
            if is_odd:
                G = self.deriv.differentiate_odd_layer(start_layer=start_layer_func(Uopt), left_matrix=left_matrix).T
            else:
                G = self.deriv.differentiate_even_layer(start_layer=start_layer_func(Uopt), left_matrix=left_matrix).T
            G += derivative_func(Uopt)
            G = G.real

            if not is_odd:
                #print(np.allclose(eval_numerical_gradient(f, Uopt), G))
                pass

            G = project_unitary_tangent(Uopt, G)
            Uopt = Uopt - eta * G
            diff_norm_arr.append(diff_norm(Uopt))
            grad_norm_arr.append(np.linalg.norm(G))
        Uopt = polar_decomp(Uopt)[0]
        self.plot_optimization(diff_norm_arr, grad_norm_arr, title)
        return Uopt

    def reshape_even_layer(self, even_V):
        if self.sim.periodic:
            matrix = kron((even_V, even_V, even_V))
            matrix = np.reshape(matrix, (2, 2 ** (self.N - 1), 2, 2 ** (self.N - 1)))
            matrix = np.transpose(matrix, (1, 0, 3, 2))
            return np.reshape(matrix, (2 ** self.N, 2 ** self.N))
        else:
            return kron((np.eye(2), even_V, even_V, np.eye(2)))

    def unitaries_to_layers(self, unitaries):
        layers = []
        for idx, U in enumerate(unitaries):
            if idx % 2 == 0:
                layers.append(kron((U, U, U)))
            else:
                layers.append(self.reshape_even_layer(U))
        return layers

    def optimize_trotterization(self, time_step: float, r: int, recycle: bool):
        file_name = "bins/time_step_{}_r_1_k_{}".format(round(time_step, 4), self.sim.order)
        if self.sim.periodic:
            file_name += "_periodic"
        try:
            if recycle:
                to_be_optimized_unitaries = pickle.load(open(file_name, "rb"))
                unoptimized_layers = self.get_unoptimized_layers(self.unitaries_to_layers(to_be_optimized_unitaries))
            else:
                return build_whole_circuit(file_name, r)
        except (OSError, IOError) as e:
            to_be_optimized_unitaries = self.non_optimized_unitaries
            unoptimized_layers = self.unoptimized_layers
        diff_arr = []
        for i in range(20):
            optimized_layers = []
            optimized_circuit = np.eye(2 ** self.N)
            unitaries = []
            for idx in range(len(to_be_optimized_unitaries)):
                print("Iteration {}: {}. layer is being processed.".format(i, idx + 1))
                unitary = to_be_optimized_unitaries[idx]
                if idx % 2 == 0:
                    # layer is odd
                    if self.sim.order == 2 and idx == 2:
                        unitary = unitaries[0]

                    U_opt = self.optimize_odd_layer(time_step, unitary, left_circuit=optimized_circuit,
                                                    right_circuit=unoptimized_layers[idx])
                    optimized_layer = kron((U_opt, U_opt, U_opt))
                    optimized_layers.append(optimized_layer)
                    optimized_circuit = optimized_circuit @ optimized_layer
                    unitaries.append(U_opt)
                else:
                    # layer is even
                    U_opt = self.optimize_even_layer(time_step, unitary, left_circuit=optimized_circuit,
                                                     right_circuit=unoptimized_layers[idx])
                    optimized_layer = self.reshape_even_layer(U_opt)
                    optimized_layers.append(optimized_layer)
                    optimized_circuit = optimized_circuit @ optimized_layer
                    unitaries.append(U_opt)

                if idx % 2 == 0 and self.sim.order == 4:
                    to_be_optimized_unitaries[-(idx + 1)] = U_opt

            to_be_optimized_unitaries = unitaries
            unoptimized_layers = self.get_unoptimized_layers(optimized_layers)
            diff = np.linalg.norm(optimized_circuit - self.sim.Href)
            print(diff)
            diff_arr.append(diff)
            if len(diff_arr) >= 2:
                if diff_arr[-1] > diff_arr[-2]:
                    print("Early stopping at iteration ", str(i + 1))
                    break

        print("optimized step diff: ", np.linalg.norm(optimized_circuit - self.sim.Href))
        pickle.dump(to_be_optimized_unitaries, open(file_name, "wb"))
        plt.plot(diff_arr)
        plt.show()

        diff_file_name = "bins/diff_arr_" + str(round(time_step, 4))
        pickle.dump(diff_arr, open(diff_file_name, "wb"))

        circuit = matrix_power(optimized_circuit, r)
        return circuit

    def get_left_symm_matrix(self, optimized_unitaries, idx, is_first_derv):
        left_matrix = np.eye(2 ** self.N)
        for U in optimized_unitaries:
            left_matrix = left_matrix @ U

        if not is_first_derv:
            for U in self.merged_layers[idx:len(self.merged_layers) - 1 - idx]:
                left_matrix = left_matrix @ U
        return left_matrix

    def get_right_symm_matrix(self, optimized_unitaries, idx, is_first_derv):
        right_matrix = np.eye(2 ** self.N)
        for U in reversed(optimized_unitaries):
            right_matrix = right_matrix @ U

        if is_first_derv:
            for U in reversed(self.merged_layers[idx + 1:len(self.merged_layers) - idx]):
                right_matrix = U @ right_matrix
        return right_matrix

    def get_opt_unitary_from_symm_unitaries(self, optimized_unitaries):
        optimized_unitary = np.eye(2 ** self.N)
        for U in optimized_unitaries:
            optimized_unitary = optimized_unitary @ U

        for U in reversed(optimized_unitaries[:-1]):
            optimized_unitary = optimized_unitary @ U
        return optimized_unitary

    def optimize_trotterization_symmetrically(self, time_step: float, r: int):
        file_name = "time_step_{}_r_1_k_{}_sym".format(round(time_step, 4), self.sim.order)
        try:
            optimized_unitary = pickle.load(open(file_name, "rb"))
        except (OSError, IOError) as e:
            optimized_unitaries = []
            for idx, unitary in enumerate(self.non_optimized_unitaries):
                if idx > (len(self.non_optimized_unitaries) - 1) / 2:
                    break

                print("{}. layer is being processed.".format(idx + 1))
                if idx == (len(self.non_optimized_unitaries) - 1) / 2:
                    # layer is even, last unoptimized layer
                    U_opt = self.optimize_even_layer(time_step, unitary,
                                                     left_circuit=self.get_left_symm_matrix(optimized_unitaries, idx,
                                                                                            True),
                                                     right_circuit=self.get_right_symm_matrix(optimized_unitaries, idx,
                                                                                              False))
                    optimized_unitaries.append(self.reshape_even_layer(U_opt))
                else:
                    left_circuit_first_derv = self.get_left_symm_matrix(optimized_unitaries, idx, True)
                    left_circuit_sec_derv = self.get_left_symm_matrix(optimized_unitaries, idx, False)

                    right_circuit_first_derv = self.get_right_symm_matrix(optimized_unitaries, idx, True)
                    right_circuit_sec_derv = self.get_right_symm_matrix(optimized_unitaries, idx, False)
                    if idx % 2 == 0:
                        # layer is odd
                        U_opt_1 = self.optimize_odd_layer(time_step, unitary, left_circuit=left_circuit_first_derv,
                                                          right_circuit=right_circuit_first_derv)
                        U_opt_2 = self.optimize_odd_layer(time_step, unitary,
                                                          left_circuit=left_circuit_sec_derv,
                                                          right_circuit=right_circuit_sec_derv)
                        U_opt = U_opt_1 + U_opt_2
                        optimized_unitaries.append(kron((U_opt_1, U_opt_1, U_opt_1)))
                    else:
                        # layer is even
                        U_opt_1 = self.optimize_even_layer(time_step, unitary,
                                                           left_circuit=left_circuit_first_derv,
                                                           right_circuit=right_circuit_first_derv)
                        U_opt_2 = self.optimize_even_layer(time_step, unitary,
                                                           left_circuit=left_circuit_sec_derv,
                                                           right_circuit=right_circuit_sec_derv)
                        U_opt = U_opt_1 + U_opt_2
                        optimized_unitaries.append(self.reshape_even_layer(U_opt_1))

            optimized_unitary = self.get_opt_unitary_from_symm_unitaries(optimized_unitaries)
            pickle.dump(optimized_unitary, open(file_name, "wb"))

        circuit = matrix_power(optimized_unitary, r)
        return circuit



if __name__ == '__main__':
    # R = [2 ** x for x in range(9, 2, -1)]
    # delta_t = [T / r for r in R]
    T = 0.01
    R = 1
    t = T / R
    order = 2
    periodic = True

    step_simulation = Simulation(6, T=t, r=1, order=order, periodic_boundary=periodic)
    print("Nonoptimized step diff ", np.linalg.norm(step_simulation.trotterization - step_simulation.Href))

    derivv = Derivative(step_simulation)

    total_simulation = Simulation(6, T=T, r=R, order=order, periodic_boundary=periodic)
    trotterization = total_simulation.trotterization
    exact_solution = total_simulation.Href
    # total_simulation.plot_eigenvalues()

    optimization = Optimization(step_simulation, derivv)
    optimized_trotterization = optimization.optimize_trotterization(time_step=t, r=R, recycle=True)

    print("Difference between trotterization and exact solution: ", np.linalg.norm(trotterization - exact_solution))
    print("Difference between opt. trott. and exact solution: ",
          np.linalg.norm(optimized_trotterization - exact_solution))

    # plot_error(N=6, T=1)
    # plot_diff_arr_from_pickle("diff_arr_0.125")
    # compare_r(T=1, R=[2 ** x for x in range(7, 3, -1)])
