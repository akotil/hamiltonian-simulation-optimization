from typing import Tuple

import numpy as np
from scipy.linalg import expm

from simulation import Simulation

from gradient_check import eval_numerical_gradient

from utils import X, Y, Z, kron


class Derivation:
    def __init__(self, simulation: Simulation, t: float):
        self.simulation = simulation
        self.N = simulation.N
        self.t = t
        self.even = simulation.get_parity_hamiltonian(t, 0)
        self.odd = simulation.get_parity_hamiltonian(t / 2, 1)
        self.Href = simulation.get_exact_solution()

    def _get_left_tensor_shape(self, V):
        V_1, single_V, V_2 = V

        if type(V_1) == int:
            no_wires_V2 = int(np.log2(V_2.shape[0]))
            return 4, 2 ** no_wires_V2, 4, 2 ** no_wires_V2

        elif type(V_2) == int:
            no_wires_V1 = int(np.log2(V_1.shape[0]))
            return 2 ** no_wires_V1, 4, 2 ** no_wires_V1, 4

        else:
            no_wires_V1, no_wires_V2 = int(np.log2(V_1.shape[0])), int(np.log2(V_2.shape[0]))
            return 2 ** no_wires_V1, 4, 2 ** no_wires_V2, 2 ** no_wires_V1, 4, 2 ** no_wires_V2


    def check_correctness(self, partial_derivation, V_1, V_2):
        # check the correctness of the differentiation
        single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
        single_V = expm(single_V * -1j * self.t / 2)
        check_trace = np.tensordot(partial_derivation, single_V, axes=([0, 1], [1, 0]))
        ref_trace = np.trace(left_matrix @ (np.kron(np.kron(V_1, single_V), V_2)))
        assert np.allclose(ref_trace, check_trace)

    def differentiate_kth_hamiltonian(self, left_matrix, left_tensor: np.ndarray,
                                      V: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        V_1, single_V, V_2 = V

        # tensor contraction without the k-th Hamiltonian
        partial_derivation = left_tensor
        # TODO: Generalize indices
        if type(V_1) != int and type(V_2) != int:
            partial_derivation = np.tensordot(left_tensor, V_1, axes=([0, 3], [1, 0]))

        elif type(V_1) != int and type(V_2) == int:
            partial_derivation = np.tensordot(left_tensor, V_1, axes=([0, 2], [1, 0]))

        if type(V_2) != int:
            partial_derivation = np.tensordot(partial_derivation, V_2, axes=([1, 3], [1, 0]))

        #self.check_correctness(partial_derivation, V_1, V_2)
        return partial_derivation

    def get_segmented_V(self, k: int, layer: [np.ndarray], t: float, single_V: np.ndarray =
    None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        V_1, V_2 = 1, 1
        for idx_1 in range(k):
            V_1 = np.kron(V_1, layer[idx_1])
        for idx_2 in range(k + 1, len(layer)):
            V_2 = np.kron(V_2, layer[idx_2])

        if single_V is None:
            # TODO: Can be extracted
            single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
            single_V = expm(single_V * -1j * t)

        return V_1, single_V, V_2

    def differentiate_odd_layer(self, start_layer: list[np.ndarray] = None, left_matrix=None) -> np.ndarray:
        if left_matrix is None:
            left_matrix = self.even @ self.odd @ (
                        -2 * (self.simulation.get_exact_solution() - kron(start_layer) @ self.even @ self.odd))

        return self._differentiate_layer(1, left_matrix, start_layer)

    def differentiate_even_layer(self, start_layer: list[np.ndarray] = None, left_matrix=None) -> np.ndarray:
        if left_matrix is None:
            left_matrix = self.odd @ self.Href @ self.odd
        return self._differentiate_layer(0, left_matrix, start_layer)

    def _differentiate_layer(self, parity, left_matrix, start_layer: list[np.ndarray] = None) -> np.ndarray:
        derivation = np.zeros((4, 4), dtype="complex")
        single_V = None
        if start_layer is not None:
            layer = start_layer
            # TODO: Generalize
            single_V = start_layer[0]
        else:
            layer = self.simulation.get_hamiltonian_layer(parity, self.t)
        if parity == 1:
            k_range = range(len(layer)) if self.N % 2 == 0 else range(len(layer) - 1)
        else:
            if self.N % 2 == 1:
                k_range = range(1, len(layer))
            elif self.simulation.periodic:
                k_range = range(len(layer))
                left_matrix = np.reshape(left_matrix, (2, 2**(self.N-1), 2, 2**(self.N-1)))
                left_matrix = np.transpose(left_matrix, (1,0,3,2))
                left_matrix = np.reshape(left_matrix, (2**self.N, 2**self.N))
            else:
                k_range = range(1, len(layer) - 1)
        for k in k_range:
            if parity == 0:
                V = self.get_segmented_V(k, layer, self.t, single_V=single_V)
            else:
                V = self.get_segmented_V(k, layer, self.t / 2, single_V=single_V)

            left_tensor = np.reshape(left_matrix, self._get_left_tensor_shape(V))
            partial_derivation = self.differentiate_kth_hamiltonian(left_matrix, left_tensor, V)
            derivation += partial_derivation
        return derivation


if __name__ == "__main__":
    N = 6
    simulation = Simulation(N, 0.01, 5, 2, True)
    # TODO: Are the times correct?
    differentiation = Derivation(simulation, 0.01)
    gradient = differentiation.differentiate_even_layer()
    print(gradient)

    single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
    single_V = expm(single_V * -1j * 0.01)

    left_matrix = differentiation.odd @ differentiation.Href @ differentiation.odd
    left_matrix = np.reshape(left_matrix, (2, 2 ** (N - 1), 2, 2 ** (N - 1)))
    left_matrix = np.transpose(left_matrix, (1, 0, 3, 2))
    left_matrix = np.reshape(left_matrix, (2 ** N, 2 ** N))
    f = lambda v: np.trace(left_matrix @ kron((v, v, v)))
    print("---------------")
    print(eval_numerical_gradient(f, single_V, h=1e-6).T)

