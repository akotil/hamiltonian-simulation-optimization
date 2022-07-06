from typing import Tuple

import numpy as np
from scipy.linalg import expm

from simulation import Simulation

from gradient_check import eval_numerical_gradient

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def kron(elements: tuple):
    kronecker_product = 1
    for element in elements:
        kronecker_product = np.kron(kronecker_product, element)
    return kronecker_product


class Optimization:
    def __init__(self, simulation: Simulation, t: float):
        self.simulation = simulation
        self.N = simulation.N
        self.t = t
        self.Href = simulation.get_exact_solution()
        self.even = simulation.get_parity_hamiltonian(t, 0)
        self.odd = simulation.get_parity_hamiltonian(t / 2, 1)

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

        # check the correctness of the differentiation
        single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
        single_V = expm(single_V * -1j * 0.01)
        check_trace = np.tensordot(partial_derivation, single_V, axes=([0, 1], [1, 0]))
        ref_trace = np.trace(left_matrix @ (np.kron(np.kron(V_1, single_V), V_2)))
        print("Calculated trace: ", check_trace)
        print("Reference trace: ", ref_trace)
        assert np.allclose(ref_trace, check_trace)
        return partial_derivation

    def get_segmented_V(self, k: int, layer: [np.ndarray], t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        V_1, V_2 = 1, 1
        for idx_1 in range(k):
            V_1 = np.kron(V_1, layer[idx_1])
        for idx_2 in range(k + 1, len(layer)):
            V_2 = np.kron(V_2, layer[idx_2])

        # TODO: Can be extracted
        single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
        single_V = expm(single_V * -1j * t)

        return V_1, single_V, V_2

    def differentiate_odd_layer(self) -> np.ndarray:
        left_matrix = self.even @ self.odd @ self.Href
        return self._differentiate_layer(1, left_matrix)

    def differentiate_even_layer(self) -> np.ndarray:
        left_matrix = self.odd @ self.Href @ self.odd
        return self._differentiate_layer(0, left_matrix)

    def _differentiate_layer(self, parity, left_matrix) -> np.ndarray:
        derivation = np.zeros((4,4), dtype="complex")
        layer = simulation.get_hamiltonian_layer(parity, self.t)
        if parity == 1:
            k_range = range(len(layer)) if self.N % 2 == 0 else range(len(layer) - 1)
        else:
            if self.N % 2 == 1:
                k_range = range(1, len(layer))
            elif self.simulation.periodic:
                k_range = range(len(layer))
            else:
                k_range = range(1, len(layer) - 1)
        for k in k_range:
            print(k)
            if parity == 0:
                V = self.get_segmented_V(k, layer, self.t)
            else:
                V = self.get_segmented_V(k, layer, self.t / 2)

            left_tensor = np.reshape(left_matrix, self._get_left_tensor_shape(V))
            partial_derivation = self.differentiate_kth_hamiltonian(left_matrix, left_tensor, V)
            derivation += partial_derivation
        return derivation


if __name__ == "__main__":
    N = 6
    simulation = Simulation(N, 0.01, 5, 2, False)
    # TODO: Are the times correct?
    optimization = Optimization(simulation, 0.01)
    print(optimization.differentiate_odd_layer().T)

    single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
    single_V = expm(single_V * -1j * 0.01 / 2)
    f = lambda v: np.trace(optimization.even @ optimization.odd @ optimization.Href @ kron((v, v, v)))
    print("---------------")
    print(eval_numerical_gradient(f, single_V, h=1e-6))
    # TODO: Check switched dimensions
    # Transposing helps. Why?
