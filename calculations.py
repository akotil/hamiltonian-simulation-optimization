from typing import Tuple

import numpy as np
from scipy.linalg import expm

from simulation import Simulation
from utils import X, Y, Z, kron, eval_numerical_gradient


class Derivative:
    def __init__(self, N, periodic: bool):
        self.periodic = periodic
        self.N = N

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

    def check_correctness(self, partial_derivation, V_1, V_2, left_matrix):
        # check the correctness of the differentiation
        single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
        single_V = expm(single_V * -1j * self.t / 2)
        check_trace = np.tensordot(partial_derivation, single_V, axes=([0, 1], [1, 0]))
        ref_trace = np.trace(left_matrix @ (np.kron(np.kron(V_1, single_V), V_2)))
        assert np.allclose(ref_trace, check_trace)

    def diff_kth_hamiltonian(self, left_tensor: np.ndarray,
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

        # self.check_correctness(partial_derivation, V_1, V_2)
        return partial_derivation

    def get_segmented_V(self, k: int, layer: [np.ndarray], single_V: np.ndarray = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:

        if k != 0:
            V_1 = kron((layer[:k]))
        else:
            V_1 = 1

        if k + 1 != len(layer):
            V_2 = kron((layer[k + 1:]))
        else:
            V_2 = 1
        return V_1, single_V, V_2

    def diff_odd_layer(self, start_layer: list[np.ndarray], left_matrix) -> np.ndarray:
        return self._diff_layer(1, left_matrix, start_layer)

    def diff_even_layer(self, start_layer: list[np.ndarray], left_matrix) -> np.ndarray:
        return self._diff_layer(0, left_matrix, start_layer)

    def _diff_layer(self, parity, left_matrix, start_layer: list[np.ndarray]) -> np.ndarray:
        derivation = np.zeros((4, 4), dtype="complex")
        layer = start_layer
        single_V = start_layer[0]  # Warning: This is due to the fact that a layer consists of identical gates

        if parity == 1:
            k_range = range(len(layer)) if self.N % 2 == 0 else range(len(layer) - 1)
        else:
            if self.N % 2 == 1:
                k_range = range(1, len(layer))
            elif self.periodic:
                k_range = range(len(layer))
                left_matrix = np.reshape(left_matrix, (2, 2 ** (self.N - 1), 2, 2 ** (self.N - 1)))
                left_matrix = np.transpose(left_matrix, (1, 0, 3, 2))
                left_matrix = np.reshape(left_matrix, (2 ** self.N, 2 ** self.N))
            else:
                k_range = range(1, len(layer) - 1)
        for k in k_range:
            V = self.get_segmented_V(k, layer, single_V=single_V)
            left_tensor = np.reshape(left_matrix, self._get_left_tensor_shape(V))
            partial_derivation = self.diff_kth_hamiltonian(left_tensor, V)
            derivation += partial_derivation
        return derivation


if __name__ == "__main__":
    test_N = 6
    t = 0.01
    test_sim = Simulation(N=6, T=t, r=1, order=2, periodic_boundary=True)
    test_derv = Derivative(test_N, test_sim.periodic)

    test_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
    test_V = expm(test_V * -1j * t)


    def reorder(layer):
        layer = np.reshape(layer, (2, 2 ** (6 - 1), 2, 2 ** (6 - 1)))
        layer = np.transpose(layer, (1, 0, 3, 2))
        layer = np.reshape(layer, (2 ** 6, 2 ** 6))
        return layer


    odd_layer = test_sim.get_parity_hamiltonian(t / 2, 1)
    f = lambda test_V: np.trace(reorder(odd_layer @ test_sim.Href @ odd_layer) @ kron((test_V, test_V, test_V)))

    exact_gradient = eval_numerical_gradient(f, test_V, h=1e-6)
    tn_gradient = test_derv.diff_even_layer([test_V] * 3, left_matrix=odd_layer @ test_sim.Href @ odd_layer)

    print(np.allclose(exact_gradient, tn_gradient)) #no transpose?
