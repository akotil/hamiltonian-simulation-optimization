from typing import Tuple

import numpy as np
from scipy.linalg import expm

from simulation import Simulation

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def get_segmented_V(k: int, layer: [np.ndarray], t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V_1, V_2 = 1, 1
    for idx_1 in range(k):
        V_1 = np.kron(V_1, layer[idx_1])
    for idx_2 in range(k + 1, len(layer)):
        V_2 = np.kron(V_2, layer[idx_2])

    # TODO: Can be extracted
    single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
    single_V = expm(single_V * -1j * t)

    return V_1, single_V, V_2


def differentiate_kth_hamiltonian(left_matrix, left_tensor: np.ndarray,
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


class Optimization:
    def __init__(self, simulation: Simulation, t: float):
        self.simulation = simulation
        self.N = simulation.N
        self.t = t
        self.Href = simulation.get_exact_solution()
        self.even = simulation.get_parity_hamiltonian(t, 0)
        self.odd = simulation.get_parity_hamiltonian(t, 1)

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

    def differentiate_odd_layer(self):
        # TODO: Specify which layer should be differentiated
        derivation = 0
        odd_layer = simulation.get_hamiltonian_layer(1, self.t)
        for k in range(len(odd_layer)):
            print(k)
            V = get_segmented_V(k, odd_layer, self.t)
            left_matrix = self.even @ self.odd @ self.Href
            left_tensor = np.reshape(left_matrix, self._get_left_tensor_shape(V))

            partial_derivation = differentiate_kth_hamiltonian(left_matrix, left_tensor, V)
            derivation += partial_derivation
        return derivation

    def differentiate_even_layer(self):
        pass


if __name__ == "__main__":
    N = 6
    # TODO: Think about periodic boundaries
    simulation = Simulation(N, 0.01, 5, 2)
    optimization = Optimization(simulation, 0.01)
    optimization.differentiate_odd_layer()
