import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from calculations import Derivation
from simulation import Simulation
from utils import *
from gradient_check import eval_numerical_gradient


def crandn(size):
    """
    Draw samples from the random complex standard normal distribution.
    """
    return (np.random.normal(size=size) + 1j * np.random.normal(size=size)) / np.sqrt(2)


def random_unitary(n):
    """
    Construct a Haar random unitary matrix.
    """
    # TODO: Replace
    return np.linalg.qr(crandn((n, n)))[0]


def polar_decomp(A):
    """
    Perform a polar decomposition of a matrix: ``A = U P``,
    with `U` unitary and `P` positive semidefinite.
    """
    u, s, vh = np.linalg.svd(A)
    return u @ vh, (vh.conj().T * s) @ vh


def antisymm(W):
    """
    Antisymmetrize a matrix by projecting it onto the antisymmetric (skew-symmetric) subspace.
    """
    return 0.5 * (W - W.conj().T)


def project_unitary_tangent(U, Z):
    """
    Project `Z` onto the tangent plane at the unitary matrix `U`.
    """
    return U @ antisymm(U.conj().T @ Z)


# Alternative parametrization of a unitary matrix via matrix exponential
def real_to_hermitian(X):
    """
    Convert a real square matrix to a complex Hermitian matrix.
    """
    return (np.tril(X) + np.tril(X, -1).T) + 1j * (np.triu(X, 1) - np.triu(X, 1).T)


def plot_optimization(diff_norm_arr, grad_norm_arr, title):
    plt.plot(diff_norm_arr, label="Diff. norm")
    plt.legend()
    plt.title(title)
    plt.show()

    plt.semilogy(grad_norm_arr, label="Grad norm")
    plt.legend()
    plt.title(title)
    plt.show()


def optimize_even_layer(U, deriv: Derivation):
    f = lambda v: np.real(np.trace((-2 * deriv.simulation.get_exact_solution().conj().T +
                                    (deriv.odd @ kron(
                                        (np.eye(2), v, v, np.eye(2))) @ deriv.odd).conj().T) @
                                   deriv.odd @ kron((np.eye(2), v, v, np.eye(2))) @ deriv.odd))

    diff_norm = lambda v: np.linalg.norm(
        deriv.odd @ kron(
            (np.eye(2), v, v, np.eye(2))) @ deriv.odd - deriv.simulation.get_exact_solution())

    start_layer_func = lambda Uopt: [np.eye(2)] + [Uopt] * 2 + [np.eye(2)]
    left_matrix = deriv.odd @ (-2 * deriv.Href.conj().T) @ deriv.odd
    derivative_func = lambda Uopt: 4 * 2 * np.trace(Uopt.conj().T @ Uopt) * 2 * Uopt
    optimize(U, diff_norm, start_layer_func, left_matrix, derivative_func, "Even optimization", False)


def optimize_odd_layer(U, deriv: Derivation, is_first_layer):
    if is_first_layer:
        f = lambda v: np.real(np.trace((-2 * deriv.simulation.get_exact_solution().conj().T +
                                        (kron((v, v, v)) @ deriv.even @ deriv.odd).conj().T) @
                                       kron((v, v, v)) @ deriv.even @ deriv.odd))
        diff_norm = lambda v: np.linalg.norm(
            kron((v, v,
                  v)) @ deriv.even @ deriv.odd - deriv.simulation.get_exact_solution())
        left_matrix = deriv.even @ deriv.odd @ (-2 * deriv.Href.conj().T)
    else:
        f = lambda v: np.real(np.trace((-2 * deriv.simulation.get_exact_solution().conj().T +
                                        (deriv.odd @ deriv.even @ kron((v, v, v))).conj().T) @
                                       deriv.odd @ deriv.even @ kron((v, v, v))))
        diff_norm = lambda v: np.linalg.norm(
            deriv.odd @ deriv.even @ kron(
                (v, v, v)) - deriv.simulation.get_exact_solution())
        left_matrix = (-2 * deriv.Href.conj().T) @ deriv.odd @ deriv.even

    title = "First odd layer optimization" if is_first_layer else "Second odd layer optimization"
    start_layer_func = lambda Uopt: [Uopt] * 3
    derivative_func = lambda Uopt: 3 * (np.trace(Uopt.conj().T @ Uopt) ** 2) * 2 * Uopt
    optimize(U, diff_norm, start_layer_func, left_matrix, derivative_func, title, True)


def optimize(U, diff_norm, start_layer_func, left_matrix, derivative_func, title, is_odd):
    eta = 1e-2
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

        # approx_G = eval_numerical_gradient(f, Uopt, h=1e-6)
        # print(np.linalg.norm(approx_G - G))
        # assert np.allclose(approx_G, G)

        G = project_unitary_tangent(Uopt, G)
        Uopt = polar_decomp(Uopt - eta * G)[0]
        diff_norm_arr.append(diff_norm(Uopt))
        grad_norm_arr.append(np.linalg.norm(G))
        print("gradient norm:", np.linalg.norm(G))

    plot_optimization(diff_norm_arr, grad_norm_arr, title)


if __name__ == '__main__':
    N = 6
    T = 0.1
    simulation = Simulation(N, T, 5, 2, False)
    deriv = Derivation(simulation, T)
    single_V = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z) + np.kron(Z, np.eye(2))
    single_V = expm(single_V * -1j * T / 2)
    optimize_odd_layer(single_V, deriv, False)
