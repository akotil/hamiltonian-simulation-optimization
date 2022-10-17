import pickle

import numpy as np
from numpy.linalg import matrix_power

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def kron(elements):
    kron_prod = 1
    for idx, elem in enumerate(elements):
        if idx == 0:
            kron_prod = elem
        else:
            shape = kron_prod.shape[0] * elem.shape[0]
            kron_prod = np.einsum('ik,jl', kron_prod, elem).reshape(shape, shape)
    return kron_prod


def build_whole_circuit(file_name, r, order=2):
    optimized_unitaries = pickle.load(open(file_name, "rb"))
    circuit = np.eye(2 ** 6) # TODO: Replace 6 with N
    for idx, U in enumerate(optimized_unitaries):
        if idx % 2 == 0:
            layer = kron((U, U, U))
        else:
            layer = kron((np.eye(2), U, U, np.eye(2)))
        circuit = circuit @ layer

    return matrix_power(circuit, r)

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

def eval_numerical_gradient(f, x, h=1e-5):
    """
    Approximate the numeric gradient of a function via
    the difference quotient (f(x + h) - f(x - h)) / (2 h).
    """
    grad = np.zeros_like(x)

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        xi_ref = x[i]
        x[i] = xi_ref + h
        fpos = f(x)         # evaluate f(x + h)
        x[i] = xi_ref - h
        fneg = f(x)         # evaluate f(x - h)
        x[i] = xi_ref       # restore
        # compute the partial derivative via centered difference quotient
        grad[i] = (fpos - fneg) / (2 * h)
        it.iternext() # step to next dimension

    return grad