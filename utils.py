import numpy as np

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def kron(elements):
    kronecker_product = 1
    for element in elements:
        kronecker_product = np.kron(kronecker_product, element)
    return kronecker_product


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
