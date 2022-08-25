import numpy as np

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def kron(elements):
    kronecker_product = 1
    for element in elements:
        kronecker_product = np.kron(kronecker_product, element)
    return kronecker_product