import numpy as np


def rel_error(x, y):
    """
    Compute the relative error between `x` and `y`.
    """
    return np.max(np.abs(x - y) / (np.maximum(np.abs(x) + np.abs(y), 1e-8)))


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
