"""Helper functions"""
import numpy as np


def check_symmetric(matrix, rtol=1e-05, atol=1e-08):
    """Return true if matrix is symmetric"""
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)
