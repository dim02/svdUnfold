"""Helper functions"""
import numpy as np


def check_symmetric(matrix, rtol=1e-05, atol=1e-08):
    """Return true if matrix is symmetric"""
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def calc_second_deriv_matrix(n, xi):
    """Return n by n second derivative matrix with small diagonal xi"""
    c_matrix = np.zeros((n, n))
    for i in range(n):
        c_matrix[i, i] = -2 + xi
        if i - 1 >= 0:
            c_matrix[i, i - 1] = 1
        if i + 1 < n:
            c_matrix[i, i + 1] = 1
    c_matrix[0, 0] = -1 + xi
    c_matrix[n - 1, n - 1] = -1 + xi
    return c_matrix

def calc_inverse_second_deriv_matrix(c_matrix):
    """Return the inverse of the second derivative matrix c_matrix"""
    return np.linalg.inv(c_matrix)
