"""
A single class module that performs data unfolding using the singular value decomposition approach
as described in https://arxiv.org/abs/hep-ph/9509307
"""

import helpers
import numpy as np


class SVDunfold:
    """
    A class to perform a singular value decomposition data unfolding
    """

    def __init__(self, x_ini, b, A, cov):
        """
        x_ini: numpy histogram with the initial Monte Carlo distribution
        used to build the response matrix
        b: numpy histogram with the measured data distribution that we want to unfold
        A: 2d numpy array with the response matrix (n_b x n_x)
        cov: 2d numpy array with the covariance matrix
        """
        self.__b_measured = b
        self.__x_ini = x_ini
        self.__response_matrix = A
        self.__covariance_matrix = cov
        n_bins_b = len(self.__b_measured[0])
        n_bins_x = len(self.__x_ini[0])
        assert(self.__response_matrix.shape[0] == n_bins_b),\
            "Wrong dimensions: bins in b != rows in response matrix"
        assert(self.__response_matrix.shape[1] == n_bins_x),\
            "Wrong dimensions: bins in x_ini != columns in response matrix"
        assert helpers.check_symmetric(self.__covariance_matrix), \
            "Covariance matrix is not symmetric"

    def unfold(self, k):
        """Perform the unfolding with regularization parameter tau=s(k)^2"""

    def get_unfolded_distribution(self):
        """Return the unfolded distribution as a 1d array"""

    def get_unfolded_cov_matrix(self):
        """Return the unfolded covariance matrix as a 2d array"""

    def get_abs_d(self):
        """Return a 1d array of the absolute value of the deconvolution coefficients d"""

    def get_singular_values(self):
        """Return an array of the singular values of the rescaled and rotated problem"""

    def __transform_system(self):
        """Rescale and rotate the system of equations"""

    def __perform_svd_on_covariance(self):
        """Return the result of the svd on the covariance matrix"""
        Q, R, QT = np.linalg.svd(
            self.__covariance_matrix, full_matrices=False)
        r = np.sqrt(R)
        return Q, r, QT

    def __transform_b_measured(self):
        """Return the rotated and rescaled measured b"""
        n_bins_b = len(self.__b_measured[0])
        return np.zeros(n_bins_b)

    def __transform_response_matrix(self):
        """Return the rotated and rescaled response matrix"""

    def __caclulate_inverse_covariance(self):
        """Return the inverse covariance of the transformed system"""

    def __perform_svd_on_transformed_system(self):
        """Return the result of svd on the transformed system"""

    def __calculate_expansion_coefficients(self):
        """Return the array of expansion coefficients d"""
