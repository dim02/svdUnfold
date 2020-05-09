"""
A single class module that performs data unfolding using the singular value decomposition approach
as described in https://arxiv.org/abs/hep-ph/9509307
"""

import numpy as np
import helpers


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
        self._b_measured = b
        self._x_ini = x_ini
        self._response_matrix = A
        self._covariance_matrix = cov
        self._X_inv = None
        self._S = None
        self._d = None
        n_bins_b = len(self._b_measured[0])
        n_bins_x = len(self._x_ini[0])
        self._C = helpers.calc_second_deriv_matrix(n_bins_x, 0.01)
        self._C_inv = helpers.calc_inverse_second_deriv_matrix(self._C)
        assert(self._response_matrix.shape[0] == n_bins_b),\
            "Wrong dimensions: bins in b != rows in response matrix"
        assert(self._response_matrix.shape[1] == n_bins_x),\
            "Wrong dimensions: bins in x_ini != columns in response matrix"
        assert helpers.check_symmetric(self._covariance_matrix), \
            "Covariance matrix is not symmetric"
        assert(n_bins_b > 1 and n_bins_x > 1),\
            "Bins should be at least 2"

    def unfold(self, k):
        """Perform the unfolding with regularization parameter tau=s(k)^2"""
        n_bins_x = len(self._x_ini[0])
        assert(k >= 0 and k < n_bins_x),\
            "Critical value k is out of bounds"
        tau = self._S[k]**2
        d_reg = self._calculate_regularized_d(tau)
        V = self._VT.T
        w_solution = self._calculate_transformed_system_solution(
            tau, d_reg, V)
        W_covariance = self._calculate_transformed_system_covariance(tau, V)
        self._x_unfolded = self._calculate_unfolded_distribution(w_solution)
        self._X_unfolded_covariance = self._calculate_unfolded_distribution_covariance(
            W_covariance)

    def get_unfolded_distribution(self):
        """Return the unfolded distribution as a 1d array"""
        return self._x_unfolded

    def get_unfolded_cov_matrix(self):
        """Return the unfolded covariance matrix as a 2d array"""
        return self._X_unfolded_covariance

    def get_abs_d(self):
        """Return a 1d array of the absolute value of the deconvolution coefficients d"""
        return np.abs(self._d)

    def get_singular_values(self):
        """Return an array of the singular values of the rescaled and rotated problem"""
        return self._S

    def transform_system(self):
        """Rescale and rotate the system of equations"""
        n_bins_x = len(self._x_ini[0])
        Q, r, _ = self._perform_svd_on_covariance()
        b_transformed = self._transform_b_measured(Q, r)
        transformed_response = self._transform_response_matrix(Q, r)
        self._X_inv = self._caclulate_inverse_covariance(
            transformed_response)
        U, self._S, self._VT = self._perform_svd_on_transformed_system(
            transformed_response)
        self._d = self._calculate_expansion_coefficients(U, b_transformed)

    def _perform_svd_on_covariance(self):
        """Return the result of the svd on the covariance matrix"""
        Q, R, QT = np.linalg.svd(
            self._covariance_matrix, full_matrices=False)
        r = np.sqrt(R)
        return Q, r, QT

    def _transform_b_measured(self, Q, r):
        """Return the rotated and rescaled measured b"""
        b_transformed = Q@self._b_measured[0]
        for i in range(b_transformed.shape[0]):
            b_transformed[i] = b_transformed[i] / r[i]
        return b_transformed

    def _transform_response_matrix(self, Q, r):
        """Return the rotated and rescaled response matrix"""
        n_bins_x = len(self._x_ini[0])
        n_bins_b = len(self._b_measured[0])
        transformed_response = Q@self._response_matrix
        for i in range(n_bins_b):
            for j in range(n_bins_x):
                transformed_response[i, j] = transformed_response[i, j] / r[i]
        return transformed_response

    def _caclulate_inverse_covariance(self, A_tilde):
        """Return the inverse covariance of the transformed system"""
        n_bins_x = len(self._x_ini[0])
        n_bins_b = len(self._b_measured[0])
        X_inv = np.zeros((n_bins_x, n_bins_x))
        for j in range(n_bins_x):
            for k in range(n_bins_x):
                for i in range(n_bins_b):
                    X_inv[j, k] += A_tilde[i, j] * A_tilde[i, k] / \
                        (self._x_ini[0][j] * self._x_ini[0][k])
        return X_inv

    def _perform_svd_on_transformed_system(self, A_tilde):
        """Return the result of svd on the transformed system"""
        A_tilde_x_C_inv = A_tilde@self._C_inv
        U, S, VT = np.linalg.svd(A_tilde_x_C_inv)
        return U, S, VT

    def _calculate_expansion_coefficients(self, U, b_transformed):
        """Return the array of expansion coefficients d"""
        d = U.T@b_transformed
        return d

    def _calculate_regularized_d(self, tau):
        """Calculate regularized expansion coefficients d_reg with tau=s(k)^2"""
        d_reg = self._d * self._S**2 / (self._S**2 + tau)
        return d_reg

    def _calculate_transformed_system_solution(self, tau, d_reg, V):
        """Return the solution of the rotated system w(tau)"""
        z = d_reg * self._S / (self._S**2 + tau)
        w_solution = self._C_inv@V@z
        return w_solution

    def _calculate_transformed_system_covariance(self, tau, V):
        """Return the covariance matrix of the rotated system W(tau)"""
        n_bins_x = len(self._x_ini[0])
        Z = np.zeros((n_bins_x, n_bins_x))
        z_diag = self._S**2 / (self._S**2 + tau)**2
        for i in range(n_bins_x):
            Z[i, i] = z_diag[i]
        W_covariance = self._C_inv@V@Z@V.T@self._C_inv
        return W_covariance

    def _calculate_unfolded_distribution(self, w_solution):
        """Calculate the unfolded distribution x(tau)"""
        x_unfolded = w_solution * self._x_ini[0]
        return x_unfolded

    def _calculate_unfolded_distribution_covariance(self, W_covariance):
        """Calculate the covariance matrix X(tau) of the unfolded distribution"""
        X_unfolded_covariance = W_covariance * \
            (np.outer(self._x_ini[0], self._x_ini[0]))
        return X_unfolded_covariance
