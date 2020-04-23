"""
A single class module that performs data unfolding using the singular value decomposition approach
as described in https://arxiv.org/abs/hep-ph/9509307
"""


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
