"""
A collection of tests for svdunfold
"""
import pytest
import numpy as np
import svdunfold


def test_exception_for_rows_in_response_matrix():
    """Test if exception is thrown when number of bins in b is not the same
    as the rows in the response matrix
    """
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=5)
    A = np.zeros((4, 5))
    cov = np.zeros((5, 5))
    with pytest.raises(AssertionError, match=r".*Wrong dimensions.*"):
        svdunfold.SVDunfold(x_ini, b, A, cov)


def test_exception_for_columns_in_response_matrix():
    """Test if exception is thrown when number of bins in x_ini is not the same
    as the columns in the response matrix
    """
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=5)
    A = np.zeros((5, 4))
    cov = np.zeros((5, 5))
    with pytest.raises(AssertionError, match=r".*Wrong dimensions.*"):
        svdunfold.SVDunfold(x_ini, b, A, cov)


def test_exception_for_symmetric_covariance():
    """Test if exception is thrown when the covariance matrix is not symmetric"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=5)
    A = np.zeros((5, 5))
    cov = np.zeros((5, 5))
    cov[0, 1] = 1
    with pytest.raises(AssertionError, match=r".*is not symmetric.*"):
        svdunfold.SVDunfold(x_ini, b, A, cov)
