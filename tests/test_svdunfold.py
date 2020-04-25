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
