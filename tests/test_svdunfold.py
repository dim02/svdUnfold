"""
A collection of tests for svdunfold
"""
import pytest
import numpy as np
import svdunfold
import helpers


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


def test_contruct_c_matrix_3d():
    """Test if 3d second derivative matrix is constructed correctly"""
    c_matrix = np.array([[-1, 1, 0], [1, -2, 1], [0, 1, -1]])
    assert np.array_equal(c_matrix, helpers.calc_second_deriv_matrix(3, 0))


def test_contruct_c_matrix_7d():
    """Test if 7d second derivative matrix is constructed correctly"""
    c_matrix = np.array([[-1, 1, 0, 0, 0, 0, 0],
                         [1, -2, 1, 0, 0, 0, 0],
                         [0, 1, -2, 1, 0, 0, 0],
                         [0, 0, 1, -2, 1, 0, 0],
                         [0, 0, 0, 1, -2, 1, 0],
                         [0, 0, 0, 0, 1, -2, 1],
                         [0, 0, 0, 0, 0, 1, -1]])
    assert np.array_equal(
        c_matrix, helpers.calc_second_deriv_matrix(7, 0))


def test_contruct_c_matrix_7d_with_xi():
    """Test if 7d second derivative matrix with x=0.1 is constructed correctly"""
    c_matrix = np.array([[-0.9, 1, 0, 0, 0, 0, 0],
                         [1, -1.9, 1, 0, 0, 0, 0],
                         [0, 1, -1.9, 1, 0, 0, 0],
                         [0, 0, 1, -1.9, 1, 0, 0],
                         [0, 0, 0, 1, -1.9, 1, 0],
                         [0, 0, 0, 0, 1, -1.9, 1],
                         [0, 0, 0, 0, 0, 1, -0.9]])
    assert np.array_equal(
        c_matrix, helpers.calc_second_deriv_matrix(7, 0.1))


def test_svd_on_covariance_matrix():
    """Test svd on covariance matrix"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=5)
    A = np.zeros((5, 5))
    cov = cov = np.array([[4, 0, 0], [0, 25, 0], [0, 0, 16]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    Q_test = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    r_test = np.array([5., 4., 2.])
    QT_test = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    Q, r, QT = unfold._SVDunfold__perform_svd_on_covariance()
    assert np.array_equal(Q, Q_test)
    assert np.array_equal(r, r_test)
    assert np.array_equal(QT, QT_test)


def test_transformed_b_measured_dimension():
    """Test if the dimensions of the transformed measured array are correct"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=4)
    A = np.zeros((4, 5))
    cov = np.zeros((4, 4))
    Q = np.zeros((4, 4))
    r = np.ones(4)
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    b_tilde = unfold._SVDunfold__transform_b_measured(Q, r)
    assert np.size(b_tilde) == 4

def test_transformed_measured_distribution():
    x_ini = np.histogram(np.zeros(10), bins=3)
    b = np.histogram([6, 7, 8, 9, 10], bins=3)
    A = np.zeros((3, 3))
    cov = cov = np.array([[4, 0, 0], [0, 25, 0], [0, 0, 16]])
    Q = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    r = np.array([5., 4., 2.])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    b_tilde = unfold._SVDunfold__transform_b_measured(Q, r)
    assert np.array_equal(np.array([2. / 5., 2. / 4., 1. / 2.]), b_tilde)
