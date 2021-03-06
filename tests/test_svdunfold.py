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
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.array([[4, 0, 0], [0, 25, 0], [0, 0, 16]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    Q_test = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    r_test = np.array([5., 4., 2.])
    QT_test = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    Q, r, QT = unfold._perform_svd_on_covariance()
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
    b_tilde = unfold._transform_b_measured(Q, r)
    assert np.size(b_tilde) == 4


def test_transformed_measured_distribution():
    """Test if the transformed measured distribution is correct"""
    x_ini = np.histogram(np.zeros(10), bins=3)
    b = np.histogram([6, 7, 8, 9, 10], bins=3)
    A = np.zeros((3, 3))
    cov = cov = np.array([[4, 0, 0], [0, 25, 0], [0, 0, 16]])
    Q = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    r = np.array([5., 4., 2.])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    b_tilde = unfold._transform_b_measured(Q, r)
    assert np.array_equal(np.array([2. / 5., 2. / 4., 1. / 2.]), b_tilde)


def test_transformed_response_correct_dimensions():
    """Test if the dimensions of the transposed response matrix are correct"""
    x_ini = np.histogram(np.zeros(10), bins=3)
    b = np.histogram(np.zeros(10), bins=7)
    A = np.zeros((7, 3))
    cov = np.eye(7)
    Q = np.zeros((7, 7))
    r = np.ones(7)
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    A_tilde = unfold._transform_response_matrix(Q, r)
    assert A_tilde.shape == (7, 3)


def test_transformed_response_correct_values():
    """Test if the transformed response matrix is correct"""
    x_ini = np.histogram(np.zeros(10), bins=3)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.histogram2d([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], bins=3)[0]
    cov = np.array([[4, 0, 0], [0, 25, 0], [0, 0, 16]])
    Q = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    r = np.array([5., 4., 2.])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    A_tilde = unfold._transform_response_matrix(Q, r)
    assert np.array_equal(
        np.array([[0, 0, 2. / 5.], [2. / 4., 0, 0], [0, 1. / 2., 0]]), A_tilde)


def test_inverse_covariance_correct_dimensions():
    """Test if the inverse covariance matrix has the correct dimensions"""
    x_ini = np.histogram(np.array([1, 2, 3, 4, 5]), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.eye(3)
    A_tilde = np.zeros((3, 5))
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    X_inv = unfold._caclulate_inverse_covariance(A_tilde)
    assert X_inv.shape == (5, 5)


def test_inverse_covariance_correct_values_3x3():
    """Test if the inverse covariance matrix is correct for A_tilde(3x3)"""
    x_ini = np.histogram(np.array([1, 2, 2, 3, 3, 3]), bins=3)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 3))
    cov = np.eye(3)
    A_tilde = np.array([[1, 0, 1], [0, 2, 2], [3, 1, 0]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    X_inv = unfold._caclulate_inverse_covariance(A_tilde)
    assert np.array_equal(X_inv, np.array(
        [[10., 3. / 2, 1. / 3.], [3. / 2., 5. / 4., 2. / 3.], [1. / 3., 2. / 3., 5. / 9.]]))


def test_inverse_covariance_correct_values_3x5():
    """Test if the inverse covariance matrix is correct for A_tilde(3x3)"""
    x_ini = np.histogram(
        np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.eye(3)
    A_tilde = np.array([[1, 0, 1, 4, 9], [0, 2, 2, 0, 5], [3, 1, 0, 1, 1]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    X_inv = unfold._caclulate_inverse_covariance(A_tilde)
    X_inv_test = np.array([[10., 3. / 2., 1. / 3., 7. / 4., 12. / 5.],
                           [3. / 2., 5. / 4., 2. / 3., 1. / 8., 11. / 10.],
                           [1. / 3., 2. / 3., 5. / 9., 1. / 3., 19. / 15.],
                           [7. / 4., 1. / 8., 1. / 3., 17. / 16., 37. / 20.],
                           [12. / 5., 11. / 10., 19. / 15., 37. / 20, 107. / 25.]])
    assert np.array_equal(X_inv, X_inv_test)


def test_svd_on_transformed_system():
    """Test svd on transformed system"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.eye(3)
    A_tilde = np.array([[1, 0, 6, 3, 8],
                        [9, 3, 6, 4, 0],
                        [1, 5, 3, 7, 5]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    U_test = np.array([[-0.50933966, 0.58444234, -0.63166467],
                       [-0.62247852, -0.75704201, -0.1985142],
                       [-0.5942168, 0.29208653, 0.74939432]])
    S_test = np.array([1580.51138731, 21.14648575, 1.60524919])
    VT_test = np.array([[-0.44869132, -0.44844725, -0.44678007, -0.44606964, -0.44607244],
                        [0.61939546, 0.33245276, 0.00384851,
                         -0.33232921, -0.62878209],
                        [0.42855687, -0.61537921, 0.31004087,
                         -0.47011051, 0.34715732],
                        [-0.35523906, -0.09660769, 0.82961209,
                         0.04126636, -0.41774758],
                        [-0.32429862, 0.54803796, 0.12645634, -0.68401197, 0.33259768]])
    U, S, VT = unfold._perform_svd_on_transformed_system(
        A_tilde)
    assert np.allclose(U, U_test)
    assert np.allclose(S, S_test)
    assert np.allclose(VT, VT_test)


def test_expansion_coefficients_dimension():
    """Test if vector of expansion coefficients has correct dimensions"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.array([1, 2, 2, 3, 3, 3]), bins=3)
    A = np.zeros((3, 5))
    cov = np.eye(3)
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    U = np.ones((3, 5))
    b_transformed = np.ones(3)
    d = unfold._calculate_expansion_coefficients(U, b_transformed)
    assert d.shape == (5,)


def test_expansion_coefficients_correct():
    """Test if vector of expansion coefficients is calculated correctly"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.array([1, 2, 2, 3, 3, 3]), bins=3)
    A = np.zeros((3, 5))
    cov = np.eye(3)
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    U = np.array([[1, 4, 8, 3, 6], [9, 5, 3, 0, 3], [3, 5, 4, 4, 8]])
    b_transformed = np.array([4, 5, 6])
    d = unfold._calculate_expansion_coefficients(U, b_transformed)
    d_test = np.array([67, 71, 71, 36, 87])
    assert np.allclose(d, d_test)


def test_singular_values():
    """Test if singular values are returned correctly with 3-bin b and 5-bin x_ini and
    second inverse derivative xi=0.01 """
    x_ini = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5], bins=5)
    b = np.histogram([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], bins=3)
    A = np.array([[1, 2, 0, 0, 0], [0, 0, 3, 2, 0], [0, 0, 0, 2, 5]])
    cov = np.array([[1.5, 0., 0.], [0., 1.5, 0.], [0., 0., 1.5]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    unfold.transform_system()
    s_test = np.array([332.72483635, 6.42101093, 1.62637087])
    s = unfold.get_singular_values()
    assert np.allclose(s_test, s)


def test_expansion_coefficients():
    """Test if expansion coefficients are returned correctly with 3-bin b and 5-bin x_ini and
    second inverse derivative xi=0.01 """
    x_ini = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5], bins=5)
    b = np.histogram([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], bins=3)
    A = np.array([[1, 2, 0, 0, 0], [0, 0, 3, 2, 0], [0, 0, 0, 2, 5]])
    cov = np.array([[1.5, 0., 0.], [0., 1.5, 0.], [0., 0., 1.5]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    unfold.transform_system()
    d_test = np.abs(np.array([-7.0796561, 1.42501533, -1.78357341]))
    d = unfold.get_abs_d()
    assert np.allclose(d_test, d)


def test_transform_system_3x5():
    """Test X_inv from transform system proceedure with 3-bin b and 5-bin x_ini and
    second inverse derivative xi=0.01
    """
    x_ini = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5], bins=5)
    b = np.histogram([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], bins=3)
    A = np.array([[1, 2, 0, 0, 0], [0, 0, 3, 2, 0], [0, 0, 0, 2, 5]])
    cov = np.array([[1.5, 0., 0.], [0., 1.5, 0.], [0., 0., 1.5]])
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    unfold.transform_system()
    X_inv = np.array([[0.66666667, 0.66666667, 0., 0., 0.],
                      [0.66666667, 0.66666667, 0., 0., 0.],
                      [0., 0., 0.66666667, 0.33333333, 0.],
                      [0., 0., 0.33333333, 0.33333333, 0.33333333],
                      [0., 0., 0., 0.33333333, 0.66666667]])
    assert np.allclose(unfold._X_inv, X_inv)


def test_regularized_expansion_coefficients():
    """Test if regularized expansion coefficients are calculated correctly"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.ones((3, 5))
    cov = np.eye(3)
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    unfold._d = np.array([5, 4, 1, 0.1, 1])
    unfold._S = np.array([10, 3, 2, 1, 0.01])
    d_reg = np.array([4.80769231, 2.76923077, 0.5, 0.02, 2.49993750e-05])
    tau = 4
    d_reg_test = unfold._calculate_regularized_d(tau)
    assert np.allclose(d_reg, d_reg_test)


def test_exception_when_k_out_of_bounds():
    """Test if  exception is thrown when critical value k is out of bounds"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.zeros((3, 3))
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    with pytest.raises(AssertionError, match=r".*out of bounds.*"):
        unfold.unfold(5)


def test_transformed_system_solution():
    """Test the calculation of the stransformed system solution"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.zeros((3, 3))
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    V = np.array([[-0.44869132, 0.61939546, 0.42855687, -0.35523906, -0.32429862],
                  [-0.44844725, 0.33245276, -0.61537921, -0.09660769, 0.54803796],
                  [-0.44678007, 0.00384851, 0.31004087, 0.82961209, 0.12645634],
                  [-0.44606964, -0.33232921, -0.47011051, 0.04126636, -0.68401197],
                  [-0.44607244, -0.62878209, 0.34715732, -0.41774758, 0.33259768]])
    d = np.array([5, 4, 1, 0.1, 1])
    unfold._S = np.array([10, 3, 2, 1, 0.01])
    tau = 4
    w = unfold._calculate_transformed_system_solution(tau, d, V)
    w_test = np.array([-23.13863676, -22.45199401,
                       -21.60395838, -20.65671022, -20.14252922])
    assert np.allclose(w, w_test)


def test_transformed_system_covariance():
    """Test the calculation of the transformed system covariance"""
    x_ini = np.histogram(np.zeros(10), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.zeros((3, 3))
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    V = np.array([[-0.44869132, 0.61939546, 0.42855687, -0.35523906, -0.32429862],
                  [-0.44844725, 0.33245276, -0.61537921, -0.09660769, 0.54803796],
                  [-0.44678007, 0.00384851, 0.31004087, 0.82961209, 0.12645634],
                  [-0.44606964, -0.33232921, -0.47011051, 0.04126636, -0.68401197],
                  [-0.44607244, -0.62878209, 0.34715732, -0.41774758, 0.33259768]])
    unfold._S = np.array([10, 3, 2, 1, 0.01])
    tau = 4
    W_cov = unfold._calculate_transformed_system_covariance(tau, V)
    W_cov_t = np.array([[18.65430, 18.58575, 18.49273, 18.40675, 18.35836],
                        [18.58575, 18.55553, 18.49723, 18.44252, 18.40052],
                        [18.49273, 18.49723, 18.50056, 18.48892, 18.47695],
                        [18.40675, 18.44252, 18.48892, 18.53564, 18.56115],
                        [18.35836, 18.40052, 18.47695, 18.56115, 18.62346]])
    assert np.allclose(W_cov, W_cov_t)


def test_unfolded_distribution():
    """Test calculation of unfolded distribution x"""
    x_ini = np.histogram(
        np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.zeros((3, 3))
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    w_solution = np.array([2, 1, 5, 6, 3])
    x = unfold._calculate_unfolded_distribution(w_solution)
    x_test = np.array([2, 2, 15, 24, 15])
    assert np.allclose(x, x_test)


def test_unfolded_covariance():
    """Test calculation of unfolded covariance X_cov_unfolded"""
    x_ini = np.histogram(
        np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]), bins=5)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 5))
    cov = np.zeros((3, 3))
    unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
    W_covar = np.array([[6, 5, 4, 4, 3],
                        [5, 5, 4, 4, 4],
                        [9, 3, 6, 2, 5],
                        [5, 2, 2, 4, 5],
                        [6, 2, 5, 5, 6]])
    X_cov = unfold._calculate_unfolded_distribution_covariance(
        W_covar)
    X_cov_test = np.array([[6, 10, 12, 16, 15],
                           [10, 20, 24, 32, 40],
                           [27, 18, 54, 24, 75],
                           [20, 16, 24, 64, 100],
                           [30, 20, 75, 100, 150]])
    assert np.allclose(X_cov, X_cov_test)


def test_x_ini_bins_at_least_two():
    """Test exception when bins in x_ini are not at least two"""
    x_ini = np.histogram(np.zeros(10), bins=1)
    b = np.histogram(np.zeros(10), bins=3)
    A = np.zeros((3, 1))
    cov = np.zeros((3, 3))
    with pytest.raises(AssertionError, match=r".*at least 2.*"):
        unfold = svdunfold.SVDunfold(x_ini, b, A, cov)


def test_b_bins_at_least_two():
    """Test exception when bins in b are not at least two"""
    x_ini = np.histogram(np.zeros(10), bins=3)
    b = np.histogram(np.zeros(10), bins=1)
    A = np.zeros((1, 3))
    cov = np.zeros((1, 1))
    with pytest.raises(AssertionError, match=r".*at least 2.*"):
        unfold = svdunfold.SVDunfold(x_ini, b, A, cov)
