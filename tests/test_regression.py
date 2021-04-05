from math import isclose

from local_polynomial_regression.smoothing import create_fit


def test_create_fit(test_data):
    X_test, y_test, y_real = test_data
    X_est, y_est, first, second, h = create_fit(X_test, y_test, h=0.3)

    # X_domain similar range as X_sim
    x_tolerance = (X_test.max() - X_test.min()) / 50
    assert isclose(X_test.min(), X_est.min(), abs_tol=x_tolerance)
    assert isclose(X_test.max(), X_est.max(), abs_tol=x_tolerance)

    # y_est similar range as y_sim: can not test the same way as for X, because of outliers
    # insted: check if y_est is smaller than y_test, but this could also overshoot?
    assert y_est.min() >= y_test.min()
    assert y_est.max() <= y_test.max()

    # fit, first, second exist
    assert y_est is not None
