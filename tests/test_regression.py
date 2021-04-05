from math import isclose

from local_polynomial_regression.base import LocalPolynomialRegression
from local_polynomial_regression.config.core import config


def test_create_fit(test_data):
    X_test, y_test, y_real = test_data

    model = LocalPolynomialRegression(
        X=X_test, y=y_test, h=config.model_config.bandwidth, kernel=config.model_config.kernel, gridsize=100
    )
    X_est, y_est, first, second, h = model.fit()

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
