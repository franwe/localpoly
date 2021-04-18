from math import isclose

from local_polynomial_regression.base import LocalPolynomialRegression


def test_local_polynomial_regression(test_data):
    X_test, y_test, y_real = test_data

    model = LocalPolynomialRegression(X=X_test, y=y_test, h=0.89, kernel="gaussian", gridsize=100)
    x = X_test.mean()
    results = model.local_polynomial_regression(x)

    assert results is not None
    for key in ["fit", "first", "second", "weight"]:
        assert key in results.keys()  # entry exists
        assert results[key] is not None  # and is not None


def test_create_fit(test_data):
    X_test, y_test, y_real = test_data

    model = LocalPolynomialRegression(X=X_test, y=y_test, h=0.89, kernel="gaussian", gridsize=100)
    prediction_interval = (X_test.min(), X_test.max())
    results = model.fit(prediction_interval)
    assert results is not None
    for key in ["X", "fit", "first", "second"]:
        assert key in results.keys()  # entry exists
        assert results[key] is not None  # and is not None

    # results["X"] similar range as X_test
    x_tolerance = (X_test.max() - X_test.min()) / 50
    assert isclose(X_test.min(), results["X"].min(), abs_tol=x_tolerance)
    assert isclose(X_test.max(), results["X"].max(), abs_tol=x_tolerance)

    # results["fit"] similar range as y_sim: can not test the same way as for X, because of outliers
    # insted: check if results["fit"] is smaller than y_test, but this could also overshoot?
    assert results["fit"].min() >= y_test.min()
    assert results["fit"].max() <= y_test.max()
