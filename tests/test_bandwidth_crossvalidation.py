import math
from local_polynomial_regression.base import LocalPolynomialRegressionCV


def test_bandwidth_has_suitable_magnitude(test_data, list_of_bandwidths):
    X_test, y_test, y_real = test_data

    # Given
    h_min = list_of_bandwidths.min()
    h_max = list_of_bandwidths.max()
    X_range = X_test.max() - X_test.min()

    # When
    assert math.isclose(h_min, X_range / 5, rel_tol=0.9)  # TODO: rethink
    assert math.isclose(h_max, X_range / 5, rel_tol=0.9)


def test_results_of_bandwidth_cv_sampling(test_data, list_of_bandwidths):
    X_test, y_test, y_real = test_data

    model = LocalPolynomialRegressionCV(
        X=X_test,
        y=y_test,
        kernel="gaussian",
        n_sections=3,
        loss="MSE",
        sampling="random",
    )

    results = model._bandwidth_cv_sampling(list_of_bandwidths)

    assert results is not None
    for key in ["MSE", "bandwidths", "h"]:
        assert key in results.keys()  # entry exists
        assert results[key] is not None  # and is not None


def test_results_of_bandwidth_cv(test_data, list_of_bandwidths):
    X_test, y_test, y_real = test_data

    model = LocalPolynomialRegressionCV(
        X=X_test,
        y=y_test,
        kernel="gaussian",
        n_sections=3,
        loss="MSE",
        sampling="random",
    )

    results = model.bandwidth_cv(list_of_bandwidths)

    assert results is not None
    for key in ["coarse results", "fine results"]:
        assert key in results.keys()  # entry exists
        assert results[key] is not None  # and is not None
