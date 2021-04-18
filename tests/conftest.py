import pytest
from pathlib import Path
import numpy as np
import local_polynomial_regression

PACKAGE_ROOT = Path(local_polynomial_regression.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
RANDOM_STATE = 1


@pytest.fixture
def test_data():
    np.random.seed(RANDOM_STATE)
    X = np.random.uniform(-np.pi, np.pi, size=30)
    y_real = np.sin(X)
    y = np.random.normal(0, 1, len(X)) + y_real
    return X, y, y_real


@pytest.fixture
def list_of_bandwidths():
    return np.linspace(0.4, 1.4, num=5)


@pytest.fixture
def tmpdir():
    return ROOT / "tmp"
