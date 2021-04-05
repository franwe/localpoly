import pytest
from pathlib import Path
import numpy as np
import local_polynomial_regression

PACKAGE_ROOT = Path(local_polynomial_regression.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent


@pytest.fixture
def test_data():
    X = np.linspace(-np.pi, np.pi, num=100)
    y_real = np.sin(X)
    y = np.random(len(X)) + y_real
    return X, y, y_real


@pytest.fixture
def tmpdir():
    return ROOT / "tmp"
