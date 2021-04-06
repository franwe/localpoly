import pytest
from pathlib import Path
import numpy as np
import local_polynomial_regression

from local_polynomial_regression.config.core import config

PACKAGE_ROOT = Path(local_polynomial_regression.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent


@pytest.fixture
def test_data():
    np.random.seed(config.model_config.random_state)
    X = np.random.uniform(-np.pi, np.pi, size=150)
    y_real = np.sin(X)
    y = np.random.normal(0, 1, len(X)) + y_real
    return X, y, y_real


@pytest.fixture
def list_of_bandwidths():
    return np.linspace(0.4, 1.4, num=10)


@pytest.fixture
def tmpdir():
    return ROOT / "tmp"
