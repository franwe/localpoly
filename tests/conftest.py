import pytest
from pathlib import Path
import numpy as np
import local_polynomial_regression

from local_polynomial_regression.config.core import config

PACKAGE_ROOT = Path(local_polynomial_regression.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent


@pytest.fixture
def test_data():
    X = np.linspace(-np.pi, np.pi, num=100)
    y_real = np.sin(X)
    np.random.seed(config.model_config.random_state)
    y = np.random.normal(0, 1, len(X)) + y_real
    return X, y, y_real


@pytest.fixture
def tmpdir():
    return ROOT / "tmp"
