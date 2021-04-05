from pathlib import Path

from local_polynomial_regression.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml,
)

import pytest
from pydantic import ValidationError


TEST_CONFIG_TEXT = """
package_name: local_polynomial_regression
pipeline_name: local_polynomial_regression
pipeline_save_file: local_polynomial_regression_output
random_state: 1
bandwidth: 0.75
allowed_kernels:
  - gaussian
  - epanechnikov
kernel: gaussian
n_sections: 15
allowed_loss_functions:
  - ls
  - MSE
loss: MSE
allowed_sampling_types:
  - random
  - slices
sampling: slices
"""

INVALID_TEST_CONFIG_TEXT = """
package_name: local_polynomial_regression
pipeline_name: local_polynomial_regression
pipeline_save_file: local_polynomial_regression_output
random_state: 1
bandwidth: 0.75
allowed_kernels:
  - epanechnikov
kernel: gaussian
n_sections: 15
allowed_loss_functions:
  - ls
loss: MSE
allowed_sampling_types:
  - random
sampling: slices
"""


def test_fetch_config_structure(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    config = create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_config_validation_raises_error_for_invalid_config(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"

    # invalid config attempts to set a prohibited loss
    # function which we validate against an allowed set of
    # loss function parameters.
    config_1.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "not in the allowed set" in str(excinfo.value)


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    TEST_CONFIG_TEXT = """package_name: local_polynomial_regression"""
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(excinfo.value)
    assert "pipeline_name" in str(excinfo.value)
