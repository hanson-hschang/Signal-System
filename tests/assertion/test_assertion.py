from typing import Any

import pytest

from assertion import (
    Validator,
    isNonNegativeInteger,
    isPositiveInteger,
    isPositiveNumber,
)


@pytest.mark.parametrize(
    "input_value, expected_result",
    [
        (1, True),
        (0, False),
        (-1, False),
        (1.0, True),
        (0.0, False),
        (-1.0, False),
        (1.5, True),
        (0.5, True),
        (-1.5, False),
        ("1", False),
    ],
)
def test_isPositiveNumber(input_value: Any, expected_result: bool) -> None:
    """Test isPositiveNumber function"""
    assert isPositiveNumber(input_value) == expected_result


@pytest.mark.parametrize(
    "input_value, expected_result",
    [
        (1, True),
        (0, False),
        (-1, False),
        (1.0, True),
        (0.0, False),
        (-1.0, False),
        (1.5, False),
        (0.5, False),
        (-1.5, False),
        ("1", False),
    ],
)
def test_isPositiveInteger(input_value: Any, expected_result: bool) -> None:
    """Test isPositiveInteger function"""
    assert isPositiveInteger(input_value) == expected_result


@pytest.mark.parametrize(
    "input_value, expected_result",
    [
        (1, True),
        (0, True),
        (-1, False),
        (1.0, True),
        (0.0, True),
        (-1.0, False),
        (1.5, False),
        (0.5, False),
        (-1.5, False),
        ("1", False),
    ],
)
def test_isNonNegativeInteger(input_value: Any, expected_result: bool) -> None:
    """Test isNonNegativeInteger function"""
    assert isNonNegativeInteger(input_value) == expected_result


def test_validator_base_class_initialization() -> None:
    """Test that base Validator class can be initialized without errors"""
    validator = Validator()
    assert hasattr(validator, "_errors")
    assert isinstance(validator._errors, list)
    assert len(validator._errors) == 0


def test_validator_post_init_failure() -> None:
    """Test that __post_init__ raises assertion error when there are errors"""

    class FailingValidator(Validator):
        def __init__(self) -> None:
            super().__init__()
            self._errors.append("Test error 1")
            self._errors.append("Test error 2")

    with pytest.raises(AssertionError) as exc_info:
        FailingValidator()
    assert "['Test error 1', 'Test error 2']" in str(exc_info.value)
