from typing import Any

import pytest

from assertion import isNonNegativeInteger, isPositiveInteger, isPositiveNumber


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
