from typing import Any

import pytest

from ss.tool.assertion import (
    is_nonnegative_integer,
    is_positive_integer,
    is_positive_number,
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
def test_is_positive_number(input_value: Any, expected_result: bool) -> None:
    """Test is_positive_number function"""
    assert is_positive_number(input_value) == expected_result


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
def test_is_positive_integer(input_value: Any, expected_result: bool) -> None:
    """Test is_positive_integer function"""
    assert is_positive_integer(input_value) == expected_result


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
def test_is_nonnegative_integer(
    input_value: Any, expected_result: bool
) -> None:
    """Test is_nonnegative_integer function"""
    assert is_nonnegative_integer(input_value) == expected_result
