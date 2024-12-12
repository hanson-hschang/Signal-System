import pytest

from ss.utility.assertion.validator import Validator


def test_validator_base_class_initialization() -> None:
    """Test that base Validator class can be initialized without errors"""
    validator = Validator()
    assert hasattr(validator, "_errors")
    assert isinstance(validator._errors, list)
    assert isinstance(validator._validate_functions, list)
    assert len(validator._errors) == 0
    assert len(validator._validate_functions) == 0


def test_validator_post_init_failure() -> None:
    """Test that __post_init__ raises assertion error when there are errors"""

    class FailingValidator(Validator):
        def __init__(self) -> None:
            super().__init__()
            self._validate_functions.append(self._validate)

        def _validate(self) -> bool:
            self._errors.append("Test error 1")
            self._errors.append("Test error 2")
            return False

    with pytest.raises(AssertionError) as exc_info:
        FailingValidator()
    assert "['Test error 1', 'Test error 2']" in str(exc_info.value)
