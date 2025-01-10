import pytest

from ss.utility.assertion.validator import Validator


def test_validator_base_class_initialization() -> None:
    validator = Validator()
    assert hasattr(validator, "_errors")
    assert isinstance(validator._errors, list)
    assert isinstance(validator._validate_functions, list)
    assert len(validator._errors) == 0
    assert len(validator._validate_functions) == 0


def test_validator_post_init_failure() -> None:
    class FailingValidator(Validator):
        def __init__(self) -> None:
            super().__init__()
            self.add_validation(self._validate)

        def _validate(self) -> bool:
            self.add_error("Test error 1")
            self.add_error("Test error 2", "Test error 3")
            return False

    with pytest.raises(AssertionError) as exc_info:
        FailingValidator()
    assert "Test error 1\nTest error 2\nTest error 3" in str(exc_info.value)
