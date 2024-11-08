from typing import Any, List, Type, TypeVar, Union


def isPositiveNumber(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number > 0


def isPositiveInteger(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number > 0 and int(number) == number


def isNonNegativeInteger(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number >= 0 and int(number) == number


# Create a TypeVar for the Validator class
T = TypeVar("T", bound="Validator")


class ValidatorMeta(type):  # pragma: no cover
    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:  # type: ignore
        # Create instance (calls __init__)
        instance: T = super().__call__(*args, **kwargs)  # type: ignore
        # Call __post_init__ after initialization
        if hasattr(instance, "__post_init__"):
            instance.__post_init__()
        return instance


class Validator(metaclass=ValidatorMeta):
    def __init__(self) -> None:
        self._errors: List[str] = []

    def __post_init__(self) -> None:
        is_valid: bool = len(self._errors) == 0
        assert is_valid, self._errors
