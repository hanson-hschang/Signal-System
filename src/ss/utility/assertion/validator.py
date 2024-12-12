from typing import Any, Callable, List, Type, TypeVar

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
        self._validate_functions: List[Callable[[], bool]] = []

    def __post_init__(self) -> None:
        for validate in self._validate_functions:
            assert validate(), self._errors
