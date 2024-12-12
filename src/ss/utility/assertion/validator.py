from typing import (
    Any,
    Callable,
    List,
    Literal,
    Type,
    TypeVar,
    Union,
    assert_never,
    overload,
)

from ss.utility.assertion import (
    is_integer,
    is_nonnegative_integer,
    is_nonnegative_number,
    is_number,
    is_positive_integer,
    is_positive_number,
)

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


class BasicScalarValidator(Validator):
    def __init__(
        self,
        value: Union[int, float],
        name: str,
        range_validation: Callable[..., bool],
    ) -> None:
        super().__init__()
        self._value = value
        self._name = name
        self._range_validation = range_validation
        self._condition = " ".join(
            self._range_validation.__name__.split("_")[1:]
        )
        self._return_type: Literal["integer", "number"] = (
            "integer"
            if self._condition.split(" ")[-1] == "integer"
            else "number"
        )
        self._validate_functions.append(self._validate_value_range)

    def _validate_value_range(self) -> bool:
        if self._range_validation(self._value):
            return True
        self._errors.append(
            f"{self._name} = {self._value} must be a {self._condition}"
        )
        return False

    @overload
    def get_value(self) -> int: ...

    @overload  # type: ignore
    def get_value(self) -> Union[int, float]: ...

    def get_value(self) -> Union[int, float]:
        match self._return_type:
            case "integer":
                return int(self._value)
            case "number":
                return self._value
            case _ as return_type:
                assert_never(return_type)


class IntegerValidator(BasicScalarValidator):
    def __init__(self, value: Union[int, float], name: str) -> None:
        super().__init__(value, name, is_integer)


class PositiveIntegerValidator(BasicScalarValidator):
    def __init__(self, value: Union[int, float], name: str) -> None:
        super().__init__(value, name, is_positive_integer)


class NonnegativeIntegerValidator(BasicScalarValidator):
    def __init__(self, value: Union[int, float], name: str) -> None:
        super().__init__(value, name, is_nonnegative_integer)


class NumberValidator(BasicScalarValidator):
    def __init__(self, value: Union[int, float], name: str) -> None:
        super().__init__(value, name, is_number)


class PositiveNumberValidator(BasicScalarValidator):
    def __init__(self, value: Union[int, float], name: str) -> None:
        super().__init__(value, name, is_positive_number)


class NonnegativeNumberValidator(BasicScalarValidator):
    def __init__(self, value: Union[int, float], name: str) -> None:
        super().__init__(value, name, is_nonnegative_number)
