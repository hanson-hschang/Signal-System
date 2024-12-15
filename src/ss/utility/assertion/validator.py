from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Type,
    TypeVar,
    Union,
    assert_never,
    overload,
)

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.utility.assertion import (
    check_directory_existence,
    check_filepath_existence,
    check_parent_directory_existence,
    is_extension_valid,
    is_filepath_valid,
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
            assert validate(), "\n".join(self._errors)

    def add_validation(self, *validations: Callable[[], bool]) -> None:
        self._validate_functions.extend(validations)

    def add_error(self, *errors: str) -> None:
        self._errors.extend(errors)


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
        self.add_validation(self._validate_value_range)

    def _validate_value_range(self) -> bool:
        if self._range_validation(self._value):
            return True
        self.add_error(
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


class SignalTrajectoryValidator(Validator):
    def __init__(self, signal_trajectory: Dict[str, ArrayLike]) -> None:
        super().__init__()
        self._signal_trajectory = signal_trajectory
        self.add_validation(
            self._validate_type,
            self._validate_time_key,
        )

    def _validate_type(self) -> bool:
        if isinstance(self._signal_trajectory, dict):
            return True
        self.add_error(
            "signal_trajectory must be a dictionary.",
            f"{type(self._signal_trajectory) = }",
        )
        return False

    def _validate_time_key(self) -> bool:
        if "time" in self._signal_trajectory:
            return True
        self.add_error(
            "'time' must be a key in signal_trajectory.",
            f"{self._signal_trajectory.keys() = }",
        )
        return False

    def get_trajectory(self) -> Dict[str, NDArray[np.float64]]:
        signal_trajectory = dict()
        for key, value in self._signal_trajectory.items():
            signal_trajectory[key] = np.array(value, dtype=np.float64)
        return signal_trajectory


class FolderPathExistenceValidator(Validator):
    def __init__(self, foldername: Union[str, Path]) -> None:
        super().__init__()
        self._foldername = foldername
        self.add_validation(self._validate_folderpath_existence)

    def _validate_folderpath_existence(self) -> bool:
        # Check if the foldername is a valid folderpath and exists
        if check_directory_existence(self._foldername):
            return True

        # If not, ask the user if they want to create the folder
        response = (
            input(
                f"folder: '{self._foldername}' does not exist. "
                "Would you like to create the folder? (y/n): "
            )
            .lower()
            .strip()
        )

        # If the user responds with 'y' or 'yes', create the folder and return True
        if response in ["y", "yes"]:
            Path(self._foldername).mkdir(parents=True, exist_ok=True)
            print(f"Folder '{self._foldername}' created successfully.")
            return True

        # If the user does not want to create the folder, show an error message and return False
        self.add_error(
            "foldername must be a valid and existed folderpath (str or Path).",
            f"foldername provided is '{self._foldername}'.",
        )
        return False

    def get_folderpath(self) -> Path:
        return Path(self._foldername)


class FilePathExistenceValidator(Validator):
    def __init__(
        self, filename: Union[str, Path], extension: Union[str, Iterable[str]]
    ) -> None:
        super().__init__()
        self._filename = filename
        self._extension = extension
        self.add_validation(self._validate_filepath_existence)

    def _validate_filepath_existence(self) -> bool:
        # Check if the extension is valid
        if not is_extension_valid(self._extension):
            self.add_error(
                f"extension = '{self._extension}' must start with a '.'."
            )
            return False

        # Check if the filename is a valid filepath and exists
        if check_filepath_existence(self._filename, self._extension):
            return True

        # If not, show an error message and return False
        self.add_error(
            f"filename must has the correct extension: {self._extension} and exist.",
            f"filename provided is {self._filename}.",
        )
        return False

    def get_filepath(self) -> Path:
        return Path(self._filename)


class FilePathValidator(Validator):
    def __init__(
        self, filename: Union[str, Path], extension: Union[str, Iterable[str]]
    ) -> None:
        super().__init__()
        self._filename = filename
        self._extension = extension
        self.add_validation(self._validate_filepath)

    def _validate_filepath(self) -> bool:
        # Check if the extension is valid
        if not is_extension_valid(self._extension):
            self.add_error(
                f"extension = '{self._extension}' must start with a '.'."
            )
            return False

        # Check if the filename is a valid filepath
        if is_filepath_valid(self._filename, self._extension):
            return True

        # If not, check if the parent directory exists
        if check_parent_directory_existence(self._filename):
            # The parent directory exists, but the filename is invalid
            # show an error message and return False
            self.add_error(
                "filename must be a valid filepath (str or Path) "
                f"with the correct extension: '{self._extension}'.",
                f"filename provided is '{self._filename}'.",
            )
            return False

        # The parent directory does not exist, ask the user if they want to create it
        folder_name = Path(self._filename).parent
        response = (
            input(
                f"folder: '{folder_name}' does not exist. "
                "Would you like to create the folder? (y/n): "
            )
            .lower()
            .strip()
        )

        # If the user responds with 'y' or 'yes', create the folder and return True
        if response in ["y", "yes"]:
            folder_name.mkdir(parents=True, exist_ok=True)
            print(f"Folder '{folder_name}' created successfully.")
            return True

        # If the user does not want to create the folder, show an error message and return False
        self.add_error(
            f"folder: '{folder_name}' does not exist. ",
            "filename must be a valid filepath (str or Path) "
            f"with the correct extension: '{self._extension}'. ",
            f"filename provided is '{self._filename}'.",
        )
        return False

    def get_filepath(self) -> Path:
        return Path(self._filename)
