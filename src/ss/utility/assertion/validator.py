from types import FrameType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    assert_never,
    cast,
    overload,
)

import inspect
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
from ss.utility.assertion.inspect import get_call_line

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
    def __init__(self, argument: Any) -> None:
        self._argument = argument
        self._name = self._get_validator_argument_name(inspect.currentframe())
        self._errors: List[str] = []
        self._validate_functions: List[Callable[[], bool]] = []

    def __post_init__(self) -> None:
        for validate in self._validate_functions:
            assert validate(), "\n".join(self._errors)

    @property
    def name(self) -> str:
        return self._name

    def add_validation(self, *validations: Callable[[], bool]) -> None:
        self._validate_functions.extend(validations)

    def add_error(self, *errors: str) -> None:
        self._errors.extend(errors)

    @staticmethod
    def _get_validator_argument_name(frame: Optional[FrameType]) -> str:

        # Get the frame of the caller
        call_line = get_call_line(frame)

        # Get the argument name from the call line
        key_phrase = "Validator("
        start_index = call_line.find(key_phrase) + len(key_phrase)
        end_index = (
            call_line.find(",", start_index)
            if "," in call_line[start_index:]
            else call_line.find(")", start_index)
        )
        argument_name = call_line[start_index:end_index].strip()
        if (end_index := argument_name.find(":=")) != -1:
            argument_name = argument_name[:end_index]
        if (end_index := argument_name.find("=")) != -1:
            argument_name = argument_name[:end_index]
        if "self." == argument_name[:5]:
            argument_name = argument_name[5:]
        if argument_name[0].isdigit():
            argument_name = "argument"
        if argument_name[0] in ["'", '"']:
            argument_name = "argument"
        return argument_name


class BasicScalarValidator(Validator):
    def __init__(
        self,
        value: Union[int, float],
        range_validation: Callable[..., bool],
    ) -> None:
        super().__init__(value)
        self._value = value
        self._range_validation = range_validation
        self._condition = "a scalar"
        self._update_condition()
        # self._condition = " ".join(
        #     self._range_validation.__name__.split("_")[1:]
        # )
        # if self._condition[0] in ["a", "e", "i", "o", "u"]:
        #     self._condition = "an " + self._condition
        # else:
        #     self._condition = "a " + self._condition
        # self._return_type: Literal["integer", "number"] = (
        #     "integer"
        #     if self._condition.split(" ")[-1] == "integer"
        #     else "number"
        # )
        self.add_validation(self._validate_value_range)

    def _update_condition(self) -> None:
        self._condition = " ".join(
            self._range_validation.__name__.split("_")[1:]
        )
        if self._condition[0] in ["a", "e", "i", "o", "u"]:
            self._condition = "an " + self._condition
        else:
            self._condition = "a " + self._condition

    def _validate_value_range(self) -> bool:
        if self._range_validation(self._value):
            return True
        self.add_error(
            f"{self._name} must be {self._condition}. "
            f"{self._name} given is {self._value}."
        )
        return False

    # @overload
    # def get_value(self) -> int: ...

    # @overload  # type: ignore
    # def get_value(self) -> float: ...

    def get_value(self) -> Union[int, float]:
        return self._value
        # match self._return_type:
        #     case "integer":
        #         return int(self._value)
        #     case "number":
        #         return self._value
        #     case _ as return_type:
        #         assert_never(return_type)


class IntegerValidator(BasicScalarValidator):
    def __init__(
        self,
        value: Union[int, float],
        range_validation: Callable[..., bool] = is_integer,
    ) -> None:
        super().__init__(value, range_validation)

    def get_value(self) -> int:
        return int(self._value)


class PositiveIntegerValidator(IntegerValidator):
    def __init__(self, value: Union[int, float]) -> None:
        super().__init__(value, is_positive_integer)


class NonnegativeIntegerValidator(IntegerValidator):
    def __init__(self, value: Union[int, float]) -> None:
        super().__init__(value, is_nonnegative_integer)


class NumberValidator(BasicScalarValidator):
    def __init__(
        self,
        value: Union[int, float],
        range_validation: Callable[..., bool] = is_number,
    ) -> None:
        super().__init__(value, range_validation)

    def get_value(self) -> float:
        return float(self._value)


class PositiveNumberValidator(NumberValidator):
    def __init__(self, value: Union[int, float]) -> None:
        super().__init__(value, is_positive_number)


class NonnegativeNumberValidator(NumberValidator):
    def __init__(self, value: Union[int, float]) -> None:
        super().__init__(value, is_nonnegative_number)


class SignalTrajectoryValidator(Validator):
    def __init__(self, signal_trajectory: Dict[str, ArrayLike]) -> None:
        super().__init__(signal_trajectory)
        self._signal_trajectory = cast(Dict[str, ArrayLike], self._argument)
        self.add_validation(
            self._validate_type,
            self._validate_time_key,
        )

    def _validate_type(self) -> bool:
        if isinstance(self._signal_trajectory, dict):
            return True
        self.add_error(
            f"{self._name} must be a dictionary.",
            f"{self._name} given is of type {type(self._signal_trajectory)}",
        )
        return False

    def _validate_time_key(self) -> bool:
        if "time" in self._signal_trajectory.keys():
            return True
        self.add_error(
            f"'time' must be a key in {self._name}.",
            f"{self._name} given has keys {self._signal_trajectory.keys()}",
        )
        return False

    def get_trajectory(self) -> Dict[str, NDArray[np.float64]]:
        signal_trajectory = dict()
        for key, value in self._signal_trajectory.items():
            signal_trajectory[key] = np.array(value, dtype=np.float64)
        return signal_trajectory


class BasePathValidator(Validator):
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        return Path(path).resolve()

    @staticmethod
    def _create_folder(foldername: Union[str, Path]) -> None:
        Path(foldername).mkdir(parents=True, exist_ok=True)
        print(f"folder: '{foldername}' is successfully created.\n")

    @staticmethod
    def _is_folder_creation(
        foldername: Union[str, Path], auto_create: bool = False
    ) -> bool:
        if auto_create:
            return True
        while True:
            response = (
                input(
                    f"\nfolder: '{foldername}' does not exist. "
                    "Would you like to create the folder? (y/n): "
                )
                .lower()
                .strip()
            )
            if response in ["y", "yes"]:
                return True
            if response in ["n", "no"]:
                return False
            print("Invalid input. Please enter 'y' or 'n'.")


class FolderPathExistenceValidator(BasePathValidator):
    def __init__(
        self, foldername: Union[str, Path], auto_create: bool = False
    ) -> None:
        super().__init__(foldername)
        self._foldername = self._resolve_path(foldername)
        self._auto_create = auto_create
        self.add_validation(self._validate_folderpath_existence)

    def _validate_folderpath_existence(self) -> bool:
        # Check if the foldername is a valid folderpath and exists
        if check_directory_existence(self._foldername):
            return True

        # Check if folder creation is True, create the folder and return True
        if self._is_folder_creation(self._foldername, self._auto_create):
            self._create_folder(self._foldername)
            return True

        # If the user does not want to create the folder, show an error message and return False
        self.add_error(
            "foldername must be a valid and existed folderpath (str or Path).",
            f"foldername provided is '{self._foldername}'.",
        )
        return False

    def get_folderpath(self) -> Path:
        return Path(self._foldername)


class FilePathExistenceValidator(BasePathValidator):
    def __init__(
        self, filename: Union[str, Path], extension: Union[str, Iterable[str]]
    ) -> None:
        super().__init__(filename)
        self._filename = self._resolve_path(filename)
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
            f"filename provided is '{self._filename}'.",
        )
        return False

    def get_filepath(self) -> Path:
        return Path(self._filename)


class FilePathValidator(BasePathValidator):
    def __init__(
        self,
        filename: Union[str, Path],
        extension: Union[str, Iterable[str]],
        auto_create_directory: bool = True,
    ) -> None:
        super().__init__(filename)
        self._filename = self._resolve_path(filename)
        self._extension = extension
        self._auto_create = auto_create_directory
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
        foldername = Path(self._filename).parent
        if self._is_folder_creation(foldername, self._auto_create):
            self._create_folder(foldername)
            return True

        # If the user does not want to create the folder, show an error message and return False
        self.add_error(
            f"folder: '{foldername}' does not exist. ",
            "filename must be a valid filepath (str or Path) "
            f"with the correct extension: '{self._extension}'. ",
            f"filename provided is '{self._filename}'.",
        )
        return False

    def get_filepath(self) -> Path:
        return Path(self._filename)
