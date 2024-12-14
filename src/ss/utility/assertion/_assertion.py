from typing import Iterable, Optional, Union

from pathlib import Path


def is_number(number: Union[int, float]) -> bool:
    return isinstance(number, (int, float))


def is_positive_number(number: Union[int, float]) -> bool:
    if isinstance(number, (int, float)):
        return number > 0
    return False


def is_nonnegative_number(number: Union[int, float]) -> bool:
    if isinstance(number, (int, float)):
        return number >= 0
    return False


def is_integer(number: Union[int, float]) -> bool:
    if isinstance(number, (int, float)):
        return int(number) == number
    return False


def is_positive_integer(number: Union[int, float]) -> bool:
    if isinstance(number, (int, float)):
        return number > 0 and int(number) == number
    return False


def is_nonnegative_integer(number: Union[int, float]) -> bool:
    if isinstance(number, (int, float)):
        return number >= 0 and int(number) == number
    return False


def check_parent_directory_existence(filename: Union[str, Path]) -> bool:
    if not isinstance(filename, (str, Path)):
        return False
    directory = Path(filename).parent
    if not directory.is_dir():
        return False
    return directory.exists()


def is_extension_valid(extension: Union[str, Iterable[str]]) -> bool:
    if isinstance(extension, str):
        return extension.startswith(".")
    if isinstance(extension, Iterable):
        return all([ext.startswith(".") for ext in extension])
    return False


def is_filepath_valid(
    filename: Union[str, Path],
    extension: Optional[Union[str, Iterable[str]]] = None,
) -> bool:
    if not isinstance(filename, (str, Path)):
        return False
    filepath = Path(filename)
    if not check_parent_directory_existence(filepath):
        return False
    if extension is not None:
        extension = (
            extension if isinstance(extension, Iterable) else (extension,)
        )
        return filepath.suffix in extension
    return filepath.is_file()


def check_filepath_existence(
    filename: Union[str, Path],
    extension: Optional[Union[str, Iterable[str]]] = None,
) -> bool:
    if not is_filepath_valid(filename, extension):
        return False
    return Path(filename).exists()
