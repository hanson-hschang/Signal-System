from typing import Union


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
