from typing import Union


def is_positive_number(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number > 0


def is_nonnegative_number(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number >= 0


def is_positive_integer(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number > 0 and int(number) == number


def is_nonnegative_integer(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number >= 0 and int(number) == number
