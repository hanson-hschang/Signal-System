from typing import Union


def isPositiveNumber(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number > 0


def isNonNegativeNumber(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number >= 0


def isPositiveInteger(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number > 0 and int(number) == number


def isNonNegativeInteger(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number >= 0 and int(number) == number
