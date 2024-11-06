from typing import Union
import numpy as np


def isPositiveInteger(number: Union[int, float]) -> bool:
    if not isinstance(number, (int, float)):
        return False
    return number >= 1 and int(number) == number
