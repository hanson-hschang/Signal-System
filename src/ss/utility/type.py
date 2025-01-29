from typing import Tuple, Type, Union

import numpy as np


def get_type_string(allowed_types: Union[Type, Tuple[Type, ...]]) -> str:
    """
    Convert type names to more readable format

    Arguments:
    ----------
        allowed_types : Union[Type, Tuple[Type, ...]]
            A type or tuple of types

    Returns:
    --------
        allowed_types_str : str
            A more readable format of the input types
    """
    if isinstance(allowed_types, type):
        type_names = [allowed_types.__name__]
    else:
        type_names = [t.__name__ for t in allowed_types]

    if len(type_names) == 0:
        return ""
    elif len(type_names) == 1:
        return type_names[0]
    elif len(type_names) == 2:
        return f"{type_names[0]} and {type_names[1]}"
    else:
        return f"{', '.join(type_names[:-1])}, or {type_names[-1]}"


def from_numpy_generic(
    value: Union[np.bool_, np.integer, np.floating, np.character],
) -> Union[bool, int, float, str]:
    """
    Convert numpy scalar to Python scalar with proper type hinting.

    Arguments:
    ----------
        value: A numpy scalar value (np.bool_, np.integer, np.floating, np.character)

    Returns:
    --------
        A Python scalar (bool, int, float, or str)

    Raises:
    -------
        ValueError: If the numpy scalar type is not supported
    """
    if np.issubdtype(type(value), np.bool_):
        return bool(value)
    if np.issubdtype(type(value), np.integer):
        return int(value)
    if np.issubdtype(type(value), np.floating):
        return float(value)
    if np.issubdtype(type(value), np.character):
        return str(value)
    raise ValueError(f"Unsupported numpy scalar type: {type(value)}")
