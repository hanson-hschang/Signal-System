from typing import Tuple, Type, Union


def get_type_string(allowed_types: Union[Type, Tuple[Type, ...]]) -> str:
    # Convert type names to more readable format
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
        return f"{', '.join(type_names[:-1])}, and {type_names[-1]}"
