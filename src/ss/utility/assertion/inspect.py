from types import FrameType, NoneType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_type_hints,
)

import inspect
from dataclasses import fields, is_dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

_array_like_types: tuple = (
    ArrayLike,
    NDArray,
    NDArray,
    List,
    Tuple,
    np.ndarray,
    list,
    tuple,
    np.ndarray[Any, np.dtype[np.float64]],
)


# Default Python types including common collections
_default_python_types: Set = {
    str,
    int,
    float,
    bool,
    bytes,
    list,
    dict,
    set,
    tuple,
    List,
    Set,
    Dict,
    Tuple,
    Tuple[int, ...],
    NoneType,
}


def inspect_arguments(
    func: Callable,
    arg_name_shape_dict: Dict[str, Tuple],
    result_shape: Tuple,
) -> Callable:
    signature = inspect.signature(func)
    arg_dict = {}
    for arg_name in arg_name_shape_dict.keys():
        assert (
            arg_name in signature.parameters
        ), f"{arg_name} should be an argument for {func.__name__}"
        param = signature.parameters[arg_name]
        assert param.annotation in _array_like_types, (
            f"{arg_name} should be of type ArrayLike "
            f"but is of type {param.annotation}"
        )
        arg_dict[arg_name] = np.zeros(arg_name_shape_dict[arg_name])
    try:
        result: NDArray = func(**arg_dict)
    except TypeError as e:
        raise AssertionError(e)
    assert (
        result.shape == result_shape
    ), f"Function {func.__name__} does not return an array of shape {result_shape}"
    return func


def get_nondefault_type_fields(
    cls: type,
) -> Dict[str, type]:
    """
    Returns a dictionary of non-default types found in the dataclass's fields.
    The keys are the field names and the values are the non-default types.

    Parameters
    ----------
    cls : type
        The dataclass to analyze.

    Returns
    -------
    nondefault_type_parameters : Dict[str, type]
        A dictionary of non-default types found in the dataclass's fields.

    Raises
    ------
    ValueError
        If the input is not a dataclass
    """
    nondefault_type_parameters: Dict[str, type] = {}
    if not is_dataclass(cls):
        raise ValueError("Argument cls must be a dataclass")

    type_hints: Dict[str, type] = get_type_hints(cls)

    for field_name, field_type in type_hints.items():
        # Handle Union types (including Optional)
        if hasattr(field_type, "__origin__"):
            field_type_origin = getattr(field_type, "__origin__")
            if field_type_origin is Union:
                field_types: Set[type] = set(getattr(field_type, "__args__"))
                nondefault_types = field_types - _default_python_types
                for i, nondefault_type in enumerate(nondefault_types):
                    nondefault_type_parameters[field_name + f"({i})"] = (
                        nondefault_type
                    )
        # Handle regular types
        elif field_type not in _default_python_types and hasattr(
            field_type, "__qualname__"
        ):
            nondefault_type_parameters[field_name] = field_type

    return nondefault_type_parameters


def is_valid_parentheses(string: str) -> bool:
    """
    Check if the parentheses in a string are valid.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_valid : bool
        True if the parentheses in the string are valid, False otherwise.
    """

    stack = []

    for char in string:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:  # Stack is empty but we found closing bracket
                return False
            stack.pop()

    return len(stack) == 0  # True if stack is empty (all brackets matched)


def resolve_call_line(caller_frame: FrameType) -> str:
    """
    Resolve the source line of the function call.

    Parameters
    ----------
    caller_frame : FrameType
        The frame of the caller.

    Returns
    -------
    call_line : str
        The source line of the function call.
    """

    context = 1
    context_complete = False

    while not context_complete:

        # Get the code context of the caller
        code_context: List[str] = getattr(
            inspect.getframeinfo(caller_frame, context=context),
            "code_context",
        )

        code_context_string = ""
        for i in range(context // 2, len(code_context)):
            code_context_string = code_context_string + code_context[i].strip()

        context_complete = is_valid_parentheses(code_context_string)

        context += 2

    # Get the source line of the function call
    call_line = code_context_string.strip()

    return call_line


def get_call_line(frame: Optional[FrameType]) -> str:
    """
    Get the source line of the function call.

    Parameters
    ----------
    frame : Optional[FrameType]
        The frame of the caller.

    Returns
    -------
    call_line : str
        The source line of the function call.
    """

    # Get the frame of the caller
    caller_frame: FrameType = getattr(frame, "f_back")

    # Get the source line of the function call
    call_line = resolve_call_line(caller_frame)

    if "super()" in call_line:
        return get_call_line(caller_frame)
    return call_line
