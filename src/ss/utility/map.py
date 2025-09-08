from typing import Callable, TypeVar

import torch
from numpy.typing import NDArray

V = TypeVar("V", torch.Tensor, NDArray)


def transform(
    value: V,
    transformation_function: Callable[[V], V],
) -> V:
    return transformation_function(value)
