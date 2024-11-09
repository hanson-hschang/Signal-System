from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TensorDescriptor:
    def __init__(
        self, *name_of_dimensions: str, squeeze_first_dimension: bool = False
    ) -> None:
        self._name_of_dimensions = list(name_of_dimensions)
        self._squeeze_first_dimension = squeeze_first_dimension

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray[np.float64]:
        value: NDArray = getattr(obj, self.private_name)
        if self._squeeze_first_dimension and value.shape[0] == 1:
            value = value[0]
        return value

    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value, dtype=np.float64)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        if (
            self._squeeze_first_dimension
            and shape[0] == 1
            and (len(value.shape) == len(shape) - 1)
        ):
            value = value[np.newaxis, :]
        assert (
            value.shape == shape
        ), f"input shape {value.shape} does not match with {self.name} shape {shape}."
        setattr(obj, self.private_name, value)


T = TypeVar("T")


class ReadOnlyDescriptor(Generic[T]):
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> T:
        value: T = getattr(obj, self.private_name)
        return value
