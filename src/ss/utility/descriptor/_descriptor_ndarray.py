import numpy as np
from numpy.typing import ArrayLike, NDArray


class NDArrayDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray:
        value: NDArray = getattr(obj, self.private_name)
        return value.copy()

    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        assert value.shape == shape, (
            f"{self.name} must be in the shape of {shape}. "
            f"{self.name} given has the shape of {value.shape}."
        )
        setattr(obj, self.private_name, value)


class MultiSystemNDArrayReadOnlyDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)
        assert len(self._name_of_dimensions) > 1, (
            "The number of dimensions must be greater than 1. "
            f"The number of dimensions given is {len(self._name_of_dimensions)} "
            f"with the name of dimensions as {self._name_of_dimensions}"
        )
        assert self._name_of_dimensions[0] == "_number_of_systems", (
            "The name of the first dimension must be '_number_of_systems'. "
            f"The name of the first dimension given is {self._name_of_dimensions[0]}."
        )

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray:
        value: NDArray = getattr(obj, self.private_name)
        if getattr(obj, "_number_of_systems") == 1:
            value = value[0]
        return value


class MultiSystemNDArrayDescriptor(MultiSystemNDArrayReadOnlyDescriptor):
    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        if (getattr(obj, "_number_of_systems") == 1) and (
            (len(shape) - value.ndim) == 1
        ):
            value = value[np.newaxis, ...]
        assert value.shape == shape, (
            f"{self.name} must be in the shape of {shape}. "
            f"{self.name} given has the shape of {value.shape}."
        )
        setattr(obj, self.private_name, value)
