import numpy as np
from numpy.typing import ArrayLike, NDArray


class NDArrayReadOnlyDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray:
        value: NDArray = getattr(obj, self.private_name)
        return value.copy()


class NDArrayDescriptor(NDArrayReadOnlyDescriptor):
    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        assert value.shape == shape, (
            f"{self.name} must be in the shape of {shape}. "
            f"{self.name} given has the shape of {value.shape}."
        )
        setattr(obj, self.private_name, value)


class BatchNDArrayReadOnlyDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)
        assert (n_dim := len(self._name_of_dimensions)) > 1, (
            "The number of dimensions must be greater than 1. "
            f"The number of dimensions given is {n_dim} "
            f"with the name of dimensions as {self._name_of_dimensions}"
        )
        assert (batch_size := self._name_of_dimensions[0]) == "_batch_size", (
            "The name of the first dimension must be '_batch_size'. "
            f"The name of the first dimension given is {batch_size}."
        )

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> NDArray:
        value: NDArray = getattr(obj, self.private_name)
        if getattr(obj, "_batch_size") == 1:
            value = value[0]
        return value


class BatchNDArrayDescriptor(BatchNDArrayReadOnlyDescriptor):
    def __set__(self, obj: object, value: ArrayLike) -> None:
        value = np.array(value)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        if (getattr(obj, "_batch_size") == 1) and (
            (len(shape) - value.ndim) == 1
        ):
            value = value[np.newaxis, ...]
        assert value.shape == shape, (
            f"{self.name} must be in the shape of {shape}. "
            f"{self.name} given has the shape of {value.shape}."
        )
        setattr(obj, self.private_name, value)
