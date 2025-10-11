import torch
from numpy.typing import ArrayLike


class TensorReadOnlyDescriptor:
    def __init__(self, *name_of_dimensions: str) -> None:
        self._name_of_dimensions = list(name_of_dimensions)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: object, obj_type: type) -> torch.Tensor:
        value: torch.Tensor = getattr(obj, self.private_name)
        return value.detach()


class TensorDescriptor(TensorReadOnlyDescriptor):
    def __set__(self, obj: object, value: ArrayLike) -> None:
        _value: torch.Tensor = torch.tensor(value)
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        assert _value.shape == shape, (
            f"{self.name} must be in the shape of {shape}. "
            f"{self.name} given has the shape of {_value.shape}."
        )
        setattr(obj, self.private_name, _value)


class BatchTensorReadOnlyDescriptor:
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

    def __get__(self, obj: object, obj_type: type) -> torch.Tensor:
        value: torch.Tensor = getattr(obj, self.private_name)
        if getattr(obj, "_batch_size") == 1:
            value = value[0]
        return value.detach()


class BatchTensorDescriptor(BatchTensorReadOnlyDescriptor):
    def __set__(self, obj: object, value: ArrayLike) -> None:
        _value: torch.Tensor = (
            value if isinstance(value, torch.Tensor) else torch.tensor(value)
        )
        shape = tuple(getattr(obj, name) for name in self._name_of_dimensions)
        if (getattr(obj, "_batch_size") == 1) and (
            (len(shape) - _value.ndim) == 1
        ):
            _value = _value.unsqueeze(0)
        assert _value.shape == shape, (
            f"{self.name} must be in the shape of {shape}. "
            f"{self.name} given has the shape of {_value.shape}."
        )
        setattr(obj, self.private_name, _value)
