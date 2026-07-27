"""
Parameter classes for handling constrained parameters in JAX.
"""

from __future__ import annotations

from typing import Protocol, Generic, TypeVar

import equinox as eqx
from jaxtyping import Array


class Transformer(Protocol):
    def forward(self, raw_value: Array) -> Array: ...
    def inverse(self, value: Array) -> Array: ...


T = TypeVar("T", bound=Transformer)


class Parameter(eqx.Module, Generic[T]):
    """
    A Parameter is a wrapper around a raw tensor that applies a Transformer
    to map between the raw tensor and a constrained value. The raw tensor is
    stored as a pytree leaf, so it can be learned. The Parameter class is
    immutable, so "setting a value" means constructing a new Parameter with
    the new value, rather than mutating the raw tensor in place.
    """

    _tensor: Array
    _transformer: T

    def __init__(self, value: Array, transformer: T) -> None:
        self._transformer = transformer
        self._tensor = self._transformer.inverse(value)

    def value(self) -> Array:
        return self._transformer.forward(self._tensor)

    def from_value(self, value: Array) -> "Parameter[T]":
        return Parameter[T](value, self._transformer)
