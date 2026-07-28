"""
Parameter classes for handling constrained parameters in JAX.
"""

from __future__ import annotations

from typing import Protocol, Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Float


class Transformer(Protocol):
    def forward(
        self, raw_value: Float[Array, "..."]
    ) -> Float[Array, "..."]: ...
    def inverse(self, value: Float[Array, "..."]) -> Float[Array, "..."]: ...


T = TypeVar("T", bound=Transformer)


class Parameter(eqx.Module, Generic[T]):
    """
    A Parameter is a wrapper around a raw tensor that applies a Transformer
    to map between the raw tensor and a constrained value. The raw tensor is
    stored as a pytree leaf, so it can be learned. The Parameter class is
    immutable, so "setting a value" means constructing a new Parameter with
    the new value, rather than mutating the raw tensor in place.
    """

    _tensor: Float[Array, "..."]
    _transformer: T

    def __init__(self, value: Float[Array, "..."], transformer: T):
        self._transformer = transformer
        self._tensor = self._transformer.inverse(value)

    def value(self) -> Float[Array, "..."]:
        return self._transformer.forward(self._tensor)

    def from_value(self, value: Float[Array, "..."]) -> "Parameter[T]":
        return Parameter[T](value, self._transformer)
