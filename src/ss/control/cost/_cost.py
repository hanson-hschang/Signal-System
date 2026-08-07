# ruff: noqa: F722

"""Cost protocols for controllers and examples."""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float


@runtime_checkable
class Cost(Protocol):
    """Minimal cost interface used by MPPI and example post-processing."""

    def running_cost(
        self,
        state: Float[Array, "*batch state_dim"],
        control: Float[Array, "*batch control_dim"],
    ) -> Float[Array, "*batch"]:
        """Instantaneous running cost."""

    def terminal_cost(
        self,
        state: Float[Array, "*batch state_dim"],
    ) -> Float[Array, "*batch"]:
        """Terminal cost at the end of a horizon or trajectory."""
