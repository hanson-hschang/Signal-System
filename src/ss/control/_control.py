from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


class Controller(eqx.Module):
    """Base class for batched stateful controllers."""

    control_dim: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.control_dim > 0, "control_dim must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"

    def init_state(self):
        return ()

    @abstractmethod
    def __call__(
        self,
        controller_state,
        time: Array,
        observation: Array,
        random_key: PRNGKeyArray,
    ):
        raise NotImplementedError
