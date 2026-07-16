"""
A discrete-state dynamical system, such as a Hidden Markov Model (HMM).
The state and observation are both discrete, and the system is defined
by a transition matrix and an emission matrix. The transition matrix defines
the probabilities of moving from one state to another, while the emission
matrix defines the probabilities of observing a particular observation given
the current state.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from ss.utility.parameter.probability import ProbabilityParameter

from ._system import System


class HiddenMarkovModel(System):
    _transition_matrix: ProbabilityParameter
    _emission_matrix: ProbabilityParameter
    discrete_state_dim: int = eqx.field(static=True)
    discrete_observation_dim: int = eqx.field(static=True)

    def __init__(
        self,
        transition_matrix: Array,
        emission_matrix: Array,
    ) -> None:
        self._transition_matrix = ProbabilityParameter(
            jnp.asarray(transition_matrix),
        )
        self._emission_matrix = ProbabilityParameter(
            jnp.asarray(emission_matrix),
        )
        self.discrete_state_dim = transition_matrix.shape[0]
        self.discrete_observation_dim = emission_matrix.shape[1]
        super().__init__(
            state_dim=1,
            observation_dim=1,
            control_dim=0,
            time_step=1.0,
        )

    @property
    def transition_matrix(self) -> Array:
        return self._transition_matrix.value()

    @property
    def emission_matrix(self) -> Array:
        return self._emission_matrix.value()

    def with_transition_matrix(
            self, transition_matrix: Array,
        ) -> "HiddenMarkovModel":
        assert transition_matrix.shape == self.transition_matrix.shape, (
            f"transition_matrix must have shape {self.transition_matrix.shape}, "
            f"got {transition_matrix.shape}"
        )
        return eqx.tree_at(
            update_transition_matrix,
            self,
            ProbabilityParameter(jnp.asarray(transition_matrix))
        )

    def with_emission_matrix(
            self, emission_matrix: Array,
        ) -> "HiddenMarkovModel":
        assert emission_matrix.shape == self.emission_matrix.shape, (
            f"emission_matrix must have shape {self.emission_matrix.shape}, "
            f"got {emission_matrix.shape}"
        )
        return eqx.tree_at(
            update_emission_matrix,
            self,
            ProbabilityParameter(jnp.asarray(emission_matrix))
        )

    def __check_init__(self) -> None:
        super().__check_init__()
        transition_matrix_shape = self.transition_matrix.shape
        assert transition_matrix_shape[0] == transition_matrix_shape[1] == self.discrete_state_dim, (
            f"transition_matrix must be square ({self.discrete_state_dim}, "
            f"{self.discrete_state_dim}), got {transition_matrix_shape}"
        )
        assert jnp.allclose(self.transition_matrix.sum(axis=1), 1.0), (
            "transition_matrix rows must sum to 1"
        )
        emission_matrix_shape = self.emission_matrix.shape
        assert emission_matrix_shape[0] == self.discrete_state_dim, (
            f"emission_matrix must have {self.discrete_state_dim} rows, "
            f"got shape {emission_matrix_shape}"
        )
        assert emission_matrix_shape[1] == self.discrete_observation_dim, (
            f"emission_matrix must have {self.discrete_observation_dim} columns, "
            f"got shape {emission_matrix_shape}"
        )
        assert jnp.allclose(self.emission_matrix.sum(axis=1), 1.0), (
            "emission_matrix rows must sum to 1"
        )

    def init_state(
        self, key: PRNGKeyArray, initial_distribution: Array | None = None
    ) -> Int[Array, ""]:
        if initial_distribution is None:
            initial_distribution = (
                jnp.ones(self.discrete_state_dim) / self.discrete_state_dim
            )
        return jax.random.categorical(key, jnp.log(initial_distribution))

    def process(
        self,
        time: float,
        state: Int[Array, ""],
        random_key: PRNGKeyArray,
        control: Array | None = None,
    ) -> tuple[float, Int[Array, ""]]:
        return time + self.time_step, jax.random.categorical(
            random_key, jnp.log(self.transition_matrix[state]),
        )

    def observe(
        self,
        time: float,
        state: Int[Array, ""],
        random_key: PRNGKeyArray,
    ) -> Int[Array, ""]:
        return jax.random.categorical(random_key, jnp.log(self.emission_matrix[state]))

    def state_one_hot(self, state: Int[Array, ""]) -> Array:
        return jax.nn.one_hot(state, self.discrete_state_dim)

    def observation_one_hot(self, observation: Int[Array, ""]) -> Array:
        return jax.nn.one_hot(observation, self.discrete_observation_dim)


def update_transition_matrix(
        model: "HiddenMarkovModel",
    ) -> "ProbabilityParameter":
    return model._transition_matrix

def update_emission_matrix(
        model: "HiddenMarkovModel",
    ) -> "ProbabilityParameter":
    return model._emission_matrix
