"""
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ss.estimation.filtering import Filter
from ss.utility.parameter.probability import ProbabilityParameter

class HmmFilter(Filter):
    _transition_matrix: ProbabilityParameter
    _emission_matrix: ProbabilityParameter
    discrete_state_dim: int = eqx.field(static=True)
    discrete_observation_dim: int = eqx.field(static=True)

    def __init__(
        self,
        transition_matrix: Float[Array, "state_dim state_dim"],
        emission_matrix: Float[Array, "state_dim observation_dim"],
    ) -> None:
        self._transition_matrix = ProbabilityParameter(
            jnp.asarray(transition_matrix),
        )
        self._emission_matrix = ProbabilityParameter(
            jnp.asarray(emission_matrix),
        )
        self.discrete_state_dim = self.transition_matrix.shape[0]
        self.discrete_observation_dim = self.emission_matrix.shape[1]
        super().__init__(
            state_dim=self.discrete_state_dim,
            observation_dim=1,
            control_dim=0,
            time_step=1.0,
        )

    @property
    def transition_matrix(self) -> Float[Array, "state_dim state_dim"]:
        return self._transition_matrix.value()

    @property
    def emission_matrix(self) -> Float[Array, "state_dim observation_dim"]:
        return self._emission_matrix.value()

    def with_transition_matrix(
            self, transition_matrix: Array,
        ) -> "HmmFilter":
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
        ) -> "HmmFilter":
        assert emission_matrix.shape == self.emission_matrix.shape, (
            f"emission_matrix must have shape {self.emission_matrix.shape}, "
            f"got {emission_matrix.shape}"
        )
        return eqx.tree_at(
            update_emission_matrix,
            self,
            ProbabilityParameter(jnp.asarray(emission_matrix))
        )

    def update(
        self,
        time: Float,
        prior: Float[Array, "state_dim"], # noqa: F821
        observation: Int[Array, "observation_dim"], # noqa: F821
    ) -> tuple[Float, Float[Array, "state_dim"]]: # noqa: F821
        """Update the belief state given a new observation."""
        likelihood = self.emission_matrix[:, observation[0]]
        # update step (unnormalized posterior given the new observation)
        updated = prior * likelihood
        # normalize -> conditional probability
        posterior = updated / jnp.sum(updated)
        # predict step (Chapman-Kolmogorov)
        belief = posterior @ self.transition_matrix
        return time + self.time_step, belief

def update_transition_matrix(
        filter: "HmmFilter",
    ) -> "ProbabilityParameter":
    return filter._transition_matrix

def update_emission_matrix(
        filter: "HmmFilter",
    ) -> "ProbabilityParameter":
    return filter._emission_matrix
