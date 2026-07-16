"""
A discrete-state dynamical system, such as a Hidden Markov Model (HMM).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ss.system.discrete import HiddenMarkovModel
from ss.system import simulate, batch_simulate

if __name__ == "__main__":


    print("=== Discrete State Dynamic System Simulation: Hidden Markov Model ===")

    system = HiddenMarkovModel(
        transition_matrix=jnp.array([[0.7, 0.3], [0.4, 0.6]]),
        emission_matrix=jnp.array([[0.8, 0.2], [0.2, 0.8]]),
    )

    print("transition_matrix:\n", system.transition_matrix)
    print("emission_matrix:\n", system.emission_matrix)

    system = system.with_transition_matrix(
        jnp.array([[0.6, 0.4], [0.5, 0.5]])
    )
    print("updated transition_matrix:\n", system.transition_matrix)

    random_key = jax.random.PRNGKey(0)

    print("=== single rollout ===")
    time_horizon = 100

    initial_state = jnp.array(0, dtype=jnp.int32)
    random_keys = jax.random.split(random_key, time_horizon)

    times, states, observations, _ = simulate(
        system, 0, initial_state, random_keys,
    )

    print("times:", times.shape)
    print("states:", states.shape)
    print("observations:", observations.shape)


    print("=== batched rollout ===")
    batch_size = 15

    init_states = jnp.tile(jnp.array(0, dtype=jnp.int32), (batch_size, 1))
    random_keys = jax.random.split(random_key, (batch_size, time_horizon))

    batch_times, batch_states, batch_observations, _ = batch_simulate(
        system, 0, init_states, random_keys
    )

    print("batch_times shape:", batch_times.shape)
    print("batch_states shape:", batch_states.shape)
    print("batch_observations shape:", batch_observations.shape)
