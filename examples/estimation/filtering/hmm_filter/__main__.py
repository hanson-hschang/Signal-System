"""
This script demonstrates the simulation of a discrete state dynamic system,
specifically a Hidden Markov Model (HMM), and the application of filtering
techniques to estimate the hidden states based on observations. The script
initializes an HMM with specified transition and emission matrices, simulates
a single rollout of the system, and then applies a filtering algorithm to
estimate the hidden states from the observed data.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ss.system.discrete import HiddenMarkovModel
from ss.system import simulate

from ss.estimation.filtering.hmm import HmmFilter
from ss.estimation.filtering import filtering, batch_filtering


if __name__ == "__main__":
    print(
        "=== Discrete State Dynamic System Simulation: Hidden Markov Model ==="
    )

    transition_matrix = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    emission_matrix = jnp.array([[0.9, 0.1], [0.2, 0.8]])

    system = HiddenMarkovModel(transition_matrix, emission_matrix)

    print(f"System: {system}")

    random_key = jax.random.PRNGKey(0)

    print("=== single rollout ===")
    time_horizon = 10

    initial_state = system.initial_state(random_key)
    random_keys = jax.random.split(random_key, time_horizon)

    times, states, observations, _ = simulate(
        system,
        0,
        initial_state,
        random_keys,
    )

    print(states)
    print(observations)
    print(observations.shape)

    filter = HmmFilter(transition_matrix, emission_matrix)

    print("=== filtering ===")
    print(f"filter: {filter}")

    initial_belief = jnp.array([0.5, 0.5])
    times, beliefs = filtering(filter, 0, initial_belief, observations)

    print("times:", times)
    print("beliefs:", beliefs)

    print("=== batch filtering ===")
    batch_size = 5
    systems = system.duplicate(batch_size=batch_size)

    initial_state_key, random_key = jax.random.split(random_key)
    initial_state_keys = jax.random.split(initial_state_key, batch_size)
    init_states = systems.initial_state(initial_state_keys)
    random_keys = jax.random.split(random_key, time_horizon)

    batch_times, batch_states, batch_observations, _ = simulate(
        systems, 0, init_states, random_keys
    )

    print("batch_times shape:", batch_times.shape)
    print("batch_states shape:", batch_states.shape)
    print("batch_observations shape:", batch_observations.shape)

    initial_beliefs = jnp.tile(jnp.array([0.5, 0.5]), (batch_size, 1))
    batch_times, batch_beliefs = batch_filtering(
        filter, 0, initial_beliefs, batch_observations
    )

    print("batch_times shape:", batch_times.shape)
    print("batch_beliefs shape:", batch_beliefs.shape)
