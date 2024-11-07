import numpy as np
import pytest

from system import DynamicSystem, System


def test_system_initialization():
    state_dim = 2
    observation_dim = 2
    system = System(state_dim, observation_dim)
    assert system.state_dim == state_dim
    assert system.observation_dim == observation_dim
    assert np.all(system.state == np.zeros(state_dim))
    assert np.all(system.observation == np.zeros(observation_dim))
    system.state = [1.0, 2.0]
    assert np.all(system.state == np.array([1, 2]))
    with pytest.raises(AssertionError):
        system.state = [1, 2, 3]


def test_dynamic_system_initialization():
    state_dim = 2
    observation_dim = 2
    system = DynamicSystem(state_dim, observation_dim)
    assert system.state_dim == state_dim
    assert system.observation_dim == observation_dim
    assert np.all(system.state == np.zeros(state_dim))
    assert np.all(system.observation == np.zeros(observation_dim))
    assert system.time_step == 1
    system.state = [1.0, 2.0]
    assert np.all(system.state == np.array([1, 2]))
    with pytest.raises(AssertionError):
        system.state = [1, 2, 3]
    system.update()
