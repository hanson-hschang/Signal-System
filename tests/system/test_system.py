import numpy as np
import pytest

from system import ControlSystem, System


def test_system_initialization():
    state_dim = 2
    observation_dim = 2
    system = System(state_dim, observation_dim)
    assert system.state_dim == state_dim
    assert system.observation_dim == observation_dim
    assert np.allclose(system.state, np.zeros(state_dim))
    assert np.allclose(system.observation, np.zeros(observation_dim))
    system.state = [1.0, 2.0]
    assert np.allclose(system.state, np.array([1, 2]))
    with pytest.raises(AssertionError):
        system.state = [1, 2, 3]


def test_control_system_initialization():
    state_dim = 2
    control_dim = 1
    observation_dim = 2
    system = ControlSystem(state_dim, control_dim, observation_dim)
    assert system.state_dim == state_dim
    assert system.control_dim == control_dim
    assert system.observation_dim == observation_dim
    assert np.allclose(system.state, np.zeros(state_dim))
    assert np.allclose(system.control, np.zeros(control_dim))
    assert np.allclose(system.observation, np.zeros(observation_dim))
    system.state = [1.0, 2.0]
    assert np.allclose(system.state, np.array([1, 2]))
    with pytest.raises(AssertionError):
        system.state = [1, 2, 3]
