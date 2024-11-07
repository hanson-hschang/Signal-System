import numpy as np
import pytest

from system import DynamicMixin, System


class TestSystem:
    state_dim = 2
    observation_dim = 2
    control_dim = 1
    system = System(state_dim, observation_dim)
    control_system = System(state_dim, observation_dim, control_dim)

    def test_system_initialization(self):
        assert self.system.state_dim == self.state_dim
        assert self.system.observation_dim == self.observation_dim
        assert np.allclose(self.system.state, np.zeros(self.state_dim))
        assert np.allclose(
            self.system.observation, np.zeros(self.observation_dim)
        )
        self.system.state = [1.0, 2.0]
        assert np.allclose(self.system.state, np.array([1, 2]))
        with pytest.raises(AssertionError):
            self.system.state = [1, 2, 3]

    def test_control_system_initialization(self):
        assert self.control_system.state_dim == self.state_dim
        assert self.control_system.observation_dim == self.observation_dim
        assert self.control_system.control_dim == self.control_dim
        assert np.allclose(self.control_system.state, np.zeros(self.state_dim))
        assert np.allclose(
            self.control_system.observation, np.zeros(self.observation_dim)
        )
        assert np.allclose(
            self.control_system.control, np.zeros(self.control_dim)
        )
        self.control_system.control = [1.0]
        assert np.allclose(self.control_system.control, np.array([1]))
        with pytest.raises(AssertionError):
            self.control_system.control = [1, 2, 3]


class TestDynamicMixin:
    def test_dynamic_mixin_initialization(self):
        dynamic_mixin = DynamicMixin(0.5)
        assert dynamic_mixin.time_step == 0.5
        with pytest.raises(AssertionError):
            dynamic_mixin = DynamicMixin(0)
        with pytest.raises(AssertionError):
            dynamic_mixin = DynamicMixin(-1)
        dynamic_mixin.update()
