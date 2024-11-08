import numpy as np
import pytest

from system import DynamicSystem, System


class TestSystem:

    @pytest.fixture
    def system(self) -> System:
        """Create a basic system with default parameters"""
        return System(
            state_dim=2,
            observation_dim=2,
            number_of_systems=3,
        )

    @pytest.fixture
    def control_system(self) -> System:
        """Create a basic control_system with default parameters"""
        return System(
            state_dim=2,
            observation_dim=2,
            control_dim=1,
        )

    def test_system_initialization(self, system: System) -> None:
        """Test the initialization of the system"""
        assert np.allclose(system.state, np.zeros(system.state_dim))
        assert np.allclose(
            system.observation,
            np.zeros((system.number_of_systems, system.observation_dim)),
        )
        state = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
        system.state = state
        assert np.allclose(system.state, np.array(state))
        with pytest.raises(AssertionError):
            system.state = [1, 2, 3]

    def test_control_system_initialization(
        self, control_system: System
    ) -> None:
        """Test the initialization of the system with control"""
        assert np.allclose(
            control_system.control, np.zeros(control_system.control_dim)
        )
        control = [1.0]
        control_system.control = control
        assert np.allclose(control_system.control, np.array(control))
        with pytest.raises(AssertionError):
            control_system.control = [1, 2, 3]


class TestDynamicSystem:

    @pytest.fixture
    def system(self) -> DynamicSystem:
        """Create a basic dynamic system with default parameters"""
        return DynamicSystem(
            state_dim=2, observation_dim=1, control_dim=1, time_step=0.1
        )

    def test_update_method(self, system: DynamicSystem) -> None:
        """Test the update method"""
        initial_time = 0.0
        new_time = system.update(initial_time)
        assert new_time == initial_time + system.time_step
        assert np.allclose(system.state, np.zeros(2))
