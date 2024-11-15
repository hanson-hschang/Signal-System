import numpy as np
import pytest

from ss.system.state_vector.dynamic_system import (
    ContinuousTimeSystem,
    DiscreteTimeSystem,
)


class TestSystem:

    @pytest.fixture
    def continuous_time_system(self) -> ContinuousTimeSystem:
        """Create a basic system with default parameters"""
        return ContinuousTimeSystem(
            time_step=0.1,
            state_dim=2,
            observation_dim=1,
            # number_of_systems=4,
            number_of_systems=1,
        )

    @pytest.fixture
    def discrete_time_control_system(self) -> DiscreteTimeSystem:
        """Create a basic control_system with default parameters"""
        return DiscreteTimeSystem(
            state_dim=2,
            observation_dim=1,
            control_dim=1,
            process_noise_covariance=[[1, 0], [0, 1]],
            observation_noise_covariance=[[1]],
        )

    def test_continuous_time_system(
        self, continuous_time_system: ContinuousTimeSystem
    ) -> None:
        """Test the continuous time system"""
        observation = continuous_time_system.observe()

        # assert continuous_time_system.state.shape == (4, 2)
        # state = [
        #     [1.0, 2.0],
        #     [3.0, 4.0],
        #     [5.0, 6.0],
        #     [7.0, 8.0],
        # ]
        # assert observation.shape == (4, 1)

        assert observation.shape == (1,)
        assert continuous_time_system.state.shape == (2,)
        state = [1.0, 2.0]

        continuous_time_system.state = state
        assert np.allclose(continuous_time_system.state, np.array(state))
        time = continuous_time_system.process(0)
        assert time == 0.1
        with pytest.raises(AssertionError):
            continuous_time_system.state = [1, 2, 3]

    def test_discrete_time_control_system(
        self, discrete_time_control_system: DiscreteTimeSystem
    ) -> None:
        """Test the discrete time control system"""
        assert discrete_time_control_system.process_noise_covariance.shape == (
            2,
            2,
        )
        assert (
            discrete_time_control_system.observation_noise_covariance.shape
            == (1, 1)
        )
        control = [1.0]
        discrete_time_control_system.control = control
        assert np.allclose(
            discrete_time_control_system.control, np.array(control)
        )
        with pytest.raises(AssertionError):
            discrete_time_control_system.control = [1, 2, 3]
