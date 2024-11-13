import numpy as np
import pytest
from numba import njit
from numpy.typing import NDArray

from ss.system.dense_state.nonlinear import (
    ContinuousTimeNonlinearSystem,
    DiscreteTimeNonlinearSystem,
)


class TestNonlinearSystem:

    @pytest.fixture
    def nonlinear_control_system(self) -> ContinuousTimeNonlinearSystem:
        """Create a basic nonlinear system with control input"""

        @njit(cache=True)  # type: ignore
        def process_function(
            state: NDArray[np.float64], control: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            process = np.zeros_like(state)
            process[:, 0] = state[:, 1] * state[:, 1]
            process[:, 1] = -state[:, 0] + control[:, 0]
            return process

        @njit(cache=True)  # type: ignore
        def observation_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return state[:, 0:1]

        @njit(cache=True)  # type: ignore
        def state_constraint_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            state[:, 1] = (state[:, 1] + np.pi) % (2 * np.pi) - np.pi
            return state

        return ContinuousTimeNonlinearSystem(
            time_step=0.1,
            state_dim=2,
            observation_dim=1,
            control_dim=1,
            process_function=process_function,
            observation_function=observation_function,
            state_constraint_function=state_constraint_function,
        )

    @pytest.fixture
    def stochastic_nonlinear_system(self) -> ContinuousTimeNonlinearSystem:
        """Create a basic stochastic nonlinear system with default parameters"""

        @njit(cache=True)  # type: ignore
        def process_function(state: NDArray[np.float64]) -> NDArray[np.float64]:
            process = np.zeros_like(state)
            process[:, 0] = state[:, 1] * state[:, 1]
            process[:, 1] = -state[:, 0]
            return process

        @njit(cache=True)  # type: ignore
        def observation_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return state[:, 0:1]

        return ContinuousTimeNonlinearSystem(
            time_step=0.1,
            state_dim=2,
            observation_dim=1,
            process_function=process_function,
            observation_function=observation_function,
            process_noise_covariance=[[1, 0], [0, 1]],
            observation_noise_covariance=[[1]],
        )

    @pytest.fixture
    def stochastic_discrete_time_nonlinear_system(
        self,
    ) -> DiscreteTimeNonlinearSystem:
        """Create a basic stochastic discrete time nonlinear system with default parameters"""

        @njit(cache=True)  # type: ignore
        def process_function(state: NDArray[np.float64]) -> NDArray[np.float64]:
            process = np.zeros_like(state)
            process[:, 0] = state[:, 1] * state[:, 1]
            process[:, 1] = -state[:, 0]
            return process

        @njit(cache=True)  # type: ignore
        def observation_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return state[:, 0:1]

        return DiscreteTimeNonlinearSystem(
            state_dim=2,
            observation_dim=1,
            process_function=process_function,
            observation_function=observation_function,
            process_noise_covariance=[[1, 0], [0, 1]],
            observation_noise_covariance=[[1]],
        )

    @pytest.fixture
    def discrete_time_nonlinear_control_system(
        self,
    ) -> DiscreteTimeNonlinearSystem:
        """Create a basic discrete time nonlinear system with control input"""

        @njit(cache=True)  # type: ignore
        def process_function(
            state: NDArray[np.float64], control: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            process = np.zeros_like(state)
            process[:, 0] = state[:, 1] * state[:, 1]
            process[:, 1] = -state[:, 0] + control[:, 0]
            return process

        @njit(cache=True)  # type: ignore
        def observation_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return state[:, 0:1]

        return DiscreteTimeNonlinearSystem(
            state_dim=2,
            observation_dim=1,
            control_dim=1,
            process_function=process_function,
            observation_function=observation_function,
        )

    def test_nonlinear_system_with_control(
        self, nonlinear_control_system: ContinuousTimeNonlinearSystem
    ) -> None:
        """Test the nonlinear system with control input"""
        observation = nonlinear_control_system.observe()

        assert observation.shape == (1,)
        assert nonlinear_control_system.state.shape == (2,)
        state = [1.0, 2.0]
        nonlinear_control_system.state = state
        assert np.allclose(nonlinear_control_system.state, np.array(state))
        nonlinear_control_system.control = [1]
        time = nonlinear_control_system.process(0)
        assert time == 0.1
        with pytest.raises(AssertionError):
            nonlinear_control_system.state = [1, 2, 3]

    def test_stochastic_nonlinear_system(
        self, stochastic_nonlinear_system: ContinuousTimeNonlinearSystem
    ) -> None:
        """Test the nonlinear system"""
        observation = stochastic_nonlinear_system.observe()

        assert observation.shape == (1,)
        assert stochastic_nonlinear_system.state.shape == (2,)
        state = [1.0, 2.0]

        stochastic_nonlinear_system.state = state
        assert np.allclose(stochastic_nonlinear_system.state, np.array(state))
        time = stochastic_nonlinear_system.process(0)
        assert time == 0.1
        with pytest.raises(AssertionError):
            stochastic_nonlinear_system.state = [1, 2, 3]

    def test_stochastic_discrete_time_nonlinear_system(
        self,
        stochastic_discrete_time_nonlinear_system: DiscreteTimeNonlinearSystem,
    ) -> None:
        """Test the discrete time nonlinear system"""
        observation = stochastic_discrete_time_nonlinear_system.observe()

        assert observation.shape == (1,)
        assert stochastic_discrete_time_nonlinear_system.state.shape == (2,)
        state = [1.0, 2.0]

        stochastic_discrete_time_nonlinear_system.state = state
        assert np.allclose(
            stochastic_discrete_time_nonlinear_system.state, np.array(state)
        )
        time = stochastic_discrete_time_nonlinear_system.process(0)
        assert time == 1
        with pytest.raises(AssertionError):
            stochastic_discrete_time_nonlinear_system.state = [1, 2, 3]

    def test_discrete_time_nonlinear_control_system(
        self,
        discrete_time_nonlinear_control_system: DiscreteTimeNonlinearSystem,
    ) -> None:
        """Test the discrete time nonlinear system with control input"""
        observation = discrete_time_nonlinear_control_system.observe()

        assert observation.shape == (1,)
        assert discrete_time_nonlinear_control_system.state.shape == (2,)
        state = [1.0, 2.0]
        discrete_time_nonlinear_control_system.state = state
        assert np.allclose(
            discrete_time_nonlinear_control_system.state, np.array(state)
        )
        discrete_time_nonlinear_control_system.control = [1]
        time = discrete_time_nonlinear_control_system.process(0)
        assert time == 1
        assert np.allclose(
            discrete_time_nonlinear_control_system.state, np.array([4, 0])
        )
        with pytest.raises(AssertionError):
            discrete_time_nonlinear_control_system.state = [1, 2, 3]
