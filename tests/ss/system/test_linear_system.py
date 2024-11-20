import numpy as np
import pytest

from ss.system.state_vector.linear import DiscreteTimeLinearSystem


class TestDiscreteTimeLinearSystem:

    @pytest.fixture
    def control_system(self) -> DiscreteTimeLinearSystem:
        """Create a basic control_system with default parameters"""
        return DiscreteTimeLinearSystem(
            state_space_matrix_A=np.array([[1, 1], [0, 1]]),
            state_space_matrix_B=np.array([[0], [1]]),
            state_space_matrix_C=np.array([[1, 0], [1, 1]]),
        )

    @pytest.fixture
    def stochastic_system(self) -> DiscreteTimeLinearSystem:
        """Create a basic stochastic_system with default parameters"""
        return DiscreteTimeLinearSystem(
            state_space_matrix_A=np.array([[1]]),
            state_space_matrix_C=np.array([[1]]),
            process_noise_covariance=np.array([[1]]),
            observation_noise_covariance=np.array([[1]]),
        )

    def test_wrong_initialization(self) -> None:

        with pytest.raises(AssertionError):
            DiscreteTimeLinearSystem(
                state_space_matrix_A=np.array([[1, 1], [0, 1], [0, 0]]),
                state_space_matrix_C=np.array([[1, 0], [1, 1]]),
            )

        with pytest.raises(AssertionError):
            DiscreteTimeLinearSystem(
                state_space_matrix_A=np.array([[1, 1], [0, 1]]),
                state_space_matrix_C=np.array([[1], [1]]),
            )

        with pytest.raises(AssertionError):
            DiscreteTimeLinearSystem(
                state_space_matrix_A=np.array([[1, 1], [0, 1]]),
                state_space_matrix_B=np.array([[0, 1]]),
                state_space_matrix_C=np.array([[1, 0], [1, 1]]),
            )

        with pytest.raises(AssertionError):
            DiscreteTimeLinearSystem(
                state_space_matrix_A=np.array([[1]]),
                state_space_matrix_C=np.array([[1]]),
                process_noise_covariance=np.array([[1, 2], [3, 4]]),
            )

        with pytest.raises(AssertionError):
            DiscreteTimeLinearSystem(
                state_space_matrix_A=np.array([[1]]),
                state_space_matrix_C=np.array([[1]]),
                observation_noise_covariance=np.array([[1, 2], [3, 4]]),
            )

    def test_control_system_initialization(
        self, control_system: DiscreteTimeLinearSystem
    ) -> None:
        """Test the initialization of the system with control"""
        assert np.all(control_system.state_dim == 2)
        assert np.all(control_system.observation_dim == 2)
        assert np.all(control_system.control_dim == 1)

    def test_stochastic_system_initialization(
        self, stochastic_system: DiscreteTimeLinearSystem
    ) -> None:
        """Test the initialization of the system with stochasticity"""
        assert stochastic_system.process_noise_covariance.shape == (1, 1)
        assert stochastic_system.observation_noise_covariance.shape == (1, 1)

    def test_control_system_process(
        self, control_system: DiscreteTimeLinearSystem
    ) -> None:
        """Test the process of the system"""
        control_system.state = np.array([1, 2])
        control_system.control = [1]
        time = control_system.process(0)
        assert np.all(control_system.state == np.array([3, 3]))
        observation = control_system.observe()
        assert np.all(observation == np.array([3, 6]))
        assert time == 1

    def test_stochastic_system_process(
        self, stochastic_system: DiscreteTimeLinearSystem
    ) -> None:
        """Test the process of the system"""
        stochastic_system.process(0)
