import numpy as np
import pytest
from scipy.linalg import expm

from system.linear import DiscreteTimeLinearSystem, MassSpringDamperSystem


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

    def test_control_system_update(
        self, control_system: DiscreteTimeLinearSystem
    ) -> None:
        """Test the update of the system"""
        control_system.state = np.array([1, 2])
        control_system.control = [1]
        time = control_system.update(0)
        assert np.all(control_system.state == np.array([3, 3]))
        assert np.all(control_system.observation == np.array([3, 6]))
        assert time == 1

    def test_stochastic_system_update(
        self, stochastic_system: DiscreteTimeLinearSystem
    ) -> None:
        """Test the update of the system"""
        stochastic_system.update(0)


class TestMassSpringDamperSystem:

    @pytest.fixture
    def one_mass_system(self) -> MassSpringDamperSystem:
        return MassSpringDamperSystem(
            number_of_connections=1,
            mass=1.0,
            spring_constant=1.0,
            damping_coefficient=1.0,
            time_step=0.01,
            observation_choice=MassSpringDamperSystem.ObservationChoice.LAST_POSITION,
        )

    @pytest.fixture
    def two_mass_system(self) -> MassSpringDamperSystem:
        return MassSpringDamperSystem(
            number_of_connections=2,
            mass=1.0,
            spring_constant=1.0,
            damping_coefficient=1.0,
            time_step=0.01,
            observation_choice=MassSpringDamperSystem.ObservationChoice.ALL_POSITIONS,
        )

    def test_one_mass_system_initialization(
        self, one_mass_system: MassSpringDamperSystem
    ) -> None:
        """Test the initialization of the system"""
        assert one_mass_system.state_dim == 2
        assert one_mass_system.observation_dim == 1
        assert one_mass_system.control_dim == 0
        assert one_mass_system.number_of_systems == 1
        assert np.all(
            one_mass_system.state_space_matrix_A
            == expm(
                np.array(
                    [
                        [0, 1],
                        [-1, -1],
                    ]
                )
                * 0.01
            )
        )
        assert np.all(
            one_mass_system.state_space_matrix_C == np.array([[0, 1]])
        )

    def test_two_mass_system_initialization(
        self, two_mass_system: MassSpringDamperSystem
    ) -> None:
        """Test the initialization of the system"""
        assert two_mass_system.state_dim == 4
        assert two_mass_system.observation_dim == 2
        assert two_mass_system.control_dim == 0
        assert np.all(
            two_mass_system.state_space_matrix_A
            == expm(
                np.array(
                    [
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [-2, 1, -2, 1],
                        [1, -1, 1, -1],
                    ]
                )
                * 0.01
            )
        )
        assert np.all(
            two_mass_system.state_space_matrix_C
            == np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        )
