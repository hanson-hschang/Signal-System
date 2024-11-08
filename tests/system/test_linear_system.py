import numpy as np
from scipy.linalg import expm

from system.linear import DiscreteTimeLinearSystem, MassSpringDamperSystem


class TestDiscreteTimeLinearSystem:
    state_space_matrix_A = np.array([[1, 1], [0, 1]])
    state_space_matrix_B = np.array([[0], [1]])
    state_space_matrix_C = np.array([[1, 0], [1, 1]])
    process_noise_covariance_Q = np.array([[1, 0], [0, 1]])
    observation_noise_covariance_R = np.array([[1, 0], [0, 1]])
    control_system = DiscreteTimeLinearSystem(
        state_space_matrix_A=state_space_matrix_A,
        state_space_matrix_B=state_space_matrix_B,
        state_space_matrix_C=state_space_matrix_C,
    )
    stochastic_system = DiscreteTimeLinearSystem(
        state_space_matrix_A=state_space_matrix_A,
        state_space_matrix_C=state_space_matrix_C,
        process_noise_covariance=process_noise_covariance_Q,
        observation_noise_covariance=observation_noise_covariance_R,
    )

    def test_initialization(self):
        assert np.all(
            self.control_system.state_space_matrix_A
            == self.state_space_matrix_A
        )
        assert np.all(
            self.control_system.state_space_matrix_B
            == self.state_space_matrix_B
        )
        assert np.all(
            self.control_system.state_space_matrix_C
            == self.state_space_matrix_C
        )
        assert np.all(
            self.control_system.process_noise_covariance
            == np.zeros(self.state_space_matrix_A.shape)
        )
        assert np.all(
            self.control_system.observation_noise_covariance
            == np.zeros(
                (
                    self.state_space_matrix_C.shape[0],
                    self.state_space_matrix_C.shape[0],
                )
            )
        )

        assert np.all(
            self.stochastic_system.state_space_matrix_A
            == self.state_space_matrix_A
        )
        assert np.all(
            self.stochastic_system.state_space_matrix_C
            == self.state_space_matrix_C
        )
        assert np.all(
            self.stochastic_system.process_noise_covariance
            == self.process_noise_covariance_Q
        )
        assert np.all(
            self.stochastic_system.observation_noise_covariance
            == self.observation_noise_covariance_R
        )

    def test_update(self):
        self.control_system.state = np.array([1, 2])
        self.control_system.control = [1]
        self.control_system.update()
        assert np.all(self.control_system.state == np.array([3, 3]))
        assert np.all(self.control_system.observation == np.array([3, 6]))

        self.stochastic_system.state = np.array([1, 2])
        self.stochastic_system.update()


class TestMassSpringDamperSystem:
    time_step = 0.01
    mass = 1.0
    spring_constant = 1.0
    damping_coefficient = 1.0
    one_mass_system = MassSpringDamperSystem(
        number_of_connections=1,
        mass=mass,
        spring_constant=spring_constant,
        damping_coefficient=damping_coefficient,
        time_step=time_step,
        observation_choice=MassSpringDamperSystem.ObservationChoice.LAST_POSITION,
    )
    two_mass_system = MassSpringDamperSystem(
        number_of_connections=2,
        mass=mass,
        spring_constant=spring_constant,
        damping_coefficient=damping_coefficient,
        time_step=time_step,
        observation_choice=MassSpringDamperSystem.ObservationChoice.ALL_POSITIONS,
    )

    def test_initialization(self):
        assert self.one_mass_system.state_dim == 2
        assert self.one_mass_system.observation_dim == 1
        assert self.one_mass_system.control_dim == 0
        assert self.one_mass_system.number_of_systems == 1
        assert np.all(
            self.one_mass_system.state_space_matrix_A
            == expm(
                np.array(
                    [
                        [0, 1],
                        [
                            -self.spring_constant / self.mass,
                            -self.damping_coefficient / self.mass,
                        ],
                    ]
                )
                * self.time_step
            )
        )
        assert np.all(
            self.one_mass_system.state_space_matrix_C == np.array([[0, 1]])
        )

        assert self.two_mass_system.state_dim == 4
        assert self.two_mass_system.observation_dim == 2
        assert self.two_mass_system.control_dim == 0
        assert np.all(
            self.two_mass_system.state_space_matrix_A
            == expm(
                np.array(
                    [
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [
                            -2 * self.spring_constant / self.mass,
                            self.spring_constant / self.mass,
                            -2 * self.damping_coefficient / self.mass,
                            self.damping_coefficient / self.mass,
                        ],
                        [
                            self.spring_constant / self.mass,
                            -self.spring_constant / self.mass,
                            self.damping_coefficient / self.mass,
                            -self.damping_coefficient / self.mass,
                        ],
                    ]
                )
                * self.time_step
            )
        )
        assert np.all(
            self.two_mass_system.state_space_matrix_C
            == np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        )
