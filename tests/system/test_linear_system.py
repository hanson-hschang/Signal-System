import numpy as np

from system.linear import DiscreteTimeLinearSystem


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
        process_noise_covariance_Q=process_noise_covariance_Q,
        observation_noise_covariance_R=observation_noise_covariance_R,
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
            self.control_system.process_noise_covariance_Q
            == np.zeros(self.state_space_matrix_A.shape)
        )
        assert np.all(
            self.control_system.observation_noise_covariance_R
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
            self.stochastic_system.process_noise_covariance_Q
            == self.process_noise_covariance_Q
        )
        assert np.all(
            self.stochastic_system.observation_noise_covariance_R
            == self.observation_noise_covariance_R
        )

    def test_update(self):
        self.control_system.state = np.array([1, 2])
        self.control_system.control = [1]
        self.control_system.update()
        assert np.all(self.control_system.state == np.array([3, 3]))
        assert np.all(self.control_system.observation == np.array([3, 6]))

        self.stochastic_system.state = np.array([1, 2])
        self.stochastic_system.control = [1]
        self.stochastic_system.update()
