import numpy as np

from system.linear import DiscreteTimeLinearSystem


def test_discrete_time_linear_system_initialization():
    state_space_matrix_A = np.array([[1, 0], [0, 1]])
    state_space_matrix_C = np.array([[1, 0], [0, 1]])
    system = DiscreteTimeLinearSystem(
        state_space_matrix_A, state_space_matrix_C
    )
    assert np.all(system.state_space_matrix_A == state_space_matrix_A)
    assert np.all(system.state_space_matrix_C == state_space_matrix_C)
    assert np.all(
        system.process_noise_covariance_Q
        == np.zeros(state_space_matrix_A.shape)
    )
    assert np.all(
        system.observation_noise_covariance_R
        == np.zeros(
            (state_space_matrix_C.shape[0], state_space_matrix_C.shape[0])
        )
    )
