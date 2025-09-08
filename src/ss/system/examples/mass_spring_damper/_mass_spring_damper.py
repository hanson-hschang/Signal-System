from typing import Optional, assert_never

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

from ss.system.linear import ContinuousTimeLinearSystem
from ss.utility.assertion import is_positive_integer


class ObservationChoice(StrEnum):
    ALL_STATES = "ALL_STATES"
    ALL_POSITIONS = "ALL_POSITIONS"
    LAST_POSITION = "LAST_POSITION"


class ControlChoice(StrEnum):
    ALL_FORCES = "ALL_FORCES"
    LAST_FORCE = "LAST_FORCE"
    NO_CONTROL = "NO_CONTROL"


class MassSpringDamperSystem(ContinuousTimeLinearSystem):
    def __init__(
        self,
        number_of_connections: int = 1,
        mass: float = 1.0,
        spring_constant: float = 1.0,
        damping_coefficient: float = 1.0,
        time_step: float = 0.01,
        observation_choice: ObservationChoice = ObservationChoice.LAST_POSITION,
        control_choice: ControlChoice = ControlChoice.NO_CONTROL,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
        batch_size: int = 1,
    ) -> None:
        assert is_positive_integer(
            number_of_connections
        ), "number_of_connections should be a positive integer"
        assert isinstance(
            observation_choice, ObservationChoice
        ), f"observation_choice should be one of {ObservationChoice.__members__.keys()}"
        assert isinstance(
            control_choice, ControlChoice
        ), f"control_choice should be one of {ControlChoice.__members__.keys()}"

        self._number_of_connections = int(number_of_connections)
        self._mass = mass
        self._spring = spring_constant
        self._damping = damping_coefficient

        matrix_A = np.zeros(
            (2 * self._number_of_connections, 2 * self._number_of_connections)
        )
        matrix_A[
            : self._number_of_connections, self._number_of_connections :
        ] = np.identity(self._number_of_connections)
        matrix_A[self._number_of_connections, 0] = -self._spring / self._mass
        matrix_A[self._number_of_connections, self._number_of_connections] = (
            -self._damping / self._mass
        )
        for i in range(1, self._number_of_connections):
            position_index = i - 1
            velocity_index = self._number_of_connections + i - 1
            matrix_A[
                velocity_index + 0, position_index : position_index + 2
            ] += (np.array([-1, 1]) * self._spring / self._mass)
            matrix_A[
                velocity_index + 1, position_index : position_index + 2
            ] -= (np.array([-1, 1]) * self._spring / self._mass)
            matrix_A[
                velocity_index + 0, velocity_index : velocity_index + 2
            ] += (np.array([-1, 1]) * self._damping / self._mass)
            matrix_A[
                velocity_index + 1, velocity_index : velocity_index + 2
            ] -= (np.array([-1, 1]) * self._damping / self._mass)

        self._observation_choice = observation_choice
        match self._observation_choice:
            case ObservationChoice.ALL_STATES:
                matrix_C = np.identity(2 * self._number_of_connections)
            case ObservationChoice.ALL_POSITIONS:
                matrix_C = np.zeros(
                    (
                        self._number_of_connections,
                        2 * self._number_of_connections,
                    )
                )
                matrix_C[:, self._number_of_connections :] = np.identity(
                    self._number_of_connections
                )
            case ObservationChoice.LAST_POSITION:
                matrix_C = np.zeros((1, 2 * self._number_of_connections))
                matrix_C[0, -1] = 1
            case _ as unmatched_observation_choice:  # pragma: no cover
                assert_never(unmatched_observation_choice)

        self._control_choice = control_choice
        match self._control_choice:
            case ControlChoice.ALL_FORCES:
                matrix_B = np.zeros(
                    (
                        2 * self._number_of_connections,
                        self._number_of_connections,
                    )
                )
                matrix_B[self._number_of_connections :, :] = (
                    np.identity(self._number_of_connections) / self._mass
                )
            case ControlChoice.LAST_FORCE:
                matrix_B = np.zeros((2 * self._number_of_connections, 1))
                matrix_B[-1, 0] = 1 / self._mass
            case ControlChoice.NO_CONTROL:
                matrix_B = np.zeros((2 * self._number_of_connections, 0))
            case _ as unmatched_control_choice:
                assert_never(unmatched_control_choice)

        super().__init__(
            time_step=time_step,
            state_space_matrix_A=matrix_A,
            state_space_matrix_B=matrix_B,
            state_space_matrix_C=matrix_C,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            batch_size=batch_size,
        )

    def duplicate(self, batch_size: int) -> "MassSpringDamperSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `batch_size: int`
            The number of systems to be created.

        Returns
        -------
        `system: MassSpringDamperSystem`
            The created multi-system.
        """
        return self.__class__(
            number_of_connections=self._number_of_connections,
            mass=self._mass,
            spring_constant=self._spring,
            damping_coefficient=self._damping,
            time_step=self._time_step,
            observation_choice=self._observation_choice,
            control_choice=self._control_choice,
            process_noise_covariance=self.process_noise_covariance,
            observation_noise_covariance=self.observation_noise_covariance,
            batch_size=batch_size,
        )
