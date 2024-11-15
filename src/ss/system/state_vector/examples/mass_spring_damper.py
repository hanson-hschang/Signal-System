from typing import Optional, Self, Tuple, assert_never

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ss.system.state_vector.linear import DiscreteTimeLinearSystem
from ss.tool.assertion import isPositiveInteger
from ss.tool.figure import TimeTrajectoryFigure


class ObservationChoice(StrEnum):
    ALL_STATES = "ALL_STATES"
    ALL_POSITIONS = "ALL_POSITIONS"
    LAST_POSITION = "LAST_POSITION"


class ControlChoice(StrEnum):
    ALL_FORCES = "ALL_FORCES"
    LAST_FORCE = "LAST_FORCE"
    NO_CONTROL = "NO_CONTROL"


class MassSpringDamperSystem(DiscreteTimeLinearSystem):
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
        number_of_systems: int = 1,
    ) -> None:
        assert isPositiveInteger(
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

        state_space_matrix_A = expm(matrix_A * time_step)
        state_space_matrix_B = state_space_matrix_A @ matrix_B * time_step

        super().__init__(
            state_space_matrix_A=state_space_matrix_A,
            state_space_matrix_B=state_space_matrix_B,
            state_space_matrix_C=matrix_C,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            number_of_systems=number_of_systems,
        )
        self._time_step = time_step

    def create_multiple_systems(
        self, number_of_systems: int
    ) -> "MassSpringDamperSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `number_of_systems: int`
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
            number_of_systems=number_of_systems,
        )


class MassSpringDamperStateTrajectoryFigure(TimeTrajectoryFigure):
    """
    Figure for plotting the state trajectories of a mass-spring-damper system.
    """

    def __init__(
        self,
        time_trajectory: ArrayLike,
        state_trajectory: ArrayLike,
        fig_size: Tuple[int, int] = (12, 8),
    ) -> None:
        state_trajectory = np.array(state_trajectory)
        assert (
            len(state_trajectory.shape) == 2
        ), "state_trajectory must be a 2D array."
        assert (
            state_trajectory.shape[0] % 2 == 0
        ), "state_trajectory must have an even number of rows."
        self._number_of_connections = state_trajectory.shape[0] // 2
        super().__init__(
            time_trajectory,
            fig_size=fig_size,
            fig_title="Mass-Spring-Damper System State Trajectory",
            fig_layout=(2, self._number_of_connections),
        )
        assert (
            state_trajectory.shape[1] == self._time_length
        ), "state_trajectory must have the same length as time_trajectory."

        self._state_trajectory = state_trajectory
        self._state_name = [
            "Position (m)",
            "Velocity (m/s)",
        ]
        self._position_range = np.array(
            [
                np.min(
                    self._state_trajectory[: self._number_of_connections, :]
                ),
                np.max(
                    self._state_trajectory[: self._number_of_connections, :]
                ),
            ]
        )
        position_range_diff = self._position_range[1] - self._position_range[0]
        self._position_range += np.array(
            [
                -0.1 * position_range_diff,
                0.1 * position_range_diff,
            ]
        )
        self._velocity_range = np.array(
            [
                np.min(
                    self._state_trajectory[self._number_of_connections :, :]
                ),
                np.max(
                    self._state_trajectory[self._number_of_connections :, :]
                ),
            ]
        )
        velocity_range_diff = self._velocity_range[1] - self._velocity_range[0]
        self._velocity_range += np.array(
            [
                -0.1 * velocity_range_diff,
                0.1 * velocity_range_diff,
            ]
        )

    def plot_figure(self) -> Self:
        for i, state_name in enumerate(self._state_name):
            self._subplots[i][0].set_ylabel(state_name)
            for j in range(self._number_of_connections):
                self._subplots[i][j].plot(
                    self._time_trajectory,
                    self._state_trajectory[
                        i * self._number_of_connections + j, :
                    ],
                )
                ylim_range = (
                    self._position_range if i == 0 else self._velocity_range
                )
                self._subplots[i][j].set_ylim(*ylim_range)
        for j in range(self._number_of_connections):
            self._subplots[0][j].set_title(f"Mass {j+1}")
        super().plot_figure()
        return self
