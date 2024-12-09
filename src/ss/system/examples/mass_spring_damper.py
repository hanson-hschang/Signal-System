from typing import Any, Dict, Optional, Self, Tuple, assert_never

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ss.system.linear import ContinuousTimeLinearSystem
from ss.utility.assertion import is_positive_integer
from ss.utility.figure import TimeTrajectoryFigure


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
        number_of_systems: int = 1,
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
            number_of_systems=number_of_systems,
        )

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

        match len(state_trajectory.shape):
            case 1:
                state_trajectory = state_trajectory[np.newaxis, np.newaxis, :]
            case 2:
                state_trajectory = state_trajectory[np.newaxis, :, :]
            case _:
                pass
        assert (
            len(state_trajectory.shape) == 3
        ), "state_trajectory in general is a 3D array with shape (number_of_systems, state_dim, time_length)."
        assert state_trajectory.shape[1] % 2 == 0, (
            "state_trajectory must have an even number of states_dim."
            "state_trajectory in general is a 3D array with shape (number_of_systems, state_dim, time_length)."
        )
        self._number_of_connections = state_trajectory.shape[1] // 2
        super().__init__(
            time_trajectory,
            number_of_systems=state_trajectory.shape[0],
            fig_size=fig_size,
            fig_title="Mass-Spring-Damper System State Trajectory",
            fig_layout=(2, self._number_of_connections),
        )
        assert state_trajectory.shape[2] == self._time_length, (
            f"state_trajectory must have the same time horizon as time_trajectory. "
            f"state_trajectory has the time horizon of {state_trajectory.shape[2]} "
            f"while time_trajectory has the time horizon of {self._time_length}."
        )

        self._state_trajectory = state_trajectory
        self._state_name = [
            "Position (m)",
            "Velocity (m/s)",
        ]
        self._position_range = self._compute_range(
            signal_trajectory=state_trajectory[
                :, : self._number_of_connections, :
            ]
        )
        self._velocity_range = self._compute_range(
            signal_trajectory=state_trajectory[
                :, self._number_of_connections :, :
            ]
        )

    def _compute_range(
        self, signal_trajectory: NDArray[np.float64]
    ) -> Tuple[float, float]:
        signal_range = np.array(
            [np.min(signal_trajectory), np.max(signal_trajectory)]
        )
        signal_range_diff = signal_range[1] - signal_range[0]
        signal_range += np.array(
            [-0.1 * signal_range_diff, 0.1 * signal_range_diff]
        )
        return signal_range[0], signal_range[1]

    def plot(self) -> Self:
        if self._number_of_systems <= 10:
            self._plot_each_system_trajectory()
        else:
            kwargs: Dict = dict(
                color=self._default_color,
                alpha=self._default_alpha,
            )
            self._plot_each_system_trajectory(
                **kwargs,
            )
            mean_trajectory, std_trajectory = (
                self._compute_system_statistics_trajectory(
                    signal_trajectory=self._state_trajectory,
                )
            )
            self._plot_systems_statistics_trajectory(
                mean_trajectory=mean_trajectory,
                std_trajectory=std_trajectory,
            )
        return super().plot()

    def _plot_each_system_trajectory(
        self,
        **kwargs: Any,
    ) -> None:
        for i in range(self._number_of_systems):
            for d in range(len(self._state_name)):
                for j in range(self._number_of_connections):
                    self._plot_signal_trajectory(
                        self._subplots[d][j],
                        self._state_trajectory[
                            i, d * self._number_of_connections + j, :
                        ],
                        **kwargs,
                    )
                    ylim_range = (
                        self._position_range if d == 0 else self._velocity_range
                    )
                    self._subplots[d][j].set_ylim(*ylim_range)
        for d, state_name in enumerate(self._state_name):
            self._subplots[d][0].set_ylabel(state_name)
        for j in range(self._number_of_connections):
            self._subplots[0][j].set_title(f"Mass {j+1}")

    def _plot_systems_statistics_trajectory(
        self,
        mean_trajectory: NDArray[np.float64],
        std_trajectory: NDArray[np.float64],
    ) -> None:
        for d in range(len(self._state_name)):
            for j in range(self._number_of_connections):
                self._plot_statistics_signal_trajectory(
                    self._subplots[d][j],
                    mean_trajectory[d * self._number_of_connections + j, :],
                    std_trajectory[d * self._number_of_connections + j, :],
                )
