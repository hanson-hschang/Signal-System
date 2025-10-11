from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.figure.trajectory import TimeTrajectoryFigure


class MassSpringDamperStateTrajectoryFigure(TimeTrajectoryFigure):
    """
    Figure for plotting the state trajectories of a mass-spring-damper system.
    """

    def __init__(
        self,
        time_trajectory: ArrayLike,
        state_trajectory: ArrayLike,
        fig_size: tuple[int, int] = (12, 8),
    ) -> None:
        state_trajectory = np.array(state_trajectory)

        match len(state_trajectory.shape):
            case 1:
                state_trajectory = state_trajectory[np.newaxis, np.newaxis, :]
            case 2:
                state_trajectory = state_trajectory[np.newaxis, :, :]
            case _:
                pass
        assert len(state_trajectory.shape) == 3, (
            "state_trajectory in general is a 3D array with "
            "shape (batch_size, state_dim, time_length)."
        )
        assert state_trajectory.shape[1] % 2 == 0, (
            "state_trajectory must have an even number of states_dim."
            "state_trajectory in general is a 3D array "
            "with shape (batch_size, state_dim, time_length)."
        )
        self._number_of_connections = state_trajectory.shape[1] // 2
        super().__init__(
            time_trajectory,
            batch_size=state_trajectory.shape[0],
            fig_size=fig_size,
            fig_title="Mass-Spring-Damper System State Trajectory",
            fig_layout=(2, self._number_of_connections),
        )
        assert state_trajectory.shape[2] == self._sequence_length, (
            "state_trajectory must have the same time horizon "
            "as time_trajectory. state_trajectory has the time horizon "
            f"of {state_trajectory.shape[2]} while time_trajectory "
            f"has the time horizon of {self._sequence_length}."
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
    ) -> tuple[float, float]:
        signal_range = np.array(
            [np.min(signal_trajectory), np.max(signal_trajectory)]
        )
        signal_range_diff = signal_range[1] - signal_range[0]
        signal_range += np.array(
            [-0.1 * signal_range_diff, 0.1 * signal_range_diff]
        )
        return signal_range[0], signal_range[1]

    def plot(self) -> Self:
        if self._batch_size <= 10:
            self._plot_each_system_trajectory()
        else:
            kwargs: dict = dict(
                color=self._default_color,
                alpha=self._default_alpha,
            )
            self._plot_each_system_trajectory(
                **kwargs,
            )
            (
                mean_trajectory,
                std_trajectory,
            ) = self._compute_system_statistics_trajectory(
                signal_trajectory=self._state_trajectory,
            )
            self._plot_systems_statistics_trajectory(
                mean_trajectory=mean_trajectory,
                std_trajectory=std_trajectory,
            )
        super().plot()
        return self

    def _plot_each_system_trajectory(
        self,
        **kwargs: Any,
    ) -> None:
        for i in range(self._batch_size):
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
                        self._position_range
                        if d == 0
                        else self._velocity_range
                    )
                    self._subplots[d][j].set_ylim(*ylim_range)
        for d, state_name in enumerate(self._state_name):
            self._subplots[d][0].set_ylabel(state_name)
        for j in range(self._number_of_connections):
            self._subplots[0][j].set_title(f"Mass {j + 1}")

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
