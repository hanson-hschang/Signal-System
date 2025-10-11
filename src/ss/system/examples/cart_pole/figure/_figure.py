from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.figure.trajectory import SequenceTrajectoryFigure


class CartPoleStateTrajectoryFigure(SequenceTrajectoryFigure):
    """
    Figure for plotting the state trajectories of a cart-pole system.
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
        assert state_trajectory.shape[1] == 4, (
            "state_trajectory must have 4 states, corresponding to "
            "the cart-pole system. state_trajectory in general is "
            "a 3D array with shape (batch_size, state_dim, time_length)."
        )
        super().__init__(
            time_trajectory,
            batch_size=state_trajectory.shape[0],
            fig_size=fig_size,
            fig_title="Cart-Pole System State Trajectory",
            fig_layout=(2, 2),
        )
        assert state_trajectory.shape[2] == self._sequence_length, (
            "state_trajectory must have the same length of time_trajectory."
            "state_trajectory in general is a 3D array with shape "
            "(batch_size, state_dim, time_length)."
        )

        self._state_trajectory = state_trajectory
        self._state_name = [
            "Cart Position (m)",
            "Cart Velocity (m/s)",
            "Pole Angle (rad)",
            "Pole Angular Velocity (rad/s)",
        ]
        self._state_subplots = [
            self._subplots[0][0],
            self._subplots[0][1],
            self._subplots[1][0],
            self._subplots[1][1],
        ]

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
            for state_subplot, state_trajectory, state_name in zip(
                self._state_subplots,
                self._state_trajectory[i, ...],
                self._state_name,
            ):
                self._plot_signal_trajectory(
                    state_subplot,
                    state_trajectory,
                    ylabel=state_name,
                    **kwargs,
                )

    def _plot_systems_statistics_trajectory(
        self,
        mean_trajectory: NDArray[np.float64],
        std_trajectory: NDArray[np.float64],
    ) -> None:
        for state_subplot, mean_state, std_state in zip(
            self._state_subplots,
            mean_trajectory,
            std_trajectory,
        ):
            self._plot_statistics_signal_trajectory(
                state_subplot,
                cast(NDArray, mean_state),
                cast(NDArray, std_state),
            )
