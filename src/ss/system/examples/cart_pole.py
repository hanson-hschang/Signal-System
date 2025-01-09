from typing import Any, Dict, Optional, Self, Tuple

from collections import defaultdict

import numpy as np
from matplotlib.axes import Axes
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ss.system.nonlinear import ContinuousTimeNonlinearSystem
from ss.utility.figure import SequenceTrajectoryFigure


class CartPoleSystem(ContinuousTimeNonlinearSystem):
    """
    Cart-pole system dynamics.

    The cart-pole system is a classic control problem where a pole is attached to a cart moving along a frictionless track.

    The number of states is 4, with the state vector including position and velocity of the cart, and angle and angular velocity of the pole.
    The pole is at its upright position when the angle is 0, and the positive direction is counter-clockwise.
    The number of observations is 4, with the observation vector being the same as the state vector.
    The number of controls is 1, with the control vector being the force applied to the cart.

    The system is described by the equations from: https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
    """

    def __init__(
        self,
        cart_mass: float = 1.0,
        pole_mass: float = 0.01,
        pole_length: float = 2.0,
        gravity: float = 9.81,
        time_step: float = 0.01,
        number_of_systems: int = 1,
    ) -> None:
        self._cart_mass = cart_mass
        self._pole_mass = pole_mass
        self._pole_length = pole_length
        self._gravity = gravity

        @njit(cache=True)  # type: ignore
        def process_function(
            state: NDArray[np.float64],
            control: NDArray[np.float64],
            cart_mass: float = self._cart_mass,
            pole_mass: float = self._pole_mass,
            pole_length: float = self._pole_length,
            gravity: float = self._gravity,
        ) -> NDArray[np.float64]:
            cart_position = state[:, 0]
            cart_velocity = state[:, 1]
            pole_angle = state[:, 2]
            pole_angular_velocity = state[:, 3]
            force = control[:, 0]
            process = np.zeros_like(state)

            total_mass = cart_mass + pole_mass
            total_mass_adjusted = cart_mass + (
                pole_mass * np.sin(pole_angle) ** 2
            )
            pole_mass_length = pole_mass * pole_length

            common_numerator = force - (
                pole_mass_length
                * np.sin(pole_angle)
                * (pole_angular_velocity**2)
            )
            process[:, 3] = (
                common_numerator * np.cos(pole_angle)
                + total_mass * gravity * np.sin(pole_angle)
            ) / (total_mass_adjusted * pole_length)
            process[:, 2] = pole_angular_velocity
            process[:, 1] = (
                common_numerator
                + pole_mass * gravity * np.sin(pole_angle) * np.cos(pole_angle)
            ) / total_mass_adjusted
            process[:, 0] = cart_velocity

            return process

        @njit(cache=True)  # type: ignore
        def state_constraint_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            state[:, 2] = np.fmod(state[:, 2] + np.pi, 2 * np.pi) - np.pi
            return state

        @njit(cache=True)  # type: ignore
        def observation_function(
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return state[:, :]

        super().__init__(
            time_step=time_step,
            state_dim=4,
            observation_dim=4,
            control_dim=1,
            process_function=process_function,
            observation_function=observation_function,
            state_constraint_function=state_constraint_function,
            number_of_systems=number_of_systems,
        )

    def duplicate(self, number_of_systems: int) -> "CartPoleSystem":
        return self.__class__(
            cart_mass=self._cart_mass,
            pole_mass=self._pole_mass,
            pole_length=self._pole_length,
            gravity=self._gravity,
            time_step=self._time_step,
            number_of_systems=number_of_systems,
        )


class CartPoleStateTrajectoryFigure(SequenceTrajectoryFigure):
    """
    Figure for plotting the state trajectories of a cart-pole system.
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
        assert state_trajectory.shape[1] == 4, (
            "state_trajectory must have 4 states, corresponding to the cart-pole system."
            "state_trajectory in general is a 3D array with shape (number_of_systems, state_dim, time_length)."
        )
        super().__init__(
            time_trajectory,
            number_of_systems=state_trajectory.shape[0],
            fig_size=fig_size,
            fig_title="Cart-Pole System State Trajectory",
            fig_layout=(2, 2),
        )
        assert state_trajectory.shape[2] == self._sequence_length, (
            "state_trajectory must have the same length of time_trajectory."
            "state_trajectory in general is a 3D array with shape (number_of_systems, state_dim, time_length)."
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
        for i in range(self._number_of_systems):
            for state_subplot, state_trajectory, state_name in zip(
                self._state_subplots,
                self._state_trajectory[i, ...],
                self._state_name,
            ):
                self._plot_signal_trajectory(
                    state_subplot,
                    state_trajectory,
                    state_name,
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
                mean_state,
                std_state,
            )
