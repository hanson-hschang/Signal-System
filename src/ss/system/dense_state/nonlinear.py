from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ss.system.dense_state import ContinuousTimeSystem
from ss.tool.assertion.inspect import inspect_arguments


class ContinuousTimeNonlinearSystem(ContinuousTimeSystem):
    def __init__(
        self,
        time_step: float,
        state_dim: int,
        observation_dim: int,
        process_function: Callable,
        observation_function: Callable,
        state_constraint_function: Optional[Callable] = None,
        control_dim: int = 0,
        number_of_systems: int = 1,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(
            time_step=time_step,
            state_dim=state_dim,
            observation_dim=observation_dim,
            control_dim=control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
        )
        arg_name_shape_dict = {"state": self._state.shape}
        self._observation_function: Callable = inspect_arguments(
            func=observation_function,
            result_shape=self._observation.shape,
            arg_name_shape_dict=arg_name_shape_dict,
        )
        if state_constraint_function is None:

            def default_state_constraint_function(
                state: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                return state

            state_constraint_function = default_state_constraint_function
        self._state_constraint_function: Callable = inspect_arguments(
            func=state_constraint_function,
            result_shape=self._state.shape,
            arg_name_shape_dict=arg_name_shape_dict,
        )
        if self._control_dim > 0:
            arg_name_shape_dict["control"] = self._control.shape
        self._process_function: Callable = inspect_arguments(
            func=process_function,
            result_shape=self._state.shape,
            arg_name_shape_dict=arg_name_shape_dict,
        )
        self._set_compute_state_process(control_flag=(control_dim > 0))

    def create_multiple_systems(
        self, number_of_systems: int
    ) -> "ContinuousTimeNonlinearSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `number_of_systems: int`
            The number of systems to be created.

        Returns
        -------
        `system: ContinuousTimeNonlinearSystem`
            The created multi-system.
        """
        return self.__class__(
            time_step=self._time_step,
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            process_function=self._process_function,
            observation_function=self._observation_function,
            state_constraint_function=self._state_constraint_function,
            control_dim=self._control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=self._process_noise_covariance,
            observation_noise_covariance=self._observation_noise_covariance,
        )

    def _set_compute_state_process(self, control_flag: bool) -> None:
        def _compute_state_process_without_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = (
                self._state_constraint_function(
                    self._state
                    + self._process_function(
                        self._state,
                    )
                    * self._time_step
                )
            )
            return state_process

        def _compute_state_process_with_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = (
                self._state_constraint_function(
                    self._state
                    + self._process_function(
                        self._state,
                        self._control,
                    )
                    * self._time_step
                )
            )
            return state_process

        self._compute_state_process: Callable[[], NDArray[np.float64]] = (
            _compute_state_process_with_control
            if control_flag
            else _compute_state_process_without_control
        )

    def _compute_observation_process(self) -> NDArray[np.float64]:
        observation: NDArray[np.float64] = self._observation_function(
            self._state
        )
        return observation


class DiscreteTimeNonlinearSystem(ContinuousTimeNonlinearSystem):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        process_function: Callable,
        observation_function: Callable,
        state_constraint_function: Optional[Callable] = None,
        control_dim: int = 0,
        number_of_systems: int = 1,
        process_noise_covariance: Optional[ArrayLike] = None,
        observation_noise_covariance: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(
            time_step=1,
            state_dim=state_dim,
            observation_dim=observation_dim,
            process_function=process_function,
            observation_function=observation_function,
            state_constraint_function=state_constraint_function,
            control_dim=control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
        )

    def create_multiple_systems(
        self, number_of_systems: int
    ) -> "DiscreteTimeNonlinearSystem":
        """
        Create multiple systems based on the current system.

        Parameters
        ----------
        `number_of_systems: int`
            The number of systems to be created.

        Returns
        -------
        `system: DiscreteTimeNonlinearSystem`
            The created multi-system.
        """
        return self.__class__(
            state_dim=self._state_dim,
            observation_dim=self._observation_dim,
            process_function=self._process_function,
            observation_function=self._observation_function,
            state_constraint_function=self._state_constraint_function,
            control_dim=self._control_dim,
            number_of_systems=number_of_systems,
            process_noise_covariance=self._process_noise_covariance,
            observation_noise_covariance=self._observation_noise_covariance,
        )

    def _set_compute_state_process(self, control_flag: bool) -> None:
        def _compute_state_process_without_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = self._process_function(
                self._state,
            )
            return state_process

        def _compute_state_process_with_control() -> NDArray[np.float64]:
            state_process: NDArray[np.float64] = self._process_function(
                self._state,
                self._control,
            )
            return state_process

        self._compute_state_process: Callable[[], NDArray[np.float64]] = (
            _compute_state_process_with_control
            if control_flag
            else _compute_state_process_without_control
        )
