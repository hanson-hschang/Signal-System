from typing import Generic, Tuple, assert_never

import torch

from ss.estimation.filtering.hmm.learning.module.config import (
    LearningHmmFilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.emission import (
    EmissionModule,
)
from ss.estimation.filtering.hmm.learning.module.estimation import (
    EstimationModule,
)
from ss.estimation.filtering.hmm.learning.module.filter import FilterModule
from ss.estimation.filtering.hmm.learning.module.transition import (
    TransitionModule,
)
from ss.utility.descriptor import (
    BatchTensorReadOnlyDescriptor,
    Descriptor,
    ReadOnlyDescriptor,
)
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class LearningHmmFilter(
    BaseLearningModule[LearningHmmFilterConfig[TC]],
    Generic[T, TC],
):
    """
    `LearningHmmFilter` module for learning the hidden Markov model and estimating the next observation.
    """

    def __init__(
        self,
        config: LearningHmmFilterConfig[TC],
    ) -> None:
        """
        Initialize the `LearningHmmFilter` module.

        Arguments:
        ----------
        config : LearningHmmFilterConfig
            dataclass containing the configuration for the module `LearningHmmFilter` class
        """
        super().__init__(config)

        # Define the dimensions of the state, observation, and the number of layers
        # self._state_dim = self._config.filter.state_dim
        # self._discrete_observation_dim = (
        #     self._config.filter.discrete_observation_dim
        # )
        # self._layer_dim = self._config.transition.layer_dim + 1
        # self._number_of_systems = 1

        self._estimation_matrix_binding = False
        if self._config.filter.estimation_dim == 0:
            self._estimation_matrix_binding = True
            self._config.filter.estimation_dim = (
                self._config.filter.discrete_observation_dim
            )

        # Define the filter module
        self._filter = FilterModule(self._config.filter)

        # Define the learnable emission, transition and estimation modules
        self._emission = EmissionModule[T, TC](
            self._config.emission, self._config.filter
        )
        self._transition = TransitionModule[T, TC](
            self._config.transition, self._config.filter
        )
        self._estimation = EstimationModule[T, TC](
            self._config.estimation, self._config.filter
        )

        if self._estimation_matrix_binding:
            self._estimation.matrix_parameter.bind_with(
                self._emission.matrix_parameter
            )

        # Initialize the estimated next state, and next observation for the inference mode
        # self._init_batch()

    # def _init_batch(
    #     self,
    #     batch_size: int = 1,
    #     device: Device = Device.CPU,
    #     is_initialized: bool = False,
    # ) -> None:
    #     self._is_initialized = is_initialized
    #     self._batch_size = batch_size

    #     with self.evaluation_mode():
    #         self._estimated_state = (
    #             torch.ones((self._batch_size, self._state_dim))
    #             / self._state_dim
    #         ).to(
    #             device=device.torch_device
    #         )  # (batch_size, state_dim)
    #         self._predicted_next_state = (
    #             torch.ones((self._batch_size, self._state_dim))
    #             / self._state_dim
    #         ).to(
    #             device=device.torch_device
    #         )  # (batch_size, state_dim)
    #         self._predicted_next_observation_probability: torch.Tensor = cast(
    #             torch.Tensor,
    #             self._emission(
    #                 self._predicted_next_state,  # (batch_size, state_dim)
    #             ),
    #         ).to(
    #             device=device.torch_device
    #         )  # (batch_size, discrete_observation_dim)
    #         # self._estimation = self._estimate()
    #         # self._estimation_shape = tuple(self._estimation.shape[1:])

    # estimated_state = BatchTensorReadOnlyDescriptor(
    #     "_batch_size", "_state_dim"
    # )
    # predicted_next_state = BatchTensorReadOnlyDescriptor(
    #     "_batch_size", "_state_dim"
    # )
    # predicted_next_observation_probability = BatchTensorReadOnlyDescriptor(
    #     "_batch_size", "_discrete_observation_dim"
    # )
    # estimation = BatchTensorReadOnlyDescriptor(
    #     "_batch_size", "*_estimation_shape"
    # )

    # state_dim = ReadOnlyDescriptor[int]()
    # discrete_observation_dim = ReadOnlyDescriptor[int]()
    # layer_dim = ReadOnlyDescriptor[int]()
    # estimation_shape = ReadOnlyDescriptor[Tuple[int, ...]]()
    # batch_size = ReadOnlyDescriptor[int]()

    @property
    def state_dim(self) -> int:
        return self._filter.state_dim

    @property
    def discrete_observation_dim(self) -> int:
        return self._filter.discrete_observation_dim

    @property
    def estimation_dim(self) -> int:
        return self._filter.estimation_dim

    @property
    def number_of_systems(self) -> int:
        return self._filter.batch_size

    @number_of_systems.setter
    def number_of_systems(self, number_of_systems: int) -> None:
        self._filter.batch_size = number_of_systems

    @property
    def emission(self) -> EmissionModule[T, TC]:
        return self._emission

    @property
    def transition(self) -> TransitionModule[T, TC]:
        return self._transition

    @property
    def estimation(self) -> EstimationModule[T, TC]:
        return self._estimation

    # @property
    # def emission_matrix(self) -> torch.Tensor:
    #     return self._emission.matrix

    # @property
    # def transition_matrix(self) -> List[torch.Tensor]:
    #     return [matrix for matrix in self._transition.matrix]

    # @property
    # def initial_state(self) -> List[List[torch.Tensor]]:
    #     return [
    #         [
    #             transition_block.initial_state
    #             for transition_block in transition_layer.blocks
    #         ]
    #         for transition_layer in self._transition.layers
    #     ]

    # @property
    # def coefficient(self) -> List[torch.Tensor]:
    #     return [
    #         transition_layer.coefficient
    #         for transition_layer in self._transition.layers
    #     ]

    # @property
    # def estimation_option(self) -> Config.EstimationConfig.Option:
    #     return self._config.estimation.option

    # @estimation_option.setter
    # def estimation_option(
    #     self,
    #     estimation_option: Config.EstimationConfig.Option,
    # ) -> None:
    #     self._config.estimation.option = estimation_option
    #     self._init_batch(batch_size=self._batch_size)

    def forward(self, observation_trajectory: torch.Tensor) -> torch.Tensor:
        """
        forward method for the `LearningHmmFilter` class

        Arguments:
        ----------
            observation_trajectory : torch.Tensor
                shape (batch_size, horizon)

        Returns:
        --------
            predicted_next_observation_log_probability_trajectory : torch.Tensor
                shape (batch_size, discrete_observation_dim, horizon)
        """
        (
            estimated_state_trajectory,
            predicted_state_trajectory,
        ) = self._forward(
            observation_trajectory  # (batch_size, horizon)
        )

        estimation_trajectory: torch.Tensor = self._estimation(
            estimated_state_trajectory,  # (batch_size, horizon, state_dim)
            predicted_state_trajectory,  # (batch_size, horizon, state_dim)
        )  # (batch_size, horizon, estimation_dim)

        # predicted_next_observation_trajectory = self._emission(
        #     predicted_next_state_trajectory,  # (batch_size, horizon, state_dim)
        #     emission_matrix,  # (state_dim, observation_dim)
        # )  # (batch_size, horizon, observation_dim)

        # predicted_next_observation_log_probability_trajectory = torch.moveaxis(
        #     torch.log(predicted_next_observation_trajectory), 1, 2
        # )  # (batch_size, discrete_observation_dim, horizon)

        # return predicted_next_observation_log_probability_trajectory

        # _, estimation_trajectory = self._forward(
        #     observation_trajectory  # (batch_size, horizon, discrete_observation_dim)
        # )  # (batch_size, horizon, estimation_dim)

        # log_estimation_trajectory = torch.moveaxis(
        #     torch.log(estimation_trajectory), 1, 2
        # )  # (batch_size, estimation_dim, horizon)

        return estimation_trajectory

    def _forward(
        self,
        observation_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        _forward method for the `LearningHmmFilter` class

        Arguments
        ---------
        observation_trajectory : torch.Tensor
            shape (batch_size, horizon)

        Returns
        -------
        estimated_state_trajectory: torch.Tensor
            shape (batch_size, horizon, state_dim)
        predicted_state_trajectory: torch.Tensor
            shape (batch_size, horizon, state_dim)
        """

        # # Get emission matrix
        # emission_matrix = (
        #     self._emission.matrix
        # )  # (state_dim, discrete_observation_dim)

        # # Get emission based on each observation in the trajectory
        # emission_trajectory = torch.moveaxis(
        #     emission_matrix[:, observation_trajectory], 0, 2
        # )  # (batch_size, horizon, state_dim)

        emission_trajectory = self._emission(observation_trajectory)

        (
            estimated_state_trajectory,  # (batch_size, horizon, state_dim)
            predicted_state_trajectory,  # (batch_size, horizon, state_dim)
        ) = self._transition(emission_trajectory)

        # return estimated_state_trajectory, estimation_trajectory

        return (
            estimated_state_trajectory,
            predicted_state_trajectory,
            # emission_matrix,
        )

    def reset(self) -> None:
        # self._is_initialized = False
        self._emission.reset()
        self._transition.reset()
        self._estimation.reset()

    # def _check_batch(self, batch_size: int, device: torch.device) -> None:
    #     if self._is_initialized:
    #         assert batch_size == self._batch_size, (
    #             f"batch_size must be the same as the initialized batch_size. "
    #             f"batch_size given is {batch_size} while the initialized batch_size is {self._batch_size}."
    #         )
    #         return

    #     _device: Device
    #     match device:
    #         case torch.device(type="cpu"):
    #             _device = Device.CPU
    #         case torch.device(type="cuda"):
    #             _device = Device.CUDA
    #         case torch.device(type="mps"):
    #             _device = Device.MPS
    #         case _:
    #             logger.warning(
    #                 f"device {device} is not supported. "
    #                 f"device will be set to CPU."
    #             )
    #             _device = Device.CPU

    #     self._init_batch(
    #         batch_size=batch_size, device=_device, is_initialized=True
    #     )

    @torch.inference_mode()
    def update(self, observation: torch.Tensor) -> None:
        """
        Update the estimated state based on the observation.

        Arguments:
        ----------
        observation : torch.Tensor
            shape = (batch_size, horizon) or (batch_size, horizon, discrete_observation_dim)
        """
        observation = self._emission.validate_observation_shape(
            observation, number_of_systems=self.number_of_systems
        )

        # if observation.ndim == 0:
        #     observation = observation.unsqueeze(0)  # (horizon=1,)
        # if observation.ndim == 1:
        #     if self._number_of_systems == 1:
        #         observation = observation.unsqueeze(
        #             0
        #         )  # (batch_size=1, horizon)
        #     else:
        #         observation = observation.unsqueeze(
        #             1
        #         )  # (batch_size, horizon=1)
        # assert observation.ndim == 2, (
        #     f"observation must be in the shape of (batch_size, horizon). "
        #     f"observation given has the shape of {observation.shape}."
        # )

        # self._estimation.check_batch(
        #     batch_size=observation_trajectory.shape[0],
        #     device=observation_trajectory.device,
        # )

        # estimated_state_trajectory, predicted_next_state_trajectory, _ = (
        #     self._forward(observation_trajectory)
        # )
        (
            estimated_state_trajectory,
            predicted_state_trajectory,
        ) = self._forward(
            observation  # (batch_size, horizon)
        )
        self._filter.estimated_state = estimated_state_trajectory[:, -1, :]
        self._filter.predicted_state = predicted_state_trajectory[:, -1, :]

        # self._estimated_state = estimated_state_trajectory[
        #     :, -1, :
        # ]  # (batch_size, state_dim)
        # self._predicted_next_state = predicted_next_state_trajectory[
        #     :, -1, :
        # ]  # (batch_size, state_dim)
        # self._predicted_next_observation_probability = self._emission(
        #     self._predicted_next_state,  # (batch_size, state_dim)
        # )  # (batch_size, discrete_observation_dim)

    @torch.inference_mode()
    def estimate(self) -> torch.Tensor:
        """
        Compute the estimation. This method should be called after the `update` method.

        Returns
        -------
        estimation: torch.Tensor
            Based on the estimation option in the configuration, the chosen estimation will be returned.
        """
        # self._estimation = self._estimate()

        # return self._estimation.result
        estimation: torch.Tensor = self._estimation(
            self._filter.estimated_state,  # (batch_size, state_dim)
            self._filter.predicted_state,  # (batch_size, state_dim)
        )
        return estimation.detach()

    # @torch.inference_mode()
    # def _estimate(self) -> torch.Tensor:
    #     match self.estimation_option:
    #         case (
    #             Config.EstimationConfig.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY_OVER_LAYERS
    #         ):
    #             estimation: torch.Tensor = self._emission(
    #                 self._transition.predicted_next_state_over_layers  # (batch_size, layer_dim, state_dim)
    #             )  # (batch_size, layer_dim, discrete_observation_dim)
    #             if self._batch_size == 1:
    #                 estimation = estimation.unsqueeze(0)
    #         case (
    #             Config.EstimationConfig.Option.PREDICTED_NEXT_STATE_OVER_LAYERS
    #         ):
    #             estimation = (
    #                 self._transition.predicted_next_state_over_layers
    #             )  # (batch_size, layer_dim, state_dim)
    #             if self._batch_size == 1:
    #                 estimation = estimation.unsqueeze(0)
    #         case Config.EstimationConfig.Option.ESTIMATED_STATE:
    #             estimation = self.estimated_state
    #         case Config.EstimationConfig.Option.PREDICTED_NEXT_STATE:
    #             estimation = self.predicted_next_state
    #         case (
    #             self._config.estimation.Option.PREDICTED_NEXT_OBSERVATION_PROBABILITY
    #         ):
    #             estimation = self.predicted_next_observation_probability
    #         case _ as _invalid_estimation_option:
    #             assert_never(_invalid_estimation_option)
    #     return estimation

    # @torch.inference_mode()
    # def predict(self) -> torch.Tensor:
    #     """
    #     Predict the next observation.

    #     Returns
    #     -------
    #     predicted_observation: torch.Tensor
    #         shape = (batch_size, 1) or (1,) if batch_size is 1
    #     """
    #     # predicted_next_observation = torch.multinomial(
    #     #     self._config.prediction.process_probability(
    #     #         self.predicted_next_observation_probability,
    #     #     ),
    #     #     1,
    #     #     replacement=True,
    #     # )  # (batch_size, 1) or (1,) if batch_size is 1

    #     predicted_next_observation = torch.multinomial(
    #         self._config.prediction.process_probability(self.estimate()),
    #         1,
    #         replacement=True,
    #     )  # (batch_size, 1) or (1,) if batch_size is 1

    #     return predicted_next_observation.detach()
