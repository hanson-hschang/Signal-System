import numpy as np

from ss.estimation.filtering.hmm.learning.module import LearningDualHmmFilter
from ss.estimation.filtering.hmm.learning.module.emission import EmissionModule
from ss.estimation.filtering.hmm.learning.module.estimation import (
    EstimationModule,
)
from ss.estimation.filtering.hmm.learning.module.transition import (
    DualTransitionModule,
)
from ss.utility.learning.process import BaseLearningProcess
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def emission_module_info(
    emission: EmissionModule,
    # layer_dim: int,
) -> None:
    emission_matrix = emission.matrix.numpy()
    # logger.info(f"(layer 0 / {layer_dim}) learned emission module:")
    logger.info("learned emission module:")
    logger.info("    emission matrix:")
    for k in range(emission_matrix.shape[0]):
        logger.info(f"        {emission_matrix[k]}")


def transition_module_info(
    transition: DualTransitionModule,
    # layer_dim: int,
) -> None:
    # for transition_layer in transition.layers:
    #     transition_layer_info(transition_layer)
    pass


def estimation_module_info(estimation: EstimationModule) -> None:
    logger.info("learned estimation module:")
    logger.info("    estimation matrix:")
    estimation_matrix = estimation.matrix.numpy()
    for k in range(estimation_matrix.shape[0]):
        logger.info(f"        {estimation_matrix[k]}")


def module_info(learning_filter: LearningDualHmmFilter) -> None:
    # logger.info(f"Module: {learning_filter}")

    logger.info("")
    logger.info("learned filter information:")
    logger.info(
        f"state dimension: {learning_filter.state_dim}", indent_level=1
    )
    logger.info(
        f"observation dimension: {learning_filter.discrete_observation_dim}",
        indent_level=1,
    )
    logger.info(
        f"estimation dimension: {learning_filter.estimation_dim}",
        indent_level=1,
    )

    # layer_dim = learning_filter.layer_dim - 1
    np.set_printoptions(precision=3, suppress=True)
    with BaseLearningProcess.inference_mode(learning_filter):
        emission_module_info(learning_filter.emission)
        transition_module_info(learning_filter.transition)
        estimation_module_info(learning_filter.estimation)

    total_parameters = 0
    trainable_parameters = 0
    non_trainable_parameters = 0
    for name, p in learning_filter.named_parameters():
        number_of_parameters = p.numel()
        total_parameters += number_of_parameters
        if p.requires_grad:
            trainable_parameters += number_of_parameters
        else:
            non_trainable_parameters += number_of_parameters

    logger.info(f"total number of parameters: {total_parameters}")
    logger.info(
        f"trainable parameters: {trainable_parameters} "
        f"({trainable_parameters / total_parameters * 100:.2f}%)",
        indent_level=1,
    )
    logger.info(
        f"non-trainable parameters: {non_trainable_parameters} "
        f"({non_trainable_parameters / total_parameters * 100:.2f}%)",
        indent_level=1,
    )
