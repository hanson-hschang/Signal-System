from typing import cast

import numpy as np
import torch

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module.emission import (
    EmissionModule,
)
from ss.estimation.filtering.hmm.learning.module.estimation import (
    EstimationModule,
)
from ss.estimation.filtering.hmm.learning.module.transition import (
    TransitionModule,
)
from ss.estimation.filtering.hmm.learning.module.transition.block import (
    BaseTransitionBlock,
    TransitionFullMatrix,
)
from ss.estimation.filtering.hmm.learning.module.transition.layer import (
    TransitionLayer,
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
    logger.info(f"learned emission module:")
    logger.info(f"    emission matrix:")
    for k in range(emission_matrix.shape[0]):
        logger.info(f"        {emission_matrix[k]}")


def transition_module_info(
    transition: TransitionModule,
    # block_dim: int,
    # show_initial_state: bool = True,
) -> None:
    start_index = len("Transition")
    block_type_name = type(transition).__name__[start_index:]

    # logger.info(
    #     f"    (block {transition.id+1} / {block_dim}) learned transition block ({block_type_name}):"
    # )
    logger.info(f"    learned transition ({block_type_name}):")
    # if show_initial_state:
    initial_state = transition.initial_state.detach().numpy()
    logger.info("        initial state:")
    logger.info(f"            {initial_state}")

    transition_matrix = transition.matrix.detach().numpy()
    logger.info(f"        transition matrix:")
    for k in range(transition_matrix.shape[0]):
        logger.info(f"            {transition_matrix[k]}")


# def transition_layer_info(
#     transition_layer: TransitionLayer,
#     # layer_dim: int,
# ) -> None:
#     block_dim = transition_layer.block_dim
#     # logger.info(
#     #     f"(layer {transition_layer.id} / {layer_dim}) learned transition module ({block_dim} block(s)):"
#     # )
#     logger.info(f"learned transition module ({block_dim} block(s)):")
#     coefficient = transition_layer.coefficient.detach().numpy()
#     if transition_layer.block_state_binding:
#         initial_state = transition_layer.initial_state.detach().numpy()
#         logger.info("    initial state:")
#         logger.info(f"        {initial_state}")
#         transition_matrix = transition_layer.matrix.detach().numpy()
#         coefficient = transition_layer.coefficient.detach().numpy()
#         logger.info(f"    equivalent transition matrix:")
#         for k in range(transition_matrix.shape[0]):
#             logger.info(
#                 f"        {transition_matrix[k]} with block coefficient(s) {coefficient[k]}"
#                 if transition_layer.block_dim > 1
#                 else f"        {transition_matrix[k]}"
#             )
#     else:
#         if transition_layer.block_dim > 1:
#             logger.info(f"    block coefficient(s):")
#             logger.info(f"        {coefficient}")

#     for transition_block in transition_layer.blocks:
#         transition_block_info(
#             transition_block,
#             block_dim,
#             show_initial_state=not transition_layer.block_state_binding,
#         )


# def transition_module_info(
#     transition: TransitionModule,
#     # layer_dim: int,
# ) -> None:
#     for transition_layer in transition.layers:
#         transition_layer_info(transition_layer)


def estimation_module_info(estimation: EstimationModule) -> None:
    logger.info("learned estimation module:")
    logger.info(f"    estimation matrix:")
    estimation_matrix = estimation.matrix.numpy()
    for k in range(estimation_matrix.shape[0]):
        logger.info(f"        {estimation_matrix[k]}")


def module_info(learning_filter: LearningHmmFilter) -> None:
    # logger.info(f"Module: {learning_filter}")

    logger.info("")
    logger.info("learned filter information:")
    logger.info(f"    state dimension: {learning_filter.state_dim}")
    logger.info(
        f"    observation dimension: {learning_filter.discrete_observation_dim}"
    )
    logger.info(f"    estimation dimension: {learning_filter.estimation_dim}")

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
        f"    trainable parameters: {trainable_parameters} ({trainable_parameters/total_parameters*100:.2f}%)"
    )
    logger.info(
        f"    non-trainable parameters: {non_trainable_parameters} ({non_trainable_parameters/total_parameters*100:.2f}%)"
    )
