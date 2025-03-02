from typing import cast

import numpy as np
import torch

from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module.emission import (
    EmissionProcess,
)
from ss.estimation.filtering.hmm.learning.module.transition import (
    TransitionProcess,
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


def emission_process_info(
    emission_process: EmissionProcess,
    layer_dim: int,
) -> None:
    emission_matrix = emission_process.matrix.numpy()
    emission_matrix_temperature = (
        cast(
            torch.Tensor,
            emission_process.matrix_parameter.transformer.temperature(),
        )
        .detach()
        .numpy()
    )
    logger.info(f"(layer 0 / {layer_dim}) learned emission process:")
    logger.info(f"    emission matrix:")
    for k in range(emission_matrix.shape[0]):
        logger.info(
            f"        {emission_matrix[k]} with temperature = {emission_matrix_temperature[k]}"
        )


def transition_block_info(
    transition_block: BaseTransitionBlock,
    block_dim: int,
) -> None:
    start_index = len("Transition")
    block_type_name = type(transition_block).__name__[start_index:]

    logger.info(
        f"    (block {transition_block.id+1} / {block_dim}) learned transition block ({block_type_name}):"
    )
    initial_state = transition_block.initial_state.detach().numpy()
    initial_state_temperature = (
        cast(
            torch.Tensor,
            transition_block.initial_state_parameter.transformer.temperature(),
        )
        .detach()
        .numpy()
    )
    logger.info("        initial state:")
    logger.info(
        f"            {initial_state} with temperature = {initial_state_temperature}"
    )

    transition_matrix = transition_block.matrix.detach().numpy()
    transition_matrix_temperature = (
        cast(
            torch.Tensor,
            transition_block.matrix_parameter.transformer.temperature(),
        )
        .detach()
        .numpy()
    )

    logger.info(f"        transition matrix:")
    for k in range(transition_matrix.shape[0]):
        _transition_matrix_temperature = (
            transition_matrix_temperature[k]
            if isinstance(transition_block, TransitionFullMatrix)
            else transition_matrix_temperature
        )
        logger.info(
            f"            {transition_matrix[k]} with temperature = {_transition_matrix_temperature}"
        )


def transition_layer_info(
    transition_layer: TransitionLayer,
    layer_dim: int,
) -> None:
    block_dim = transition_layer.block_dim
    logger.info(
        f"(layer {transition_layer.id} / {layer_dim}) learned transition process ({block_dim} block(s)):"
    )

    transition_matrix = transition_layer.matrix.detach().numpy()
    logger.info(f"    equivalent transition matrix:")
    for k in range(transition_matrix.shape[0]):
        logger.info(f"        {transition_matrix[k]}")

    coefficient = transition_layer.coefficient.detach().numpy()
    coefficient_temperature = (
        cast(
            torch.Tensor,
            transition_layer.coefficient_parameter.transformer.temperature(),
        )
        .detach()
        .numpy()
    )
    logger.info(f"    block coefficient(s) for each state:")
    for k in range(coefficient.shape[0]):
        logger.info(
            f"        state {k+1}: {coefficient[k]} with temperature = {coefficient_temperature[k]}"
        )

    for transition_block in transition_layer.blocks:
        transition_block_info(transition_block, block_dim)


def transition_process_info(
    transition_process: TransitionProcess,
    layer_dim: int,
) -> None:
    for transition_layer in transition_process.layers:
        transition_layer_info(transition_layer, layer_dim)


def module_info(learning_filter: LearningHmmFilter) -> None:
    # logger.info(f"Module: {learning_filter}")

    logger.info("")
    logger.info("learned filter information:")
    logger.info(f"    state dimension: {learning_filter.state_dim}")
    logger.info(
        f"    observation dimension: {learning_filter.discrete_observation_dim}"
    )

    layer_dim = learning_filter.layer_dim - 1
    np.set_printoptions(precision=3, suppress=True)
    with BaseLearningProcess.inference_mode(learning_filter):
        emission_process_info(learning_filter.emission_process, layer_dim)
        transition_process_info(learning_filter.transition_process, layer_dim)

    logger.info(
        f"total number of parameters: {sum(p.numel() for p in learning_filter.parameters())}"
    )
