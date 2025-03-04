import torch
import torch.nn as nn


def prediction(
    estimated_state: torch.Tensor,
    transition_matrix: torch.Tensor,
) -> torch.Tensor:
    predicted_next_state = torch.matmul(estimated_state, transition_matrix)
    return predicted_next_state


def update(
    prior_state: torch.Tensor,
    likelihood_state: torch.Tensor,
) -> torch.Tensor:
    # update step based on likelihood_state (conditional probability)
    posterior_state = nn.functional.normalize(
        prior_state * likelihood_state,
        p=1,
        dim=1,
    )  # (batch_size, state_dim)
    return posterior_state
