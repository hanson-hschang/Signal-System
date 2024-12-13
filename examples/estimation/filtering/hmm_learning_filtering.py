import os
from pathlib import Path

import torch

from lss.estimation.filtering.hmm_filtering import (
    LearningHiddenMarkovModelFilter,
    LearningHiddenMarkovModelFilterParameters,
)


def main() -> None:
    params = LearningHiddenMarkovModelFilterParameters(
        state_dim=1,
        observation_dim=2,
        horizon_of_observation_history=5,
    )
    filter = LearningHiddenMarkovModelFilter(params)

    # parent_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    # result_folder_directory = parent_directory / Path(__file__).stem
    # filter.save(result_folder_directory / "filter.pt")

    x = torch.tensor([1.0, 2.0])
    for _ in range(5):
        filter.update(observation_trajectory=x)
        filter.estimate()
        print(filter.estimated_next_observation)


if __name__ == "__main__":
    main()
