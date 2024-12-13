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

    x = torch.tensor([1.0, 2.0])
    for _ in range(5):
        filter.update(observation=x)
        x = filter.estimate()
        print(x)


if __name__ == "__main__":
    main()
