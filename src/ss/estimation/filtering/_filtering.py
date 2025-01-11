from typing import Callable, Optional

from ss.estimation import Estimator


class Filter(Estimator):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        estimation_model: Optional[Callable] = None,
        number_of_systems: int = 1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            horizon_of_observation_history=1,
            estimation_model=estimation_model,
            number_of_systems=number_of_systems,
        )
