from ss.estimation.estimator import Estimator


class Filter(Estimator):

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        number_of_systems: int = 1,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            observation_dim=observation_dim,
            horizon_of_observation_history=1,
            number_of_systems=number_of_systems,
        )
