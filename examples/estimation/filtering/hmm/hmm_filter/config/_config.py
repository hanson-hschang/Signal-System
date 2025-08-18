from typing import Optional

from dataclasses import dataclass

from ss.utility.click import BaseClickConfig
from ss.utility.learning.process.config import ProcessConfig


@dataclass
class UserConfig(BaseClickConfig):
    simulation_time_steps: int = 30  # The simulation time steps
    step_skip: int = 1
    state_dim: int = 3
    discrete_observation_dim: int = 7
    number_of_systems: int = 1
    random_seed: int = 2024  # The random seed (non-negative integers)
