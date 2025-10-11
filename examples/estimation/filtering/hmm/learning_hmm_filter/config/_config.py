from dataclasses import dataclass

from ss.utility.click import BaseClickConfig
from ss.utility.learning.process.config import ProcessConfig


@dataclass
class UserConfig(BaseClickConfig):
    mode: ProcessConfig.Mode = (
        ProcessConfig.Mode.INFERENCE
    )  # The learning process mode [ training | analysis | inference ]
    data_foldername: str = (
        "hmm_filter_result"  # The foldername where the data is stored
    )
    data_filename: str = "system_train.hdf5"  # The filename of the data
    model_foldername: str | None = (
        None  # The foldername where the model is stored
    )
    model_filename: str = "learning_filter"  # The filename of the model
    continue_training: bool = False  # Whether to continue training the model
