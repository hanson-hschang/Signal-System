from typing import Optional

from dataclasses import dataclass

from ss.utility.click import BaseClickConfig
from ss.utility.learning.process import BaseLearningProcess


@dataclass
class ClickConfig(BaseClickConfig):
    mode: BaseLearningProcess.Mode = (
        BaseLearningProcess.Mode.INFERENCE
    )  # The learning process mode [ training | analysis | inference ]
    data_foldername: str = (
        "hmm_filter_result"  # The foldername where the data is stored
    )
    data_filename: str = "system_train.hdf5"  # The filename of the data
    model_foldername: Optional[str] = (
        None  # The foldername where the model is stored
    )
    model_filename: str = "learning_filter"  # The filename of the model
    continue_training: bool = False  # Whether to continue training the model
