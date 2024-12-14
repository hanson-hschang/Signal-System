from lss._lss import BaseLearningModule, BaseLearningParameters, Mode
from lss.utility.registration import register_subclasses

register_subclasses(BaseLearningParameters, "lss")
