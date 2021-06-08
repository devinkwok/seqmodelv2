import abc
import torch.nn as nn
from seqmodel.hparam import Hparams


class HparamModule(nn.Module, Hparams, abc.ABC):

    @staticmethod
    def _default_hparams(parser):
        return parser
