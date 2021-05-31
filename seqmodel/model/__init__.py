import abc
import torch.nn as nn
from seqmodel.hparam import Hparams


class HparamModule(nn.Module, Hparams, abc.ABC):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--dropout', default=0., type=float,
                            help='proportion between [0., 1.] of dropout to apply between module layers.')
        return parser
