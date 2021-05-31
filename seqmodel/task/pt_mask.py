from argparse import ArgumentParser
from seqmodel.task import Task
import pytorch_lightning as pl
from seqmodel.hparam import Hparams


def masked_sequence():
    # generate mask
    # replace masked positions
    # randomize positions
    # make tensors
    pass #TODO


class PtMask(Task):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--keep_prop', default=0.01, type=float,
                            help='proportion between [0., 1.] of sequence positions to apply identity loss.')
        parser.add_argument('--mask_prop', default=0.13, type=float,
                            help='proportion between [0., 1.] of sequence positions to mask.')
        parser.add_argument('--random_prop', default=0.01, type=float,
                            help='proportion between [0., 1.] of sequence positions to randomize.')
        parser.add_argument('--val_keep_prop', default=None, type=float,
                            help='same as keep_prop for validation, use keep_prop if None.')
        parser.add_argument('--val_mask_prop', default=None, type=float,
                            help='same as mask_prop for validation, use mask_prop if None.')
        parser.add_argument('--val_random_prop', default=None, type=float,
                            help='same as random_prop for validation, use random_prop if None.')
        return parser

    def __init__(self):
        pass #TODO

    def forward(self):
        pass #TODO

    def training_step(self):
        pass #TODO

    def validation_step(self):
        pass #TODO

    def validation_epoch_end(self):
        pass #TODO
