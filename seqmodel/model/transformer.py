import torch.nn as nn
from seqmodel import Hparams


class PositionEncoder(Hparams, nn.Module):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--posencoder_dropout', default=0., type=float,
                            help='dropout after positional encoder')
        return parser

    def forward(self, x):
        return x #TODO


class TransformerEncoder(Hparams, nn.Module):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--repr_dims', default=512, type=int,
                            help='number of dimensions in representation layer')
        parser.add_argument('--feedforward_dims', default=None, type=int,
                            help='number of dimensions in feedforward (fully connected) layer, ' +
                                'if None set to 2*repr_dims')
        parser.add_argument('--n_heads', default=4, type=int,
                            help='number of attention heads')
        parser.add_argument('--n_layers', default=4, type=int,
                            help='number of attention layers')
        parser.add_argument('--dropout', default=0., type=float,
                            help='proportion between [0., 1.] of dropout to apply between module layers.')
        return parser

    def forward(self, x):
        return x #TODO
