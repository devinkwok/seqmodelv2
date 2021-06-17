import torch.nn as nn
from seqmodel import Hparams
# from seqmodel.model.abstract_module import AbstractModule


class LinearDecoder(Hparams, nn.Module):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--decode_dims', default=None, type=int,
                            help='number of dimensions in intermediate layers, ' + \
                                'if None set to 2*in_dims')
        parser.add_argument('--n_decode_layers', default=2, type=int,
                            help='number of linear layers')
        parser.add_argument('--decode_dropout', default=0., type=float,
                            help='dropout between linear layers')
        return parser

    def __init__(self,
        in_dims: int,
        out_dims: int,
        ActivationFunction: nn.Module = nn.ReLU,
        DropoutFunction: nn.Module = nn.Dropout,
        **hparams,
    ):
        super().__init__(**hparams)
        # define layer dimensions
        layers = []
        decode_dims = self.hparams.decode_dims
        if self.hparams.decode_dims is None:
            decode_dims = in_dims * 2
        dims = [decode_dims] * (self.hparams.n_decode_layers + 1)
        dims[0] = in_dims
        dims[-1] = out_dims
        # create layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Conv1d(d_in, d_out, 1))
            layers.append(ActivationFunction())
            if self.hparams.decode_dropout > 0:
                layers.append(DropoutFunction(
                    p=self.hparams.decode_dropout, inplace=True))

        layers = layers[:-1]  # remove last activation layer
        if self.hparams.decode_dropout > 0:
            layers = layers[:-1]  # remove last dropout layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # transpose (N, L, C) to (N, C, L) for Conv1d layers
        x = x.transpose(1, 2)
        x = self.model.forward(x)
        # transpose back to (N, L, C)
        return x.transpose(1, 2)
