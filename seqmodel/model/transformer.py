from seqmodel.model import HparamModule


class PositionEncoder(HparamModule):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--posencoder_dropout', default=None, type=float,
                            help='dropout after positional encoder, set to --dropout if None')
        return parser

    def forward(self, x):
        return x #TODO


class TransformerEncoder(HparamModule):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--n_repr_dims', default=512, type=int,
                            help='number of dimensions in representation layer')
        parser.add_argument('--n_feedforward_dims', default=None, type=int,
                            help='number of dimensions in feedforward (fully connected) layer, ' +
                                'if None set to 2*n_repr_dims')
        parser.add_argument('--n_heads', default=4, type=int,
                            help='number of attention heads')
        parser.add_argument('--n_layers', default=4, type=int,
                            help='number of attention layers')
        parser.add_argument('--dropout', default=0., type=float,
                            help='proportion between [0., 1.] of dropout to apply between module layers.')
        return parser

    def forward(self, x):
        return x #TODO
