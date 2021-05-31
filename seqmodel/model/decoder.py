from seqmodel.model import HparamModule


class LinearDecoder(HparamModule):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--decode_dropout', default=None, type=float,
                            help='dropout between linear layers, set to --dropout if None')
        parser.add_argument('--n_decode_layers', default=2, type=int,
                            help='number of linear layers')
        return parser

    def __init__(in_dims: int, out_dims: int, **hparams):
        # note: Task defines number of input/output dimensions
        super().__init__(**hparams)
        #TODO

    def forward(self, x):
        return x #TODO
