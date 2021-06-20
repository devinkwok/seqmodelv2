import torch
import torch.nn as nn
from seqmodel import Hparams


class LinearDecoder(Hparams, nn.Module):
    """Combines multiple linear layers.
    """
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
        ActivationFn: nn.Module = nn.ReLU,
        DropoutFn: nn.Module = nn.Dropout,
        **hparams,
    ):
        """
        Args:
            in_dims (int): input dimensions
            out_dims (int): output dimensions
            ActivationFn (nn.Module): type of activation function
                to apply between feedforward (Linear) layers, use lambda
                function to specify args if needed, since instances
                are created as `ActivationFn()`.
            DropoutFn (nn.Module): type of dropout to apply between
                feedforward (Linear) layers.
        """
        super().__init__(**hparams)
        # define layer dimensions
        layers = []
        decode_dims = self.hparams.decode_dims
        if decode_dims is None:
            decode_dims = 2 * in_dims
        dims = [decode_dims] * (self.hparams.n_decode_layers + 1)
        dims[0] = in_dims
        dims[-1] = out_dims
        # create layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out, 1))
            layers.append(ActivationFn())
            if self.hparams.decode_dropout > 0:
                layers.append(DropoutFn(
                    p=self.hparams.decode_dropout, inplace=True))

        layers = layers[:-1]  # remove last activation layer
        if self.hparams.decode_dropout > 0:
            layers = layers[:-1]  # remove last dropout layer
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layers in the order Linear, Activation, Dropout,
        except in the final layer where only Linear is applied.

        Args:
            x (torch.Tensor): input whose last dimension is `in_dims`

        Returns:
            torch.Tensor: output whose last dimension is `out_dims`,
                remaining dimensions are identical to input.
        """
        return self.model.forward(x)
