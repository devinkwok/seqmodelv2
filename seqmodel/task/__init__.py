import abc
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from seqmodel.hparam import Hparams
from seqmodel.dataset import Dataset


class Task(pl.LightningModule, Hparams, abc.ABC):
    """A trainable task following pytorch-lightning structure
    """
    @staticmethod
    def _default_hparams(parser):
        # batch
        parser.add_argument('--batch_size', default=16, type=int,
                            help='number of samples in each training minibatch')
        parser.add_argument('--valid_batch_size', default=None, type=int,
                            help='number of samples in each validation minibatch,' +
                            ' set to --batch_size if None')
        parser.add_argument('--test_batch_size', default=None, type=int,
                            help='number of samples in each test minibatch,' +
                            ' set to --batch_size if None')
        parser.add_argument('--accumulate_grad_batches', default=1, type=int,
                            help='average over this many batches before backprop')
        # optimizer
        parser.add_argument('--lr', default=3e-4, type=float,
                            help='learning rate')
        parser.add_argument('--adam_beta_1', default=0.9, type=float,
                            help='beta 1 parameter for Adam optimizer')
        parser.add_argument('--adam_beta_2', default=0.99, type=float,
                            help='beta 2 parameter for Adam optimizer')
        parser.add_argument('--adam_eps', default=1e-6, type=float,
                            help='epsilon parameter for Adam optimizer')
        parser.add_argument('--weight_decay', default=0.01, type=float,
                            help='weight decay for Adam optimizer')
        parser.add_argument('--gradient_clip_val', default=10., type=float,
                            help='limit max abs gradient value, no clipping if 0')
        return parser

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Wrapper to call default_hparams for pytorch-lightning.

        Args:
            parent_parser (ArgumentParser): parser

        Returns:
            ArgumentParser: parser with default_hparams added
        """
        return default_hparams(parent_parser)

    def __init__(self, dataset: Dataset, encoder: nn.Module, decoder: nn.Module, **hparams):
        super(pl.LigtningModule, self).__init__(**hparams)
        self.dataset = dataset
        self.encoder = encoder
        self.decoder = decoder

    @abc.abstractmethod
    def loss_fn(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss (defined by subclass).

        Args:
            model_output (torch.Tensor): output of `self.model.forward()`
            target (torch.Tensor): [description]
        """
        pass

    def train_dataloader(self):
        return self.dataset.dataloader('train')

    def val_dataloader(self):
        return self.dataset.dataloader('valid')

    def test_dataloader(self):
        return self.dataset.dataloader('test')

    def forward(self, x):
        repr = self.encoder.forward(x)
        out = self.decoder.forward(x)
        return out, repr

    def training_step(self):
        # forward
        # loss
        # logging
        pass #TODO

    def validation_step(self):
        pass #TODO

    def validation_epoch_end(self):
        pass #TODO

    def configure_optimizers(self):
        pass #TODO
