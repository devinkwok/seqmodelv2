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
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--adam_beta_1', default=0.9, type=float)
        parser.add_argument('--adam_beta_2', default=0.99, type=float)
        parser.add_argument('--adam_eps', default=1e-6, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
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
    def loss_fn(self, model_output: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
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
