import abc
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from seqmodel import Hparams
from seqmodel.dataset import Dataset


class Task(pl.LightningModule, Hparams, abc.ABC):
    """A trainable task following pytorch-lightning structure
    """
    @staticmethod
    def _default_hparams(parser):
        # batch
        parser.add_argument('--accumulate_grad_batches', default=1, type=int,
                            help='average over this many batches before backprop (pytorch_lightning)')
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
                            help='limit max abs gradient value, ' + \
                                'no clipping if 0 (pytorch lightning)')
        return parser

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        """Wrapper to call default_hparams for pytorch-lightning.

        Args:
            parent_parser (ArgumentParser): parser

        Returns:
            ArgumentParser: parser with default_hparams added
        """
        return cls.default_hparams(parent_parser)

    def __init__(self, train_dataset: Dataset,
        valid_dataset: Dataset, test_dataset: Dataset,
        encoder: nn.Module, decoder: nn.Module, **hparams
    ):
        super(pl.LigtningModule, self).__init__(**hparams)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = self.configure_loss_fn()

    @abc.abstractmethod
    def configure_loss_fn(self) -> nn.Module:
        """Calculate loss (defined by subclass).

        Args:
            model_output (torch.Tensor): output of `self.model.forward()`
            target (torch.Tensor): [description]
        """
        pass

    def train_dataloader(self):
        return self.train_dataset.dataloader()

    def val_dataloader(self):
        return self.valid_dataset.dataloader()

    def test_dataloader(self):
        return self.test_dataset.dataloader()

    def forward(self, x, **encoder_options):
        repr, intermediates = self.encoder.forward(x, **encoder_options)
        out = self.decoder.forward(x)
        return out, repr, intermediates

    def step(self, batch, batch_idx, compute_log_stats=True):
        # separate batch
        src, tgt, metadata = batch
        out, repr, intermediates = self.forward(src)
        # don't need to mask src/tgt here because ignore_index is set
        loss = self.loss_fn(out, tgt)
        # logging 
        #TODO use compute_log_stats here
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, compute_log_stats=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, compute_log_stats=False)

    def validation_epoch_end(self):
        pass #TODO compute log stats

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [self.encoder, self.decoder],
            lr=self.hparams.lr,
            betas=(self.hparams.adam_beta_1, self.hparams.adam_beta_2),
            eps=self.hparams.adam_eps,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer  #TODO lr scheduler

    def n_steps(self):
        """Return effective number of iterations, as true number of
        backpropagation steps since model initialization.
        Include iterations from restored checkpoint, but not pretrained encoder.
        """
        pass #TODO
