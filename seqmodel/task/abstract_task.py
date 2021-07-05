import abc
from itertools import chain
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from seqmodel.hparam import TaskHparams
from seqmodel.dataset import Dataset


class Task(pl.LightningModule, abc.ABC):
    """A trainable task following pytorch-lightning structure.
    """
    def __init__(self,
        hparams: TaskHparams,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.hp = hparams
        self.save_hyperparameters('hparams')
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
        print(self.decoder.parameters())
        optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.hp.lr,
            betas=(self.hp.adam_beta_1, self.hp.adam_beta_2),
            eps=self.hp.adam_eps,
            weight_decay=self.hp.weight_decay,
        )
        return optimizer  #TODO lr scheduler

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        """Wrapper to call default_hparams for pytorch-lightning.

        Args:
            parent_parser (ArgumentParser): parser

        Returns:
            ArgumentParser: parser with default_hparams added
        """
        return TaskHparams.to_parser(parent_parser)
