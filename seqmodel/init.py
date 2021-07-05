import abc
from argparse import ArgumentParser
from seqmodel.dataset.seq import FastaSequence
import torch.nn as nn
import pytorch_lightning as pl
from seqmodel.hparam import *
from seqmodel.dataset import DnaAlphabet
from seqmodel.dataset import SeqIntervalDataset
from seqmodel.model import LinearDecoder
from seqmodel.model import PositionEncoder
from seqmodel.model import TransformerEncoder
from seqmodel.task import Task
from seqmodel.task import PtMask
from seqmodel.task import MatFileDataset
from seqmodel.task import FtDeepSea


class Initializer(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def hparam_collection() -> type:
        """Collection of hparams to initialize.

        Returns:
            type: a subclass of HparamCollection
        """
        return RunHparamCollection

    def __init__(self, **hparams):
        """Creates task objects from hparams.
        """
        collection_class = self.hparam_collection()
        self.hparams = collection_class(**hparams)
        self.alphabet = DnaAlphabet()
        self.repr_dims = self.hparams[TransformerEncoderHparams]['repr_dims']
        self.pos_encoder = PositionEncoder(self.hparams[PositionEncoderHparams],
                                        len(self.alphabet), self.repr_dims)
        self.encoder = TransformerEncoder(self.hparams[TransformerEncoderHparams],
                                    self.pos_encoder, ActivationFn=nn.GELU,
                                    DropoutFn=nn.Dropout,
                                    LayerNormFn=nn.LayerNorm)

    def run(self, args):
        pl.seed_everything(0)
        # args['callbacks'] = []
        trainer = pl.Trainer.from_argparse_args(args)
        hparams = vars(args)
        if hparams['init_mode'] == 'train':
            trainer.fit(self.task)
        elif hparams['init_mode'] == 'test':
            trainer.test(self.task)


class PtMaskInitializer(Initializer):

    @staticmethod
    def hparam_collection() -> type:
        return PtMaskHparamCollection

    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.dataset = SeqIntervalDataset(self.hparams[SeqIntervalDatasetHparams],
                                            self.alphabet)
        self.decoder = LinearDecoder(self.hparams[LinearDecoderHparams],
                                self.repr_dims, self.alphabet.n_char)
        self.task = PtMask(self.hparams[PtMaskHparams],
                        self.dataset, self.dataset, self.dataset,
                        self.encoder, self.decoder)


class FtDeepSeaInitializer(Initializer):

    @staticmethod
    def hparam_collection() -> type:
        return FtDeepSeaHparamCollection

    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.dataset = MatFileDataset(**hparams)
        self.decoder = LinearDecoder(self.hparams[LinearDecoderHparams],
                                self.repr_dims, self.dataset.target_dims)
        self.task = FtDeepSea(self.hparams[FtDeepSeaHparams],
                            self.dataset, self.dataset, self.dataset,
                            self.encoder, self.decoder)
