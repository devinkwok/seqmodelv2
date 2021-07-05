import typing
import torch
from torch import random
import torch.nn as nn
from seqmodel.hparam import PtMaskHparams
from seqmodel.dataset import Dataset
from seqmodel.dataset.transforms import DataTransform
from seqmodel.task.abstract_task import Task


class MaskSequence(DataTransform):

    def __init__(self, keep_prop, mask_prop, random_prop):
        super().__init__({0})
        if keep_prop < 0 or mask_prop < 0 or random_prop < 0:
            raise ValueError('Proportions less than 0: ' + \
                f'keep={keep_prop}, mask={mask_prop}, random={random_prop}.')
        EPSILON = 1e-6
        if keep_prop + mask_prop + random_prop > 1. + EPSILON:
            raise ValueError('Proportions sum to greater than 1: ' + \
                f'keep={keep_prop}, mask={mask_prop}, random={random_prop}.')
        # cutoff values for CDF of discrete distribution
        self.random_cutoff = random_prop                # smallest
        self.mask_cutoff = self.random_prop + mask_prop
        self.keep_cutoff = self.mask_cutoff + keep_prop # largest
        self.alphabet = alphabet.add_control_tokens(['mask'])

    #TODO switch to numpy ops?
    def _transform(self, src: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # generate mask
            mask = torch.rand(src, 4, requires_grad=False)
            tgt = src.detach().clone()
            # mask in place
            tgt.masked_fill_((mask < self.mask_cutoff), self.alphabet['mask'])
            # random shift by at least 1 to avoid mapping char back to itself
            random_shift = torch.randint_like(src,
                self.dataset.alphabet.n_char - 1, requires_grad=False)
            random_chars = torch.remainder(src + random_shift + 1,
                                    self.dataset.alphabet.n_char)
            # randomize in place
            tgt.masked_scatter_((mask < self.random_cutoff), random_chars)
            # fill no loss positions with `none` token, these will be ignored
            tgt.masked_fill_((mask > self.keep_cutoff), self.alphabet['none'])
        return src.contiguous(), tgt.contiguous()

    @property
    def no_loss_token(self):
        return self.dataset.alphabet['none']


class PtMask(Task):

    def __init__(self,
        hparams: PtMaskHparams,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__(hparams, train_dataset, valid_dataset, test_dataset,
                        encoder, decoder)
        mask = MaskSequence(
            self.hparams.keep_prop,
            self.hparams.mask_prop,
            self.hparams.random_prop)
        self.train_dataset.transform.append_transforms(mask)
        self.valid_dataset.transform.append_transforms(mask)
        self.test_dataset.transform.append_transforms(mask)

    def configure_loss_fn(self) -> nn.Module:
        # remove loss where tgt class is no_loss_token
        return nn.CrossEntropyLoss(ignore_index=self.dataset.alphabet['none'])
