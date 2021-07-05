import typing
import torch
from torch import random
import torch.nn as nn
from seqmodel.hparam import PtMaskHparams
from seqmodel.dataset import Dataset
from seqmodel.dataset.transforms import DataTransform
from seqmodel.task.abstract_task import Task


class MaskSequence(DataTransform):

    def __init__(self, keep_prop, mask_prop, random_prop, alphabet):
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
        self.mask_cutoff = self.random_cutoff + mask_prop
        self.keep_cutoff = self.mask_cutoff + keep_prop # largest
        self.alphabet = alphabet.add_control_tokens(['mask'])

    #TODO switch to numpy ops?
    def _transform(self, src: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # generate mask
            mask = torch.rand_like(src, dtype=torch.float, requires_grad=False)
            tgt = src.detach().clone()
            # mask in place
            tgt.masked_fill_((mask < self.mask_cutoff), self.alphabet.tokens_to_idx['mask'])
            # random shift by at least 1 to avoid mapping char back to itself
            random_shift = torch.randint_like(src,
                self.alphabet.n_char - 1, requires_grad=False)
            random_chars = torch.remainder(src + random_shift + 1,
                                    self.alphabet.n_char)
            # randomize in place
            tgt.masked_scatter_((mask < self.random_cutoff), random_chars)
            # fill no loss positions with `none` token, these will be ignored
            tgt.masked_fill_((mask > self.keep_cutoff), self.alphabet.none_idx)
        return src.contiguous(), tgt.contiguous()

    @property
    def no_loss_token(self):
        return self.alphabet.none_idx


class PtMask(Task):

    def __init__(self,
        hparams: PtMaskHparams,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        self.mask = MaskSequence(
            hparams.keep_prop,
            hparams.mask_prop,
            hparams.random_prop,
            train_dataset.alphabet)
        super().__init__(hparams, train_dataset, valid_dataset, test_dataset,
                        encoder, decoder)
        self.train_dataset.transform.append_transforms(self.mask)
        self.valid_dataset.transform.append_transforms(self.mask)
        self.test_dataset.transform.append_transforms(self.mask)

    def configure_loss_fn(self) -> nn.Module:
        # remove loss where tgt class is no_loss_token
        return nn.CrossEntropyLoss(ignore_index=self.mask.no_loss_token)
