import re
import math
import typing
import numpy as np
import torch
from seqmodel.hparam import SeqIntervalDatasetHparams
from seqmodel.dataset.seq import Alphabet
from seqmodel.dataset.seq import Intervals
from seqmodel.dataset.seq import FastaSequence
from seqmodel.dataset.abstract_dataset import MapDataset
from seqmodel.dataset.transforms import DataTransform
from seqmodel.dataset.transforms import Compose
from seqmodel.dataset.transforms import Uppercase
from seqmodel.dataset.transforms import RandomFlip
from seqmodel.dataset.transforms import ArrayToTensor
from seqmodel.dataset.transforms import SequenceToTensor


class SeqIntervalDataset(MapDataset):

    def __init__(self,
        hparams: SeqIntervalDatasetHparams,
        alphabet: Alphabet,
        transform: DataTransform = None,
        override_start_offset: int = None,
    ):
        """Dataset which indexes over sequences at specified intervals.
        Sequences are defined by Sequence and optional Interval objects.

        Args:
            hparams (SeqIntervalDatasetHparams): hyperparameters (tracked, see hparam.py).
            transform (DataTransform): transforms to apply after sequence
                is augmented by random reverse/complement and
                converted to a tuple of (sequence, metadata) tensors.
            seq_source (Sequence): source of sequence data.
            sample_intervals (Intervals): source of interval data.
            override_start_offset (int, optional): how much to move sample start in each interval,
                this is mainly for testing/debugging and overrides random value
                from setting hparam `randomize_start_offsets=True`. If None,
                sets override to 0 or random value depending on
                `randomize_start_offsets`. Defaults to None.
        """
        super().__init__(hparams, transform)

        if self.hparams.skip_len is None:
            self.hparams.skip_len = self.hparams.seq_len
        if self.hparams.min_len is None:
            self.hparams.min_len = self.hparams.seq_len
        offset = override_start_offset
        if offset is None:
            offset = 0
            if self.hparams.randomize_start_offsets:
                offset = torch.randint(self.hparams.skip_len, (1,)).item()
        assert offset < self.hparams.skip_len
        if self.hparams.drop_incomplete:
            start_pos = self.hparams.skip_len
        else:
            start_pos = 0
        self.offset = start_pos - offset

        self.seq_source = FastaSequence(self.hparams.seq_file)
        if self.hparams.intervals is None:
            self.intervals = Intervals.from_fasta_obj(self.seq_source.fasta)
        else:
            self.intervals = Intervals.from_bed_file(self.hparams.intervals)
        self.alphabet = alphabet
        self._verify_intervals_in_seq()
        self.indexes = self._index_intervals()

        self.transform = Compose(
            RandomFlip([0], self.hparams.reverse_prop, self.hparams.complement_prop),
            Uppercase([0]),
            SequenceToTensor([0], self.alphabet),
            ArrayToTensor([1], dtype=torch.long),
            self.transform,
        )

    def _verify_intervals_in_seq(self):
        for name, start, end in self.intervals:
            if not self.seq_source.exists(name, start, end):
                raise ValueError(f'Interval not in seq: {name} {start} {end}')

    def _index_intervals(self):
        # ith interval starts at indexes[i] and ends at indexes[i+1]
        indexes = [0]
        for _, start, end in self.intervals:
            divisible_len = end - start + self.offset \
                            - self.hparams.min_len + self.hparams.skip_len
            # number of indexable samples in interval
            n_samples = divisible_len / self.hparams.skip_len
            if self.hparams.drop_incomplete:
                n_samples = int(math.floor(n_samples))
            else:
                n_samples = int(math.ceil(n_samples))
            # index of end of current interval (exclusive)
            # is also index of start of next interval (inclusive)
            indexes.append(n_samples + indexes[-1])
        return indexes

    def get_sample(self, idx: int) -> typing.Tuple[str, int, int]:
        print(self.indexes, self.intervals)
        # find the nearest start index to idx
        i = np.searchsorted(self.indexes, idx, side='left')
        # use this to retrieve the interval containing idx
        name, start, end = self.intervals[i]
        # also retrieve start index of interval
        interval_idx_start = self.indexes[i]
        idx_to_coord = (idx - interval_idx_start) * self.hparams.skip_len
        sample_start = start + self.offset + idx_to_coord
        sample_end = sample_start + self.hparams.seq_len

        # truncate and pad seq if coords outside of interval
        if sample_start < start:
            pass #TODO pad with empty char
            sample_start = start
        if sample_end > end:
            pass #TODO pad with empty char
            sample_end = end
        seq = self.seq_source.get(name, sample_start, sample_end)
        # apply transforms using superclass
        return super().get_sample(
            seq, self.seq_source.name_to_id(name), sample_start)

    def __len__(self):
        return self.indexes[-1]
