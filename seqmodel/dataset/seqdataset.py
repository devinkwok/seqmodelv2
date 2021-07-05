import re
import math
import typing
import numpy as np
import torch
from seqmodel import Hparams
from seqmodel.dataset.seq import Sequence
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

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--seq_file', default='data/seq', type=str,
                            help='path to sequence file')
        parser.add_argument('--intervals', default=None, type=str,
                            help='path to interval files for training split, use all sequences if None.')
        parser.add_argument('--seq_len', default=2000, type=str,
                            help='length of sampled sequence')
        parser.add_argument('--skip_len', default=None, type=str,
                            help='how many characters to skip before next sample, ' + \
                                'defaults to seq_len')
        parser.add_argument('--min_len', default=None, type=str,
                            help='smallest interval length to sample from, ' + \
                                'defaults to seq_len')
        parser.add_argument('--randomize_start_offsets', default=True, type=Hparams.str2bool,
                            help='move sampling start position by random number ' + \
                                'less than seq_len')
        parser.add_argument('--drop_incomplete', default=True, type=Hparams.str2bool,
                            help='remove first and/or last samples with length ' + \
                                'less than seq_len, if False and start_offset > 0, ' + \
                                'this will pad the start of the first sample.')
        parser.add_argument('--reverse_prop', default=0.5, type=float,
                            help='proportion of samples to reverse.')
        parser.add_argument('--complement_prop', default=0.5, type=float,
                            help='proportion of samples to complement.')
        return parser

    def __init__(self,
        transform: DataTransform,
        seq_source: Sequence,
        sample_intervals: Intervals,
        override_start_offset: int = None,
        **hparams
    ):
        """Dataset which indexes over sequences at specified intervals.
        Sequences are defined by Sequence and optional Interval objects.

        Args:
            seq_source (Sequence): source of sequence data.
            sample_intervals (Intervals): source of interval data.
            override_start_offset (int, optional): how much to move sample start in each interval,
                this is mainly for testing/debugging and overrides random value
                from setting hparam `randomize_start_offsets=True`. If None,
                sets override to 0 or random value depending on
                `randomize_start_offsets`. Defaults to None.
        """
        transforms = Compose([
            RandomFlip([0], self.hparams.reverse_prop, self.hparams.complement_prop),
            Uppercase([0]),
            SequenceToTensor([0], self.alphabet),
            ArrayToTensor([1], dtype=torch.Long),
            transform,
        ])
        super().__init__(transforms, **hparams)

        if self.hparams.skip_len is None:
            self.hparams.skip_len = self.hparams.seq_len
        if self.hparams.min_len is None:
            self.hparams.min_len = self.hparams.seq_len
        offset = override_start_offset
        if offset is None:
            offset = 0
            if self.hparams.randomize_start_offsets:
                offset = torch.randint(self.hparams.skip_len).item()
        assert offset < self.hparams.skip_len
        if self.hparams.drop_incomplete:
            start_pos = self.hparams.skip_len
        else:
            start_pos = 0
        self.offset = start_pos - offset

        self.seq_source = seq_source
        self.alphabet = seq_source.alphabet
        self.intervals = sample_intervals
        self._verify_intervals_in_seq()
        self.indexes = self._index_intervals()

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
            n_samples = divisible_len / self.skip_len
            if self.hparams.drop_incomplete:
                n_samples = int(math.floor(n_samples))
            else:
                n_samples = int(math.ceil(n_samples))
            # index of end of current interval (exclusive)
            # is also index of start of next interval (inclusive)
            indexes.append(n_samples + indexes[-1])
        return indexes

    def get_sample(self, idx: int) -> typing.Tuple[str, int, int]:
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
            pass #TODO
            sample_start = start
        if sample_end > end:
            pass #TODO
            sample_end = end
        seq = self.interval.get(sample_start, sample_end)
        # apply transforms using superclass
        return super().get_sample(
            seq, self.seq_source.name_to_id(name), sample_start)

    def __len__(self):
        return self.indexes[-1]

    def dataloader(self, dataset_hparams: dict = {}):
        interval_file = self.hparams.intervals
        source = FastaSequence(self.hparams.seq_file)
        if interval_file is None:
            intervals = Intervals.from_fasta_obj(source.fasta)
        else:
            intervals = Intervals.from_bed_file(interval_file)

        return super().dataloader(self.transforms, source, intervals,
                        dataset_hparams=dataset_hparams, type=type)
