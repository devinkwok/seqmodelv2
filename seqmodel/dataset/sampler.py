from seqmodel.dataset.abstract_dataset import UnsupervisedDataset


class StridedSeqSampler(UnsupervisedDataset):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--seq_path', default='data/seq', type=str,
                            help='path to sequence files')
        parser.add_argument('--seq_len', default='data/seq', type=str,
                            help='length of sampled sequence')
        parser.add_argument('--skip_len', default='data/seq', type=str,
                            help='how many positions to skip between samples')
        parser.add_argument('--train_intervals', default=None, type=str,
                            help='path to interval files for training split, use all sequences if None.')
        parser.add_argument('--valid_intervals', default=None, type=str,
                            help='path to interval files for validation split, use all sequences if None.')
        parser.add_argument('--test_intervals', default=None, type=str,
                            help='path to interval files for test split, use all sequences if None.')
        return parser

    def dataloader(type: str = 'train'):
        return None #TODO

    @property
    def source_dims(self) -> int:
        return len(self.alphabet)  #TODO
