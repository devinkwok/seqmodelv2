from seqmodel.dataset import Dataset


class StridedSeqSampler(Dataset):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--seq_path', default='data/seq', type=str,
                            help='path to sequence files')
        parser.add_argument('--train_intervals', default=None, type=str,
                            help='path to interval files for training split, use all sequences if None.')
        parser.add_argument('--valid_intervals', default=None, type=str,
                            help='path to interval files for validation split, use all sequences if None.')
        parser.add_argument('--test_intervals', default=None, type=str,
                            help='path to interval files for test split, use all sequences if None.')
        return parser

    def dataloader(type: str = 'train'):
        return None #TODO
