from seqmodel.task import Task
from seqmodel.dataset import SupervisedDataset


class MatFileDataset(SupervisedDataset):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--mat_file', default='data/train.mat', type=str,
                            help='path to matlab file containing training data')
        return parser

    def dataloader(type: str = 'train'):
        """Returns dataloader for train/valid/test split.

        Args:
            type (str): one of 'train', 'valid', or 'test'.

        Returns:
            torch.utils.DataLoader: data loader object
        """
        return None #TODO

    @property
    def source_dims(self) -> int:
        return len(self.alphabet)  #TODO

    @property
    def target_dims(self) -> int:
        return self.n_class  #TODO


class FtDeepSEA(Task):

    @staticmethod
    def _default_hparams(parser):
        #TODO
        return parser

    def __init__(self):
        pass #TODO
