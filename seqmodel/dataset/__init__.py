import abc
from torch.utils.data.dataloader import DataLoader
from seqmodel.hparam import Hparams


class Dataset(Hparams, abc.ABC):

    @staticmethod
    def _default_hparams(parser):
        return parser

    @abc.abstractmethod
    def dataloader(type: str = 'train') -> DataLoader:
        """Returns dataloader for train/valid/test split.

        Args:
            type (str): one of 'train', 'valid', or 'test'.

        Returns:
            torch.utils.DataLoader: data loader object
        """
        return None #TODO
