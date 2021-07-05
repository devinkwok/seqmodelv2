from seqmodel.task import Task
from seqmodel.hparam import MatFileDatasetHparams
from seqmodel.dataset import SupervisedDataset


class MatFileDataset(SupervisedDataset):

    def __init__(self, hparams: MatFileDatasetHparams):
        self.hparams = hparams

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


class FtDeepSea(Task):

    def __init__(self):
        pass #TODO
