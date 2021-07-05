import abc
import torch
import typing
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from seqmodel.hparam import DatasetHparams
from seqmodel.dataset.transforms import Compose


class Dataset(abc.ABC):

    def __init__(self, hparams: DatasetHparams, transform: Compose = None):
        """Unsupervised dataset with transforms.

        Args:
            hparams (DatasetHparams): hyperparameters (tracked, see hparam.py).
            transform (DataTransform, optional):
                transforms to apply after sampling.
        """
        self.hparams = hparams
        if transform is None:
            transform = Compose()
        self.transform = transform

    @abc.abstractmethod
    def get_sample(self, *args
    )-> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Function signature for retrieving sample using optional arguments
        such as index position.
        This allows either map-style or iter-style implementation.
        Applies transform to sample.

        Returns:
            typing.tuple[torch.Tensor, torch.Tensor]: sample tensor
                of type torch.long and dimension `S` (seq len),
                associated metadata as tensor.
        """
        return self.transform(*args)

    def dataloader(self) -> DataLoader:
        """Returns dataloader for train/valid/test split.
        Creates new copy of self, incorporat.

        Args:
            params: any parameters needed to initialize dataset object,
                should be filled by subclass calling super().dataloader.
            dataset_hparams (dict): hparams to override for dataset object.
            type (str): one of 'train', 'valid', or 'test'.

        Returns:
            torch.utils.DataLoader: data loader object
        """
        # sampler = DistributedSampler(self)  #TODO gpu > 1
        # pytorch_lightning adds shuffle=True/False
        dataloader = DataLoader(
            self,
            batch_size=self.hparams.batch_size,
            # sampler=sampler,  #TODO gpu > 1
            num_workers=1,  # hardcode number of workers for now
            pin_memory=True,
            drop_last=True)
        return dataloader


class MapDataset(Dataset, torch.utils.data.Dataset, abc.ABC):
    """Uses index to retrieve samples, extends `torch.utils.data.Dataset`.
    """
    def __getitem__(self, index):
        return self.get_sample(index)

    @abc.abstractmethod
    def __len__(self):
        return None


class IterableDataset(Dataset, torch.utils.data.IterableDataset, abc.ABC):
    """Retrieves samples sequentially, extends `torch.utils.data.IterableDataset`.
    """
    def __iter__(self):
        return self.get_sample()


class SupervisedDataset(Dataset, abc.ABC):
    """Supervised version of Dataset.
    """
    @property
    @abc.abstractmethod
    def target_dims(self) -> int:
        """Number of dimensions per target position.
        """
        return None

    @abc.abstractmethod
    def get_sample(self, *args
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            typing.tuple[torch.Tensor, torch.Tensor, torch.Tensor]: sample tensor
                of type torch.long and dimension `S` (seq len),
                target tensor of dimension `S, T` where `T` is `target_dims`,
                and associated metadata.
        """
        return super().get_sample(*args)
