import abc
import torch
import typing
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from seqmodel import Hparams


class Dataset(Hparams, abc.ABC):

    @staticmethod
    def _default_hparams(parser):
        return parser

    def __init__(self, transform, **hparams):
        """Unsupervised dataset with transforms.

        Args:
            transform (DataTransform): transforms to apply after sampling.
        """
        super().__init__(**hparams)
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


class DataTransform(abc.ABC):
    def __init__(self, indexes: typing.List[int]):
        """Generic transformation applied to sample.

        Args:
            indexes (typing.List[int]): which indexes in args to transform,
                remaining args are returned as is.
        """
        self.indexes = indexes

    def transform(self, *args):
        """Transforms sampled data, note it takes any number of arguments.
        E.g. unsupervised Dataset will give sequence and metadata as args,
        whereas SupervisedDataset will give sequence, target, metadata.
        
        Applies `_transform` to positions in self.indexes
        """
        for i in self.indexes:
            args[i] = self._transform(args[i])
        return args

    @abc.abstractmethod
    def _transform(self, arg: typing.Any):
        return arg
    
    def __call__(self, *args):
        return self.transform(*args)


class Compose(DataTransform):

    def __init__(self, *transforms: typing.List[DataTransform]):
        """Applies multiple DataTransform objects sequentially.

        Args:
            transforms (list[DataTransform]): if empty, functions as identity,
                else applies transforms sequentially in list order.
        """
        self.transforms = []
        self.append_transforms(*transforms)
    
    def append_transforms(self, *transforms: typing.List[DataTransform]):
        """Add transforms to Compose instance. Applies after current transforms.

        Args:
            transforms (list): list of transforms to append.
        """
        for t in transforms:
            if type(t) is Compose:
                self.append_transforms(t.transforms)
            else:
                self.transforms.append(t)

    def transform(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ArrayToTensor(DataTransform):

    def __init__(self, indexes: typing.List[int], dtype: type = torch.float):
        """Transforms arguments to tensors.

        Args:
            indexes (typing.List[int]): which indexes in args to transform,
                remaining args are returned as is.
            dtype (type, optional): type of tensors. Defaults to torch.float.
        """
        super().__init__(indexes)
        self.dtype = dtype

    def _transform(self, array: typing.List[typing.Any]) -> torch.Tensor:
        return torch.tensor(array, dtype=self.dtype)


class DataManager(Hparams, abc.ABC):
    """Factory for DataLoader objects which copies Dataset objects.
    """

    TRAIN = 'train'
    VALIDATE = 'valid'
    TEST = 'test'

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--batch_size', default=16, type=int,
                            help='number of samples in each training minibatch')
        parser.add_argument('--valid_batch_size', default=None, type=int,
                            help='number of samples in each validation minibatch, ' +
                            'set to --batch_size if None')
        parser.add_argument('--test_batch_size', default=None, type=int,
                            help='number of samples in each test minibatch, ' +
                            'set to --batch_size if None')
        return parser

    def __init__(self, dataset_class: type, dataset_hparams: dict, **hparams):
        """
        Args:
            dataset_class (type): dataset object to create.
            dataset_hparams (dict): dataset hyperparameters.
        """
        super().__init__(**hparams)
        self.dataset_class = dataset_class
        self.dataset_hparams = dataset_hparams

    @abc.abstractmethod
    def dataloader(self, *params,
        dataset_hparams: dict = {},
        type: str = TRAIN
    ):
        """Returns dataloader for train/valid/test split.
        Creates new copy of ref_dataset object.

        Args:
            params: any parameters needed to initialize dataset object,
                should be filled by subclass calling super().dataloader.
            dataset_hparams (dict): hparams to override for dataset object.
            type (str): one of 'train', 'valid', or 'test'.

        Returns:
            torch.utils.DataLoader: data loader object
        """
        batch_size = None
        if type == DataManager.VALIDATE:
            batch_size = self.hparams.valid_batch_size
        elif type == DataManager.TEST:
            batch_size = self.hparams.test_batch_size
        if batch_size is None:  # also, if type == 'train'
            batch_size = self.hparams.batch_size

        dataset_hparams = {k: dataset_hparams[k] if k in dataset_hparams else v \
                        for k, v in self.dataset_hparams.items()}
        dataset = self.dataset_class(*params, **dataset_hparams)

        sampler = DistributedSampler(dataset)
        # pytorch_lightning adds shuffle=True/False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=1,  # hardcode number of workers for now
            pin_memory=True,
            drop_last=True)
        return dataloader
