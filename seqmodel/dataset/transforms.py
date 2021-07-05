import abc
import typing
import torch
import pyfaidx
from seqmodel.dataset.seq import Alphabet


class DataTransform(abc.ABC):

    def __init__(self, indexes: typing.Set[int]):
        """Generic transformation applied to sample.

        Args:
            indexes (typing.List[int]): which indexes in args to transform,
                remaining args are returned as is.
        """
        self.indexes = indexes

    @abc.abstractmethod
    def _transform(self, arg: typing.Any):
        return arg

    def transform(self, *args):
        """Transforms sampled data, note it takes any number of arguments.
        E.g. unsupervised Dataset will give sequence and metadata as args,
        whereas SupervisedDataset will give sequence, target, metadata.
        
        Applies `_transform` to positions in self.indexes
        """
        output = []
        for i, arg in enumerate(args):
            if i in self.indexes:
                arg = self._transform(arg)
            if type(arg) == tuple:
                [output.append(a) for a in arg]
            else:
                output.append(arg)
        if len(output) == 1:
            return output[0]
        return tuple(output)

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
                self.append_transforms(*t.transforms)
            elif issubclass(type(t), DataTransform):
                self.transforms.append(t)
            else:
                raise ValueError(f'must append DataTransform type: {type(t)}')

    def _transform(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def transform(self, *args):
        return self._transform(*args)


class Uppercase(DataTransform):
    """Makes all letters uppercase in pyfaidx.Sequence object.
    """
    def _transform(self, sequence: str) -> torch.Tensor:
        sequence.seq = sequence.seq.upper()
        return sequence


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


class SequenceToTensor(DataTransform):

    def __init__(self, indexes: typing.List[int], alphabet: Alphabet):
        super().__init__(indexes)
        self.alphabet = alphabet

    def _transform(self, sequence: str) -> torch.Tensor:
        return torch.tensor(self.alphabet.to_idx(sequence), dtype=torch.long)


class RandomFlip(DataTransform):

    def __init__(self, indexes: typing.List[int],
        reverse_prop: float = 0.5,
        complement_prop: float = 0.5
    ):
        super().__init__(indexes)
        self.reverse_prop = reverse_prop
        self.complement_prop = complement_prop

    def _transform(self, sequence: pyfaidx.Sequence) -> pyfaidx.Sequence:
        if torch.rand(1).item() < self.reverse_prop:
            sequence = sequence.reverse
        if torch.rand(1).item() < self.complement_prop:
            sequence = sequence.complement
        return sequence
