import os
import abc
import typing
import pyfaidx
import torch
from seqmodel import Hparams
from seqmodel.dataset.abstract_dataset import DataTransform


class Alphabet():

    def __init__(self, tokens: typing.List[str], control_tokens: typing.List[str]):
        """Maps tokens (chars in a string) to positive integer indexes.
    Randomly selectable tokens should be in the range $\{0, 1, \dots, n-1\}$,
    where n is number of tokens. None or empty token should have value n.
    Integers above n are reserved for control tokens.

        Args:
            tokens (typing.List[str]): list of tokens,
                each token should be a single char.
            control_tokens (typing.List[str]): list of control tokens,
                first token should be empty/null/none. If token is more than 1 char,
                first char is used in sequence representations.
        """
        self.tokens = tokens + control_tokens
        self.n_char = len(tokens)
        assert len({*self.tokens}) == len(self.tokens), 'Tokens not unique!'
        self.tokens_to_idx = {t: i for i, t in enumerate(self.tokens)}
    
    def __len__(self):
        return len(self.tokens)

    def add_control_tokens(self, new_tokens: typing.List[str]):
        """Creates a new `Alphabet` with control tokens added to end of list.

        Args:
            new_tokens (typing.List[str]): control tokens to add.

        Returns:
            Alphabet: new copy of this object, with new_tokens appended
        """
        for t in new_tokens:
            assert t not in self.tokens, f'Token {t} already in alphabet'
        tokens = self.tokens[:self.n_char]
        control_tokens = self.tokens[self.n_char:] + new_tokens
        return Alphabet(tokens, control_tokens)

    def to_tokens(self, seq: typing.Iterable[int], single_chars=False) -> typing.List[str]:
        """Converts sequence of token indexes into str tokens.

        Args:
            sequence (typing.Iterable[int]): sequence to convert.
            single_chars (bool): if True, shorten each token to first character only.
                This may cause `to_tokens` to no longer be the inverse of `to_idx`.

        Returns:
            typing.List[str]: list of tokens as strings,
                or string if single_chars==True.
        """
        if single_chars:
            return ''.join([self.tokens[i][:1] for i in seq])
        return [self.tokens[i] for i in seq]

    def to_idx(self, seq: typing.Iterable[str]) -> typing.List[int]:
        """Converts sequence in str or list of str form into token indexes.

        Args:
            sequence (typing.Iterable[str]): sequence to convert.

        Returns:
            typing.List[int]: integer indexes of tokens
        """
        return [self.tokens_to_idx[str(t)] if str(t) in self.tokens_to_idx \
                else self.none_idx for t in seq]

    @property
    def none_token(self) -> int:
        """
        Returns:
            str: name of empty/null/none token
        """
        return self.tokens[self.n_char]

    @property
    def none_idx(self) -> int:
        """
        Returns:
            int: index of empty/null/none token
        """
        return self.n_char


class DnaAlphabet(Alphabet):
    """Standard DNA alphabet (uppercase).
    """
    TOKENS = ['A', 'G', 'C', 'T']
    CONTROL_TOKENS = ['N']

    def __init__(self):
        super().__init__(self.TOKENS, self.CONTROL_TOKENS)


class Sequence(abc.ABC):
    """A collection of sequences such as a FASTA file.
    """
    def __init__(self):
        self.name_ids = {self.name_to_id(name): name \
                        for name in self.names}

    @property
    @abc.abstractmethod
    def names(self) -> typing.List[str]:
        """Lists names of all sequences, ordering is consistent.

        Returns:
            typing.List[str]: list of all names
        """
        return None

    def name_to_id(self, name: str) -> int:
        """Converts sequence name into unique integer identifier.

        Args:
            name (str): name of sequence

        Returns:
            int: unique id for name
        """
        return hash(name)

    def id_to_name(self, id: int) -> str:
        """Retrieves sequence name from integer identifier.
        Inverse of `name_to_id`.

        Args:
            id (int): unique id of sequence name

        Raises:
            ValueError: if id not found among sequence names in this object.

        Returns:
            str: name of sequence with this id
        """
        if id not in self.name_ids:
            raise ValueError(f'No name with id {id} found.')
        return self.name_ids[id]

    @abc.abstractmethod
    def get(self, name: str, coord_start: int, coord_end: int) -> str:
        """Retrieves sequence as string.

        Args:
            name (str): name of sequence
            coord_start (int): zero-indexed start coordinate (inclusive)
            coord_end (int): zero-idnexed end coordinate (exclusive)

        Returns:
            str: string of length (coord_end - coord_start) containing
                sequence.
        """
        return None

    @abc.abstractmethod
    def exists(self, name: str, coord_start: int, coord_end: int) -> str:
        """Checks whether subsequence exists.
        Unlike `get`, this avoids loading into memory.

        Args:
            name (str): name of sequence
            coord_start (int): zero-indexed start coordinate (inclusive)
            coord_end (int): zero-idnexed end coordinate (exclusive)

        Returns:
            bool: True if sequence with name, coord_start, and coord_end exists.
        """
        return None


class FastaSequence(Sequence):

    def __init__(self, file_path: os.PathLike):
        self.fasta = pyfaidx.Fasta(file_path)
        super().__init__()

    @property
    def names(self) -> typing.Iterable[str]:
        return self.fasta.keys()

    def get(self, name, coord_start: int, coord_end: int) -> str:
        return self.fasta[name][coord_start:coord_end]

    def exists(self, name, coord_start: int, coord_end: int) -> str:
        return name in self.fasta and \
            coord_start > 0 and \
            coord_end < len(self.fasta[name])


class Intervals():
    """A list of contiguous subsequences with name, start, end coordinates.
    Extend to include other data.
    """
    def __init__(self,
        names: typing.List[str] = [],
        starts: typing.List[int] = [],
        ends: typing.List[int] = []
    ):
        assert len(names) == len(starts) and len(starts) == len(ends), \
            'Number of names, starts, ends differ.'
        self.names = names
        self.starts = starts
        self.ends = ends

    @classmethod
    def from_fasta_obj(self, fasta_obj: pyfaidx.Fasta):
        intervals = Intervals()
        for key, record in fasta_obj.items():
            intervals.append(key, 0, len(record))
        return intervals

    @classmethod
    def from_bed_file(self, path: os.PathLike):
        intervals = Intervals()
        with open(path) as f:
            # skip header lines starting with 'browser', 'track', or '#'
            line = '#'
            while line.startswith('browser') or \
                    line.startswith('track') or \
                    line.startswith('#'):
                line = f.readline()
            while line:  # to end of file
                row = line.split()
                # take first 3 columns of each line as name, start, end
                intervals.append(row[0], row[1], row[2])
        return intervals

    def append(self, name: str, start: int, end: int):
        assert end > start, 'Interval length less than 1.'
        self.names.append(name)
        self.starts.append(start)
        self.ends.append(end)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> typing.Tuple:
        return self.names[i], self.starts[i], self.ends[i]


class Uppercase(DataTransform):

    def _transform(self, sequence: str) -> torch.Tensor:
        return str.upper(sequence)


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
        if torch.rand(1).item() > self.reverse_prop:
            sequence = sequence.reverse
        if torch.rand(1).item() > self.complement_prop:
            sequence = sequence.complement
        return sequence


class Variant():
    pass  #TODO
