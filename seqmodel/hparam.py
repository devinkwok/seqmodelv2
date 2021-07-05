import abc
import shlex
import typing
from argparse import ArgumentParser
from argparse import ArgumentTypeError


class Hparams(abc.ABC, typing.Dict):
    """
    Abstract class for hyperparameter containing object.
    Default registered hyperparameters are stored 
    Hyperparameters can be instantiated from `dict` or `ArgumentParser`,
    and accessed in a dict-style interface.
    Only hparams in hparam.py are tracked by canonical string/path in job.py.
    """

    def __init__(self, **hparams):
        """Adds hparams to self.hparams (similar to pytorch-lightning).
        However, only adds keys defined in `default_hparams` and also parses `hparams`
        as a dict. Keys not in `default_hparams` are ignored.

        Args:
            hparams (dict): hyperparameters.
        """
        super().__init__()
        known_hparams = self.parse_known(**hparams)
        for k, v in known_hparams.items():
            self[k] = v

    # Number of significant digits to round floats for canonical hparam string.
    # Floats (therefore hparams) are considered equal if they differ by less
    # precision than FLOAT_SIG_DIGITS, making their canonical paths identical.
    FLOAT_SIG_DIGITS = 3

    # type replacing bool for argparse, see below link for justification:
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')

    @staticmethod
    @abc.abstractmethod
    def _default_hparams() -> typing.Dict[str, typing.Tuple[typing.Any, str, type, str]]:
        """Defines default hparams as a dict of tuples including default value,
        type, and help message. Do not call this, use `default_hparams`.
        The shorthand is included in ArgumentParser, and also used to
        generate string representation of hparams.

        Returns:
            dict: of the form
            {hparam_names: (default_value, shorthand, type, help_string)}
        """
        return None

    @classmethod
    def default_hparams(cls) -> typing.Dict[str, typing.Any]:
        """Dict of default_values from all superclasses of cls
        that subclass Hparams. Calls _default_hparams().

        Returns:
            dict: default values including from superclasses, of the form
            {hparam_names: (default_value, shorthand, type, help_string)}
        """
        hparams = {}
        for super_cls in cls.mro():
            if issubclass(super_cls, Hparams) and super_cls != Hparams:
                for k, v in super_cls._default_hparams().items():
                    if k in hparams:
                        raise ValueError(f'Subclass {cls} ' + \
                            f'overriding default value of hparam {k}')
                    else:
                        hparams[k] = v
        return hparams

    @classmethod
    def to_parser(cls, parser: ArgumentParser = None) -> ArgumentParser:
        """Creates ArgumentParser for hparams.

        Args:
            parser (ArgumentParser, optional): Existing parser to add to,
                if None creates new one. Defaults to None.

        Returns:
            ArgumentParser: parser containing hyperparameters.
        """
        if parser is None:
            parser = ArgumentParser()
        for name, (value, shorthand, t, help) in cls.default_hparams().items():
            if t == bool:
                t = Hparams.str2bool
            parser.add_argument(f'-{shorthand}', f'--{name}',
                default=value, type=t, help=help)
        return parser

    @classmethod
    def _parse(cls, ignore_unknown: bool, **hparams) -> dict:
        commands = cls._to_args(**hparams)
        parser = cls.to_parser()
        if ignore_unknown:
            args, _ = parser.parse_known_args(shlex.split(commands))
        else:
            args = parser.parse_args(shlex.split(commands))
        return vars(args)

    @classmethod
    def parse(cls, **hparams) -> dict:
        """Parses `hparams` as if it were command line arguments.
        Returns default values from cls if `hparams={}`.

        Args:
            hparams (dict): to compare against `cls.default_hparams()`.
                If `hparams` is None or empty, returns default values only.

        Returns:
            dict: contains `hparams` combined with `cls.default_hparams()`
        """
        return cls._parse(False, **hparams)

    @classmethod
    def parse_known(cls, **hparams) -> dict:
        """Parses `hparams` but ignoring keys not in default_hparams.

        Args:
            hparams (dict): to compare against `cls.default_hparams()`.
                If `hparams` is None or empty, returns default values only.

        Returns:
            dict: contains `hparams` combined with `cls.default_hparams()`
        """
        return cls._parse(True, **hparams)

    def changed_hparams(self) -> dict:
        """Returns values which differ from `default_hparams`.

        Returns:
            dict[str, Any]: non-default hparam values.
        """
        changed = {}
        defaults = self.default_hparams()
        for k, v in self.items():
            if defaults[k][0] != v:
                changed[k] = v
        return changed

    @staticmethod
    def _to_args(**hparams) -> str:
        commands = []
        for k, v in hparams.items():
            if v is not None:
                commands.append(f'--{k}={shlex.quote(str(v))}')
        return ' '.join(commands)

    def to_args(self, include_default: bool = False) -> str:
        """Converts hparams to command line flags.

        Args:
            include_default (bool): if False, apply changed_hparams to
                exclude hparams with default values. Default is False.

        Returns:
            str: command line flags.
        """
        if include_default:
            return self._to_args(**self)
        else:
            return self._to_args(**self.changed_hparams())

    @classmethod
    def sortable_name(cls):
        return cls.__name__

    @staticmethod
    def _hparam_to_str(key, value):
        if value is None:
            value = 'None'
        elif type(value) == bool:
            value = 'T' if value else 'F'
        elif type(value) == float:
            # use string format code to return scientific notation
            # with FLOAT_SIG_DIGITS - 1 decimals
            value = ("%." + str(Hparams.FLOAT_SIG_DIGITS - 1) + "E") % (value)
        elif type(value) == str:  # no spaces
            value = value.replace(' ', '_')
        return key + '=' + str(value)

    def __str__(self):
        """Outputs canonical string format of hparams. Sorts hparams by name.
        Changes display of some types as follows:
        - None becomes 'None'
        - bool becomes 'T' or 'F'known_hparams
        - float is displayed in scientific notation rounded to FLOAT_SIG_DIGITS - 1

        Returns:
            str: string in the format
                '{shorthand0}={value0},{shorthand1}={value1}, ...',
                if empty (all default values), return sortable_name().
        """
        defaults = self.default_hparams()
        str_repr = []
        changed_hparams = self.changed_hparams()
        keys = sorted(changed_hparams.keys())
        for k in keys:
            shorthand = defaults[k][1]
            value = changed_hparams[k]
            str_repr.append(self._hparam_to_str(shorthand, value))
        if len(str_repr) == 0:
            return self.sortable_name()
        return ','.join(str_repr)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key, val):
        self[key] = val


class HparamCollection(Hparams):
    """An ordered list of Hparams classes and objects.
    """
    def __init__(self, **hparams):
        super(Hparams, self).__init__()
        hparams_dict = self.parse(**hparams)
        for module in self.hparam_list():
            self[module] = module(**hparams_dict)

    @staticmethod
    @abc.abstractmethod
    def _hparam_list() -> typing.List[type]:
        """Define which classes to include in this collection.
        Call hparam_list to get canonical (sorted) version.

        Returns:
            typing.List[type]: list of classes inheriting from Hparams.
        """
        return None

    @classmethod
    def hparam_list(cls) -> typing.List[type]:
        """Canonical (sorted) list of hparam classes.
        Combines `_hparam_list()` from superclasses.

        Returns:
            typing.List[type]: sorted list of classes inheriting from Hparams.
        """
        hparams = []
        for super_cls in cls.mro():
            if issubclass(super_cls, HparamCollection) \
                    and super_cls != HparamCollection:
                hparams += super_cls._hparam_list()
        return sorted(hparams, key=lambda h: h.sortable_name())

    @classmethod
    def default_hparams(cls) -> typing.Dict[str, typing.Any]:
        hparams = {}
        for h in cls.hparam_list():
            hparams = {**hparams, **h.default_hparams()}
        return hparams

    @classmethod
    def to_parser(cls, parser: ArgumentParser = None) -> ArgumentParser:
        for h in cls.hparam_list():
            parser = h.to_parser(parser)
        return parser

    def changed_hparams(self) -> dict:
        hparams = {}
        for h in self.hparams_obj:
            hparams = {**hparams, **h.changed_hparams()}
        return hparams

    def to_args(self, include_default: bool = False) -> str:
        args = [h.to_args(include_default) for h in self.hparam_list()]
        return ' '.join(args)

    def __str__(self):
        """Converts dict of hyperparameters into ordered list of strings.
        Only non-default hyperparameters are recorded.
        Hparams are arranged by sorted_name order, and separated by ' '.

        Returns:
            str: canonical string form of hparams in canonical order.
        """
        output = []
        for _, hparam_obj in self.items():
            output.append(str(hparam_obj))
        return ' '.join(output)


class DatasetHparams(Hparams):
    def _default_hparams():
        return {
            'batch_size': (16, 'bs', int,
                'number of samples in each training minibatch'),
        }

class SeqIntervalDatasetHparams(DatasetHparams):
    def _default_hparams():
        return {
            'seq_file': ('data/seq', 'seq', str,
                'path to sequence file'),
            'intervals': (None, 'int', str,
                'path to interval files for training split, use all sequences if None.'),
            'seq_len': (2000, 'len', int,
                'length of sampled sequence'),
            'skip_len': (None, 'skip', str,
                'how many characters to skip before next sample, ' + \
                                'defaults to seq_len'),
            'min_len': (None, 'mlen', str,
                'smallest interval length to sample from, ' + \
                                'defaults to seq_len'),
            'randomize_start_offsets': (True, 'randstart', bool,
                'move sampling start position by random number ' + \
                                'less than seq_len'),
            'drop_incomplete': (True, 'noinc', bool,
                'remove first and/or last samples with length ' + \
                                'less than seq_len, if False and start_offset > 0, ' + \
                                'this will pad the start of the first sample.'),
            'reverse_prop': (0.5, 'prev', float,
                'proportion of samples to reverse.'),
            'complement_prop': (0.5, 'pcompl', float,
                'proportion of samples to complement.'),
        }

class LinearDecoderHparams(Hparams):
    def _default_hparams():
        return {
            'decode_dims': (None, 'ddec', int,
                'number of dimensions in intermediate layers, ' + \
                                'if None set to 2*in_dims'),
            'n_decode_layers': (2, 'ndec', int,
                'number of linear layers'),
            'decode_dropout': (0., 'ddrop', float,
                'dropout between linear layers'),
        }

class PositionEncoderHparams(Hparams):
    def _default_hparams():
        return {
            'posencoder_dropout': (0., 'posdrop', float,
                'dropout after positional encoder'),
        }

class TransformerEncoderHparams(Hparams):
    def _default_hparams():
        return {
            'repr_dims': (512, 'd', int,
                'number of dimensions in representation layer'),
            'feedforward_dims': (None, 'dff', int,
                'number of dimensions in feedforward (fully connected) layer, ' +
                                'if None set to 2*repr_dims'),
            'n_heads': (4, 'nh', int,
                'number of attention heads'),
            'n_layers': (4, 'nl', int,
                'number of attention layers'),
            'dropout': (0., 'drop', float,
                'proportion between [0., 1.] of dropout to apply between module layers.'),
        }

class TaskHparams(Hparams):
    def _default_hparams():
        return {
            # batch
            'accumulate_grad_batches': (1, 'bacc', int,
                'average over this many batches before backprop (pytorch_lightning)'),
            # optimizer
            'lr': (3e-4, 'lr', float,
                'learning rate'),
            'adam_beta_1': (0.9, 'b1', float,
                'beta 1 parameter for Adam optimizer'),
            'adam_beta_2': (0.99, 'b2', float,
                'beta 2 parameter for Adam optimizer'),
            'adam_eps': (1e-6, 'eps', float,
                'epsilon parameter for Adam optimizer'),
            'weight_decay': (0.01, 'wd', float,
                'weight decay for Adam optimizer'),
            'gradient_clip_val': (10., 'clip', float,
                'limit max abs gradient value, ' + \
                                'no clipping if 0 (pytorch lightning)'),
        }

class PtMaskHparams(TaskHparams):
    def _default_hparams():
        return {
            'keep_prop': (0.01, 'pk', float,
                'proportion of sequence positions to apply identity loss.'),
            'mask_prop': (0.13, 'pm', float,
                'proportion of sequence positions to mask.'),
            'random_prop': (0.01, 'pr', float,
                'proportion of sequence positions to randomize.'),
        }

class MatFileDatasetHparams(DatasetHparams):
    def _default_hparams():
        return {
            'mat_file': ('data/train.mat', 'mat', str,
                'path to matlab file containing training data'),
        }

class FtDeepSEAHparams(TaskHparams):
    def _default_hparams():
        return {
            #TODO
        }

class InitializerHparams(Hparams):
    def _default_hparams():
        return {
            'init_version': (None, 'ver', str,
                'code version number, increment if hparam functionality changes'),
            'init_task': (None, 'tsk', str,
                '[ptmask, ftdeepsea] objects to load'),
            'init_mode': ('train', 'mode', str,
                '[train, test] whether to run training or inference'),
            'precision': (16, 'pre', int,
                '32 or 16 bit training (pytorch lightning)'),
            'load_encoder_from_checkpoint': (None, 'cpenc', str,
                'path to encoder checkpoint, replaces encoder from ' + \
                'load_from_checkpoint or resume_from_checkpoint'),
        }


class RunHparamCollection(HparamCollection):
    def _hparam_list():
        return [
            InitializerHparams,
            PositionEncoderHparams,
            TransformerEncoderHparams,
            LinearDecoderHparams,
        ]

class PtMaskHparamCollection(RunHparamCollection):
    def _hparam_list():
        return [
            SeqIntervalDatasetHparams,
            PtMaskHparams,
        ]

class FtDeepSEAHparamCollection(RunHparamCollection):
    def _hparam_list():
        return [
            MatFileDatasetHparams,
            FtDeepSEAHparams,
        ]
