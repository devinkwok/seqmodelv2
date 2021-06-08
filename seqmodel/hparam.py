import abc
import shlex
import typing
from argparse import ArgumentParser, ArgumentTypeError


# type replacing bool for argparse, see below link for justification:
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def _to_args(hparams: dict) -> str:
    """Converts hparams to command line arg string. Doesn't check for valid values.

    Args:
        hparams (dict): hparams

    Returns:
        str: commands
    """
    commands = [f'--{k}={shlex.quote(str(v))}' for k, v in hparams.items()]
    return ' '.join(commands)

def parse_dict(hparams: dict, default_hparams: ArgumentParser) -> dict:
    """Parses `hparams` as if it were command line arguments.
    Returns default values only if `hparams={}`.

    Args:
        hparams (dict): to compare against `default_hparams`.
            If `hparams` is None or empty, returns default values only.
        default_hparams (ArgumentParser): containing default hparams.

    Returns:
        dict: contains `hparams` combined with `default_hparams`
    """
    commands = _to_args(hparams)
    args = default_hparams.parse_args(shlex.split(commands))
    return vars(args)

def parse_known_dict(hparams: dict, default_hparams: ArgumentParser) -> dict:
    """Parses `hparams` but ignoring keys not in default_hparams.

    Args:
        hparams (dict): to compare against `default_hparams`.
            If `hparams` is None or empty, returns default values only.
        default_hparams (ArgumentParser): containing default hparams.

    Returns:
        dict: contains `hparams` combined with `default_hparams`
    """
    known_hparams = {}
    for k in default_to_dict(default_hparams).keys():
        if k in hparams:
            known_hparams[k] = hparams[k]
    return parse_dict(known_hparams, default_hparams)

def default_to_dict(default_hparams: ArgumentParser) -> dict:
    """Calls `parse_dict` where `hparams` is not set.

    Args:
        default_hparams (ArgumentParser): containing default hparams.

    Returns:
        dict: contains `hparams` combined with `default_hparams`
    """
    return parse_dict({}, default_hparams)

def changed_hparams(hparams: dict, default_hparams: ArgumentParser) -> dict:
    """Returns items in `hparams` which differ from defaults in `default_hparams`.
    Ignores items which are not present in `hparams` but exist in `default_hparams`.

    Args:
        default_hparams (ArgumentParser): containing default hparams.
        hparams (dict): to compare against default_hparams.

    Returns:
        dict[str, Any]: items in `hparams` with keys in `default_hparams`,
            whose values are different.
    """
    defaults = default_to_dict(default_hparams)
    changed = {}
    for k, v in hparams.items():
        if k not in defaults:
            raise ValueError(f'Undefined hparam {k}: {v} not in {defaults}')
        if defaults[k] != v:
            changed[k] = v
    return changed

def to_args(hparams: dict, default_hparams: ArgumentParser = None, include_default: bool = False) -> str:
    """Converts hparams to command line flags.

    Args:
        default_hparams (ArgumentParser): containing default hparams.
        hparams (dict): to compare against default_hparams.
        include_default (bool): if False, apply changed_hparams() first to exclude defaults

    Returns:
        str: command line flags.
    """
    if default_hparams is not None:
        if include_default:
            hparams = parse_dict(hparams, default_hparams)
        else:
            hparams = changed_hparams(hparams, default_hparams)
    return _to_args(hparams)


class Hparams(abc.ABC):
    """
    Abstract class for objects which register hyperparameters.
    Default registered hyperparameters are stored as an `ArgumentParser`.
    Hyperparameters can be passed as `dict` and compared against defaults
    by calling `parse_dict` and `changed_hparams`.
    Hyperparameters are required for object instantiation and stored in `self.hparams`.
    """

    @abc.abstractstaticmethod
    def _default_hparams(parser: ArgumentParser) -> ArgumentParser:
        """Defines default hparam values by adding them to parser.
        Do not call this to get defaults, instead call `default_hparams` which
        includes defaults from superclass.

        Args:
            parser (ArgumentParser): parser object.

        Returns:
            ArgumentParser: parser with default values of registered hparams.
        """
        return None

    @classmethod
    def default_hparams(cls, parser: ArgumentParser = None) -> ArgumentParser:
        """Adds _default_hparams from all superclasses of cls that subclass Hparams.

        Args:
            parser (ArgumentParser): parser object. If None, creates a new ArgumentParser.

        Returns:
            ArgumentParser: parser with default values of registered hparams.
        """
        if parser is None:
            parser = ArgumentParser()
        for super_cls in cls.mro():
            if issubclass(super_cls, Hparams) and super_cls != Hparams:
                parser = super_cls._default_hparams(parser)
        return parser

    def __init__(self, **hparams):
        """Adds hparams to self.hparams (similar to pytorch-lightning).
        However, only adds keys defined in `default_hparams` and also parses `hparams`
        as a dict. Keys not in `default_hparams` are ignored

        Args:
            hparams (dict): dict of hparams
        """
        known_hparams = parse_known_dict(hparams, self.default_hparams())
        self.hparams = AttributeDict(known_hparams)


class AttributeDict(typing.Dict):
    """Identical to pytorch_lightning.utilities.parsing.AttributeDict,
    but copied in order to avoid importing pytorch_lightning when defining jobs.

    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/parsing.py

    Copyright The PyTorch Lightning team.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        if not len(self):
            return ""
        max_key_length = max([len(str(k)) for k in self])
        tmp_name = '{:' + str(max_key_length + 3) + 's} {}'
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = '\n'.join(rows)
        return out
