import abc
from argparse import ArgumentParser


class Hparams(abc.ABC):
    """
    Abstract class for objects which register hyperparameters.
    Default registered hyperparameters are stored as an `ArgumentParser`.
    Hyperparameters can be passed as `dict` and compared against defaults
    by calling `parse_dict` and `changed_hparams`.
    Hyperparameters are required for object instantiation and stored in `self.hparams`.
    """

    @abc.abstractstaticmethod
    def _default_hparams(parser: ArgumentParser)-> ArgumentParser:
        """Defines default hparam values by adding them to parser.
        Do not call this to get defaults, instead call `default_hparams` which
        includes defaults from superclass.

        Args:
            parser (ArgumentParser): parser object.

        Returns:
            ArgumentParser: parser with default values of registered hparams.
        """
        return None

    @staticmethod
    def parse_dict(hparams: dict, default_hparams: ArgumentParser)-> dict:
        """Parses `hparams` as if it were command line arguments.

        Args:
            default_hparams (ArgumentParser): containing default hparams.
            hparams (dict): to compare against `default_hparams`.

        Returns:
            dict: contains `hparams` combined with `default_hparams`
        """
        return None #TODO

    @staticmethod
    def changed_hparams(hparams: dict, default_hparams: ArgumentParser)-> dict:
        """Returns items in `hparams` which differ from defaults in `default_hparams`.

        Args:
            default_hparams (ArgumentParser): containing default hparams.
            hparams (dict): to compare against default_hparams.

        Returns:
            dict[str, Any]: items in `hparams` with keys in `default_hparams`,
                whose values are different.
        """
        return None #TODO

    @staticmethod
    def to_args(hparams: dict, default_hparams: ArgumentParser, include_default=False):
        """Converts hparams to command line flags.

        Args:
            default_hparams (ArgumentParser): containing default hparams.
            hparams (dict): to compare against default_hparams.
            include_default (bool): if False, apply changed_hparams() first to exclude defaults

        Returns:
            str: command line flags.
        """
        return None #TODO

    @classmethod
    def default_hparams(cls, parser: ArgumentParser):
        """Adds _default_hparams from all superclasses of cls that subclass Hparams.

        Args:
            parser (ArgumentParser): parser object.

        Returns:
            ArgumentParser: parser with default values of registered hparams.
        """
        return None #TODO

    def __init__(self, **hparams):
        """Adds hparams to self.hparams (same as pytorch-lightning).

        Args:
            hparams (dict): dict of hparams
        """
        pass #TODO
