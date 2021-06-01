import unittest
import os
import inspect
import importlib
from seqmodel.hparam import Hparams
import argparse


def path_to_module_name(path: os.PathLike, filename: os.PathLike)-> str:
    """Converts path to .py file into a python module name for importing.
    Examples: `path_to_module_name('seqmodel/task/', '__init__.py`)` returns `seqmodel.task`,
    and `path_to_module_name('seqmodel/task/', 'ft.py`)` returns `seqmodel.task.ft`.

    Args:
        path (os.PathLike): path of directory containing file
        filename (os.PathLike): name of file including extension `.py`

    Returns:
        str: name of module for python import
    """
    filepath = os.path.normpath(os.path.join(path, filename))
    path_names = filepath.split(os.sep)
    name, ext = os.path.splitext(path_names.pop())
    if not ext == '.py':
        return None
    if not name == '__init__':
        path_names += [name]
    return '.'.join(path_names)


class TestHparams(unittest.TestCase):


    def test_hparams_unique(self):
        """Add all hparams from all objects in seqmodel into ArgumentParser.
        Check for correct default_hparam definitions and no namespace duplicates.
        Also check that every hparam has a help string.
        Note: this assumes each subclass of Hparams has a unique name!
        """
        parser = argparse.ArgumentParser()
        checked_member_names = ['Hparams']  # don't check the abstract base class Hparams

        for path, _, files in os.walk('.'):  # recursively check root dir

            for file in files:
                module_name = path_to_module_name(path, file)
                if module_name is None:
                    continue  # only import .py files
                module = importlib.import_module(module_name)

                for member_name, member in inspect.getmembers(module):
                    if not inspect.isclass(member):
                        continue  # only look at classes...
                    if not issubclass(member, Hparams):
                        continue  # ...that subclass Hparams
                    if member_name in checked_member_names:
                        continue # ...and hasn't been checked

                    checked_member_names.append(member_name)
                    parser = member._default_hparams(parser)

                    # check that the parser was returned
                    if not isinstance(parser, argparse.ArgumentParser):
                        raise ValueError(f'`{member_name}._default_hparams(parser)` should' +
                            f' return type `ArgumentParser`, instead got `{parser}`.')

        # check every hparam for help string
        for a in parser._actions:
            if a.help is None or a.help == '':
                raise ValueError(f'Help string for hparam `{a.dest}` empty or not defined.')
