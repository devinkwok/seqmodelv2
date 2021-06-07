import os
import typing
import inspect
import importlib


def path_to_module_name(filepath: os.PathLike) -> str:
    """Converts path to .py file into a python module name for importing.
    Examples: `path_to_module_name('seqmodel/task/', '__init__.py`)` returns `seqmodel.task`,
    and `path_to_module_name('seqmodel/task/', 'ft.py`)` returns `seqmodel.task.ft`.

    Args:
        path (os.PathLike): path of directory containing file
        filename (os.PathLike): name of file including extension `.py`

    Returns:
        str: name of module for python import
    """
    filepath = os.path.normpath(filepath)
    path_names = filepath.split(os.path.sep)
    name, ext = os.path.splitext(path_names.pop())
    if not ext == '.py':
        return None
    if not name == '__init__':
        path_names += [name]
    return '.'.join(path_names)


def find_subclasses(
        super_class: type,
        search_paths: typing.List[os.PathLike],
        exclude: typing.List[type] = [],
    ) -> type:

    checked_modules = exclude

    files = []
    for path in search_paths:
        if os.path.isdir(path):  # if dir, recursively add all files to list
            for d, _, fs in os.walk(path):  # lists files `fs` in each dir `d`
                files += [os.path.join(d, f) for f in fs]
        elif os.path.isfile(path):  # if file, add to list
            files += [path]

    for filepath in files:
        module_name = path_to_module_name(filepath)
        if module_name is None:
            continue  # only import .py files
        module = importlib.import_module(module_name)

        for _, member in inspect.getmembers(module):
            if not inspect.isclass(member):
                continue  # only look at classes...
            if not issubclass(member, super_class):
                continue  # ...that subclass Hparams
            if member in checked_modules:
                continue # ...and hasn't been checked

            checked_modules.append(member)
            yield member
