import unittest
import os
import argparse
from seqmodel import job
from seqmodel import hparam
from test import find_subclasses


def identical_change(param):
    if type(param) == int:
        return 123  # arbitrary int that doesn't appear in defaults
    if type(param) == float:
        return 123.  # arbitrary non-default float
    if type(param) is bool:
        return not param
    return '123'

def variable_change(param):
    if param is None:
        return '1'
    if type(param) == int:
        return param + 1
    if type(param) == float:
        return (param + 1) * 0.1
    if type(param) == str:
        return param + ' CHANGE'
    if type(param) is bool:
        return not param

def apply_all(hparams: dict, change_function):
    return {k: change_function(v) for k, v in hparams.items()}

def remove_nones(hparams: dict):
    output_dict = {}
    for k, v in hparams.items():
        if v is not None:
            output_dict[k] = v
    return output_dict

def merge_one(hparams: dict, change_source: dict, merge_bool=True):
    for k, v in hparams.items():
        changed_hparams = dict(hparams)
        changed_hparams[k] = change_source[k]
        if merge_bool or type(v) is not bool:
            yield k, remove_nones(changed_hparams)


class TestJob(unittest.TestCase):

    def test_hparams_to_canonical_str(self):
        """Check every registered hparam in `dataset`, `model`, `task`, and `run.py`
        to see if a unique representation is returned by `hparams_to_canonical_str`.

        Raises:
            ValueError: if duplicate paths are found
        """
        paths = {}
        parser = argparse.ArgumentParser()
        for module in find_subclasses(hparam.Hparams, search_paths=[
            'seqmodel/dataset/', 'seqmodel/model/', 'seqmodel/task/', 'seqmodel/run.py'],
            exclude=[hparam.Hparams]):
            parser = module._default_hparams(parser)

        # create test cases
        # base dicts to generate cases from: defaults, with variable changes, with identical changes
        default_hparams = hparam.default_to_dict(parser)
        hparams_changed = apply_all(default_hparams, variable_change)
        hparams_identical = apply_all(default_hparams, identical_change)
        # cases: one non-default only
        version_dc = {'dc ' + k: v for k, v in merge_one(default_hparams, hparams_changed)}
        # cases: all but one non-default
        version_cd = {'cd ' + k: v for k, v in merge_one(hparams_changed, default_hparams)}
        # cases: one identically non-default
        version_di = {'di ' + k: v for k, v in merge_one(default_hparams, hparams_identical, 
                    merge_bool=False)}  # exclude bools as they have same value as hparams_changed
        # cases: all identically non-default, except one
        version_id = {'id ' + k: v for k, v in merge_one(hparams_identical, default_hparams)}
        # cases: none changed, all changed, all identically changed
        version_other = {
                'd_ ': remove_nones(default_hparams),
                'c_ ': remove_nones(hparams_changed),
                'i_ ': remove_nones(hparams_identical),
            }
        hparam_versions = {**version_dc, **version_cd, **version_di, **version_id, **version_other}

        # check each case for unique path
        for key, version in hparam_versions.items():
            strings = job.hparams_to_canonical_str(version)
            [self.is_valid_path_component(s) for s in strings]
            canonical_path = os.path.join(*strings)
            if canonical_path in paths:
                raise ValueError(f'Path identical for key `{key}`:\n{canonical_path}\n' +
                                f'{paths[canonical_path]}\nand\n{version}')
            paths[canonical_path] = version

    def is_valid_path_component(self, component: str):
        # # check that component is non-empty
        # self.assertNotEqual(component, '')
        # check that component cannot be split into several dirs
        self.assertTrue(os.sep not in component)
        # https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
        # according to the link above, `os.stat()` will raise an exception
        # other than FileNotFoundError for invalid path components
        # since this is a unit test, assume there is no adversarial actor
        # hence running in working dir is not an issue
        try:
            os.stat(component)
        except FileNotFoundError:
            pass  # this is fine, only checking that the component is valid
