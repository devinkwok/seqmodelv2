import unittest
import os
import shutil
from seqmodel import job
from seqmodel import hparam
from seqmodel.util import find_subclasses


def identical_change(param):
    if type(param) == int:
        return 123  # arbitrary int that doesn't appear in defaults
    if type(param) == float:
        return 123.  # arbitrary non-default float
    if type(param) == bool:
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
    if type(param) == bool:
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

    TEST_DATA_SOURCE = 'test/data'
    TEST_DATA = 'test/out'

    def setUp(self) -> None:
        self.job_obj = job.ShellJob(job.LocalInterface(), job_out_dir=self.TEST_DATA)
        if os.path.exists(self.TEST_DATA):
            shutil.rmtree(self.TEST_DATA)
        shutil.copytree(self.TEST_DATA_SOURCE, self.TEST_DATA)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(self.TEST_DATA):
            shutil.rmtree(self.TEST_DATA)
        return super().tearDown()

    def test_hparams_to_canonical_path(self):
        for collection in find_subclasses(
                        hparam.HparamCollection, ['seqmodel/hparam.py'],
                        exclude=[hparam.HparamCollection]):
            self.collection_gives_unique_paths(collection)

    def collection_gives_unique_paths(self, hparam_collection: hparam.HparamCollection):
        """Check every registered hparam in `hparam.py`
        to see if a unique representation is returned by
        `hparams_to_canonical_path`.
        """
        default_hparams = {}

        for module in hparam_collection.hparam_list():
            default_hparams = {**default_hparams, **module()}
        # create test cases
        # base dicts to generate cases from: defaults, with variable changes, with identical changes
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

        paths = {}
        # check each case for unique path
        for key, version in hparam_versions.items():
            hparams = hparam_collection(**version)
            canonical_path = self.job_obj.hparams_to_canonical_path(hparams)
            for s in canonical_path.split(self.job_obj.os.sep):
                self.assert_valid_path_component(s)
            self.assertFalse(canonical_path in paths, f'Path identical for key `' + \
                f'`{key}`:\n{canonical_path}\nand\n{version}')
            paths[canonical_path] = version

    def assert_valid_path_component(self, component: str):
        """
        According to:
        https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
        `os.stat()` will raise an exception other than FileNotFoundError
        for invalid path components. Since this is a unit test,
        assume there is no adversarial actor hence running in working dir
        is not an issue.

        Args:
            component (str): dir or file name to check for validity
        """
        # # check that component is non-empty
        self.assertNotEqual(component, '')
        # check that component cannot be split into several dirs
        self.assertTrue(os.sep not in component)
        try:
            os.stat(component)
        except FileNotFoundError:
            pass  # this is fine, only checking that the component is valid

    def test_replicates(self):
        # look in test/data/replicates
        path = self.job_obj.os.join(self.TEST_DATA, 'replicates')
        # ignore replicate 0 (index from 1), replicates 2-3 do not exist
        self.assert_replicate_list_equal(*self.job_obj.replicates(path),
            ['01', '4', '05', '06', '99', '107', '108'])
        # replicate 1 is empty
        self.assert_replicate_list_equal(
            *self.job_obj.replicates(path, include={'empty'}), ['01'])
        # replicate 4 exists but is not run
        self.assert_replicate_list_equal(
            *self.job_obj.replicates(path, include={'created'}), ['4'])
        # replicate 5 started
        self.assert_replicate_list_equal(
            *self.job_obj.replicates(path, include={'started'}), ['05'])
        # replicate 6 is running
        self.assert_replicate_list_equal(
            *self.job_obj.replicates(path, include={'running'}), ['06'])
        # replicate 99 completed successfully
        self.assert_replicate_list_equal(
            *self.job_obj.replicates(path, include={'complete'}), ['99'])
        # replicate 107 terminated with error
        self.assert_replicate_list_equal(
            *self.job_obj.replicates(path, include={'error'}), ['107'])
        # replicate 108 terminated with timeout
        self.assert_replicate_list_equal(
            *self.job_obj.replicates(path, include={'timeout'}), ['108'])

    def assert_replicate_list_equal(self, replicates, paths, target):
        self.assertEqual(len(replicates), len(target))
        self.assertEqual(len(paths), len(target))
        for dir in target:
            self.assertTrue(int(dir) in replicates)
            self.assertTrue((dir + self.job_obj.os.sep) in paths)

    def test_new_replicate(self):
        # dir without replicates, next replicate is 1
        next_replicate = self.job_obj.new_replicate('test/out/no-replicates')
        self.assertEqual(next_replicate, self.job_obj.os.join(
            self.TEST_DATA, 'no-replicates', '01'))
        # dir with replicates, next replicate is 109
        next_replicate = self.job_obj.new_replicate('test/out/replicates')
        self.assertEqual(next_replicate, self.job_obj.os.join(
            self.TEST_DATA, 'replicates', '109'))

    def test_list_checkpoints_by_iter(self): 
        ckpts = self.job_obj.list_checkpoints_by_iter(
            self.job_obj.os.join(self.TEST_DATA, 'replicates'))
        self.assert_ckpt_list_equal(ckpts, [
            ('test/out/replicates/05/checkpoints/epoch=0-step=1000.ckpt', 0, 1000),
            ('test/out/replicates/05/checkpoints/N-Step-Checkpoint_0_20000.ckpt', 0, 20000),
            ('test/out/replicates/4/checkpoints/epoch=0-step=30000.ckpt', 0, 30000),
            ('test/out/replicates/05/checkpoints/N-Step-Checkpoint_1_10000.ckpt', 1, 10000)
            ])
        ckpts = self.job_obj.list_checkpoints_by_iter(
            self.job_obj.os.join(self.TEST_DATA, 'replicates', '05'))
        self.assert_ckpt_list_equal(ckpts, [
            ('test/out/replicates/05/checkpoints/epoch=0-step=1000.ckpt', 0, 1000),
            ('test/out/replicates/05/checkpoints/N-Step-Checkpoint_0_20000.ckpt', 0, 20000),
            ('test/out/replicates/05/checkpoints/N-Step-Checkpoint_1_10000.ckpt', 1, 10000)
            ])
        ckpts = self.job_obj.list_checkpoints_by_iter(
            self.job_obj.os.join(self.TEST_DATA, 'no-replicates'))
        self.assert_ckpt_list_equal(ckpts, [])

    def assert_ckpt_list_equal(self, ckpts, target):
        self.assertEqual(len(ckpts), len(target))
        for ckpt, tgt in zip(ckpts, target):
            self.assertEqual(len(ckpt), len(tgt))
            for i, j in zip(ckpt, tgt):
                self.assertEqual(i, j)

    def test_replace_latest_ckpt_paths(self):
        hparams = {
            'a': self.job_obj.LATEST_CKPT_SHORTHAND,
            'b': 'test/out/replicates/4/' + self.job_obj.LATEST_CKPT_SHORTHAND,
            'c': 'test/out/no-replicates/' + self.job_obj.LATEST_CKPT_SHORTHAND,
            'd': self.job_obj.LATEST_CKPT_SHORTHAND + '/test',
            'e': 'test/out/no-replicates/',
            'f': 10,
        }
        tgt_hparams = {
            'b': 'test/out/replicates/4/checkpoints/epoch=0-step=30000.ckpt',
            'd': self.job_obj.LATEST_CKPT_SHORTHAND + '/test',
            'e': 'test/out/no-replicates/',
            'f': 10,
        }
        # test each hparam individually
        for k, v in hparams.items():
            hp = {k: v}
            new_hp = self.job_obj._replace_latest_ckpt_paths(hp)
            if k in tgt_hparams:
                self.assertDictEqual(new_hp, {k: tgt_hparams[k]})
            else:
                self.assertDictEqual(new_hp, {})
        # test all hparams together
        new_hp = self.job_obj._replace_latest_ckpt_paths(hparams)
        self.assertDictEqual(new_hp, tgt_hparams)


if __name__ == '__main__':
    unittest.main()
