import unittest
import shlex
import warnings
import argparse
from seqmodel.hparam import HparamCollection, Hparams
from seqmodel.util import find_subclasses


class TestHparams(unittest.TestCase):

    def test_hparams_unique(self):
        """Add all hparams from all objects in seqmodel into ArgumentParser.
        Check for correct default_hparam definitions and no namespace duplicates.
        Also check that every hparam has a help string.
        Note: this assumes each subclass of Hparams has a unique name!
        """
        all_hparams, all_shorthands = set(), set()
        # recursively check source root dir, excluding base class Hparams
        members = find_subclasses(Hparams, ['seqmodel/'], [Hparams, HparamCollection])
        for member in members:
            hparams = member._default_hparams()
            if len(hparams) == 0:
                warnings.warn(f'`{member.__name__}` class has no default hparams.')
            for name, v in hparams.items():
                self.assertEqual(len(v), 4,
                    f'Missing value, shorthand, type, or help for hparam {name}')
                value, shorthand, t, help = v
                self.assertIsInstance(name, str, f'Wrong type for {name}')
                self.assertIsInstance(shorthand, str, f'Wrong type for {name}: {shorthand}')
                self.assertIsInstance(t, type, f'Wrong type for {name}: {t}')
                self.assertIsInstance(help, str, f'Wrong type for {name}: {help}')
                self.assertFalse(help is None or help == '',
                    f'Help string for hparam `{name}` missing.')
                self.assertTrue(value is None or isinstance(value, t),
                    f'Default {value} for hparam {name} has type {type(value)}, ' + \
                    f'should be {type}')
                # check for unique name and shorthand
                self.assertFalse(name in all_hparams,
                    f'Duplicate hparam {name} in {member.__name__}.')
                self.assertFalse(shorthand in all_shorthands,
                    f'Duplicate shorthand {shorthand} for hparam {name}.')
                all_hparams.add(name)
                all_shorthands.add(shorthand)
            # check that ArgumentParser works on defaults
            _ = member.parse()

    class TestClass(Hparams):
        def _default_hparams():
            return {'a': (0, 'a', int, 'a'),
                    'b': (True, 'b', bool, 'b'),
                    's': ('test', 's', str, 's'),}
    class TestSubClassA(TestClass):
        def _default_hparams():
            return {'c': (1, 'c', int, 'c')}
    class TestSubClassB(TestSubClassA):
        def _default_hparams():
            return {'d': (2, 'd', int, 'd')}
    class TestSubClassC(TestClass):
        def _default_hparams():
            return {'e': (3, 'e', int, 'e')}
    class TestClassOverwrite(TestClass):
        def _default_hparams():
            return {'a': (4, 'a', int, 'f')}
    class TestClassExtra(Hparams):
        def _default_hparams():
            return {'e': (3, 'e', int, 'f')}

    def test_default_hparams(self):
        # subclass inherits hparams
        hparams = self.TestSubClassA()
        self.assertDictEqual(hparams, {'a': 0, 'b': True, 's': 'test', 'c': 1})
        # sub-subclass inherits from subclass
        hparams = self.TestSubClassB()
        self.assertDictEqual(hparams, {'a': 0, 'b': True, 's': 'test', 'c': 1, 'd': 2})
        # another subclass inherits independently
        hparams = self.TestSubClassC()
        self.assertDictEqual(hparams, {'a': 0, 'b': True, 's': 'test', 'e': 3})
        # cannot redefine superclass hparam
        with self.assertRaises(ValueError):
            self.TestClassOverwrite()

    def test_init(self):
        # get defaults
        hparams = self.TestClass()
        self.assertDictEqual(hparams, {'a': 0, 'b': True, 's': 'test'})
        # correct args
        args = {'a': -5, 'b': False, 's': 'test2'}
        hparams = self.TestClass(**args)
        self.assertDictEqual(hparams, args)
        # subset of hparams
        hparams = self.TestClass(**{'b': False})
        self.assertDictEqual(hparams, {'a': 0, 'b': False, 's': 'test'})
        # wrong type
        with self.assertRaises(SystemExit):
            self.TestClass(**{'a': 'string'})
        # ignore undefined hparam
        hparams = self.TestClass(**{'c': 0})
        self.assertDictEqual(hparams, {'a': 0, 'b': True, 's': 'test'})

    def test_to_parser(self):
        # defaults
        parser = self.TestClass.to_parser()
        hparams = self.TestClass()
        args = hparams.to_args(include_default=True)
        self.assertDictEqual(vars(parser.parse_args(shlex.split(args))), hparams)
        # no args is equal to defaults
        args = hparams.to_parser().parse_args([])
        self.assertDictEqual(vars(args), {'a': 0, 'b': True, 's': 'test'})
        # parser can string together independent subclasses
        parser = self.TestClass.to_parser()
        parser = self.TestClassExtra.to_parser(parser)
        hparams = vars(parser.parse_args([]))
        self.assertDictEqual(hparams, {'a': 0, 'b': True, 's': 'test', 'e': 3})
        # cannot combine non-unique hparams
        with self.assertRaises(argparse.ArgumentError):
            parser = self.TestSubClassC.to_parser()
            parser = self.TestClassExtra.to_parser(parser)

    def test_changed_hparams(self):
        # compare no hparams
        hparams = self.TestClass()
        self.assertDictEqual(hparams.changed_hparams(), {})
        # compare unchanged hparams
        hparams = self.TestClass(**{'a': 0, 'b': True, 's': 'test'})
        self.assertDictEqual(hparams.changed_hparams(), {})
        # compare partially changed hparams
        hparams = self.TestClass(**{'a': 1, 'b': True, 's': 'test', })
        self.assertDictEqual(hparams.changed_hparams(), {'a': 1})
        # omit some hparams
        hparams = self.TestClass(**{'a': 1})
        self.assertDictEqual(hparams.changed_hparams(), {'a': 1})
        # compare all changed hparams
        hparams = self.TestClass(**{'a': 1, 'b': False, 's': 'test2', })
        self.assertDictEqual(hparams.changed_hparams(),
                            {'a': 1, 'b': False, 's': 'test2', })

    def test_to_args(self):
        hparams = self.TestClass()
        # generate command equivalent to defaults
        command = hparams.to_args()
        self.assertEqual(command, '')
        # include defaults in command
        command = hparams.to_args(include_default=True)
        self.assertEqual(command, '--a=0 --b=True --s=test')
        args = hparams.to_parser().parse_args(command.split(' '))
        self.assertDictEqual(vars(args), {'a': 0, 'b': True, 's': 'test'})
        # override some defaults, and include spaces in command string arg
        hparams = self.TestSubClassA(**{'c': 10, 'b': False, 's': 'test 2'})
        command = hparams.to_args()
        self.assertEqual(command, '--c=10 --b=False --s=\'test 2\'')
        args = hparams.to_parser().parse_args(shlex.split(command))
        self.assertDictEqual(vars(args), {'a': 0, 'b': False, 's': 'test 2', 'c': 10})


if __name__ == '__main__':
    unittest.main()
