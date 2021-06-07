import unittest
import shlex
import warnings
import argparse
from seqmodel import hparam
from test import find_subclasses


class TestHparams(unittest.TestCase):

    def test_hparams_unique(self):
        """Add all hparams from all objects in seqmodel into ArgumentParser.
        Check for correct default_hparam definitions and no namespace duplicates.
        Also check that every hparam has a help string.
        Note: this assumes each subclass of Hparams has a unique name!
        """
        parser = argparse.ArgumentParser()
        # recursively check source root dir, excluding base class Hparams
        for member in find_subclasses(hparam.Hparams, ['seqmodel/'], [hparam.Hparams]):
            parser = member._default_hparams(parser)
            # check that the parser was returned
            if not isinstance(parser, argparse.ArgumentParser):
                raise ValueError(f'`{member.__name__}._default_hparams(parser)` should' +
                    f' return type `ArgumentParser`, instead got `{parser}`.')
            # check that the class generates a valid set of default hparams
            new_parser = member.default_hparams()
            self.assertIsInstance(new_parser, argparse.ArgumentParser)
            hparams = hparam.default_to_dict(new_parser)
            self.assertIsInstance(hparams, dict)
            if len(hparams) == 0:
                warnings.warn(f'`{member.__name__}` class extends `Hparams` but has no default hparams.')

        # check every hparam for help string
        for a in parser._actions:
            if a.help is None or a.help == '':
                raise ValueError(f'Help string for hparam `{a.dest}` empty or not defined.')

    class TestClass(hparam.Hparams):
        def _default_hparams(parser):
            parser.add_argument('--a', default=0, type=int, help='a')
            parser.add_argument('--b', default='test', type=str, help='b')
            return parser

    class TestSubClassA(TestClass):
        def __init__(self, param_A, **hparams):
            super().__init__(**hparams)
            self.param_A = param_A

        def _default_hparams(parser):
            parser.add_argument('--c', default=1, type=int, help='c')
            return parser

    class TestSubClassB(TestSubClassA):
        def __init__(self, param_A, param_B, **hparams):
            super().__init__(param_A, **hparams)
            self.param_B = param_B

        def _default_hparams(parser):
            parser.add_argument('--d', default=2, type=int, help='d')
            return parser

    def test_parse_dict(self):
        parser = argparse.ArgumentParser()
        parser = self.TestClass._default_hparams(parser)
        # get defaults as dict
        hparams = hparam.default_to_dict(parser)
        self.assertDictEqual(hparams, {'a': 0, 'b': 'test'})
        # parse correct args
        hparams = hparam.parse_dict({'a': -5, 'b': 'test2'}, parser)
        self.assertDictEqual(hparams, {'a': -5, 'b': 'test2'})
        # parse subset of hparams
        hparams = hparam.parse_dict({'a': -5}, parser)
        self.assertDictEqual(hparams, {'a': -5, 'b': 'test'})
        # parse wrong type
        with self.assertRaises(SystemExit):
            hparam.parse_dict({'a': 'string'}, parser)
        with self.assertRaises(SystemExit):
            hparam.parse_dict({'c': -4}, parser)
        # parse known hparams only
        hparams = hparam.parse_known_dict({'c': 'string'}, parser)
        self.assertDictEqual(hparams, {'a': 0, 'b': 'test'})

    def test_default_hparams(self):
        # subclass inherits hparams
        parser = self.TestSubClassA.default_hparams()
        hparams = hparam.default_to_dict(parser)
        self.assertDictEqual(hparams, {'a': 0, 'b': 'test', 'c': 1})

        # sub-subclass inherits from subclass
        parser = self.TestSubClassB.default_hparams()
        hparams = hparam.default_to_dict(parser)
        self.assertDictEqual(hparams, {'a': 0, 'b': 'test', 'c': 1, 'd': 2})

        class TestSubClassC(self.TestClass):
            def _default_hparams(parser):
                parser.add_argument('--e', default=3, type=int, help='e')
                return parser
        # another subclass inherits independently
        parser = TestSubClassC.default_hparams()
        hparams = hparam.default_to_dict(parser)
        self.assertDictEqual(hparams, {'a': 0, 'b': 'test', 'e': 3})

        class TestClassExtra(hparam.Hparams):
            def _default_hparams(parser):
                parser.add_argument('--e', default=3, type=int, help='f')
                return parser
        # parser can string together independent subclasses
        parser = self.TestClass.default_hparams()
        parser = TestClassExtra.default_hparams(parser)
        hparams = hparam.default_to_dict(parser)
        self.assertDictEqual(hparams, {'a': 0, 'b': 'test', 'e': 3})

        # cannot combine non-unique hparams
        with self.assertRaises(argparse.ArgumentError):
            parser = TestSubClassC.default_hparams()
            parser = TestClassExtra.default_hparams(parser)

        class TestClassOverwrite(self.TestClass):
            def _default_hparams(parser):
                parser.add_argument('--a', default=4, type=int, help='f')
                return parser
        # cannot redefine superclass hparam
        with self.assertRaises(argparse.ArgumentError):
            TestClassOverwrite.default_hparams()

    def test_changed_hparams(self):
        parser = self.TestClass.default_hparams()
        # compare no hparams
        non_default = hparam.changed_hparams({}, parser)
        self.assertDictEqual(non_default, {})
        # compare unchanged hparams
        non_default = hparam.changed_hparams(hparam.default_to_dict(parser), parser)
        self.assertDictEqual(non_default, {})
        # compare partially changed hparams
        non_default = hparam.changed_hparams({'a': 1, 'b': 'test'}, parser)
        self.assertDictEqual(non_default, {'a': 1})
        # omit some hparams
        non_default = hparam.changed_hparams({'a': 1}, parser)
        self.assertDictEqual(non_default, {'a': 1})
        # compare all changed hparams
        non_default = hparam.changed_hparams({'a': 1, 'b': 'test2'}, parser)
        self.assertDictEqual(non_default, {'a': 1, 'b': 'test2'})
        # compare undefined hparam
        with self.assertRaises(ValueError):
            hparam.changed_hparams({'c': 0}, parser)

    def test_to_args(self):
        parser = self.TestClass.default_hparams()
        # generate arbitrary command
        command = hparam.to_args({'a': 0, 'b': 'test'})
        self.assertEqual(command, '--a=0 --b=test')
        args = parser.parse_args(command.split(' '))
        self.assertDictEqual(vars(args), {'a': 0, 'b': 'test'})
        # generate command equivalent to defaults
        command = hparam.to_args({'a': 0, 'b': 'test'}, default_hparams=parser)
        self.assertEqual(command, '')
        args = parser.parse_args([])
        self.assertDictEqual(vars(args), {'a': 0, 'b': 'test'})
        # include defaults in command
        command = hparam.to_args({'a': 0, 'b': 'test'}, default_hparams=parser, include_default=True)
        self.assertEqual(command, '--a=0 --b=test')
        args = parser.parse_args(command.split(' '))
        self.assertDictEqual(vars(args), {'a': 0, 'b': 'test'})
        # override some defaults, and include spaces in command string arg
        parser = self.TestSubClassA.default_hparams()
        command = hparam.to_args({'b': 'test 2', 'c': 10}, default_hparams=parser)
        self.assertEqual(command, '--b=\'test 2\' --c=10')
        args = parser.parse_args(shlex.split(command))
        self.assertDictEqual(vars(args), {'a': 0, 'b': 'test 2', 'c': 10})

    def assertHasAttr(self, obj: any, attributes: dict):
        self.assertTrue(hasattr(obj, 'hparams'))
        for k, v in attributes.items():
            self.assertEqual(getattr(obj.hparams, k), v)
        self.assertDictEqual(obj.hparams, attributes)

    def test_init(self):
        # initialize with defaults
        default_hparams = hparam.default_to_dict(self.TestClass.default_hparams())
        obj = self.TestClass(**default_hparams)
        self.assertHasAttr(obj, default_hparams)
        # ignore undefined hparam
        obj = self.TestClass(x='test')
        self.assertHasAttr(obj, default_hparams)
        # initialize with other values
        obj = self.TestClass(a=1, b='test 2', x='test')
        self.assertHasAttr(obj, {'a': 1, 'b': 'test 2',})
        # handle some parameters differently
        hparams = {'a': 0, 'b': 'test 2'}
        obj = self.TestSubClassA('param A value', **hparams)
        self.assertHasAttr(obj, {'a': 0, 'b': 'test 2', 'c': 1})
        self.assertEqual(obj.param_A, 'param A value')
        # wrong type
        hparams = {'a': 'string', 'b': 'test'}
        with self.assertRaises(SystemExit):
            self.TestClass(**hparams)
        # make sure inheritance is set up correctly
        hparams = {'a': 0, 'b': 'test 2', 'x': 'test'}
        obj = self.TestSubClassB('param A value', 'param B value', **hparams)
        self.assertHasAttr(obj, {'a': 0, 'b': 'test 2', 'c': 1, 'd': 2})
        self.assertEqual(obj.param_A, 'param A value')
        self.assertEqual(obj.param_B, 'param B value')
