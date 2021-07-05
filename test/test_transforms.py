import unittest
import torch
import numpy as np
import pyfaidx
from numpy.testing import assert_array_almost_equal
from seqmodel.dataset import DnaAlphabet
from seqmodel.dataset import transforms


class TestTransforms(unittest.TestCase):

    def setUp(self) -> None:
        self.seq = self.make_seq()
        self.metadata = list(torch.randint(50, [len(self.seq)]).detach().numpy())
        self.alphabet = DnaAlphabet()
        self.idx_seq = self.alphabet.to_idx(self.seq)

    def make_seq(self, lowercase=False):
        fasta = pyfaidx.Fasta('test/data/seq/test.fasta')
        sequence = fasta[0][:100]
        if lowercase:
            sequence.seq = sequence.seq.lower()
        return sequence

    def test_Uppercase(self):
        lower = self.make_seq(lowercase=True)
        transform = transforms.Uppercase({0})
        self.assertEqual(transform(lower).seq, self.seq.seq)

        args = [self.make_seq(lowercase=True), self.make_seq(lowercase=True)]
        transform = transforms.Uppercase({1})
        arg0, arg1 = transform(*args)
        self.assertEqual(arg0.seq, self.make_seq(lowercase=True).seq)
        self.assertEqual(arg1.seq, self.seq.seq)

        transform = transforms.Uppercase({0, 1})
        arg0, arg1 = transform(*args)
        self.assertEqual(arg0.seq, self.seq.seq)
        self.assertEqual(arg1.seq, self.seq.seq)

    def test_ArrayToTensor(self):
        args = [0.1, 0.2, 0.3], [1, 2, 3], [10, -5, -10]

        transform = transforms.ArrayToTensor({0, 1})
        out = transform(*args)
        self.assertEqual(out[0].dtype, torch.float)
        self.assertEqual(out[1].dtype, torch.float)
        [assert_array_almost_equal(t, i) for t, i in zip(out, args)]

        transform = transforms.ArrayToTensor({1, 2}, torch.long)
        out = transform(*args)
        self.assertEqual(out[1].dtype, torch.long)
        self.assertEqual(out[2].dtype, torch.long)
        [assert_array_almost_equal(t, i) for t, i in zip(out, args)]

    def test_SequenceToTensor(self):
        args = self.metadata, self.seq, self.seq
        transform = transforms.SequenceToTensor({2}, self.alphabet)
        out = transform(*args)
        self.assertEqual(out[0], self.metadata)
        self.assertEqual(out[1], self.seq)
        self.assertEqual(out[2].dtype, torch.long)
        assert_array_almost_equal(out[2].detach().numpy(),
                                self.alphabet.to_idx(self.seq))

    def test_RandomFlip(self):
        args = self.seq, self.seq, self.metadata
        # for complement sequence, subtract indexes from this number:
        max_tk = self.alphabet.n_char - 1

        transform = transforms.RandomFlip({1}, reverse_prop=0., complement_prop=0.)
        out = transform(*args)
        self.assertEqual(out[0], self.seq)
        self.assertEqual(out[1], self.seq)
        self.assertEqual(out[2], self.metadata)

        transform = transforms.RandomFlip({0}, reverse_prop=1., complement_prop=0.)
        out = transform(*args)
        assert_array_almost_equal(np.array(
                                self.alphabet.to_idx(out[0]))[::-1], self.idx_seq)
        self.assertEqual(out[1], self.seq)
        self.assertEqual(out[2], self.metadata)

        transform = transforms.RandomFlip({0, 1}, reverse_prop=0., complement_prop=1.)
        out = transform(*args)
        assert_array_almost_equal(max_tk - np.array(
                                self.alphabet.to_idx(out[0])), self.idx_seq)
        assert_array_almost_equal(max_tk - np.array(
                                self.alphabet.to_idx(out[1])), self.idx_seq)
        self.assertEqual(out[2], self.metadata)

        transform = transforms.RandomFlip({0}, reverse_prop=1., complement_prop=1.)
        out = transform(*args)
        assert_array_almost_equal(max_tk - np.array(
                                self.alphabet.to_idx(out[0]))[::-1], self.idx_seq)
        self.assertEqual(out[1], self.seq)
        self.assertEqual(out[2], self.metadata)

    def test_Compose(self):
        args = self.seq, self.metadata, self.seq

        transform1 = transforms.Compose(
                transforms.Uppercase({0}),
                transforms.ArrayToTensor({1}, torch.long),
                transforms.RandomFlip({2}, complement_prop=1.),
            )
        out1 = transform1(*args)
        self.assertEqual(out1[0], self.seq)
        self.assertEqual(out1[1].dtype, torch.long)
        self.assertEqual(len(out1[2]), len(self.seq))
        assert_array_almost_equal(out1[1].detach().numpy(), self.metadata)
        self.assertNotEqual(out1[2], self.seq)

        transform2 = transforms.Compose(
                transforms.Uppercase({2}),
                transforms.SequenceToTensor({0, 2}, self.alphabet),
            )
        out2 = transform2(*args)
        self.assertEqual(out2[0].dtype, torch.long)
        assert_array_almost_equal(out2[0].detach().numpy(), self.idx_seq)
        self.assertEqual(out2[1], self.metadata)
        assert_array_almost_equal(out2[2].detach().numpy(), self.idx_seq)

        combined = transforms.Compose(transform1, transform2)
        out = combined(*args)
        self.assertEqual(out[0].dtype, torch.long)
        assert_array_almost_equal(out[0].detach().numpy(), self.idx_seq)
        self.assertFalse(torch.all(out[0] == out[2]))
        self.assertEqual(out[1].dtype, torch.long)
        assert_array_almost_equal(out[1].detach().numpy(), self.metadata)
        self.assertEqual(out[2].dtype, torch.long)


if __name__ == '__main__':
    unittest.main()
