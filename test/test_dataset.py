import unittest
from pyfaidx import Fasta
from seqmodel import dataset
from seqmodel.dataset import transforms


class TestSequence(unittest.TestCase):

    def setUp(self):
        self.fasta = 'test/data/seq/test.fasta'

    def test_Alphabet(self):
        tokens = ['a', 'b', 'mask']
        control_tokens = ['none', 'test']
        alphabet = dataset.Alphabet(tokens, control_tokens)
        self.assertEqual(len(alphabet), len(tokens) + len(control_tokens))
        self.assertEqual(alphabet.n_char, len(tokens))
        self.assertEqual(alphabet.none_token, control_tokens[0])

        tokens = ['a', 'a', 'b', 'x', 'mask', 'unknown']
        tgt = ['a', 'a', 'b', 'none', 'mask', 'none']
        ints = [0, 0, 1, 3, 2, 3]
        token2int = alphabet.to_idx(tokens)
        self.assertListEqual(token2int, ints)
        self.assertListEqual(alphabet.to_tokens(token2int), tgt)
        int2token = alphabet.to_tokens(ints)
        self.assertListEqual(int2token, tgt)
        self.assertListEqual(alphabet.to_idx(int2token), ints)

        token_str = alphabet.to_tokens(ints, single_chars=True)
        self.assertEqual(token_str, 'aabnmn')

    def test_FastaSequence(self):
        seq = dataset.FastaSequence(self.fasta)
        names = ['CIZ78533', 'CCZ78532', 'CFZ78531', 'CMZ78530', 'CLZ78529']
        self.assertListEqual(list(seq.names), names)
        self.assertTrue(seq.exists(names[0], 10, 20))
        self.assertFalse(seq.exists(names[0], 1000, 2000))
        self.assertFalse(seq.exists(names[0], 0, 1000))
        self.assertFalse(seq.exists(names[0], -10, 20))
        self.assertFalse(seq.exists('test', 0, 100))

        ids = [seq.name_to_id(n) for n in names]
        ids2names = [seq.id_to_name(i) for i in ids]
        self.assertListEqual(ids2names, names)
        with self.assertRaises(ValueError):
            seq.id_to_name(123456)

        subseq = seq.get(names[0], 70, 180)
        self.assertEqual(len(subseq), 180 - 70)
        self.assertEqual(subseq,
            'AATCCGGAGGACCGGTGTACTCAGCTCACCGGGGGCATTGCTCCCGTGGTGACCC' + \
            'TGATTTGTTGTTGGGCCGCCTCGGGAGCGTCCATGGCGGGTTTGAACCTCTAGCC')

        alphabet = dataset.DnaAlphabet()
        self.assertEqual(alphabet.to_idx(subseq[:10]),
                        [0, 0, 3, 2, 2, 1, 1, 0, 1, 1])

    def test_Intervals(self):
        fasta = Fasta(self.fasta)
        intervals = dataset.Intervals.from_fasta_obj(fasta)
        self.assertEqual(len(intervals), len(fasta.keys()))
        for interval, (k, v) in zip(intervals, fasta.items()):
            name, start, end = interval
            self.assertEqual(name, k)
            self.assertEqual(start, 0)
            self.assertEqual(end, len(v))

        with self.assertRaises(ValueError):
            intervals = dataset.Intervals.from_bed_file('test/data/seq/invalid.bed')
        intervals = dataset.Intervals.from_bed_file('test/data/seq/test.bed')
        ref_intervals = [('CIZ78533', 70, 180), ('CCZ78532', 0, 10), ('CCZ78532', 700, 763)]
        for interval, ref in zip(intervals, ref_intervals):
            name, start, end = interval
            self.assertEqual(name, ref[0])
            self.assertEqual(start, ref[1])
            self.assertEqual(end, ref[2])

    def test_SeqIntervalDataset(self):
        intervals = dataset.Intervals.from_bed_file('test/data/seq/out-of-bounds.bed')
        #TODO


if __name__ == '__main__':
    unittest.main()
